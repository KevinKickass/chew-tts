use super::{TalkerDecoderLayer, TalkerLayerKvCache, TalkerLayerScratch};
use crate::{QwenDType, TalkerConfig};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

/// GPU-resident Qwen3-TTS talker transformer.
pub struct TalkerTransformer<T: QwenDType = f16> {
    layers: Vec<TalkerDecoderLayer<T>>,
    final_norm: Option<CudaSlice<T>>,
}

/// Persistent KV caches and scratch buffers for talker generation.
pub struct TalkerGenerationSession<T: QwenDType = f16> {
    caches: Vec<TalkerLayerKvCache>,
    scratch: TalkerLayerScratch<T>,
    max_seq_len: usize,
}

impl<T: QwenDType> TalkerTransformer<T> {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(
                TalkerDecoderLayer::load(model_dir, layer_index, config, stream)
                    .with_context(|| format!("could not load talker layer {layer_index}"))?,
            );
        }
        let (norm_shape, final_norm) = T::load(model_dir, "talker.model.norm.weight", stream)
            .context("could not load talker final norm")?;
        ensure!(
            norm_shape == [config.hidden_size],
            "talker final norm has shape {:?}, expected [{}]",
            norm_shape,
            config.hidden_size
        );
        Ok(Self {
            layers,
            final_norm: Some(final_norm),
        })
    }

    /// Load a standard Qwen2 stack rooted at `prefix`, for example
    /// `model.language_model` or `model.tts_language_model`.
    pub fn load_qwen2(
        model_dir: impl AsRef<Path>,
        prefix: &str,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
        has_final_norm: bool,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(
                TalkerDecoderLayer::load_qwen2_from_prefix(
                    model_dir,
                    &format!("{prefix}.layers.{layer_index}"),
                    config,
                    stream,
                )
                .with_context(|| format!("could not load Qwen2 layer {layer_index}"))?,
            );
        }
        let final_norm = if has_final_norm {
            let name = format!("{prefix}.norm.weight");
            let (norm_shape, final_norm) = T::load(model_dir, &name, stream)
                .with_context(|| format!("could not load {name}"))?;
            ensure!(
                norm_shape == [config.hidden_size],
                "{name} has shape {:?}, expected [{}]",
                norm_shape,
                config.hidden_size
            );
            Some(final_norm)
        } else {
            None
        };
        Ok(Self { layers, final_norm })
    }

    /// Load a MiniCPM4 stack. MiniCPM4 uses bias-free projections and
    /// per-frequency LongRoPE factors, but otherwise shares the same GQA and
    /// SwiGLU geometry as the native Qwen path.
    pub fn load_minicpm(
        model_dir: impl AsRef<Path>,
        prefix: &str,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
        rope_factors: Option<&[f32]>,
        apply_rope: bool,
        causal_attention: bool,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        if let Some(factors) = rope_factors {
            ensure!(
                factors.len() == config.head_dim / 2,
                "MiniCPM LongRoPE has {} factors, expected {}",
                factors.len(),
                config.head_dim / 2
            );
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(
                TalkerDecoderLayer::load_minicpm_from_prefix(
                    model_dir,
                    &format!("{prefix}.layers.{layer_index}"),
                    config,
                    stream,
                    rope_factors,
                    apply_rope,
                    causal_attention,
                )
                .with_context(|| format!("could not load MiniCPM layer {layer_index}"))?,
            );
        }
        let name = format!("{prefix}.norm.weight");
        let (norm_shape, final_norm) =
            T::load(model_dir, &name, stream).with_context(|| format!("could not load {name}"))?;
        ensure!(
            norm_shape == [config.hidden_size],
            "{name} has shape {:?}, expected [{}]",
            norm_shape,
            config.hidden_size
        );
        Ok(Self {
            layers,
            final_norm: Some(final_norm),
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn start_session(
        &self,
        max_seq_len: usize,
        max_batch_tokens: usize,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<TalkerGenerationSession<T>> {
        ensure!(max_seq_len > 0, "maximum sequence length must be non-zero");
        ensure!(
            max_batch_tokens > 0,
            "maximum batch token count must be non-zero"
        );
        let caches = (0..self.layers.len())
            .map(|_| TalkerLayerKvCache::allocate(max_seq_len, config, stream))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(TalkerGenerationSession {
            caches,
            scratch: TalkerLayerScratch::allocate(max_batch_tokens, config, stream)?,
            max_seq_len,
        })
    }

    /// Append prepared embeddings to a persistent generation session.
    pub fn forward_session(
        &self,
        session: &mut TalkerGenerationSession<T>,
        hidden_host: &[f32],
        seq_len: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            seq_len <= session.scratch.max_tokens,
            "sequence has {seq_len} tokens, scratch holds {}",
            session.scratch.max_tokens
        );
        ensure!(
            hidden_host.len() == seq_len * config.hidden_size,
            "hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * config.hidden_size
        );
        let position = session.position();
        ensure!(
            position + seq_len <= session.max_seq_len,
            "session length {} exceeds maximum {}",
            position + seq_len,
            session.max_seq_len
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        for ((layer, cache), expected_position) in self
            .layers
            .iter()
            .zip(&mut session.caches)
            .zip(std::iter::repeat(position))
        {
            ensure!(
                cache.position() == expected_position,
                "talker layer caches are out of sync"
            );
            layer.forward_cached_device(
                &mut hidden,
                seq_len,
                config,
                kernels,
                cache,
                &mut session.scratch,
            )?;
        }
        if let Some(final_norm) = &self.final_norm {
            T::rms_norm_f32in(
                kernels,
                &hidden,
                final_norm,
                &mut session.scratch.norm,
                seq_len as u32,
                config.hidden_size as u32,
                config.rms_norm_eps as f32,
            )?;
            stream.synchronize()?;
            let mut output = vec![T::zero(); seq_len * config.hidden_size];
            stream.memcpy_dtoh(
                &session.scratch.norm.slice(..seq_len * config.hidden_size),
                &mut output,
            )?;
            Ok(output.into_iter().map(T::to_f32).collect())
        } else {
            stream.synchronize()?;
            let mut output = vec![0.0f32; seq_len * config.hidden_size];
            stream.memcpy_dtoh(&hidden, &mut output)?;
            Ok(output)
        }
    }

    /// Correctness-first stack execution from prepared talker embeddings.
    ///
    /// Hidden state and scratch stay on the GPU for the complete stack. This
    /// method creates a fresh cache; a persistent generation session follows.
    pub fn forward_hidden(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        max_seq_len: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            hidden_host.len() == seq_len * config.hidden_size,
            "hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * config.hidden_size
        );
        ensure!(
            max_seq_len >= seq_len,
            "maximum sequence length {max_seq_len} is below prompt length {seq_len}"
        );

        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = TalkerLayerScratch::allocate(seq_len, config, &stream)?;
        let mut caches = (0..self.layers.len())
            .map(|_| TalkerLayerKvCache::allocate(max_seq_len, config, &stream))
            .collect::<anyhow::Result<Vec<_>>>()?;

        for (layer, cache) in self.layers.iter().zip(&mut caches) {
            layer.forward_cached_device(
                &mut hidden,
                seq_len,
                config,
                kernels,
                cache,
                &mut scratch,
            )?;
        }
        if let Some(final_norm) = &self.final_norm {
            T::rms_norm_f32in(
                kernels,
                &hidden,
                final_norm,
                &mut scratch.norm,
                seq_len as u32,
                config.hidden_size as u32,
                config.rms_norm_eps as f32,
            )?;
            stream.synchronize()?;
            let mut output_native = vec![T::zero(); seq_len * config.hidden_size];
            stream.memcpy_dtoh(
                &scratch.norm.slice(..seq_len * config.hidden_size),
                &mut output_native,
            )?;
            Ok(output_native.into_iter().map(T::to_f32).collect())
        } else {
            stream.synchronize()?;
            let mut output = vec![0.0f32; seq_len * config.hidden_size];
            stream.memcpy_dtoh(&hidden, &mut output)?;
            Ok(output)
        }
    }

    pub fn forward_hidden_batched_full(
        &self,
        hidden_host: &[f32],
        tokens_per_batch: usize,
        batches: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            tokens_per_batch > 0 && batches > 0,
            "empty transformer batch"
        );
        let total_rows = tokens_per_batch * batches;
        ensure!(
            hidden_host.len() == total_rows * config.hidden_size,
            "batched hidden input geometry disagrees"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = TalkerLayerScratch::allocate(total_rows, config, &stream)?;
        let mut caches = (0..self.layers.len())
            .map(|_| TalkerLayerKvCache::allocate(total_rows, config, &stream))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for (layer, cache) in self.layers.iter().zip(&mut caches) {
            layer.forward_batched_full_device(
                &mut hidden,
                total_rows,
                batches,
                config,
                kernels,
                cache,
                &mut scratch,
            )?;
        }
        if let Some(final_norm) = &self.final_norm {
            T::rms_norm_f32in(
                kernels,
                &hidden,
                final_norm,
                &mut scratch.norm,
                total_rows as u32,
                config.hidden_size as u32,
                config.rms_norm_eps as f32,
            )?;
            stream.synchronize()?;
            let mut output = vec![T::zero(); total_rows * config.hidden_size];
            stream.memcpy_dtoh(&scratch.norm, &mut output)?;
            Ok(output.into_iter().map(T::to_f32).collect())
        } else {
            stream.synchronize()?;
            let mut output = vec![0.0f32; total_rows * config.hidden_size];
            stream.memcpy_dtoh(&hidden, &mut output)?;
            Ok(output)
        }
    }
}

impl<T: QwenDType> TalkerGenerationSession<T> {
    pub fn position(&self) -> usize {
        self.caches.first().map_or(0, TalkerLayerKvCache::position)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Seed a session from a safe, host-side Hugging Face KV snapshot.
    ///
    /// Source tensors use `[heads, tokens, head_dim]` ordering and BF16
    /// storage. Attention kernels use position-major FP16 caches internally,
    /// so conversion and the small layout transpose happen once per request.
    pub fn load_prompt_kv<'a>(
        &mut self,
        layers: impl IntoIterator<Item = (&'a [half::bf16], &'a [half::bf16])>,
        tokens: usize,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<()> {
        let layers = layers.into_iter().collect::<Vec<_>>();
        ensure!(
            layers.len() == self.caches.len(),
            "prompt has {} KV layers, session expects {}",
            layers.len(),
            self.caches.len()
        );
        for (cache, (key, value)) in self.caches.iter_mut().zip(layers) {
            cache.load_prompt_bf16(
                key,
                value,
                tokens,
                config.num_key_value_heads,
                config.head_dim,
                stream,
            )?;
        }
        Ok(())
    }
}
