use crate::cuda::{TalkerDecoderLayer, TalkerLayerKvCache, TalkerLayerScratch};
use crate::{CodePredictorConfig, QwenDType, TalkerConfig};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The five-layer multi-codebook decoder used after every talker step.
pub struct CodePredictorTransformer<T: QwenDType = f16> {
    layers: Vec<TalkerDecoderLayer<T>>,
    final_norm: CudaSlice<T>,
    talker_codec_embedding: CudaSlice<T>,
    codec_embeddings: Vec<CudaSlice<T>>,
    projection_weight: Option<CudaSlice<T>>,
    projection_bias: Option<CudaSlice<T>>,
    lm_heads: Vec<CudaSlice<T>>,
    geometry: TalkerConfig,
    codec_embed_dim: usize,
}

/// Reusable GPU allocations for one code-predictor worker.
pub struct CodePredictorGenerationSession<T: QwenDType = f16> {
    caches: Vec<TalkerLayerKvCache>,
    scratch: TalkerLayerScratch<T>,
    talker_hidden: CudaSlice<T>,
    semantic_id: CudaSlice<i32>,
    semantic_embed: CudaSlice<T>,
    predictor_input: CudaSlice<T>,
    projected: CudaSlice<T>,
    hidden: CudaSlice<f32>,
    norm_token: CudaSlice<T>,
    logits: CudaSlice<T>,
    logits_f16: CudaSlice<f16>,
    current_code: CudaSlice<i32>,
    all_codes: CudaSlice<i32>,
    code_embed: CudaSlice<T>,
    embedding_sum: CudaSlice<f32>,
}

impl<T: QwenDType> CodePredictorTransformer<T> {
    pub fn load(
        model_dir: impl AsRef<Path>,
        talker_config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = &talker_config.code_predictor_config;
        let geometry = predictor_geometry(config);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("talker.code_predictor.model.layers.{layer_index}");
            layers.push(
                TalkerDecoderLayer::load_from_prefix(model_dir, &prefix, &geometry, stream)
                    .with_context(|| {
                        format!("could not load code predictor layer {layer_index}")
                    })?,
            );
        }
        let (norm_shape, final_norm) =
            T::load(model_dir, "talker.code_predictor.model.norm.weight", stream)
                .context("could not load code predictor final norm")?;
        ensure!(
            norm_shape == [config.hidden_size],
            "code predictor final norm has shape {:?}, expected [{}]",
            norm_shape,
            config.hidden_size
        );
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<T>> {
            let (shape, tensor) = T::load(model_dir, name, stream)?;
            ensure!(
                shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                shape
            );
            Ok(tensor)
        };
        let codec_embed_dim = talker_config.hidden_size;
        let acoustic_groups = config.num_code_groups - 1;
        let talker_codec_embedding = load(
            "talker.model.codec_embedding.weight",
            &[talker_config.vocab_size, codec_embed_dim],
        )?;
        let (projection_weight, projection_bias) = if config.hidden_size != codec_embed_dim {
            (
                Some(load(
                    "talker.code_predictor.small_to_mtp_projection.weight",
                    &[config.hidden_size, codec_embed_dim],
                )?),
                Some(load(
                    "talker.code_predictor.small_to_mtp_projection.bias",
                    &[config.hidden_size],
                )?),
            )
        } else {
            (None, None)
        };
        let mut codec_embeddings = Vec::with_capacity(acoustic_groups);
        let mut lm_heads = Vec::with_capacity(acoustic_groups);
        for group in 0..acoustic_groups {
            codec_embeddings.push(load(
                &format!("talker.code_predictor.model.codec_embedding.{group}.weight"),
                &[config.vocab_size, codec_embed_dim],
            )?);
            lm_heads.push(load(
                &format!("talker.code_predictor.lm_head.{group}.weight"),
                &[config.vocab_size, config.hidden_size],
            )?);
        }
        Ok(Self {
            layers,
            final_norm,
            talker_codec_embedding,
            codec_embeddings,
            projection_weight,
            projection_bias,
            lm_heads,
            geometry,
            codec_embed_dim,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn start_generation_session(
        &self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<CodePredictorGenerationSession<T>> {
        let codec_embed_dim = self.codec_embed_dim;
        let hidden = self.geometry.hidden_size;
        let groups = self.geometry.num_code_groups - 1;
        Ok(CodePredictorGenerationSession {
            caches: (0..self.layers.len())
                .map(|_| TalkerLayerKvCache::allocate(17, &self.geometry, stream))
                .collect::<anyhow::Result<Vec<_>>>()?,
            scratch: TalkerLayerScratch::allocate(2, &self.geometry, stream)?,
            talker_hidden: stream.alloc_zeros::<T>(codec_embed_dim)?,
            semantic_id: stream.alloc_zeros::<i32>(1)?,
            semantic_embed: stream.alloc_zeros::<T>(codec_embed_dim)?,
            predictor_input: stream.alloc_zeros::<T>(2 * codec_embed_dim)?,
            projected: stream.alloc_zeros::<T>(2 * hidden)?,
            hidden: stream.alloc_zeros::<f32>(2 * hidden)?,
            norm_token: stream.alloc_zeros::<T>(hidden)?,
            logits: stream.alloc_zeros::<T>(self.geometry.vocab_size)?,
            logits_f16: stream.alloc_zeros::<f16>(self.geometry.vocab_size)?,
            current_code: stream.alloc_zeros::<i32>(1)?,
            all_codes: stream.alloc_zeros::<i32>(groups)?,
            code_embed: stream.alloc_zeros::<T>(codec_embed_dim)?,
            embedding_sum: stream.alloc_zeros::<f32>(codec_embed_dim)?,
        })
    }

    /// Execute one prepared code-predictor sequence.
    pub fn forward_hidden(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        max_seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            hidden_host.len() == seq_len * self.geometry.hidden_size,
            "hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * self.geometry.hidden_size
        );
        ensure!(max_seq_len >= seq_len, "KV cache is smaller than the input");

        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = TalkerLayerScratch::allocate(seq_len, &self.geometry, &stream)?;
        let mut caches = (0..self.layers.len())
            .map(|_| TalkerLayerKvCache::allocate(max_seq_len, &self.geometry, &stream))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for (layer, cache) in self.layers.iter().zip(&mut caches) {
            layer.forward_cached_device(
                &mut hidden,
                seq_len,
                &self.geometry,
                kernels,
                cache,
                &mut scratch,
            )?;
        }
        T::rms_norm_f32in(
            kernels,
            &hidden,
            &self.final_norm,
            &mut scratch.norm,
            seq_len as u32,
            self.geometry.hidden_size as u32,
            self.geometry.rms_norm_eps as f32,
        )?;
        stream.synchronize()?;

        let output_len = seq_len * self.geometry.hidden_size;
        let mut output = vec![T::zero(); output_len];
        stream.memcpy_dtoh(&scratch.norm.slice(..output_len), &mut output)?;
        Ok(output.into_iter().map(T::to_f32).collect())
    }

    /// Generate the 15 acoustic codebooks for one semantic talker token.
    ///
    /// Argmax is used as a deterministic correctness baseline. All intermediate
    /// token IDs remain on the GPU; only the completed frame is copied back.
    pub fn generate_acoustic_codes_argmax(
        &self,
        talker_hidden: &[f32],
        semantic_token: i32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let mut session = self.start_generation_session(&stream)?;
        self.generate_acoustic_codes(&mut session, talker_hidden, semantic_token, None, kernels)
    }

    /// Sample the 15 acoustic codebooks with the model's subtalker settings.
    pub fn generate_acoustic_codes_sampled(
        &self,
        talker_hidden: &[f32],
        semantic_token: i32,
        temperature: f32,
        top_k: usize,
        seed: &mut u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let mut session = self.start_generation_session(&stream)?;
        self.generate_acoustic_codes_sampled_with_session(
            &mut session,
            talker_hidden,
            semantic_token,
            temperature,
            top_k,
            seed,
            kernels,
        )
    }

    pub fn generate_acoustic_codes_sampled_with_session(
        &self,
        session: &mut CodePredictorGenerationSession<T>,
        talker_hidden: &[f32],
        semantic_token: i32,
        temperature: f32,
        top_k: usize,
        seed: &mut u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        self.generate_acoustic_codes(
            session,
            talker_hidden,
            semantic_token,
            Some((temperature, top_k, seed)),
            kernels,
        )
    }

    pub fn generate_acoustic_codes_argmax_with_session(
        &self,
        session: &mut CodePredictorGenerationSession<T>,
        talker_hidden: &[f32],
        semantic_token: i32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        self.generate_acoustic_codes(session, talker_hidden, semantic_token, None, kernels)
    }

    fn generate_acoustic_codes(
        &self,
        session: &mut CodePredictorGenerationSession<T>,
        talker_hidden: &[f32],
        semantic_token: i32,
        mut sampling: Option<(f32, usize, &mut u64)>,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        let codec_embed_dim = self.codec_embed_dim;
        let hidden_dim = self.geometry.hidden_size;
        let vocab_size = self.geometry.vocab_size;
        let groups = self.geometry.num_code_groups - 1;
        ensure!(
            talker_hidden.len() == codec_embed_dim,
            "talker hidden has {} values, expected {codec_embed_dim}",
            talker_hidden.len()
        );
        ensure!(
            semantic_token >= 0 && semantic_token < 3072,
            "semantic token {semantic_token} is outside the codec vocabulary"
        );

        let stream = Arc::clone(kernels.ops.stream());
        let talker_hidden_native = talker_hidden
            .iter()
            .copied()
            .map(T::from_f32)
            .collect::<Vec<_>>();
        stream.memcpy_htod(&talker_hidden_native, &mut session.talker_hidden)?;
        stream.memcpy_htod(&[semantic_token], &mut session.semantic_id)?;
        T::gather(
            kernels,
            &self.talker_codec_embedding,
            &session.semantic_id,
            &mut session.semantic_embed,
            1,
            codec_embed_dim as u32,
        )?;

        stream.memcpy_dtod(
            &session.talker_hidden,
            &mut session.predictor_input.slice_mut(..codec_embed_dim),
        )?;
        stream.memcpy_dtod(
            &session.semantic_embed,
            &mut session.predictor_input.slice_mut(codec_embed_dim..),
        )?;
        if let (Some(weight), Some(bias)) = (&self.projection_weight, &self.projection_bias) {
            T::matmul(
                kernels,
                &session.predictor_input,
                weight,
                &mut session.projected,
                2,
                hidden_dim as u32,
                codec_embed_dim as u32,
            )?;
            T::add_bias(kernels, &mut session.projected, bias, 2, hidden_dim as u32)?;
        } else {
            stream.memcpy_dtod(&session.predictor_input, &mut session.projected)?;
        }

        T::to_f32_device(
            kernels,
            &session.projected,
            &mut session.hidden,
            (2 * hidden_dim) as u32,
        )?;
        for cache in &mut session.caches {
            cache.reset();
        }
        for (layer, cache) in self.layers.iter().zip(&mut session.caches) {
            layer.forward_cached_device(
                &mut session.hidden,
                2,
                &self.geometry,
                kernels,
                cache,
                &mut session.scratch,
            )?;
        }
        T::rms_norm_f32in(
            kernels,
            &session.hidden,
            &self.final_norm,
            &mut session.scratch.norm,
            2,
            hidden_dim as u32,
            self.geometry.rms_norm_eps as f32,
        )?;

        stream.memcpy_dtod(
            &session.scratch.norm.slice(hidden_dim..2 * hidden_dim),
            &mut session.norm_token,
        )?;
        T::gemv(
            kernels,
            &session.norm_token,
            &self.lm_heads[0],
            &mut session.logits,
            vocab_size as u32,
            hidden_dim as u32,
        )?;
        T::to_f16(
            kernels,
            &session.logits,
            &mut session.logits_f16,
            vocab_size as u32,
        )?;
        select_acoustic_code(
            &session.logits_f16,
            &mut session.current_code,
            vocab_size,
            sampling
                .as_mut()
                .map(|(temperature, top_k, seed)| (*temperature, *top_k, &mut **seed)),
            kernels,
        )?;
        stream.memcpy_dtod(
            &session.current_code,
            &mut session.all_codes.slice_mut(0..1),
        )?;

        for group in 1..groups {
            T::gather(
                kernels,
                &self.codec_embeddings[group - 1],
                &session.current_code,
                &mut session.code_embed,
                1,
                codec_embed_dim as u32,
            )?;
            if let (Some(weight), Some(bias)) = (&self.projection_weight, &self.projection_bias) {
                T::gemv(
                    kernels,
                    &session.code_embed,
                    weight,
                    &mut session.projected,
                    hidden_dim as u32,
                    codec_embed_dim as u32,
                )?;
                T::add_bias(kernels, &mut session.projected, bias, 1, hidden_dim as u32)?;
            } else {
                stream.memcpy_dtod(
                    &session.code_embed,
                    &mut session.projected.slice_mut(..hidden_dim),
                )?;
            }
            T::to_f32_device(
                kernels,
                &session.projected,
                &mut session.hidden,
                hidden_dim as u32,
            )?;
            for (layer, cache) in self.layers.iter().zip(&mut session.caches) {
                layer.forward_cached_device(
                    &mut session.hidden,
                    1,
                    &self.geometry,
                    kernels,
                    cache,
                    &mut session.scratch,
                )?;
            }
            T::rms_norm_f32in(
                kernels,
                &session.hidden,
                &self.final_norm,
                &mut session.scratch.norm,
                1,
                hidden_dim as u32,
                self.geometry.rms_norm_eps as f32,
            )?;
            T::gemv(
                kernels,
                &session.scratch.norm,
                &self.lm_heads[group],
                &mut session.logits,
                vocab_size as u32,
                hidden_dim as u32,
            )?;
            T::to_f16(
                kernels,
                &session.logits,
                &mut session.logits_f16,
                vocab_size as u32,
            )?;
            select_acoustic_code(
                &session.logits_f16,
                &mut session.current_code,
                vocab_size,
                sampling
                    .as_mut()
                    .map(|(temperature, top_k, seed)| (*temperature, *top_k, &mut **seed)),
                kernels,
            )?;
            stream.memcpy_dtod(
                &session.current_code,
                &mut session.all_codes.slice_mut(group..group + 1),
            )?;
        }
        stream.synchronize()?;
        let mut codes = vec![0i32; groups];
        stream.memcpy_dtoh(&session.all_codes, &mut codes)?;
        Ok(codes)
    }

    /// Sum the 15 group-specific acoustic embeddings for one codec frame.
    pub fn acoustic_embeddings_sum(
        &self,
        codes: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let codec_embed_dim = self.codec_embed_dim;
        ensure!(
            codes.len() == self.codec_embeddings.len(),
            "expected {} acoustic codes, got {}",
            self.codec_embeddings.len(),
            codes.len()
        );
        for code in codes {
            ensure!(
                *code >= 0 && (*code as usize) < self.geometry.vocab_size,
                "acoustic code {code} is outside 0..{}",
                self.geometry.vocab_size
            );
        }

        let stream = Arc::clone(kernels.ops.stream());
        let mut sum = stream.alloc_zeros::<f32>(codec_embed_dim)?;
        let mut embedding = stream.alloc_zeros::<T>(codec_embed_dim)?;
        for (code, table) in codes.iter().zip(&self.codec_embeddings) {
            let id = stream.clone_htod(&[*code])?;
            T::gather(
                kernels,
                table,
                &id,
                &mut embedding,
                1,
                codec_embed_dim as u32,
            )?;
            T::add_residual(kernels, &mut sum, &embedding, codec_embed_dim as u32)?;
        }
        stream.synchronize()?;
        let mut host = vec![0.0f32; codec_embed_dim];
        stream.memcpy_dtoh(&sum, &mut host)?;
        Ok(host)
    }

    /// Sum the last generated frame's acoustic embeddings without re-uploading IDs.
    pub fn acoustic_embeddings_sum_with_session(
        &self,
        session: &mut CodePredictorGenerationSession<T>,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let codec_embed_dim = self.codec_embed_dim;
        let stream = Arc::clone(kernels.ops.stream());
        for (group, table) in self.codec_embeddings.iter().enumerate() {
            stream.memcpy_dtod(
                &session.all_codes.slice(group..group + 1),
                &mut session.current_code,
            )?;
            T::gather(
                kernels,
                table,
                &session.current_code,
                &mut session.code_embed,
                1,
                codec_embed_dim as u32,
            )?;
            if group == 0 {
                T::to_f32_device(
                    kernels,
                    &session.code_embed,
                    &mut session.embedding_sum,
                    codec_embed_dim as u32,
                )?;
            } else {
                T::add_residual(
                    kernels,
                    &mut session.embedding_sum,
                    &session.code_embed,
                    codec_embed_dim as u32,
                )?;
            }
        }
        stream.synchronize()?;
        let mut host = vec![0.0f32; codec_embed_dim];
        stream.memcpy_dtoh(&session.embedding_sum, &mut host)?;
        Ok(host)
    }
}

fn select_acoustic_code(
    logits: &CudaSlice<f16>,
    token: &mut CudaSlice<i32>,
    vocab_size: usize,
    sampling: Option<(f32, usize, &mut u64)>,
    kernels: &mut GpuKernels,
) -> anyhow::Result<()> {
    if let Some((temperature, top_k, seed)) = sampling {
        ensure!(
            top_k <= 64,
            "GPU acoustic sampling supports top-k up to 64, got {top_k}"
        );
        kernels.ops.sample_top_k_small(
            logits,
            token,
            vocab_size as u32,
            temperature,
            top_k as u32,
            crate::sampling::next_seed_u32(seed),
        )?;
    } else {
        kernels.ops.argmax_f16(logits, token, vocab_size as u32)?;
    }
    Ok(())
}

fn predictor_geometry(config: &CodePredictorConfig) -> TalkerConfig {
    TalkerConfig {
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        vocab_size: config.vocab_size,
        text_vocab_size: 0,
        text_hidden_size: config.hidden_size,
        num_code_groups: config.num_code_groups,
        max_position_embeddings: config.max_position_embeddings,
        rope_theta: config.rope_theta,
        rms_norm_eps: config.rms_norm_eps,
        code_predictor_config: config.clone(),
        codec_language_id: HashMap::new(),
        spk_id: HashMap::new(),
    }
}
