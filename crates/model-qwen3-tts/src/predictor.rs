use crate::cuda::{TalkerDecoderLayer, TalkerLayerKvCache, TalkerLayerScratch};
use crate::{CodePredictorConfig, TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The five-layer multi-codebook decoder used after every talker step.
pub struct CodePredictorTransformer {
    layers: Vec<TalkerDecoderLayer>,
    final_norm: CudaSlice<f16>,
    talker_codec_embedding: CudaSlice<f16>,
    codec_embeddings: Vec<CudaSlice<f16>>,
    projection_weight: CudaSlice<f16>,
    projection_bias: CudaSlice<f16>,
    lm_heads: Vec<CudaSlice<f16>>,
    geometry: TalkerConfig,
}

/// Reusable GPU allocations for one code-predictor worker.
pub struct CodePredictorGenerationSession {
    caches: Vec<TalkerLayerKvCache>,
    scratch: TalkerLayerScratch,
    talker_hidden: CudaSlice<f16>,
    semantic_id: CudaSlice<i32>,
    semantic_embed: CudaSlice<f16>,
    predictor_input: CudaSlice<f16>,
    projected: CudaSlice<f16>,
    hidden: CudaSlice<f32>,
    norm_token: CudaSlice<f16>,
    logits: CudaSlice<f16>,
    current_code: CudaSlice<i32>,
    all_codes: CudaSlice<i32>,
    code_embed: CudaSlice<f16>,
}

impl CodePredictorTransformer {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &CodePredictorConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
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
        let norm = load_f16_tensor(model_dir, "talker.code_predictor.model.norm.weight")
            .context("could not load code predictor final norm")?;
        ensure!(
            norm.shape == [config.hidden_size],
            "code predictor final norm has shape {:?}, expected [{}]",
            norm.shape,
            config.hidden_size
        );
        let final_norm = stream
            .clone_htod(&norm.values)
            .context("could not upload code predictor final norm")?;
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let tensor = load_f16_tensor(model_dir, name)
                .with_context(|| format!("could not load {name}"))?;
            ensure!(
                tensor.shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                tensor.shape
            );
            stream
                .clone_htod(&tensor.values)
                .with_context(|| format!("could not upload {name}"))
        };
        let codec_embed_dim = 2048;
        let acoustic_groups = config.num_code_groups - 1;
        let talker_codec_embedding = load(
            "talker.model.codec_embedding.weight",
            &[3072, codec_embed_dim],
        )?;
        let projection_weight = load(
            "talker.code_predictor.small_to_mtp_projection.weight",
            &[config.hidden_size, codec_embed_dim],
        )?;
        let projection_bias = load(
            "talker.code_predictor.small_to_mtp_projection.bias",
            &[config.hidden_size],
        )?;
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
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn start_generation_session(
        &self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<CodePredictorGenerationSession> {
        const CODEC_EMBED_DIM: usize = 2048;
        let hidden = self.geometry.hidden_size;
        let groups = self.geometry.num_code_groups - 1;
        Ok(CodePredictorGenerationSession {
            caches: (0..self.layers.len())
                .map(|_| TalkerLayerKvCache::allocate(17, &self.geometry, stream))
                .collect::<anyhow::Result<Vec<_>>>()?,
            scratch: TalkerLayerScratch::allocate(2, &self.geometry, stream)?,
            talker_hidden: stream.alloc_zeros::<f16>(CODEC_EMBED_DIM)?,
            semantic_id: stream.alloc_zeros::<i32>(1)?,
            semantic_embed: stream.alloc_zeros::<f16>(CODEC_EMBED_DIM)?,
            predictor_input: stream.alloc_zeros::<f16>(2 * CODEC_EMBED_DIM)?,
            projected: stream.alloc_zeros::<f16>(2 * hidden)?,
            hidden: stream.alloc_zeros::<f32>(2 * hidden)?,
            norm_token: stream.alloc_zeros::<f16>(hidden)?,
            logits: stream.alloc_zeros::<f16>(self.geometry.vocab_size)?,
            current_code: stream.alloc_zeros::<i32>(1)?,
            all_codes: stream.alloc_zeros::<i32>(groups)?,
            code_embed: stream.alloc_zeros::<f16>(CODEC_EMBED_DIM)?,
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
        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.final_norm,
            &mut scratch.norm,
            seq_len as u32,
            self.geometry.hidden_size as u32,
            self.geometry.rms_norm_eps as f32,
        )?;
        stream.synchronize()?;

        let output_len = seq_len * self.geometry.hidden_size;
        let mut output = vec![f16::ZERO; output_len];
        stream.memcpy_dtoh(&scratch.norm.slice(..output_len), &mut output)?;
        Ok(output.into_iter().map(f16::to_f32).collect())
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
        session: &mut CodePredictorGenerationSession,
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
        session: &mut CodePredictorGenerationSession,
        talker_hidden: &[f32],
        semantic_token: i32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        self.generate_acoustic_codes(session, talker_hidden, semantic_token, None, kernels)
    }

    fn generate_acoustic_codes(
        &self,
        session: &mut CodePredictorGenerationSession,
        talker_hidden: &[f32],
        semantic_token: i32,
        mut sampling: Option<(f32, usize, &mut u64)>,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<i32>> {
        const CODEC_EMBED_DIM: usize = 2048;
        let hidden_dim = self.geometry.hidden_size;
        let vocab_size = self.geometry.vocab_size;
        let groups = self.geometry.num_code_groups - 1;
        ensure!(
            talker_hidden.len() == CODEC_EMBED_DIM,
            "talker hidden has {} values, expected {CODEC_EMBED_DIM}",
            talker_hidden.len()
        );
        ensure!(
            semantic_token >= 0 && semantic_token < 3072,
            "semantic token {semantic_token} is outside the codec vocabulary"
        );

        let stream = Arc::clone(kernels.ops.stream());
        let talker_hidden_f16 = talker_hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        stream.memcpy_htod(&talker_hidden_f16, &mut session.talker_hidden)?;
        stream.memcpy_htod(&[semantic_token], &mut session.semantic_id)?;
        kernels.ops.gather_rows_f16(
            &self.talker_codec_embedding,
            &session.semantic_id,
            &mut session.semantic_embed,
            1,
            CODEC_EMBED_DIM as u32,
        )?;

        stream.memcpy_dtod(
            &session.talker_hidden,
            &mut session.predictor_input.slice_mut(..CODEC_EMBED_DIM),
        )?;
        stream.memcpy_dtod(
            &session.semantic_embed,
            &mut session.predictor_input.slice_mut(CODEC_EMBED_DIM..),
        )?;
        kernels.gemm.matmul_f16(
            &session.predictor_input,
            &self.projection_weight,
            &mut session.projected,
            2,
            hidden_dim as u32,
            CODEC_EMBED_DIM as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut session.projected,
            &self.projection_bias,
            2,
            hidden_dim as u32,
        )?;

        kernels.ops.copy_f16_to_f32(
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
        kernels.ops.rms_norm_f32in(
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
        kernels.gemv.gemv_f16(
            &session.norm_token,
            &self.lm_heads[0],
            &mut session.logits,
            vocab_size as u32,
            hidden_dim as u32,
        )?;
        select_acoustic_code(
            &session.logits,
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
            kernels.ops.gather_rows_f16(
                &self.codec_embeddings[group - 1],
                &session.current_code,
                &mut session.code_embed,
                1,
                CODEC_EMBED_DIM as u32,
            )?;
            kernels.gemv.gemv_f16(
                &session.code_embed,
                &self.projection_weight,
                &mut session.projected,
                hidden_dim as u32,
                CODEC_EMBED_DIM as u32,
            )?;
            kernels.ops.add_bias_f16_inplace(
                &mut session.projected,
                &self.projection_bias,
                1,
                hidden_dim as u32,
            )?;
            kernels.ops.copy_f16_to_f32(
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
            kernels.ops.rms_norm_f32in(
                &session.hidden,
                &self.final_norm,
                &mut session.scratch.norm,
                1,
                hidden_dim as u32,
                self.geometry.rms_norm_eps as f32,
            )?;
            kernels.gemv.gemv_f16(
                &session.scratch.norm,
                &self.lm_heads[group],
                &mut session.logits,
                vocab_size as u32,
                hidden_dim as u32,
            )?;
            select_acoustic_code(
                &session.logits,
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
        const CODEC_EMBED_DIM: usize = 2048;
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
        let mut sum = stream.alloc_zeros::<f32>(CODEC_EMBED_DIM)?;
        let mut embedding = stream.alloc_zeros::<f16>(CODEC_EMBED_DIM)?;
        for (code, table) in codes.iter().zip(&self.codec_embeddings) {
            let id = stream.clone_htod(&[*code])?;
            kernels
                .ops
                .gather_rows_f16(table, &id, &mut embedding, 1, CODEC_EMBED_DIM as u32)?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut sum, &embedding, CODEC_EMBED_DIM as u32)?;
        }
        stream.synchronize()?;
        let mut host = vec![0.0f32; CODEC_EMBED_DIM];
        stream.memcpy_dtoh(&sum, &mut host)?;
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
    }
}
