use crate::{
    ChatterboxConditioning, HIDDEN_SIZE, MAX_SPEECH_TOKENS, MAX_TEXT_TOKENS, SPEECH_VOCAB_SIZE,
    START_SPEECH_TOKEN, TEXT_VOCAB_SIZE,
};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_safetensors::MappedSafetensors;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

pub struct ChatterboxT3Prefix {
    pub conditional: Vec<f32>,
    pub unconditional: Vec<f32>,
    pub tokens: usize,
}

pub struct ChatterboxT3Frontend {
    text_embedding: CudaSlice<f16>,
    speech_embedding: CudaSlice<f16>,
    text_position: CudaSlice<f16>,
    speech_position: CudaSlice<f16>,
    speech_head: CudaSlice<f16>,
    speaker_weight: CudaSlice<f16>,
    speaker_bias: CudaSlice<f16>,
    emotion_weight: CudaSlice<f16>,
    perceiver_query: CudaSlice<f16>,
    perceiver_norm_weight: CudaSlice<f16>,
    perceiver_norm_bias: CudaSlice<f16>,
    perceiver_q_weight: CudaSlice<f16>,
    perceiver_q_bias: CudaSlice<f16>,
    perceiver_k_weight: CudaSlice<f16>,
    perceiver_k_bias: CudaSlice<f16>,
    perceiver_v_weight: CudaSlice<f16>,
    perceiver_v_bias: CudaSlice<f16>,
    perceiver_out_weight: CudaSlice<f16>,
    perceiver_out_bias: CudaSlice<f16>,
}

impl ChatterboxT3Frontend {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("t3_mtl23ls_v3.safetensors"))?;
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let (shape, values) = weights
                .tensor_f16(name)
                .with_context(|| format!("could not load Chatterbox T3 {name}"))?;
            ensure!(
                shape == expected,
                "Chatterbox T3 {name} has shape {shape:?}, expected {expected:?}"
            );
            Ok(stream.clone_htod(&values)?)
        };
        Ok(Self {
            text_embedding: load("text_emb.weight", &[TEXT_VOCAB_SIZE, HIDDEN_SIZE])?,
            speech_embedding: load("speech_emb.weight", &[SPEECH_VOCAB_SIZE, HIDDEN_SIZE])?,
            text_position: load(
                "text_pos_emb.emb.weight",
                &[MAX_TEXT_TOKENS + 2, HIDDEN_SIZE],
            )?,
            speech_position: load(
                "speech_pos_emb.emb.weight",
                &[MAX_SPEECH_TOKENS + 4, HIDDEN_SIZE],
            )?,
            speech_head: load("speech_head.weight", &[SPEECH_VOCAB_SIZE, HIDDEN_SIZE])?,
            speaker_weight: load("cond_enc.spkr_enc.weight", &[HIDDEN_SIZE, 256])?,
            speaker_bias: load("cond_enc.spkr_enc.bias", &[HIDDEN_SIZE])?,
            emotion_weight: load("cond_enc.emotion_adv_fc.weight", &[HIDDEN_SIZE, 1])?,
            perceiver_query: load(
                "cond_enc.perceiver.pre_attention_query",
                &[1, 32, HIDDEN_SIZE],
            )?,
            perceiver_norm_weight: load("cond_enc.perceiver.attn.norm.weight", &[HIDDEN_SIZE])?,
            perceiver_norm_bias: load("cond_enc.perceiver.attn.norm.bias", &[HIDDEN_SIZE])?,
            perceiver_q_weight: load(
                "cond_enc.perceiver.attn.to_q.weight",
                &[HIDDEN_SIZE, HIDDEN_SIZE],
            )?,
            perceiver_q_bias: load("cond_enc.perceiver.attn.to_q.bias", &[HIDDEN_SIZE])?,
            perceiver_k_weight: load(
                "cond_enc.perceiver.attn.to_k.weight",
                &[HIDDEN_SIZE, HIDDEN_SIZE],
            )?,
            perceiver_k_bias: load("cond_enc.perceiver.attn.to_k.bias", &[HIDDEN_SIZE])?,
            perceiver_v_weight: load(
                "cond_enc.perceiver.attn.to_v.weight",
                &[HIDDEN_SIZE, HIDDEN_SIZE],
            )?,
            perceiver_v_bias: load("cond_enc.perceiver.attn.to_v.bias", &[HIDDEN_SIZE])?,
            perceiver_out_weight: load(
                "cond_enc.perceiver.attn.proj_out.weight",
                &[HIDDEN_SIZE, HIDDEN_SIZE],
            )?,
            perceiver_out_bias: load("cond_enc.perceiver.attn.proj_out.bias", &[HIDDEN_SIZE])?,
        })
    }

    pub fn build_prefix(
        &self,
        text_tokens: &[i32],
        conditioning: &ChatterboxConditioning,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<ChatterboxT3Prefix> {
        ensure!(!text_tokens.is_empty(), "Chatterbox text tokens are empty");
        ensure!(
            text_tokens.len() <= MAX_TEXT_TOKENS + 2,
            "Chatterbox text token limit exceeded"
        );
        let mut condition = self.condition_embeddings(conditioning, kernels)?;
        let text = self.positioned_embeddings(
            &self.text_embedding,
            TEXT_VOCAB_SIZE,
            &self.text_position,
            text_tokens,
            0,
            kernels,
        )?;
        let bos = self.positioned_embeddings(
            &self.speech_embedding,
            SPEECH_VOCAB_SIZE,
            &self.speech_position,
            &[START_SPEECH_TOKEN as i32],
            0,
            kernels,
        )?;
        let condition_tokens = condition.len() / HIDDEN_SIZE;
        let tokens = condition_tokens + text_tokens.len() + 2;
        let mut conditional = Vec::with_capacity(tokens * HIDDEN_SIZE);
        conditional.append(&mut condition);
        conditional.extend_from_slice(&text);
        conditional.extend_from_slice(&bos);
        conditional.extend_from_slice(&bos);

        let mut unconditional = conditional.clone();
        let text_start = condition_tokens * HIDDEN_SIZE;
        let text_end = text_start + text.len();
        unconditional[text_start..text_end].fill(0.0);
        Ok(ChatterboxT3Prefix {
            conditional,
            unconditional,
            tokens,
        })
    }

    pub fn speech_embedding(
        &self,
        token: i32,
        position: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        self.positioned_embeddings(
            &self.speech_embedding,
            SPEECH_VOCAB_SIZE,
            &self.speech_position,
            &[token],
            position,
            kernels,
        )
    }

    pub fn speech_logits(
        &self,
        hidden: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f16>> {
        ensure!(
            hidden.len() == HIDDEN_SIZE,
            "T3 logit input has {} values, expected {HIDDEN_SIZE}",
            hidden.len()
        );
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let hidden = stream.clone_htod(&hidden)?;
        let mut logits = stream.alloc_zeros::<f16>(SPEECH_VOCAB_SIZE)?;
        kernels.gemv.gemv_f16(
            &hidden,
            &self.speech_head,
            &mut logits,
            SPEECH_VOCAB_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; SPEECH_VOCAB_SIZE];
        stream.memcpy_dtoh(&logits, &mut host)?;
        Ok(host)
    }

    pub fn speech_logits_pair(
        &self,
        hidden_a: &[f32],
        hidden_b: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<(Vec<f16>, Vec<f16>)> {
        ensure!(
            hidden_a.len() == HIDDEN_SIZE && hidden_b.len() == HIDDEN_SIZE,
            "paired T3 logits expect two hidden rows"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden_a
            .iter()
            .chain(hidden_b)
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let hidden = stream.clone_htod(&hidden)?;
        let mut logits = stream.alloc_zeros::<f16>(2 * SPEECH_VOCAB_SIZE)?;
        kernels.gemm.matmul_f16(
            &hidden,
            &self.speech_head,
            &mut logits,
            2,
            SPEECH_VOCAB_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; 2 * SPEECH_VOCAB_SIZE];
        stream.memcpy_dtoh(&logits, &mut host)?;
        Ok((
            host[..SPEECH_VOCAB_SIZE].to_vec(),
            host[SPEECH_VOCAB_SIZE..].to_vec(),
        ))
    }

    fn condition_embeddings(
        &self,
        conditioning: &ChatterboxConditioning,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            conditioning.speaker_embedding.len() == 256,
            "Chatterbox speaker embedding must have 256 values"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let speaker = conditioning
            .speaker_embedding
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let speaker = stream.clone_htod(&speaker)?;
        let mut speaker_out = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        kernels.gemv.gemv_f16(
            &speaker,
            &self.speaker_weight,
            &mut speaker_out,
            HIDDEN_SIZE as u32,
            256,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut speaker_out,
            &self.speaker_bias,
            1,
            HIDDEN_SIZE as u32,
        )?;

        let emotion = stream.clone_htod(&[f16::from_f32(conditioning.emotion_exaggeration)])?;
        let mut emotion_out = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        kernels.gemv.gemv_f16(
            &emotion,
            &self.emotion_weight,
            &mut emotion_out,
            HIDDEN_SIZE as u32,
            1,
        )?;
        let prompt = self.positioned_embeddings_device(
            &self.speech_embedding,
            SPEECH_VOCAB_SIZE,
            &self.speech_position,
            &conditioning.prompt_speech_tokens,
            0,
            kernels,
        )?;
        let prompt = self.perceiver(&prompt, conditioning.prompt_speech_tokens.len(), kernels)?;

        stream.synchronize()?;
        let mut result = device_f16_to_f32(&speaker_out, &stream)?;
        result.extend(device_f16_to_f32(&prompt, &stream)?);
        result.extend(device_f16_to_f32(&emotion_out, &stream)?);
        Ok(result)
    }

    fn positioned_embeddings(
        &self,
        table: &CudaSlice<f16>,
        vocab_size: usize,
        positions: &CudaSlice<f16>,
        token_ids: &[i32],
        position: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let device = self.positioned_embeddings_device(
            table, vocab_size, positions, token_ids, position, kernels,
        )?;
        let stream = Arc::clone(kernels.ops.stream());
        stream.synchronize()?;
        device_f16_to_f32(&device, &stream)
    }

    fn positioned_embeddings_device(
        &self,
        table: &CudaSlice<f16>,
        vocab_size: usize,
        positions: &CudaSlice<f16>,
        token_ids: &[i32],
        position: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        ensure!(!token_ids.is_empty(), "embedding token list is empty");
        ensure!(
            token_ids
                .iter()
                .all(|token| *token >= 0 && (*token as usize) < vocab_size),
            "embedding token is outside vocabulary"
        );
        ensure!(
            position + token_ids.len() <= MAX_SPEECH_TOKENS + 4,
            "position embedding limit exceeded"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let ids = stream.clone_htod(token_ids)?;
        let position_ids = (position..position + token_ids.len())
            .map(|value| value as i32)
            .collect::<Vec<_>>();
        let position_ids = stream.clone_htod(&position_ids)?;
        let mut embedded = stream.alloc_zeros::<f16>(token_ids.len() * HIDDEN_SIZE)?;
        let mut positioned = stream.alloc_zeros::<f16>(token_ids.len() * HIDDEN_SIZE)?;
        let mut output = stream.alloc_zeros::<f16>(token_ids.len() * HIDDEN_SIZE)?;
        kernels.ops.gather_rows_f16(
            table,
            &ids,
            &mut embedded,
            token_ids.len() as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.gather_rows_f16(
            positions,
            &position_ids,
            &mut positioned,
            token_ids.len() as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.add_f16(
            &embedded,
            &positioned,
            &mut output,
            (token_ids.len() * HIDDEN_SIZE) as u32,
        )?;
        Ok(output)
    }

    fn perceiver(
        &self,
        prompt: &CudaSlice<f16>,
        prompt_tokens: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let stream = Arc::clone(kernels.ops.stream());
        let mut query = stream.alloc_zeros::<f16>(32 * HIDDEN_SIZE)?;
        kernels.ops.copy_f16(
            &self.perceiver_query,
            &mut query.slice_mut(..),
            (32 * HIDDEN_SIZE) as u32,
        )?;
        let cross = self.perceiver_attention(&query, 32, prompt, prompt_tokens, kernels)?;
        self.perceiver_attention(&cross, 32, &cross, 32, kernels)
    }

    fn perceiver_attention(
        &self,
        query: &CudaSlice<f16>,
        query_tokens: usize,
        context: &CudaSlice<f16>,
        context_tokens: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        const HEADS: usize = 4;
        const HEAD_DIM: usize = HIDDEN_SIZE / HEADS;
        let stream = Arc::clone(kernels.ops.stream());
        let mut query_f32 = stream.alloc_zeros::<f32>(query_tokens * HIDDEN_SIZE)?;
        let mut context_f32 = stream.alloc_zeros::<f32>(context_tokens * HIDDEN_SIZE)?;
        kernels
            .ops
            .copy_f16_to_f32(query, &mut query_f32, (query_tokens * HIDDEN_SIZE) as u32)?;
        kernels.ops.copy_f16_to_f32(
            context,
            &mut context_f32,
            (context_tokens * HIDDEN_SIZE) as u32,
        )?;
        let mut query_norm = stream.alloc_zeros::<f16>(query_tokens * HIDDEN_SIZE)?;
        let mut context_norm = stream.alloc_zeros::<f16>(context_tokens * HIDDEN_SIZE)?;
        kernels.ops.layer_norm_f32in(
            &query_f32,
            &self.perceiver_norm_weight,
            &self.perceiver_norm_bias,
            &mut query_norm,
            query_tokens as u32,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        kernels.ops.layer_norm_f32in(
            &context_f32,
            &self.perceiver_norm_weight,
            &self.perceiver_norm_bias,
            &mut context_norm,
            context_tokens as u32,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        let mut q = stream.alloc_zeros::<f16>(query_tokens * HIDDEN_SIZE)?;
        let mut k = stream.alloc_zeros::<f16>(context_tokens * HIDDEN_SIZE)?;
        let mut v = stream.alloc_zeros::<f16>(context_tokens * HIDDEN_SIZE)?;
        kernels.gemm.matmul_f16(
            &query_norm,
            &self.perceiver_q_weight,
            &mut q,
            query_tokens as u32,
            HIDDEN_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut q,
            &self.perceiver_q_bias,
            query_tokens as u32,
            HIDDEN_SIZE as u32,
        )?;
        for (weight, bias, output) in [
            (&self.perceiver_k_weight, &self.perceiver_k_bias, &mut k),
            (&self.perceiver_v_weight, &self.perceiver_v_bias, &mut v),
        ] {
            kernels.gemm.matmul_f16(
                &context_norm,
                weight,
                output,
                context_tokens as u32,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
            kernels.ops.add_bias_f16_inplace(
                output,
                bias,
                context_tokens as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        let mut attention = stream.alloc_zeros::<f16>(query_tokens * HIDDEN_SIZE)?;
        kernels.ops.mha_naive_full(
            &q,
            &k.slice(..),
            &v.slice(..),
            &mut attention,
            HEAD_DIM as u32,
            HEADS as u32,
            HEADS as u32,
            query_tokens as u32,
            context_tokens as u32,
            1.0 / (HEAD_DIM as f32).sqrt(),
            0.0,
        )?;
        let mut projected = stream.alloc_zeros::<f16>(query_tokens * HIDDEN_SIZE)?;
        kernels.gemm.matmul_f16(
            &attention,
            &self.perceiver_out_weight,
            &mut projected,
            query_tokens as u32,
            HIDDEN_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.perceiver_out_bias,
            query_tokens as u32,
            HIDDEN_SIZE as u32,
        )?;
        let mut output = stream.alloc_zeros::<f16>(query_tokens * HIDDEN_SIZE)?;
        kernels.ops.add_f16(
            query,
            &projected,
            &mut output,
            (query_tokens * HIDDEN_SIZE) as u32,
        )?;
        Ok(output)
    }
}

fn device_f16_to_f32(
    values: &CudaSlice<f16>,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Vec<f32>> {
    let mut host = vec![f16::ZERO; values.len()];
    stream.memcpy_dtoh(values, &mut host)?;
    Ok(host.into_iter().map(f16::to_f32).collect())
}
