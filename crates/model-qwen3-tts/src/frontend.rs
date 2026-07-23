use crate::{TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

/// Reusable GPU storage for semantic speech sampling.
pub struct SemanticSamplingSession {
    hidden: CudaSlice<f16>,
    logits: CudaSlice<f16>,
    previous: CudaSlice<i32>,
    token: CudaSlice<i32>,
    max_previous: usize,
}

/// Native text/codec embeddings and projections surrounding the talker stack.
pub struct TalkerFrontend {
    text_embedding: CudaSlice<f16>,
    text_fc1_weight: CudaSlice<f16>,
    text_fc1_bias: CudaSlice<f16>,
    text_fc2_weight: CudaSlice<f16>,
    text_fc2_bias: CudaSlice<f16>,
    codec_embedding: CudaSlice<f16>,
    codec_head: CudaSlice<f16>,
    text_vocab_size: usize,
    text_hidden_size: usize,
    hidden_size: usize,
    codec_vocab_size: usize,
}

/// Prepared VoiceDesign prompt plus the text states consumed during generation.
pub struct VoiceDesignInputs {
    pub prefill: Vec<f32>,
    pub prefill_tokens: usize,
    pub trailing_text: Vec<f32>,
    pub trailing_tokens: usize,
    pub text_pad: Vec<f32>,
}

impl TalkerFrontend {
    pub fn start_semantic_sampling_session(
        &self,
        max_previous: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<SemanticSamplingSession> {
        ensure!(
            max_previous > 0,
            "semantic sampler must hold at least one token"
        );
        Ok(SemanticSamplingSession {
            hidden: stream.alloc_zeros::<f16>(self.hidden_size)?,
            logits: stream.alloc_zeros::<f16>(self.codec_vocab_size)?,
            previous: stream.alloc_zeros::<i32>(max_previous)?,
            token: stream.alloc_zeros::<i32>(1)?,
            max_previous,
        })
    }

    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let tensor = load_f16_tensor(model_dir, name)
                .with_context(|| format!("could not load {name}"))?;
            ensure!(
                tensor.shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                tensor.shape
            );
            Ok(stream.clone_htod(&tensor.values)?)
        };
        let text_hidden = config.text_hidden_size;
        let hidden = config.hidden_size;
        Ok(Self {
            text_embedding: load(
                "talker.model.text_embedding.weight",
                &[config.text_vocab_size, text_hidden],
            )?,
            text_fc1_weight: load(
                "talker.text_projection.linear_fc1.weight",
                &[text_hidden, text_hidden],
            )?,
            text_fc1_bias: load("talker.text_projection.linear_fc1.bias", &[text_hidden])?,
            text_fc2_weight: load(
                "talker.text_projection.linear_fc2.weight",
                &[hidden, text_hidden],
            )?,
            text_fc2_bias: load("talker.text_projection.linear_fc2.bias", &[hidden])?,
            codec_embedding: load(
                "talker.model.codec_embedding.weight",
                &[config.vocab_size, hidden],
            )?,
            codec_head: load("talker.codec_head.weight", &[config.vocab_size, hidden])?,
            text_vocab_size: config.text_vocab_size,
            text_hidden_size: text_hidden,
            hidden_size: hidden,
            codec_vocab_size: config.vocab_size,
        })
    }

    /// Embed and project text token IDs into talker hidden states.
    pub fn project_text_tokens(
        &self,
        token_ids: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(!token_ids.is_empty(), "at least one text token is required");
        for token in token_ids {
            ensure!(
                *token >= 0 && (*token as usize) < self.text_vocab_size,
                "text token {token} is outside 0..{}",
                self.text_vocab_size
            );
        }
        let rows = token_ids.len();
        let stream = Arc::clone(kernels.ops.stream());
        let ids = stream.clone_htod(token_ids)?;
        let mut embeddings = stream.alloc_zeros::<f16>(rows * self.text_hidden_size)?;
        kernels.ops.gather_rows_f16(
            &self.text_embedding,
            &ids,
            &mut embeddings,
            rows as u32,
            self.text_hidden_size as u32,
        )?;
        let mut fc1 = stream.alloc_zeros::<f16>(rows * self.text_hidden_size)?;
        kernels.gemm.matmul_f16(
            &embeddings,
            &self.text_fc1_weight,
            &mut fc1,
            rows as u32,
            self.text_hidden_size as u32,
            self.text_hidden_size as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut fc1,
            &self.text_fc1_bias,
            rows as u32,
            self.text_hidden_size as u32,
        )?;
        let mut activated = stream.alloc_zeros::<f16>(rows * self.text_hidden_size)?;
        kernels
            .ops
            .silu_act_f16(&fc1, &mut activated, (rows * self.text_hidden_size) as u32)?;
        let mut projected = stream.alloc_zeros::<f16>(rows * self.hidden_size)?;
        kernels.gemm.matmul_f16(
            &activated,
            &self.text_fc2_weight,
            &mut projected,
            rows as u32,
            self.hidden_size as u32,
            self.text_hidden_size as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.text_fc2_bias,
            rows as u32,
            self.hidden_size as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; rows * self.hidden_size];
        stream.memcpy_dtoh(&projected, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    /// Look up talker codec embeddings for semantic or control tokens.
    pub fn codec_embeddings(
        &self,
        token_ids: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            !token_ids.is_empty(),
            "at least one codec token is required"
        );
        for token in token_ids {
            ensure!(
                *token >= 0 && (*token as usize) < self.codec_vocab_size,
                "codec token {token} is outside 0..{}",
                self.codec_vocab_size
            );
        }
        let stream = Arc::clone(kernels.ops.stream());
        let ids = stream.clone_htod(token_ids)?;
        let mut embeddings = stream.alloc_zeros::<f16>(token_ids.len() * self.hidden_size)?;
        kernels.ops.gather_rows_f16(
            &self.codec_embedding,
            &ids,
            &mut embeddings,
            token_ids.len() as u32,
            self.hidden_size as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; token_ids.len() * self.hidden_size];
        stream.memcpy_dtoh(&embeddings, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    /// Project one normalized talker hidden state and return its argmax token.
    pub fn semantic_argmax(&self, hidden: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<i32> {
        ensure!(
            hidden.len() == self.hidden_size,
            "talker hidden has {} values, expected {}",
            hidden.len(),
            self.hidden_size
        );
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let hidden = stream.clone_htod(&hidden)?;
        let mut logits = stream.alloc_zeros::<f16>(self.codec_vocab_size)?;
        let mut token = stream.alloc_zeros::<i32>(1)?;
        kernels.gemv.gemv_f16(
            &hidden,
            &self.codec_head,
            &mut logits,
            self.codec_vocab_size as u32,
            self.hidden_size as u32,
        )?;
        kernels
            .ops
            .argmax_f16(&logits, &mut token, self.codec_vocab_size as u32)?;
        stream.synchronize()?;
        let mut host = [0i32];
        stream.memcpy_dtoh(&token, &mut host)?;
        Ok(host[0])
    }

    /// Project one hidden state and select a speech token or codec EOS.
    ///
    /// Qwen's semantic head also contains control IDs. During generation those
    /// IDs are suppressed: valid output is 0..2048 or EOS 2150.
    pub fn semantic_speech_argmax(
        &self,
        hidden: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<i32> {
        const SPEECH_VOCAB_SIZE: usize = 2048;
        const CODEC_EOS: usize = 2150;
        ensure!(
            hidden.len() == self.hidden_size,
            "talker hidden has {} values, expected {}",
            hidden.len(),
            self.hidden_size
        );
        ensure!(
            self.codec_vocab_size > CODEC_EOS,
            "codec vocabulary does not contain EOS {CODEC_EOS}"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let hidden = stream.clone_htod(&hidden)?;
        let mut logits = stream.alloc_zeros::<f16>(self.codec_vocab_size)?;
        kernels.gemv.gemv_f16(
            &hidden,
            &self.codec_head,
            &mut logits,
            self.codec_vocab_size as u32,
            self.hidden_size as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; self.codec_vocab_size];
        stream.memcpy_dtoh(&logits, &mut host)?;
        let mut best_token = CODEC_EOS;
        let mut best_logit = host[CODEC_EOS].to_f32();
        for (token, logit) in host.iter().take(SPEECH_VOCAB_SIZE).enumerate() {
            let logit = logit.to_f32();
            if logit > best_logit {
                best_logit = logit;
                best_token = token;
            }
        }
        Ok(best_token as i32)
    }

    /// Sample one semantic speech token using Qwen's suppression rules.
    pub fn semantic_speech_sample(
        &self,
        hidden: &[f32],
        previous: &[i32],
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        seed: &mut u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<i32> {
        const SPEECH_VOCAB_SIZE: usize = 2048;
        const CODEC_EOS: usize = 2150;
        ensure!(
            hidden.len() == self.hidden_size,
            "talker hidden has {} values, expected {}",
            hidden.len(),
            self.hidden_size
        );
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let hidden = stream.clone_htod(&hidden)?;
        let mut logits = stream.alloc_zeros::<f16>(self.codec_vocab_size)?;
        kernels.gemv.gemv_f16(
            &hidden,
            &self.codec_head,
            &mut logits,
            self.codec_vocab_size as u32,
            self.hidden_size as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; self.codec_vocab_size];
        stream.memcpy_dtoh(&logits, &mut host)?;
        Ok(crate::sampling::sample_top_k(
            &host,
            |token| token < SPEECH_VOCAB_SIZE || token == CODEC_EOS,
            temperature,
            top_k,
            previous,
            repetition_penalty,
            seed,
        ))
    }

    /// Sample one semantic speech token while keeping logits and scratch on GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn semantic_speech_sample_with_session(
        &self,
        session: &mut SemanticSamplingSession,
        hidden: &[f32],
        previous: &[i32],
        temperature: f32,
        top_k: usize,
        repetition_penalty: f32,
        seed: &mut u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<i32> {
        const SPEECH_VOCAB_SIZE: usize = 2048;
        const CODEC_EOS: usize = 2150;
        ensure!(
            hidden.len() == self.hidden_size,
            "talker hidden has {} values, expected {}",
            hidden.len(),
            self.hidden_size
        );
        ensure!(
            previous.len() <= session.max_previous,
            "semantic history has {} tokens, session holds {}",
            previous.len(),
            session.max_previous
        );
        ensure!(top_k <= 64, "GPU semantic sampling supports top-k up to 64");
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        stream.memcpy_htod(&hidden, &mut session.hidden)?;
        if !previous.is_empty() {
            stream.memcpy_htod(previous, &mut session.previous.slice_mut(..previous.len()))?;
        }
        kernels.gemv.gemv_f16(
            &session.hidden,
            &self.codec_head,
            &mut session.logits,
            self.codec_vocab_size as u32,
            self.hidden_size as u32,
        )?;
        kernels.ops.sample_top_k_small_filtered(
            &session.logits,
            &session.previous,
            &mut session.token,
            self.codec_vocab_size as u32,
            SPEECH_VOCAB_SIZE as u32,
            CODEC_EOS as u32,
            previous.len() as u32,
            temperature,
            repetition_penalty,
            top_k as u32,
            crate::sampling::next_seed_u32(seed),
        )?;
        stream.synchronize()?;
        let mut token = [0i32];
        stream.memcpy_dtoh(&session.token, &mut token)?;
        Ok(token[0])
    }

    /// Build the exact Qwen3-TTS VoiceDesign prefill embedding layout.
    pub fn build_voice_design_inputs(
        &self,
        text_ids: &[i32],
        instruction_ids: &[i32],
        language_codec_id: i32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<VoiceDesignInputs> {
        const IM_START: i32 = 151_644;
        const ASSISTANT: i32 = 77_091;
        const NEWLINE: i32 = 198;
        const TTS_PAD: i32 = 151_671;
        const TTS_BOS: i32 = 151_672;
        const TTS_EOS: i32 = 151_673;
        const CODEC_PAD: i32 = 2_148;
        const CODEC_BOS: i32 = 2_149;
        const CODEC_THINK: i32 = 2_154;
        const CODEC_THINK_BOS: i32 = 2_156;
        const CODEC_THINK_EOS: i32 = 2_157;

        ensure!(!text_ids.is_empty(), "VoiceDesign text must not be empty");
        ensure!(
            !instruction_ids.is_empty(),
            "VoiceDesign instruction must not be empty"
        );

        let instruction = self.project_text_tokens(instruction_ids, kernels)?;
        let role = self.project_text_tokens(&[IM_START, ASSISTANT, NEWLINE], kernels)?;
        let control_text =
            self.project_text_tokens(&[TTS_PAD, TTS_PAD, TTS_PAD, TTS_PAD, TTS_BOS], kernels)?;
        let control_codec = self.codec_embeddings(
            &[
                CODEC_THINK,
                CODEC_THINK_BOS,
                language_codec_id,
                CODEC_THINK_EOS,
                CODEC_PAD,
            ],
            kernels,
        )?;
        let first_text = self.project_text_tokens(&text_ids[..1], kernels)?;
        let codec_bos = self.codec_embeddings(&[CODEC_BOS], kernels)?;

        let mut prefill = Vec::with_capacity(instruction.len() + role.len() + 6 * self.hidden_size);
        prefill.extend(instruction);
        prefill.extend(role);
        prefill.extend(
            control_text
                .iter()
                .zip(control_codec)
                .map(|(text, codec)| text + codec),
        );
        prefill.extend(
            first_text
                .iter()
                .zip(codec_bos)
                .map(|(text, codec)| text + codec),
        );
        let prefill_tokens = prefill.len() / self.hidden_size;

        let mut trailing_ids = text_ids[1..].to_vec();
        trailing_ids.push(TTS_EOS);
        let trailing_tokens = trailing_ids.len();
        let trailing_text = self.project_text_tokens(&trailing_ids, kernels)?;
        let text_pad = self.project_text_tokens(&[TTS_PAD], kernels)?;

        Ok(VoiceDesignInputs {
            prefill,
            prefill_tokens,
            trailing_text,
            trailing_tokens,
            text_pad,
        })
    }
}
