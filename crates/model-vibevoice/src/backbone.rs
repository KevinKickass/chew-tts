use crate::{VibeVoiceConfig, VibeVoicePrompt, VibeVoicePromptBranch};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{
    Bf16, CodePredictorConfig, TalkerConfig, TalkerGenerationSession, TalkerTransformer,
};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The two Qwen2 stacks used by VibeVoice-Realtime. Both stay resident on the
/// GPU; only their KV sessions are created per stream.
pub struct VibeVoiceBackbones {
    pub text: TalkerTransformer<Bf16>,
    pub tts: TalkerTransformer<Bf16>,
    pub text_config: TalkerConfig,
    pub tts_config: TalkerConfig,
}

pub struct VibeVoiceBackboneSession {
    text: TalkerGenerationSession<Bf16>,
    tts: TalkerGenerationSession<Bf16>,
    negative_tts: TalkerGenerationSession<Bf16>,
    positive_condition: Vec<f32>,
    negative_condition: Vec<f32>,
}

impl VibeVoiceBackbones {
    pub fn load(
        model_dir: &Path,
        config: &VibeVoiceConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let text_config = transformer_config(config, config.text_layers());
        let tts_config = transformer_config(config, config.tts_backbone_num_hidden_layers);
        let text = TalkerTransformer::load_qwen2(
            model_dir,
            "model.language_model",
            &text_config,
            stream,
            false,
        )
        .context("could not load VibeVoice text backbone")?;
        let tts = TalkerTransformer::load_qwen2(
            model_dir,
            "model.tts_language_model",
            &tts_config,
            stream,
            true,
        )
        .context("could not load VibeVoice TTS backbone")?;
        Ok(Self {
            text,
            tts,
            text_config,
            tts_config,
        })
    }

    /// Execute one deterministic token through both stacks. This validates the
    /// BF16 GEMMs, standard Qwen2 attention (without Q/K norm), RoPE, and
    /// SwiGLU before prompt-cache integration.
    pub fn smoke(&self, kernels: &mut GpuKernels) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let hidden = (0..self.text_config.hidden_size)
            .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.125)
            .collect::<Vec<_>>();
        let text = self
            .text
            .forward_hidden(&hidden, 1, 1, &self.text_config, kernels)?;
        let tts = self
            .tts
            .forward_hidden(&text, 1, 1, &self.tts_config, kernels)?;
        ensure!(
            text.iter().chain(&tts).all(|value| value.is_finite()),
            "VibeVoice backbone produced non-finite output"
        );
        Ok((text, tts))
    }

    /// Resume all four transformer branches from an official cached voice.
    /// This exercises the BF16-HF to position-major attention-cache bridge.
    pub fn prompt_cache_smoke(
        &self,
        prompt: &VibeVoicePrompt,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let hidden = (0..self.text_config.hidden_size)
            .map(|index| ((index as f32 + 1.0) * 0.011).cos() * 0.125)
            .collect::<Vec<_>>();
        let positive_text =
            self.resume_branch(&self.text, &self.text_config, &prompt.lm, &hidden, kernels)?;
        let negative_text = self.resume_branch(
            &self.text,
            &self.text_config,
            &prompt.negative_lm,
            &hidden,
            kernels,
        )?;
        let positive_tts = self.resume_branch(
            &self.tts,
            &self.tts_config,
            &prompt.tts_lm,
            &positive_text,
            kernels,
        )?;
        let negative_tts = self.resume_branch(
            &self.tts,
            &self.tts_config,
            &prompt.negative_tts_lm,
            &negative_text,
            kernels,
        )?;
        ensure!(
            positive_tts
                .iter()
                .chain(&negative_tts)
                .all(|value| value.is_finite()),
            "VibeVoice cached prompt produced non-finite output"
        );
        Ok((positive_tts, negative_tts))
    }

    fn resume_branch(
        &self,
        transformer: &TalkerTransformer<Bf16>,
        config: &TalkerConfig,
        branch: &VibeVoicePromptBranch,
        input: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let mut session = transformer.start_session(branch.tokens + 1, 1, config, &stream)?;
        session.load_prompt_kv(
            branch
                .layers
                .iter()
                .map(|layer| (layer.key.as_slice(), layer.value.as_slice())),
            branch.tokens,
            config,
            &stream,
        )?;
        transformer.forward_session(&mut session, input, 1, config, kernels)
    }

    pub fn start_prompt_session(
        &self,
        prompt: &VibeVoicePrompt,
        max_new_tokens: usize,
        max_window_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<VibeVoiceBackboneSession> {
        ensure!(
            max_new_tokens > 0,
            "VibeVoice session needs generation capacity"
        );
        let mut text = self.text.start_session(
            prompt.lm.tokens + max_new_tokens,
            max_window_tokens,
            &self.text_config,
            stream,
        )?;
        load_prompt(&mut text, &prompt.lm, &self.text_config, stream)?;
        let mut tts = self.tts.start_session(
            prompt.tts_lm.tokens + max_new_tokens,
            max_window_tokens,
            &self.tts_config,
            stream,
        )?;
        load_prompt(&mut tts, &prompt.tts_lm, &self.tts_config, stream)?;
        let mut negative_tts = self.tts.start_session(
            prompt.negative_tts_lm.tokens + max_new_tokens,
            max_window_tokens,
            &self.tts_config,
            stream,
        )?;
        load_prompt(
            &mut negative_tts,
            &prompt.negative_tts_lm,
            &self.tts_config,
            stream,
        )?;
        Ok(VibeVoiceBackboneSession {
            text,
            tts,
            negative_tts,
            positive_condition: last_hidden(&prompt.tts_lm, self.tts_config.hidden_size),
            negative_condition: last_hidden(&prompt.negative_tts_lm, self.tts_config.hidden_size),
        })
    }

    pub fn push_text(
        &self,
        session: &mut VibeVoiceBackboneSession,
        embeddings: &[f32],
        tokens: usize,
        text_type: impl FnOnce(&mut [f32]) -> anyhow::Result<()>,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        let mut text = self.text.forward_session(
            &mut session.text,
            embeddings,
            tokens,
            &self.text_config,
            kernels,
        )?;
        text_type(&mut text)?;
        session.positive_condition =
            self.tts
                .forward_session(&mut session.tts, &text, tokens, &self.tts_config, kernels)?;
        session.positive_condition =
            last_row(&session.positive_condition, self.tts_config.hidden_size);
        Ok(())
    }

    pub fn push_speech(
        &self,
        session: &mut VibeVoiceBackboneSession,
        mut acoustic_embedding: Vec<f32>,
        speech_type: impl Fn(&mut [f32]) -> anyhow::Result<()>,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        speech_type(&mut acoustic_embedding)?;
        session.positive_condition = self.tts.forward_session(
            &mut session.tts,
            &acoustic_embedding,
            1,
            &self.tts_config,
            kernels,
        )?;
        session.negative_condition = self.tts.forward_session(
            &mut session.negative_tts,
            &acoustic_embedding,
            1,
            &self.tts_config,
            kernels,
        )?;
        Ok(())
    }
}

impl VibeVoiceBackboneSession {
    pub fn positive_condition(&self) -> &[f32] {
        &self.positive_condition
    }

    pub fn negative_condition(&self) -> &[f32] {
        &self.negative_condition
    }
}

fn load_prompt(
    session: &mut TalkerGenerationSession<Bf16>,
    branch: &VibeVoicePromptBranch,
    config: &TalkerConfig,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<()> {
    session.load_prompt_kv(
        branch
            .layers
            .iter()
            .map(|layer| (layer.key.as_slice(), layer.value.as_slice())),
        branch.tokens,
        config,
        stream,
    )
}

fn last_hidden(branch: &VibeVoicePromptBranch, hidden: usize) -> Vec<f32> {
    branch.last_hidden_state[(branch.tokens - 1) * hidden..branch.tokens * hidden]
        .iter()
        .copied()
        .map(half::bf16::to_f32)
        .collect()
}

fn last_row(values: &[f32], hidden: usize) -> Vec<f32> {
    values[values.len() - hidden..].to_vec()
}

fn transformer_config(config: &VibeVoiceConfig, layers: usize) -> TalkerConfig {
    let decoder = &config.decoder_config;
    // The shared native transformer implementation only reads the standard
    // decoder fields below. Code-predictor fields are inert for Qwen2 stacks.
    let placeholder_predictor = CodePredictorConfig {
        hidden_size: decoder.hidden_size,
        intermediate_size: decoder.intermediate_size,
        num_hidden_layers: 1,
        num_attention_heads: decoder.num_attention_heads,
        num_key_value_heads: decoder.num_key_value_heads,
        head_dim: config.head_dim(),
        vocab_size: 1,
        num_code_groups: 2,
        max_position_embeddings: decoder.max_position_embeddings,
        rope_theta: decoder.rope_theta,
        rms_norm_eps: decoder.rms_norm_eps,
    };
    TalkerConfig {
        hidden_size: decoder.hidden_size,
        intermediate_size: decoder.intermediate_size,
        num_hidden_layers: layers,
        num_attention_heads: decoder.num_attention_heads,
        num_key_value_heads: decoder.num_key_value_heads,
        head_dim: config.head_dim(),
        vocab_size: decoder.vocab_size,
        text_vocab_size: decoder.vocab_size,
        text_hidden_size: decoder.hidden_size,
        num_code_groups: 2,
        max_position_embeddings: decoder.max_position_embeddings,
        rope_theta: decoder.rope_theta,
        rms_norm_eps: decoder.rms_norm_eps,
        code_predictor_config: placeholder_predictor,
        codec_language_id: HashMap::new(),
        spk_id: HashMap::new(),
    }
}
