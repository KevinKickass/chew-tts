use crate::VibeVoiceConfig;
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, CodePredictorConfig, TalkerConfig, TalkerTransformer};
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
