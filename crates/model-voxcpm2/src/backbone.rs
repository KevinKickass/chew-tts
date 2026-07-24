use crate::VoxCpm2Config;
use anyhow::ensure;
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, CodePredictorConfig, TalkerConfig, TalkerTransformer};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// GPU-resident VoxCPM2 semantic MiniCPM4 backbone.
///
/// This first correctness target covers the complete 28-layer base LM with
/// BF16 weights, bias-free GQA, LongRoPE, SwiGLU, and FP32 residuals.
pub struct VoxCpm2BaseBackbone {
    transformer: TalkerTransformer<Bf16>,
    config: TalkerConfig,
}

pub struct VoxCpm2TransformerBackbones {
    pub base: VoxCpm2BaseBackbone,
    residual: TalkerTransformer<Bf16>,
    encoder: TalkerTransformer<Bf16>,
    dit: TalkerTransformer<Bf16>,
    residual_config: TalkerConfig,
    local_config: TalkerConfig,
}

pub struct VoxCpm2TransformerSmoke {
    pub base: Vec<f32>,
    pub residual: Vec<f32>,
    pub encoder: Vec<f32>,
    pub dit: Vec<f32>,
}

impl VoxCpm2BaseBackbone {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let transformer_config = transformer_config(
            config,
            config.lm_config.hidden_size,
            config.lm_config.intermediate_size,
            config.lm_config.num_hidden_layers,
            config.lm_config.num_attention_heads,
            config.lm_config.kv_channels,
        );
        let transformer = TalkerTransformer::load_minicpm(
            model_dir,
            "base_lm",
            &transformer_config,
            stream,
            Some(&config.lm_config.rope_scaling.short_factor),
            true,
            true,
        )?;
        Ok(Self {
            transformer,
            config: transformer_config,
        })
    }

    pub fn smoke(&self, kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let hidden = (0..self.config.hidden_size)
            .map(|index| ((index as f32 + 1.0) * 0.007).sin() * 0.125)
            .collect::<Vec<_>>();
        let output = self
            .transformer
            .forward_hidden(&hidden, 1, 1, &self.config, kernels)?;
        ensure!(
            output.iter().all(|value| value.is_finite()),
            "VoxCPM2 base LM produced non-finite output"
        );
        Ok(output)
    }
}

impl VoxCpm2TransformerBackbones {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let base = VoxCpm2BaseBackbone::load(model_dir, config, stream)?;
        let residual_config = transformer_config(
            config,
            config.lm_config.hidden_size,
            config.lm_config.intermediate_size,
            config.residual_lm_num_layers,
            config.lm_config.num_attention_heads,
            config.lm_config.kv_channels,
        );
        let residual = TalkerTransformer::load_minicpm(
            model_dir,
            "residual_lm",
            &residual_config,
            stream,
            None,
            !config.residual_lm_no_rope,
            true,
        )?;
        let local_config = transformer_config(
            config,
            config.encoder_config.hidden_dim,
            config.encoder_config.ffn_dim,
            config.encoder_config.num_layers,
            config.encoder_config.num_heads,
            config.encoder_config.kv_channels,
        );
        let factors = &config.lm_config.rope_scaling.short_factor;
        let encoder = TalkerTransformer::load_minicpm(
            model_dir,
            "feat_encoder.encoder",
            &local_config,
            stream,
            Some(factors),
            true,
            false,
        )?;
        let dit_config = transformer_config(
            config,
            config.dit_config.hidden_dim,
            config.dit_config.ffn_dim,
            config.dit_config.num_layers,
            config.dit_config.num_heads,
            config.dit_config.kv_channels,
        );
        ensure!(
            local_config.hidden_size == dit_config.hidden_size
                && local_config.intermediate_size == dit_config.intermediate_size
                && local_config.num_hidden_layers == dit_config.num_hidden_layers,
            "VoxCPM2 local encoder and DiT transformer geometries differ"
        );
        let dit = TalkerTransformer::load_minicpm(
            model_dir,
            "feat_decoder.estimator.decoder",
            &dit_config,
            stream,
            Some(factors),
            true,
            false,
        )?;
        Ok(Self {
            base,
            residual,
            encoder,
            dit,
            residual_config,
            local_config,
        })
    }

    pub fn smoke(&self, kernels: &mut GpuKernels) -> anyhow::Result<VoxCpm2TransformerSmoke> {
        let base = self.base.smoke(kernels)?;
        let residual_input = deterministic_input(self.residual_config.hidden_size, 1, 0.009);
        let residual =
            self.residual
                .forward_hidden(&residual_input, 1, 1, &self.residual_config, kernels)?;
        let encoder_tokens = 5;
        let encoder_input =
            deterministic_input(self.local_config.hidden_size, encoder_tokens, 0.005);
        let encoder = self.encoder.forward_hidden(
            &encoder_input,
            encoder_tokens,
            encoder_tokens,
            &self.local_config,
            kernels,
        )?;
        let dit_tokens = 10;
        let dit_input = deterministic_input(self.local_config.hidden_size, dit_tokens, 0.003);
        let dit = self.dit.forward_hidden(
            &dit_input,
            dit_tokens,
            dit_tokens,
            &self.local_config,
            kernels,
        )?;
        ensure!(
            residual
                .iter()
                .chain(&encoder)
                .chain(&dit)
                .all(|value| value.is_finite()),
            "VoxCPM2 transformer stack produced non-finite output"
        );
        Ok(VoxCpm2TransformerSmoke {
            base,
            residual,
            encoder,
            dit,
        })
    }
}

fn deterministic_input(hidden: usize, tokens: usize, step: f32) -> Vec<f32> {
    (0..hidden * tokens)
        .map(|index| ((index as f32 + 1.0) * step).sin() * 0.125)
        .collect()
}

fn transformer_config(
    config: &VoxCpm2Config,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
) -> TalkerConfig {
    let lm = &config.lm_config;
    let placeholder_predictor = CodePredictorConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers: 1,
        num_attention_heads,
        num_key_value_heads: lm.num_key_value_heads,
        head_dim,
        vocab_size: 1,
        num_code_groups: 2,
        max_position_embeddings: lm.max_position_embeddings,
        rope_theta: lm.rope_theta,
        rms_norm_eps: lm.rms_norm_eps,
    };
    TalkerConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: lm.num_key_value_heads,
        head_dim,
        vocab_size: lm.vocab_size,
        text_vocab_size: lm.vocab_size,
        text_hidden_size: hidden_size,
        num_code_groups: 2,
        max_position_embeddings: lm.max_position_embeddings,
        rope_theta: lm.rope_theta,
        rms_norm_eps: lm.rms_norm_eps,
        code_predictor_config: placeholder_predictor,
        codec_language_id: HashMap::new(),
        spk_id: HashMap::new(),
    }
}
