use anyhow::ensure;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceConfig {
    pub acoustic_vae_dim: usize,
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,
    pub architectures: Vec<String>,
    pub decoder_config: DecoderConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub model_type: String,
    pub torch_dtype: String,
    pub tts_backbone_num_hidden_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTokenizerConfig {
    pub causal: bool,
    pub channels: usize,
    pub decoder_n_filters: usize,
    pub decoder_ratios: Vec<usize>,
    pub layernorm_eps: f64,
    pub model_type: String,
    pub vae_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub torch_dtype: String,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    pub ddpm_num_inference_steps: usize,
    pub ddpm_num_steps: usize,
    pub head_ffn_ratio: f64,
    pub head_layers: usize,
    pub hidden_size: usize,
    pub latent_size: usize,
    pub model_type: String,
    pub prediction_type: String,
    pub rms_norm_eps: f64,
    pub speech_vae_dim: usize,
}

impl VibeVoiceConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.model_type == "vibevoice_streaming",
            "unsupported VibeVoice model type {:?}",
            self.model_type
        );
        ensure!(
            self.architectures
                .iter()
                .any(|name| name == "VibeVoiceStreamingForConditionalGenerationInference"),
            "checkpoint is not a VibeVoice streaming inference model"
        );
        ensure!(
            self.decoder_config.model_type == "qwen2",
            "unsupported VibeVoice decoder {:?}",
            self.decoder_config.model_type
        );
        ensure!(
            self.acoustic_tokenizer_config.model_type == "vibevoice_acoustic_tokenizer",
            "unsupported VibeVoice acoustic tokenizer {:?}",
            self.acoustic_tokenizer_config.model_type
        );
        ensure!(
            self.diffusion_head_config.model_type == "vibevoice_diffusion_head",
            "unsupported VibeVoice diffusion head {:?}",
            self.diffusion_head_config.model_type
        );
        ensure!(
            self.torch_dtype == "bfloat16" && self.decoder_config.torch_dtype == "bfloat16",
            "VibeVoice-Realtime native path requires BF16 weights"
        );
        let decoder = &self.decoder_config;
        ensure!(
            decoder.hidden_size % decoder.num_attention_heads == 0,
            "VibeVoice hidden size is not divisible by its attention heads"
        );
        ensure!(
            decoder.num_attention_heads % decoder.num_key_value_heads == 0,
            "VibeVoice attention heads are not divisible by KV heads"
        );
        ensure!(
            self.tts_backbone_num_hidden_layers < decoder.num_hidden_layers,
            "VibeVoice TTS backbone must leave at least one text-only layer"
        );
        ensure!(
            self.acoustic_vae_dim == self.acoustic_tokenizer_config.vae_dim
                && self.acoustic_vae_dim == self.diffusion_head_config.latent_size
                && self.acoustic_vae_dim == self.diffusion_head_config.speech_vae_dim,
            "VibeVoice acoustic latent dimensions disagree"
        );
        ensure!(
            self.diffusion_head_config.hidden_size == decoder.hidden_size,
            "VibeVoice diffusion and decoder hidden sizes disagree"
        );
        ensure!(
            self.acoustic_tokenizer_config.causal,
            "VibeVoice-Realtime requires a causal acoustic decoder"
        );
        ensure!(
            self.acoustic_tokenizer_config.channels == 1,
            "only mono VibeVoice audio is supported"
        );
        ensure!(
            !self.acoustic_tokenizer_config.decoder_ratios.is_empty()
                && self
                    .acoustic_tokenizer_config
                    .decoder_ratios
                    .iter()
                    .all(|ratio| *ratio > 0),
            "VibeVoice decoder ratios must be non-zero"
        );
        ensure!(
            self.diffusion_head_config.ddpm_num_inference_steps > 0
                && self.diffusion_head_config.ddpm_num_steps
                    >= self.diffusion_head_config.ddpm_num_inference_steps,
            "invalid VibeVoice diffusion schedule"
        );
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.decoder_config.hidden_size / self.decoder_config.num_attention_heads
    }

    pub fn text_layers(&self) -> usize {
        self.decoder_config.num_hidden_layers - self.tts_backbone_num_hidden_layers
    }

    pub fn samples_per_latent(&self) -> usize {
        self.acoustic_tokenizer_config
            .decoder_ratios
            .iter()
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_official_realtime_geometry() {
        let config: VibeVoiceConfig =
            serde_json::from_str(include_str!("../../../tests/data/vibevoice-config.json"))
                .expect("fixture parses");
        config.validate().expect("official geometry validates");
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.text_layers(), 4);
        assert_eq!(config.samples_per_latent(), 3_200);
    }
}
