use anyhow::ensure;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VoxCpm2Config {
    pub architecture: String,
    pub lm_config: MiniCpm4Config,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: usize,
    pub residual_lm_num_layers: usize,
    pub residual_lm_no_rope: bool,
    pub encoder_config: LocalTransformerConfig,
    pub dit_config: DitConfig,
    pub audio_vae_config: AudioVaeConfig,
    pub max_length: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpm4Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub kv_channels: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub scale_depth: f64,
    pub scale_emb: f64,
    pub dim_model_base: usize,
    pub use_mup: bool,
    pub rope_scaling: RopeScalingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(rename = "type")]
    pub kind: String,
    pub short_factor: Vec<f32>,
    pub long_factor: Vec<f32>,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LocalTransformerConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: usize,
    #[serde(alias = "mean_mode")]
    pub mean_mode: bool,
    pub cfm_config: CfmConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CfmConfig {
    pub sigma_min: f64,
    pub solver: String,
    pub t_scheduler: String,
    pub inference_cfg_rate: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
}

impl VoxCpm2Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.architecture == "voxcpm2",
            "unsupported VoxCPM architecture"
        );
        ensure!(self.dtype == "bfloat16", "native VoxCPM2 requires BF16");
        let lm = &self.lm_config;
        ensure!(
            lm.hidden_size == lm.num_attention_heads * lm.kv_channels,
            "MiniCPM query geometry disagrees"
        );
        ensure!(
            lm.num_attention_heads % lm.num_key_value_heads == 0,
            "MiniCPM grouped-query geometry disagrees"
        );
        ensure!(
            !lm.use_mup,
            "MiniCPM muP residual scaling is not implemented"
        );
        ensure!(
            lm.rope_scaling.kind == "longrope"
                && lm.rope_scaling.short_factor.len() == lm.kv_channels / 2
                && lm.rope_scaling.long_factor.len() == lm.kv_channels / 2,
            "invalid MiniCPM LongRoPE geometry"
        );
        ensure!(
            lm.max_position_embeddings == lm.rope_scaling.original_max_position_embeddings,
            "scaled LongRoPE amplitude is not implemented"
        );
        ensure!(
            self.patch_size > 0 && self.feat_dim > 0 && self.scalar_quantization_latent_dim > 0,
            "invalid VoxCPM feature geometry"
        );
        ensure!(
            self.encoder_config.hidden_dim % self.encoder_config.num_heads == 0
                && self.dit_config.hidden_dim % self.dit_config.num_heads == 0,
            "local transformer head geometry disagrees"
        );
        ensure!(
            self.dit_config.cfm_config.solver == "euler"
                && self.dit_config.cfm_config.t_scheduler == "log-norm",
            "unsupported VoxCPM flow schedule"
        );
        ensure!(
            self.audio_vae_config.latent_dim == self.feat_dim,
            "VoxCPM feature and VAE latent dimensions disagree"
        );
        ensure!(
            self.audio_vae_config.sample_rate == 16_000
                && self.audio_vae_config.out_sample_rate == 48_000,
            "unexpected VoxCPM2 audio rates"
        );
        Ok(())
    }

    pub fn total_transformer_layers(&self) -> usize {
        self.lm_config.num_hidden_layers
            + self.residual_lm_num_layers
            + self.encoder_config.num_layers
            + self.dit_config.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_official_geometry() {
        let config: VoxCpm2Config =
            serde_json::from_str(include_str!("../../../tests/data/voxcpm2-config.json")).unwrap();
        config.validate().unwrap();
        assert_eq!(config.total_transformer_layers(), 60);
        assert_eq!(config.audio_vae_config.out_sample_rate, 48_000);
    }
}
