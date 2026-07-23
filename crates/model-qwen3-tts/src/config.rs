use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    pub model_type: String,
    pub tokenizer_type: String,
    pub tts_model_size: String,
    pub tts_model_type: ModelType,
    pub talker_config: TalkerConfig,
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

impl Qwen3TtsConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "qwen3_tts" {
            return Err(format!(
                "expected model_type qwen3_tts, got {}",
                self.model_type
            ));
        }
        if self.talker_config.num_code_groups < 2 {
            return Err("num_code_groups must be at least 2".into());
        }
        self.talker_config.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ModelType {
    #[serde(rename = "base")]
    Base,
    #[serde(rename = "custom_voice", alias = "customvoice")]
    CustomVoice,
    #[serde(rename = "voice_design", alias = "voicedesign")]
    VoiceDesign,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    pub enc_dim: usize,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    pub num_code_groups: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub code_predictor_config: CodePredictorConfig,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
}

impl TalkerConfig {
    fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
        {
            return Err("talker attention geometry must be non-zero".into());
        }
        if self.num_key_value_heads > self.num_attention_heads {
            return Err("talker KV heads exceed query heads".into());
        }
        if self.code_predictor_config.num_code_groups != self.num_code_groups {
            return Err("talker and code predictor disagree on code groups".into());
        }
        if self
            .spk_id
            .values()
            .any(|id| *id as usize >= self.vocab_size)
        {
            return Err("speaker ID is outside the codec vocabulary".into());
        }
        self.code_predictor_config.validate()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_code_groups: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl CodePredictorConfig {
    fn validate(&self) -> Result<(), String> {
        if self.num_key_value_heads > self.num_attention_heads {
            return Err("code predictor KV heads exceed query heads".into());
        }
        if self.vocab_size == 0 || self.num_hidden_layers == 0 {
            return Err("code predictor must have layers and a vocabulary".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talker(hidden_size: usize) -> TalkerConfig {
        TalkerConfig {
            hidden_size,
            intermediate_size: 3_072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 3_072,
            text_vocab_size: 151_936,
            text_hidden_size: 1_024,
            num_code_groups: 16,
            max_position_embeddings: 32_768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            code_predictor_config: CodePredictorConfig {
                hidden_size: 1_024,
                intermediate_size: 3_072,
                num_hidden_layers: 5,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 64,
                vocab_size: 2_048,
                num_code_groups: 16,
                max_position_embeddings: 32_768,
                rope_theta: 1_000_000.0,
                rms_norm_eps: 1e-6,
            },
            codec_language_id: HashMap::new(),
            spk_id: HashMap::new(),
        }
    }

    #[test]
    fn accepts_official_06b_rectangular_query_projection() {
        let config = talker(1_024);
        assert_eq!(
            config.num_attention_heads * config.head_dim,
            2_048,
            "the official 0.6B talker intentionally projects queries wider than hidden"
        );
        assert!(config.validate().is_ok());
    }
}
