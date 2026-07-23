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
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Base,
    CustomVoice,
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
}

impl TalkerConfig {
    fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads * self.head_dim != self.hidden_size {
            return Err(format!(
                "talker query geometry mismatch: {} heads * {} != hidden {}",
                self.num_attention_heads, self.head_dim, self.hidden_size
            ));
        }
        if self.num_key_value_heads > self.num_attention_heads {
            return Err("talker KV heads exceed query heads".into());
        }
        if self.code_predictor_config.num_code_groups != self.num_code_groups {
            return Err("talker and code predictor disagree on code groups".into());
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
