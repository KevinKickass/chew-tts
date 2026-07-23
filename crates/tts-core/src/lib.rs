use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: String,
    pub voice: Option<String>,
    pub instruction: Option<String>,
    pub speed: f32,
    pub temperature: f32,
}

impl SynthesisRequest {
    pub fn validate(&self) -> Result<(), TtsError> {
        if self.text.trim().is_empty() {
            return Err(TtsError::InvalidRequest("text must not be empty".into()));
        }
        if self.language.trim().is_empty() {
            return Err(TtsError::InvalidRequest(
                "language must not be empty".into(),
            ));
        }
        if !self.speed.is_finite() || self.speed <= 0.0 {
            return Err(TtsError::InvalidRequest(
                "speed must be finite and greater than zero".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(TtsError::InvalidRequest(
                "temperature must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate) / f64::from(self.channels)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub voice_cloning: bool,
    pub voice_design: bool,
    pub preset_voices: bool,
    pub streaming: bool,
}

pub trait TtsModel: Send {
    fn capabilities(&self) -> ModelCapabilities;
    fn synthesize(&mut self, request: &SynthesisRequest) -> Result<AudioBuffer, TtsError>;
}

pub trait TtsModelLoader {
    type Model: TtsModel;

    fn load(model_dir: &Path) -> Result<Self::Model, TtsError>;
}

#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Backend(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_accounts_for_channels() {
        let audio = AudioBuffer {
            samples: vec![0.0; 48_000],
            sample_rate: 24_000,
            channels: 2,
        };
        assert_eq!(audio.duration_seconds(), 1.0);
    }

    #[test]
    fn rejects_empty_text_and_invalid_speed() {
        let mut request = SynthesisRequest {
            text: " ".into(),
            language: "de".into(),
            voice: None,
            instruction: None,
            speed: 1.0,
            temperature: 0.9,
        };
        assert!(request.validate().is_err());
        request.text = "Hallo".into();
        request.speed = f32::NAN;
        assert!(request.validate().is_err());
    }
}
