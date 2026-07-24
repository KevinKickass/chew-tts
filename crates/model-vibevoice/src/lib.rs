mod backbone;
mod config;
mod voice;

pub use backbone::VibeVoiceBackbones;
pub use config::{AcousticTokenizerConfig, DecoderConfig, DiffusionHeadConfig, VibeVoiceConfig};
pub use voice::{VibeVoicePrompt, VibeVoicePromptBranch, VibeVoicePromptKv};

use chew_safetensors::{MappedSafetensors, TensorInfo};
use std::fs;
use std::path::{Path, PathBuf};

pub struct VibeVoiceInspection {
    pub config: VibeVoiceConfig,
    pub weight_path: PathBuf,
    pub tensors: Vec<TensorInfo>,
    pub total_weight_bytes: u64,
}

pub fn inspect_model(model_dir: impl AsRef<Path>) -> Result<VibeVoiceInspection, Error> {
    let model_dir = model_dir.as_ref();
    let config: VibeVoiceConfig =
        serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    config.validate().map_err(Error::InvalidConfig)?;

    let weight_path = model_dir.join("model.safetensors");
    if !weight_path.is_file() {
        return Err(Error::MissingWeights(weight_path));
    }
    let weights = MappedSafetensors::open(&weight_path)?;
    let tensors = weights.tensor_infos()?;
    validate_required_tensors(&config, &tensors)?;
    let total_weight_bytes = tensors.iter().map(|tensor| tensor.bytes as u64).sum();
    Ok(VibeVoiceInspection {
        config,
        weight_path,
        tensors,
        total_weight_bytes,
    })
}

fn validate_required_tensors(
    config: &VibeVoiceConfig,
    tensors: &[TensorInfo],
) -> Result<(), Error> {
    for required in [
        "model.language_model.embed_tokens.weight",
        "model.tts_input_types.weight",
        "model.acoustic_connector.fc1.weight",
        "model.prediction_head.noisy_images_proj.weight",
        "tts_eos_classifier.fc1.weight",
    ] {
        require(tensors, required)?;
    }
    require(
        tensors,
        &format!(
            "model.language_model.layers.{}.self_attn.q_proj.weight",
            config.text_layers() - 1
        ),
    )?;
    require(
        tensors,
        &format!(
            "model.tts_language_model.layers.{}.self_attn.q_proj.weight",
            config.tts_backbone_num_hidden_layers - 1
        ),
    )?;
    Ok(())
}

fn require(tensors: &[TensorInfo], name: &str) -> Result<(), Error> {
    if tensors.iter().any(|tensor| tensor.name == name) {
        Ok(())
    } else {
        Err(Error::MissingTensor(name.to_owned()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Safetensors: {0}")]
    Safetensors(#[from] chew_safetensors::Error),
    #[error("invalid VibeVoice configuration: {0}")]
    InvalidConfig(#[source] anyhow::Error),
    #[error("missing VibeVoice weights {0}")]
    MissingWeights(PathBuf),
    #[error("VibeVoice checkpoint is missing tensor {0}")]
    MissingTensor(String),
}
