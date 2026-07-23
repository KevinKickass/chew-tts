mod config;
mod cuda;

pub use config::{
    CodePredictorConfig, ModelType, Qwen3TtsConfig, SpeakerEncoderConfig, TalkerConfig,
};
pub use cuda::TalkerDecoderLayer;

use chew_safetensors::{MappedSafetensors, TensorInfo};
use std::fs;
use std::path::{Path, PathBuf};

pub struct ModelInspection {
    pub config: Qwen3TtsConfig,
    pub weight_files: Vec<PathBuf>,
    pub tensors: Vec<TensorInfo>,
    pub total_weight_bytes: u64,
}

pub struct HostF16Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<half::f16>,
}

pub fn inspect_model(model_dir: impl AsRef<Path>) -> Result<ModelInspection, Error> {
    let model_dir = model_dir.as_ref();
    let config: Qwen3TtsConfig = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    config.validate().map_err(Error::InvalidConfig)?;

    let mut weight_files = fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    weight_files.sort();
    if weight_files.is_empty() {
        return Err(Error::MissingWeights);
    }

    let mut tensors = Vec::new();
    let mut total_weight_bytes = 0u64;
    for path in &weight_files {
        let mapped = MappedSafetensors::open(path)?;
        let file_infos = mapped.tensor_infos()?;
        total_weight_bytes += file_infos
            .iter()
            .map(|tensor| tensor.bytes as u64)
            .sum::<u64>();
        tensors.extend(file_infos);
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    validate_required_tensors(&tensors)?;

    Ok(ModelInspection {
        config,
        weight_files,
        tensors,
        total_weight_bytes,
    })
}

pub fn load_f16_tensor(
    model_dir: impl AsRef<Path>,
    tensor_name: &str,
) -> Result<HostF16Tensor, Error> {
    let model_dir = model_dir.as_ref();
    let mut weight_files = fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    weight_files.sort();
    for path in weight_files {
        let mapped = MappedSafetensors::open(path)?;
        if mapped
            .tensor_infos()?
            .iter()
            .any(|tensor| tensor.name == tensor_name)
        {
            let (shape, values) = mapped.tensor_f16(tensor_name)?;
            return Ok(HostF16Tensor { shape, values });
        }
    }
    Err(Error::TensorNotFound(tensor_name.to_string()))
}

fn validate_required_tensors(tensors: &[TensorInfo]) -> Result<(), Error> {
    for required in [
        "talker.model.codec_embedding.weight",
        "talker.model.text_embedding.weight",
        "talker.codec_head.weight",
        "talker.code_predictor.model.norm.weight",
    ] {
        if !tensors.iter().any(|tensor| tensor.name == required) {
            return Err(Error::MissingTensor(required));
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("weights: {0}")]
    Weights(#[from] chew_safetensors::Error),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("no .safetensors weight files found")]
    MissingWeights,
    #[error("required tensor is missing: {0}")]
    MissingTensor(&'static str),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
}
