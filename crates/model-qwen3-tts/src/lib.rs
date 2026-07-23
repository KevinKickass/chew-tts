mod codec;
mod config;
mod cuda;
mod frontend;
mod predictor;
mod sampling;

pub use codec::{CodecQuantizer, CodecTransformerSession};
pub use config::{
    CodePredictorConfig, ModelType, Qwen3TtsConfig, SpeakerEncoderConfig, TalkerConfig,
};
pub use cuda::{
    TalkerDecoderLayer, TalkerGenerationSession, TalkerLayerKvCache, TalkerLayerScratch,
    TalkerTransformer,
};
pub use frontend::{TalkerFrontend, VoiceDesignInputs};
pub use predictor::{CodePredictorGenerationSession, CodePredictorTransformer};

use chew_safetensors::{MappedSafetensors, TensorInfo};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

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

pub struct HostF32Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

struct WeightSet {
    files: Vec<MappedSafetensors>,
    paths: Vec<PathBuf>,
    tensors: Vec<TensorInfo>,
    locations: HashMap<String, usize>,
}

static WEIGHT_SETS: OnceLock<Mutex<HashMap<PathBuf, Arc<WeightSet>>>> = OnceLock::new();

fn weight_set(model_dir: &Path) -> Result<Arc<WeightSet>, Error> {
    let key = fs::canonicalize(model_dir)?;
    let cache = WEIGHT_SETS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(weights) = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .cloned()
    {
        return Ok(weights);
    }

    let mut paths = fs::read_dir(&key)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(Error::MissingWeights);
    }
    let mut files = Vec::with_capacity(paths.len());
    let mut tensors = Vec::new();
    let mut locations = HashMap::new();
    for (file_index, path) in paths.iter().enumerate() {
        let mapped = MappedSafetensors::open(path)?;
        for tensor in mapped.tensor_infos()? {
            locations.insert(tensor.name.clone(), file_index);
            tensors.push(tensor);
        }
        files.push(mapped);
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let weights = Arc::new(WeightSet {
        files,
        paths,
        tensors,
        locations,
    });
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, Arc::clone(&weights));
    Ok(weights)
}

pub fn inspect_model(model_dir: impl AsRef<Path>) -> Result<ModelInspection, Error> {
    let model_dir = model_dir.as_ref();
    let config: Qwen3TtsConfig = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    config.validate().map_err(Error::InvalidConfig)?;

    let weights = weight_set(model_dir)?;
    validate_required_tensors(&weights.tensors)?;
    let total_weight_bytes = weights
        .tensors
        .iter()
        .map(|tensor| tensor.bytes as u64)
        .sum();

    Ok(ModelInspection {
        config,
        weight_files: weights.paths.clone(),
        tensors: weights.tensors.clone(),
        total_weight_bytes,
    })
}

pub fn load_f16_tensor(
    model_dir: impl AsRef<Path>,
    tensor_name: &str,
) -> Result<HostF16Tensor, Error> {
    let weights = weight_set(model_dir.as_ref())?;
    let file_index = weights
        .locations
        .get(tensor_name)
        .copied()
        .ok_or_else(|| Error::TensorNotFound(tensor_name.to_string()))?;
    let (shape, values) = weights.files[file_index].tensor_f16(tensor_name)?;
    Ok(HostF16Tensor { shape, values })
}

pub fn load_f32_tensor(
    model_dir: impl AsRef<Path>,
    tensor_name: &str,
) -> Result<HostF32Tensor, Error> {
    let weights = weight_set(model_dir.as_ref())?;
    let file_index = weights
        .locations
        .get(tensor_name)
        .copied()
        .ok_or_else(|| Error::TensorNotFound(tensor_name.to_string()))?;
    let (shape, values) = weights.files[file_index].tensor_f32(tensor_name)?;
    Ok(HostF32Tensor { shape, values })
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
