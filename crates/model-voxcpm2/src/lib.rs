mod backbone;
mod config;
mod decoder;
mod projections;

pub use backbone::{VoxCpm2BaseBackbone, VoxCpm2TransformerBackbones, VoxCpm2TransformerSmoke};
pub use config::{
    AudioVaeConfig, CfmConfig, DitConfig, LocalTransformerConfig, MiniCpm4Config,
    RopeScalingConfig, VoxCpm2Config,
};
pub use decoder::VoxCpm2AudioDecoder;
pub use projections::{VoxCpm2ProjectionOutputs, VoxCpm2Projections};

use chew_safetensors::{MappedSafetensors, TensorInfo};
use std::fs;
use std::path::{Path, PathBuf};

pub struct VoxCpm2Inspection {
    pub config: VoxCpm2Config,
    pub weight_path: PathBuf,
    pub audio_vae_path: PathBuf,
    pub tensors: Vec<TensorInfo>,
    pub total_weight_bytes: u64,
}

pub fn inspect_model(model_dir: impl AsRef<Path>) -> Result<VoxCpm2Inspection, Error> {
    let model_dir = model_dir.as_ref();
    let config: VoxCpm2Config = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    config.validate().map_err(Error::InvalidConfig)?;
    let weight_path = model_dir.join("model.safetensors");
    let audio_vae_path = model_dir.join("audiovae.pth");
    if !weight_path.is_file() {
        return Err(Error::MissingFile(weight_path));
    }
    if !audio_vae_path.is_file() {
        return Err(Error::MissingFile(audio_vae_path));
    }
    let weights = MappedSafetensors::open(&weight_path)?;
    let tensors = weights.tensor_infos()?;
    for required in [
        "base_lm.embed_tokens.weight",
        "residual_lm.layers.0.self_attn.q_proj.weight",
        "feat_encoder.encoder.layers.0.self_attn.q_proj.weight",
        "feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.weight",
        "stop_head.weight",
    ] {
        require(&tensors, required)?;
    }
    let total_weight_bytes = tensors.iter().map(|tensor| tensor.bytes as u64).sum();
    Ok(VoxCpm2Inspection {
        config,
        weight_path,
        audio_vae_path,
        tensors,
        total_weight_bytes,
    })
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
    #[error("invalid VoxCPM2 configuration: {0}")]
    InvalidConfig(#[source] anyhow::Error),
    #[error("missing VoxCPM2 file {0}")]
    MissingFile(PathBuf),
    #[error("VoxCPM2 checkpoint is missing tensor {0}")]
    MissingTensor(String),
}
