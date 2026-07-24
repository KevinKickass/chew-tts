mod backbone;
mod config;
mod decoder;
mod dit;
mod engine;
mod projections;

pub use backbone::{VoxCpm2BaseBackbone, VoxCpm2TransformerBackbones, VoxCpm2TransformerSmoke};
pub use config::{
    AudioVaeConfig, CfmConfig, DitConfig, LocalTransformerConfig, MiniCpm4Config,
    RopeScalingConfig, VoxCpm2Config,
};
pub use decoder::{VoxCpm2AudioDecoder, VoxCpm2AudioEncoder};
pub use dit::VoxCpm2FlowDecoder;
pub use engine::{VoxCpm2Engine, VoxCpm2Generation};
pub use projections::{VoxCpm2ProjectionOutputs, VoxCpm2Projections};

/// Convert the official AudioVAE checkpoint once into mmap-friendly
/// Safetensors without a Python/PyTorch installation.
pub fn convert_audiovae_checkpoint(source: &Path, destination: &Path) -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use candle_core::{DType as CandleDType, pickle::PthTensors};
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize_to_file};
    use std::collections::HashMap;

    ensure!(source.is_file(), "missing checkpoint {}", source.display());
    let checkpoint = PthTensors::new(source, Some("state_dict"))
        .with_context(|| format!("could not read {}", source.display()))?;
    ensure!(
        !checkpoint.tensor_infos().is_empty(),
        "AudioVAE checkpoint contains no tensors"
    );
    struct OwnedTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }
    let mut owned = HashMap::new();
    for name in checkpoint.tensor_infos().keys() {
        let tensor = checkpoint
            .get(name)?
            .with_context(|| format!("AudioVAE tensor {name:?} disappeared"))?;
        let (dtype, bytes) = match tensor.dtype() {
            CandleDType::F32 => {
                let values = tensor.flatten_all()?.to_vec1::<f32>()?;
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                (Dtype::F32, bytes)
            }
            CandleDType::I32 => {
                let values = tensor.flatten_all()?.to_vec1::<i32>()?;
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                (Dtype::I32, bytes)
            }
            other => anyhow::bail!("unsupported AudioVAE dtype {other:?} for {name}"),
        };
        owned.insert(
            name.clone(),
            OwnedTensor {
                dtype,
                shape: tensor.dims().to_vec(),
                bytes,
            },
        );
    }
    // Candle intentionally skips PyTorch's IntStorage. This immutable buffer is
    // also declared in the official config, so reconstruct it from that source.
    let config_path = source
        .parent()
        .context("AudioVAE checkpoint has no parent directory")?
        .join("config.json");
    let config: VoxCpm2Config = serde_json::from_slice(
        &std::fs::read(&config_path)
            .with_context(|| format!("could not read {}", config_path.display()))?,
    )?;
    let boundaries = config.audio_vae_config.sr_bin_boundaries;
    let mut boundary_bytes = Vec::with_capacity(boundaries.len() * 4);
    for boundary in &boundaries {
        boundary_bytes.extend_from_slice(&(*boundary as i32).to_le_bytes());
    }
    owned.insert(
        "decoder.sr_bin_boundaries".to_string(),
        OwnedTensor {
            dtype: Dtype::I32,
            shape: vec![boundaries.len()],
            bytes: boundary_bytes,
        },
    );
    let views = owned
        .iter()
        .map(|(name, tensor)| {
            Ok((
                name.as_str(),
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes)?,
            ))
        })
        .collect::<Result<HashMap<_, _>, safetensors::SafeTensorError>>()?;
    serialize_to_file(
        &views,
        Some(HashMap::from([
            ("format".into(), "pt".into()),
            ("source".into(), "openbmb/VoxCPM2 audiovae.pth".into()),
        ])),
        destination,
    )?;
    Ok(())
}

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
    let safe_audio_vae_path = model_dir.join("audiovae.safetensors");
    let audio_vae_path = if safe_audio_vae_path.is_file() {
        safe_audio_vae_path
    } else {
        model_dir.join("audiovae.pth")
    };
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
