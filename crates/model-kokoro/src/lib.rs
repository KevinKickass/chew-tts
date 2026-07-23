use anyhow::{Context, ensure};
use candle_core::{DType, pickle::PthTensors};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

mod albert;
pub use albert::KokoroAlbert;
mod lstm;
pub use lstm::KokoroBiLstm;
mod prosody;
pub use prosody::{KokoroProsody, KokoroProsodyFrontend, load_default_voice};

pub const CHECKPOINT_GROUPS: [&str; 5] = [
    "bert",
    "bert_encoder",
    "predictor",
    "decoder",
    "text_encoder",
];

#[derive(Debug, Clone, Deserialize)]
pub struct KokoroConfig {
    pub istftnet: IstftNetConfig,
    pub dim_in: usize,
    pub hidden_dim: usize,
    pub max_conv_dim: usize,
    pub max_dur: usize,
    pub multispeaker: bool,
    pub n_layer: usize,
    pub n_mels: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub plbert: AlbertConfig,
    pub vocab: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IstftNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlbertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
}

#[derive(Debug, Clone)]
pub struct KokoroTensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Debug, Clone)]
pub struct KokoroInspection {
    pub config: KokoroConfig,
    pub checkpoint: PathBuf,
    pub tensors: Vec<KokoroTensorInfo>,
    pub total_weight_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct KokoroTokens {
    pub ids: Vec<usize>,
    pub phoneme_count: usize,
    pub skipped_phonemes: usize,
}

#[derive(Debug, Clone)]
pub struct KokoroVoice {
    styles: Vec<f32>,
    entries: usize,
    style_width: usize,
}

pub struct KokoroCheckpoint {
    path: PathBuf,
    groups: BTreeMap<&'static str, PthTensors>,
}

impl KokoroConfig {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let path = model_dir.join("config.json");
        let config: Self = serde_json::from_slice(
            &std::fs::read(&path)
                .with_context(|| format!("could not read Kokoro config {}", path.display()))?,
        )
        .with_context(|| format!("could not parse Kokoro config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.n_token > 0, "Kokoro vocabulary must not be empty");
        ensure!(
            self.vocab.values().all(|token| *token < self.n_token),
            "Kokoro vocabulary contains a token outside 0..{}",
            self.n_token
        );
        ensure!(
            self.plbert.hidden_size % self.plbert.num_attention_heads == 0,
            "Kokoro Albert hidden size must be divisible by its head count"
        );
        ensure!(
            self.plbert.max_position_embeddings >= 2,
            "Kokoro context must hold boundary tokens"
        );
        ensure!(
            self.istftnet.upsample_rates.len() == self.istftnet.upsample_kernel_sizes.len(),
            "Kokoro iSTFTNet upsample rates and kernels differ in length"
        );
        ensure!(
            self.istftnet.resblock_kernel_sizes.len()
                == self.istftnet.resblock_dilation_sizes.len(),
            "Kokoro iSTFTNet residual kernels and dilation groups differ in length"
        );
        Ok(())
    }

    /// Mirrors KModel.forward: unknown phonemes are skipped and token zero is
    /// inserted at both sequence boundaries.
    pub fn tokenize_phonemes(&self, phonemes: &str) -> anyhow::Result<KokoroTokens> {
        let mut ids = Vec::with_capacity(phonemes.chars().count() + 2);
        let mut skipped_phonemes = 0;
        ids.push(0);
        for phoneme in phonemes.chars() {
            if let Some(token) = self.vocab.get(&phoneme.to_string()) {
                ids.push(*token);
            } else {
                skipped_phonemes += 1;
            }
        }
        let phoneme_count = ids.len() - 1;
        ids.push(0);
        ensure!(
            ids.len() <= self.plbert.max_position_embeddings,
            "Kokoro phoneme sequence requires {} tokens, model context is {}",
            ids.len(),
            self.plbert.max_position_embeddings
        );
        ensure!(
            phoneme_count > 0,
            "Kokoro phoneme sequence contains no known tokens"
        );
        Ok(KokoroTokens {
            ids,
            phoneme_count,
            skipped_phonemes,
        })
    }
}

impl KokoroCheckpoint {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut groups = BTreeMap::new();
        for group in CHECKPOINT_GROUPS {
            let tensors = PthTensors::new(&path, Some(group))
                .with_context(|| format!("could not read Kokoro checkpoint group {group:?}"))?;
            ensure!(
                !tensors.tensor_infos().is_empty(),
                "Kokoro checkpoint group {group:?} is empty"
            );
            groups.insert(group, tensors);
        }
        Ok(Self { path, groups })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensor_infos(&self) -> Vec<KokoroTensorInfo> {
        let mut infos = self
            .groups
            .iter()
            .flat_map(|(group, tensors)| {
                tensors.tensor_infos().values().map(move |tensor| {
                    let bytes = tensor.layout.shape().elem_count() * tensor.dtype.size_in_bytes();
                    KokoroTensorInfo {
                        name: format!("{group}.{}", tensor.name),
                        dtype: tensor.dtype,
                        shape: tensor.layout.dims().to_vec(),
                        bytes,
                    }
                })
            })
            .collect::<Vec<_>>();
        infos.sort_by(|left, right| left.name.cmp(&right.name));
        infos
    }

    pub fn tensor_f32(&self, group: &str, name: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
        let tensors = self
            .groups
            .get(group)
            .with_context(|| format!("unknown Kokoro checkpoint group {group:?}"))?;
        let tensor = tensors
            .get(name)
            .with_context(|| format!("could not load Kokoro tensor {group}.{name}"))?
            .with_context(|| format!("Kokoro tensor {group}.{name} is missing"))?;
        let shape = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((shape, values))
    }

    pub fn tensor_f16(
        &self,
        group: &str,
        name: &str,
    ) -> anyhow::Result<(Vec<usize>, Vec<half::f16>)> {
        let (shape, values) = self.tensor_f32(group, name)?;
        Ok((shape, values.into_iter().map(half::f16::from_f32).collect()))
    }

    /// Materialize a small tensor from every checkpoint group. This catches
    /// corrupt ZIP storage records that metadata-only inspection cannot see.
    pub fn validate_storage(&self) -> anyhow::Result<()> {
        for (group, name) in [
            ("bert", "module.embeddings.LayerNorm.bias"),
            ("bert_encoder", "module.bias"),
            ("predictor", "module.duration_proj.linear_layer.bias"),
            ("decoder", "module.generator.conv_post.bias"),
            ("text_encoder", "module.embedding.weight"),
        ] {
            self.tensor_f32(group, name)?;
        }
        Ok(())
    }
}

impl KokoroVoice {
    /// Load the raw float32 tensor written by torch.save for an official
    /// Kokoro voice pack. The pack has shape [context - 2, 1, style * 2].
    pub fn load(path: &Path, config: &KokoroConfig) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("could not open Kokoro voice {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
            .with_context(|| format!("could not read Kokoro voice ZIP {}", path.display()))?;
        let file_names = archive.file_names().map(str::to_owned).collect::<Vec<_>>();

        if let Some(byteorder_name) = file_names.iter().find(|name| name.ends_with("/byteorder")) {
            let mut byteorder = String::new();
            archive
                .by_name(byteorder_name)?
                .read_to_string(&mut byteorder)?;
            ensure!(
                byteorder.trim() == "little",
                "unsupported Kokoro voice byte order {:?}",
                byteorder.trim()
            );
        }

        let storage_name = file_names
            .iter()
            .find(|name| name.ends_with("/data/0"))
            .with_context(|| {
                format!(
                    "Kokoro voice {} has no primary tensor storage",
                    path.display()
                )
            })?;
        let mut bytes = Vec::new();
        archive.by_name(storage_name)?.read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() % std::mem::size_of::<f32>() == 0,
            "Kokoro voice storage has a partial float32 value"
        );
        let styles = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let style_width = config.style_dim * 2;
        ensure!(
            styles.len() % style_width == 0,
            "Kokoro voice width does not match model style dimension {}",
            config.style_dim
        );
        let entries = styles.len() / style_width;
        ensure!(
            entries >= config.plbert.max_position_embeddings - 2,
            "Kokoro voice contains {entries} contexts, expected at least {}",
            config.plbert.max_position_embeddings - 2
        );
        ensure!(
            styles.iter().all(|value| value.is_finite()),
            "Kokoro voice contains non-finite style values"
        );
        Ok(Self {
            styles,
            entries,
            style_width,
        })
    }

    pub fn entries(&self) -> usize {
        self.entries
    }

    pub fn style_width(&self) -> usize {
        self.style_width
    }

    /// Kokoro indexes the voice pack by the phoneme-string length minus one.
    pub fn style_for_phoneme_count(&self, phoneme_count: usize) -> anyhow::Result<&[f32]> {
        ensure!(
            phoneme_count > 0,
            "Kokoro voice requires at least one phoneme"
        );
        let index = phoneme_count - 1;
        ensure!(
            index < self.entries,
            "Kokoro voice has no style for {phoneme_count} phonemes"
        );
        let offset = index * self.style_width;
        Ok(&self.styles[offset..offset + self.style_width])
    }
}

pub fn inspect_model(model_dir: &Path) -> anyhow::Result<KokoroInspection> {
    let config = KokoroConfig::load(model_dir)?;
    let checkpoint = find_checkpoint(model_dir)?;
    let weights = KokoroCheckpoint::open(&checkpoint)?;
    weights.validate_storage()?;
    let tensors = weights.tensor_infos();
    let total_weight_bytes = tensors.iter().map(|tensor| tensor.bytes).sum();
    Ok(KokoroInspection {
        config,
        checkpoint,
        tensors,
        total_weight_bytes,
    })
}

fn find_checkpoint(model_dir: &Path) -> anyhow::Result<PathBuf> {
    for name in ["kokoro-v1_0.pth", "kokoro-v1_1-zh.pth", "model.pth"] {
        let path = model_dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "no Kokoro checkpoint found in {}; expected kokoro-v1_0.pth, kokoro-v1_1-zh.pth, or model.pth",
        model_dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_vocab_geometry() {
        let config: KokoroConfig = serde_json::from_str(
            r#"{
                "istftnet": {
                    "upsample_kernel_sizes": [20, 12],
                    "upsample_rates": [10, 6],
                    "gen_istft_hop_size": 5,
                    "gen_istft_n_fft": 20,
                    "resblock_dilation_sizes": [[1], [1]],
                    "resblock_kernel_sizes": [3, 7],
                    "upsample_initial_channel": 512
                },
                "dim_in": 64,
                "hidden_dim": 512,
                "max_conv_dim": 512,
                "max_dur": 50,
                "multispeaker": true,
                "n_layer": 3,
                "n_mels": 80,
                "n_token": 2,
                "style_dim": 128,
                "text_encoder_kernel_size": 5,
                "plbert": {
                    "hidden_size": 768,
                    "num_attention_heads": 12,
                    "intermediate_size": 2048,
                    "max_position_embeddings": 512,
                    "num_hidden_layers": 12
                },
                "vocab": {"a": 2}
            }"#,
        )
        .unwrap();
        assert!(config.validate().is_err());
    }
}
