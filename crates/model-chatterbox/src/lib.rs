use anyhow::{Context, ensure};
use candle_core::{DType, pickle::PthTensors};
use chew_safetensors::{MappedSafetensors, TensorInfo};
use std::path::{Path, PathBuf};

mod cuda;
mod frontend;
mod tokenizer;

pub use cuda::{ChatterboxT3Layer, ChatterboxT3Transformer};
pub use frontend::{ChatterboxT3Frontend, ChatterboxT3Prefix};
pub use tokenizer::{ChatterboxTokenizer, normalize_multilingual_text};

pub const TEXT_VOCAB_SIZE: usize = 2_454;
pub const SPEECH_VOCAB_SIZE: usize = 8_194;
pub const HIDDEN_SIZE: usize = 1_024;
pub const INTERMEDIATE_SIZE: usize = 4_096;
pub const LAYERS: usize = 30;
pub const ATTENTION_HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
pub const MAX_TEXT_TOKENS: usize = 2_048;
pub const MAX_SPEECH_TOKENS: usize = 4_096;
pub const SPEECH_CONDITION_TOKENS: usize = 150;
pub const START_TEXT_TOKEN: usize = 255;
pub const STOP_TEXT_TOKEN: usize = 0;
pub const START_SPEECH_TOKEN: usize = 6_561;
pub const STOP_SPEECH_TOKEN: usize = 6_562;

#[derive(Debug, Clone)]
pub struct ChatterboxInspection {
    pub t3_path: PathBuf,
    pub s3gen_path: PathBuf,
    pub voice_encoder_path: PathBuf,
    pub t3_tensors: Vec<TensorInfo>,
    pub s3gen_tensor_count: usize,
    pub voice_encoder_tensor_count: usize,
    pub total_weight_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ChatterboxConditioning {
    pub speaker_embedding: Vec<f32>,
    pub prompt_speech_tokens: Vec<i32>,
    pub emotion_exaggeration: f32,
    pub s3_prompt_tokens: Vec<i32>,
    pub s3_prompt_features: Vec<f32>,
    pub s3_prompt_feature_frames: usize,
    pub s3_embedding: Vec<f32>,
}

impl ChatterboxConditioning {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let t3 = PthTensors::new(path, Some("t3"))
            .with_context(|| format!("could not read T3 conditioning {}", path.display()))?;
        let s3 = PthTensors::new(path, Some("gen"))
            .with_context(|| format!("could not read S3Gen conditioning {}", path.display()))?;
        let speaker_embedding = pth_f32(&t3, "speaker_emb", &[1, 256])?;
        let prompt_speech_tokens = pth_i32(&t3, "cond_prompt_speech_tokens", &[1, 150])?;
        let emotion = pth_f32(&t3, "emotion_adv", &[1, 1, 1])?;
        let s3_prompt_tokens = pth_i32(&s3, "prompt_token", &[1, 157])?;
        let s3_prompt_features = pth_f32(&s3, "prompt_feat", &[1, 314, 80])?;
        let s3_embedding = pth_f32(&s3, "embedding", &[1, 192])?;
        Ok(Self {
            speaker_embedding,
            prompt_speech_tokens,
            emotion_exaggeration: emotion[0],
            s3_prompt_tokens,
            s3_prompt_features,
            s3_prompt_feature_frames: 314,
            s3_embedding,
        })
    }
}

pub fn inspect_model(model_dir: &Path) -> anyhow::Result<ChatterboxInspection> {
    let t3_path = model_dir.join("t3_mtl23ls_v3.safetensors");
    ensure!(
        t3_path.is_file(),
        "missing Chatterbox V3 T3 weights {}",
        t3_path.display()
    );
    let t3 = MappedSafetensors::open(&t3_path)?;
    let t3_tensors = t3.tensor_infos()?;
    validate_t3(&t3, &t3_tensors)?;

    let (s3gen_path, s3gen_tensor_count, s3gen_bytes) =
        inspect_s3gen(model_dir).context("could not inspect Chatterbox S3Gen")?;
    let voice_encoder_path = model_dir.join("ve.pt");
    let voice_encoder = PthTensors::new(&voice_encoder_path, None).with_context(|| {
        format!(
            "could not read Chatterbox voice encoder {}",
            voice_encoder_path.display()
        )
    })?;
    ensure!(
        voice_encoder.tensor_infos().len() == 16,
        "Chatterbox voice encoder has {} tensors, expected 16",
        voice_encoder.tensor_infos().len()
    );
    let projection = voice_encoder
        .get("proj.weight")?
        .context("Chatterbox voice encoder is missing proj.weight")?;
    ensure!(
        projection.dims() == [256, 256],
        "Chatterbox voice encoder projection has shape {:?}, expected [256, 256]",
        projection.dims()
    );
    let voice_encoder_bytes = voice_encoder
        .tensor_infos()
        .values()
        .map(|tensor| tensor.layout.shape().elem_count() * tensor.dtype.size_in_bytes())
        .sum::<usize>();
    let t3_bytes = t3_tensors.iter().map(|tensor| tensor.bytes).sum::<usize>();

    Ok(ChatterboxInspection {
        t3_path,
        s3gen_path,
        voice_encoder_path,
        t3_tensors,
        s3gen_tensor_count,
        voice_encoder_tensor_count: voice_encoder.tensor_infos().len(),
        total_weight_bytes: t3_bytes + s3gen_bytes + voice_encoder_bytes,
    })
}

fn validate_t3(t3: &MappedSafetensors, tensors: &[TensorInfo]) -> anyhow::Result<()> {
    for (name, expected) in [
        ("text_emb.weight", &[TEXT_VOCAB_SIZE, HIDDEN_SIZE][..]),
        ("speech_emb.weight", &[SPEECH_VOCAB_SIZE, HIDDEN_SIZE]),
        ("text_head.weight", &[TEXT_VOCAB_SIZE, HIDDEN_SIZE]),
        ("speech_head.weight", &[SPEECH_VOCAB_SIZE, HIDDEN_SIZE]),
        (
            "text_pos_emb.emb.weight",
            &[MAX_TEXT_TOKENS + 2, HIDDEN_SIZE],
        ),
        (
            "speech_pos_emb.emb.weight",
            &[MAX_SPEECH_TOKENS + 4, HIDDEN_SIZE],
        ),
        ("cond_enc.spkr_enc.weight", &[HIDDEN_SIZE, 256]),
    ] {
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("Chatterbox V3 T3 is missing {name}"))?;
        ensure!(
            tensor.shape == expected,
            "Chatterbox V3 {name} has shape {:?}, expected {expected:?}",
            tensor.shape
        );
    }
    for layer in [0, LAYERS - 1] {
        let name = format!("tfmr.layers.{layer}.self_attn.q_proj.weight");
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("Chatterbox V3 T3 is missing {name}"))?;
        ensure!(
            tensor.shape == [HIDDEN_SIZE, HIDDEN_SIZE],
            "Chatterbox V3 {name} has unexpected shape {:?}",
            tensor.shape
        );
    }
    // Materialize a representative tensor so inspection also validates its
    // data range rather than trusting the Safetensors header alone.
    let (_, values) = t3.tensor_f32("cond_enc.emotion_adv_fc.weight")?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "Chatterbox V3 emotion conditioning contains non-finite weights"
    );
    Ok(())
}

fn inspect_s3gen(model_dir: &Path) -> anyhow::Result<(PathBuf, usize, usize)> {
    for name in ["s3gen_v3.safetensors", "s3gen.safetensors"] {
        let path = model_dir.join(name);
        if path.is_file() {
            let weights = MappedSafetensors::open(&path)?;
            let tensors = weights.tensor_infos()?;
            validate_s3gen_names(tensors.iter().map(|tensor| tensor.name.as_str()))?;
            let (_, sentinel) = weights.tensor_f32("flow.input_embedding.weight")?;
            ensure!(
                sentinel.iter().all(|value| value.is_finite()),
                "Chatterbox S3Gen input embedding contains non-finite weights"
            );
            let bytes = tensors.iter().map(|tensor| tensor.bytes).sum();
            return Ok((path, tensors.len(), bytes));
        }
    }

    let path = model_dir.join("s3gen.pt");
    ensure!(
        path.is_file(),
        "missing s3gen_v3.safetensors, s3gen.safetensors, or s3gen.pt"
    );
    let weights = PthTensors::new(&path, None)?;
    validate_s3gen_names(weights.tensor_infos().keys().map(String::as_str))?;
    let sentinel = weights
        .get("flow.input_embedding.weight")?
        .context("Chatterbox S3Gen is missing flow.input_embedding.weight")?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        sentinel.iter().all(|value| value.is_finite()),
        "Chatterbox S3Gen input embedding contains non-finite weights"
    );
    let bytes = weights
        .tensor_infos()
        .values()
        .map(|tensor| tensor.layout.shape().elem_count() * tensor.dtype.size_in_bytes())
        .sum();
    Ok((path, weights.tensor_infos().len(), bytes))
}

fn validate_s3gen_names<'a>(names: impl Iterator<Item = &'a str>) -> anyhow::Result<()> {
    let names = names.collect::<std::collections::HashSet<_>>();
    for required in [
        "flow.input_embedding.weight",
        "flow.decoder.estimator.down_blocks.0.0.block1.block.0.weight",
        "mel2wav.conv_pre.parametrizations.weight.original1",
    ] {
        ensure!(
            names.contains(required),
            "Chatterbox S3Gen is missing {required}"
        );
    }
    Ok(())
}

fn pth_f32(tensors: &PthTensors, name: &str, expected: &[usize]) -> anyhow::Result<Vec<f32>> {
    let tensor = tensors
        .get(name)?
        .with_context(|| format!("conditioning tensor {name} is missing"))?;
    ensure!(
        tensor.dims() == expected,
        "conditioning tensor {name} has shape {:?}, expected {expected:?}",
        tensor.dims()
    );
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn pth_i32(tensors: &PthTensors, name: &str, expected: &[usize]) -> anyhow::Result<Vec<i32>> {
    let tensor = tensors
        .get(name)?
        .with_context(|| format!("conditioning tensor {name} is missing"))?;
    ensure!(
        tensor.dims() == expected,
        "conditioning tensor {name} has shape {:?}, expected {expected:?}",
        tensor.dims()
    );
    tensor
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|value| i32::try_from(value).context("conditioning token exceeds i32"))
        .collect()
}
