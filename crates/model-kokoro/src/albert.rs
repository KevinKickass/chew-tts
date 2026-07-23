use crate::{KokoroCheckpoint, KokoroConfig};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const EMBEDDING: usize = 128;
const HIDDEN: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const INTERMEDIATE: usize = 2048;

struct Linear {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
}

/// Kokoro's parameter-shared twelve-pass ALBERT text backbone.
pub struct KokoroAlbert {
    word_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    token_type_embedding: Vec<f32>,
    embedding_norm_weight: CudaSlice<f16>,
    embedding_norm_bias: CudaSlice<f16>,
    embedding_projection: Linear,
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    attention_norm_weight: CudaSlice<f16>,
    attention_norm_bias: CudaSlice<f16>,
    ffn: Linear,
    ffn_output: Linear,
    final_norm_weight: CudaSlice<f16>,
    final_norm_bias: CudaSlice<f16>,
    output_projection: Linear,
    passes: usize,
}

impl KokoroAlbert {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let config = KokoroConfig::load(model_dir)?;
        ensure!(
            config.plbert.hidden_size == HIDDEN
                && config.plbert.num_attention_heads == HEADS
                && config.plbert.intermediate_size == INTERMEDIATE,
            "unsupported Kokoro ALBERT geometry"
        );
        let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
        let (word_shape, word_embeddings) =
            checkpoint.tensor_f32("bert", "module.embeddings.word_embeddings.weight")?;
        let (position_shape, position_embeddings) =
            checkpoint.tensor_f32("bert", "module.embeddings.position_embeddings.weight")?;
        let (token_type_shape, token_types) =
            checkpoint.tensor_f32("bert", "module.embeddings.token_type_embeddings.weight")?;
        ensure!(
            word_shape == [config.n_token, EMBEDDING],
            "invalid ALBERT word embeddings"
        );
        ensure!(
            position_shape == [config.plbert.max_position_embeddings, EMBEDDING],
            "invalid ALBERT position embeddings"
        );
        ensure!(
            token_type_shape == [2, EMBEDDING],
            "invalid ALBERT token-type embeddings"
        );
        let prefix = "module.encoder.albert_layer_groups.0.albert_layers.0";
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embedding: token_types[..EMBEDDING].to_vec(),
            embedding_norm_weight: load_vector(
                &checkpoint,
                "bert",
                "module.embeddings.LayerNorm.weight",
                EMBEDDING,
                stream,
            )?,
            embedding_norm_bias: load_vector(
                &checkpoint,
                "bert",
                "module.embeddings.LayerNorm.bias",
                EMBEDDING,
                stream,
            )?,
            embedding_projection: load_linear(
                &checkpoint,
                "bert",
                "module.encoder.embedding_hidden_mapping_in",
                EMBEDDING,
                HIDDEN,
                stream,
            )?,
            query: load_linear(
                &checkpoint,
                "bert",
                &format!("{prefix}.attention.query"),
                HIDDEN,
                HIDDEN,
                stream,
            )?,
            key: load_linear(
                &checkpoint,
                "bert",
                &format!("{prefix}.attention.key"),
                HIDDEN,
                HIDDEN,
                stream,
            )?,
            value: load_linear(
                &checkpoint,
                "bert",
                &format!("{prefix}.attention.value"),
                HIDDEN,
                HIDDEN,
                stream,
            )?,
            attention_output: load_linear(
                &checkpoint,
                "bert",
                &format!("{prefix}.attention.dense"),
                HIDDEN,
                HIDDEN,
                stream,
            )?,
            attention_norm_weight: load_vector(
                &checkpoint,
                "bert",
                &format!("{prefix}.attention.LayerNorm.weight"),
                HIDDEN,
                stream,
            )?,
            attention_norm_bias: load_vector(
                &checkpoint,
                "bert",
                &format!("{prefix}.attention.LayerNorm.bias"),
                HIDDEN,
                stream,
            )?,
            ffn: load_linear(
                &checkpoint,
                "bert",
                &format!("{prefix}.ffn"),
                HIDDEN,
                INTERMEDIATE,
                stream,
            )?,
            ffn_output: load_linear(
                &checkpoint,
                "bert",
                &format!("{prefix}.ffn_output"),
                INTERMEDIATE,
                HIDDEN,
                stream,
            )?,
            final_norm_weight: load_vector(
                &checkpoint,
                "bert",
                &format!("{prefix}.full_layer_layer_norm.weight"),
                HIDDEN,
                stream,
            )?,
            final_norm_bias: load_vector(
                &checkpoint,
                "bert",
                &format!("{prefix}.full_layer_layer_norm.bias"),
                HIDDEN,
                stream,
            )?,
            output_projection: load_linear(
                &checkpoint,
                "bert_encoder",
                "module",
                HIDDEN,
                config.hidden_dim,
                stream,
            )?,
            passes: config.plbert.num_hidden_layers,
        })
    }

    /// Return frame-major `[tokens, 512]` duration-encoder input.
    pub fn encode(&self, ids: &[usize], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "Kokoro ALBERT input is empty");
        ensure!(
            ids.iter()
                .all(|id| *id * EMBEDDING < self.word_embeddings.len()),
            "Kokoro ALBERT token is outside vocabulary"
        );
        ensure!(
            ids.len() * EMBEDDING <= self.position_embeddings.len(),
            "Kokoro ALBERT sequence exceeds position embeddings"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut embedding = Vec::with_capacity(ids.len() * EMBEDDING);
        for (position, id) in ids.iter().copied().enumerate() {
            for channel in 0..EMBEDDING {
                embedding.push(
                    self.word_embeddings[id * EMBEDDING + channel]
                        + self.position_embeddings[position * EMBEDDING + channel]
                        + self.token_type_embedding[channel],
                );
            }
        }
        let embedding = stream.clone_htod(&embedding)?;
        let mut normalized = stream.alloc_zeros::<f16>(ids.len() * EMBEDDING)?;
        kernels.ops.layer_norm_f32in(
            &embedding,
            &self.embedding_norm_weight,
            &self.embedding_norm_bias,
            &mut normalized,
            ids.len() as u32,
            EMBEDDING as u32,
            1e-12,
        )?;
        let mut hidden = linear(
            &normalized,
            ids.len(),
            EMBEDDING,
            HIDDEN,
            &self.embedding_projection,
            kernels,
        )?;
        for _ in 0..self.passes {
            hidden = self.layer(hidden, ids.len(), kernels)?;
        }
        let projected = linear(
            &hidden,
            ids.len(),
            HIDDEN,
            512,
            &self.output_projection,
            kernels,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; ids.len() * 512];
        stream.memcpy_dtoh(&projected, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    fn layer(
        &self,
        hidden: CudaSlice<f16>,
        tokens: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let stream = Arc::clone(kernels.ops.stream());
        let q = linear(&hidden, tokens, HIDDEN, HIDDEN, &self.query, kernels)?;
        let k = linear(&hidden, tokens, HIDDEN, HIDDEN, &self.key, kernels)?;
        let v = linear(&hidden, tokens, HIDDEN, HIDDEN, &self.value, kernels)?;
        let mut attention = stream.alloc_zeros::<f16>(tokens * HIDDEN)?;
        kernels.ops.mha_naive_full(
            &q,
            &k.slice(..),
            &v.slice(..),
            &mut attention,
            HEAD_DIM as u32,
            HEADS as u32,
            HEADS as u32,
            tokens as u32,
            tokens as u32,
            1.0 / (HEAD_DIM as f32).sqrt(),
            0.0,
        )?;
        let attention = linear(
            &attention,
            tokens,
            HIDDEN,
            HIDDEN,
            &self.attention_output,
            kernels,
        )?;
        let attention = residual_norm(
            &hidden,
            &attention,
            tokens,
            &self.attention_norm_weight,
            &self.attention_norm_bias,
            kernels,
        )?;
        let ffn = linear(&attention, tokens, HIDDEN, INTERMEDIATE, &self.ffn, kernels)?;
        let mut activated = stream.alloc_zeros::<f16>(tokens * INTERMEDIATE)?;
        kernels
            .ops
            .gelu_erf_f16(&ffn, &mut activated, (tokens * INTERMEDIATE) as u32)?;
        let output = linear(
            &activated,
            tokens,
            INTERMEDIATE,
            HIDDEN,
            &self.ffn_output,
            kernels,
        )?;
        residual_norm(
            &attention,
            &output,
            tokens,
            &self.final_norm_weight,
            &self.final_norm_bias,
            kernels,
        )
    }
}

fn load_vector(
    checkpoint: &KokoroCheckpoint,
    group: &str,
    name: &str,
    size: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f16>> {
    let (shape, values) = checkpoint
        .tensor_f16(group, name)
        .with_context(|| format!("could not load {group}.{name}"))?;
    ensure!(shape == [size], "invalid {group}.{name} shape {shape:?}");
    Ok(stream.clone_htod(&values)?)
}

fn load_linear(
    checkpoint: &KokoroCheckpoint,
    group: &str,
    prefix: &str,
    input: usize,
    output: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Linear> {
    let (weight_shape, weight) = checkpoint.tensor_f16(group, &format!("{prefix}.weight"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16(group, &format!("{prefix}.bias"))?;
    ensure!(
        weight_shape == [output, input],
        "invalid {group}.{prefix}.weight shape {weight_shape:?}"
    );
    ensure!(
        bias_shape == [output],
        "invalid {group}.{prefix}.bias shape {bias_shape:?}"
    );
    Ok(Linear {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
    })
}

fn linear(
    input: &CudaSlice<f16>,
    rows: usize,
    input_width: usize,
    output_width: usize,
    layer: &Linear,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(rows * output_width)?;
    kernels.gemm.matmul_f16(
        input,
        &layer.weight,
        &mut output,
        rows as u32,
        output_width as u32,
        input_width as u32,
    )?;
    kernels
        .ops
        .add_bias_f16_inplace(&mut output, &layer.bias, rows as u32, output_width as u32)?;
    Ok(output)
}

fn residual_norm(
    residual: &CudaSlice<f16>,
    update: &CudaSlice<f16>,
    rows: usize,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let elements = residual.len();
    let mut sum = stream.alloc_zeros::<f16>(elements)?;
    kernels
        .ops
        .add_f16(residual, update, &mut sum, elements as u32)?;
    let mut sum_f32 = stream.alloc_zeros::<f32>(elements)?;
    kernels
        .ops
        .copy_f16_to_f32(&sum, &mut sum_f32, elements as u32)?;
    let mut output = stream.alloc_zeros::<f16>(elements)?;
    kernels.ops.layer_norm_f32in(
        &sum_f32,
        weight,
        bias,
        &mut output,
        rows as u32,
        HIDDEN as u32,
        1e-12,
    )?;
    Ok(output)
}
