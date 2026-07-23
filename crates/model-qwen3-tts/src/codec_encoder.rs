use crate::load_f16_tensor;
use anyhow::{Context, ensure};
use chew_kernel::{GpuKernels, SpeakerKernels};
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const HIDDEN: usize = 512;
const FF: usize = 2048;
const HEADS: u32 = 8;
const HEAD_DIM: u32 = 64;
const CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;

struct Conv1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
}

struct ResidualBlock {
    first: Conv1d,
    second: Conv1d,
}

struct EncoderStage {
    residual: ResidualBlock,
    downsample: Conv1d,
}

struct TransformerLayer {
    input_norm_weight: CudaSlice<f16>,
    input_norm_bias: CudaSlice<f16>,
    q_proj: CudaSlice<f16>,
    k_proj: CudaSlice<f16>,
    v_proj: CudaSlice<f16>,
    o_proj: CudaSlice<f16>,
    attention_scale: CudaSlice<f16>,
    post_norm_weight: CudaSlice<f16>,
    post_norm_bias: CudaSlice<f16>,
    fc1: CudaSlice<f16>,
    fc2: CudaSlice<f16>,
    mlp_scale: CudaSlice<f16>,
}

struct Quantizer {
    projection: CudaSlice<f16>,
    codebooks: Vec<CudaSlice<f16>>,
}

pub struct CodecEncoder {
    initial: Conv1d,
    stages: Vec<EncoderStage>,
    final_conv: Conv1d,
    transformer: Vec<TransformerLayer>,
    downsample: Conv1d,
    semantic: Quantizer,
    acoustic: Quantizer,
    utility: SpeakerKernels,
    stream: Arc<CudaStream>,
}

impl CodecEncoder {
    pub fn load(tokenizer_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let stage_geometry = [
            (1usize, 3usize, 1usize, 4usize),
            (4, 6, 2, 5),
            (7, 9, 3, 6),
            (10, 12, 4, 8),
        ];
        let mut stages = Vec::with_capacity(stage_geometry.len());
        for (residual_index, downsample_index, dilation, stride) in stage_geometry {
            stages.push(EncoderStage {
                residual: ResidualBlock {
                    first: load_conv(
                        tokenizer_dir,
                        &format!("encoder.encoder.layers.{residual_index}.block.1.conv"),
                        1,
                        dilation,
                        true,
                        stream,
                    )?,
                    second: load_conv(
                        tokenizer_dir,
                        &format!("encoder.encoder.layers.{residual_index}.block.3.conv"),
                        1,
                        1,
                        true,
                        stream,
                    )?,
                },
                downsample: load_conv(
                    tokenizer_dir,
                    &format!("encoder.encoder.layers.{downsample_index}.conv"),
                    stride,
                    1,
                    true,
                    stream,
                )?,
            });
        }
        let mut transformer = Vec::with_capacity(8);
        for layer in 0..8 {
            let prefix = format!("encoder.encoder_transformer.layers.{layer}");
            transformer.push(TransformerLayer {
                input_norm_weight: load(
                    tokenizer_dir,
                    &format!("{prefix}.input_layernorm.weight"),
                    stream,
                )?,
                input_norm_bias: load(
                    tokenizer_dir,
                    &format!("{prefix}.input_layernorm.bias"),
                    stream,
                )?,
                q_proj: load(
                    tokenizer_dir,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    stream,
                )?,
                k_proj: load(
                    tokenizer_dir,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    stream,
                )?,
                v_proj: load(
                    tokenizer_dir,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    stream,
                )?,
                o_proj: load(
                    tokenizer_dir,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    stream,
                )?,
                attention_scale: load(
                    tokenizer_dir,
                    &format!("{prefix}.self_attn_layer_scale.scale"),
                    stream,
                )?,
                post_norm_weight: load(
                    tokenizer_dir,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    stream,
                )?,
                post_norm_bias: load(
                    tokenizer_dir,
                    &format!("{prefix}.post_attention_layernorm.bias"),
                    stream,
                )?,
                fc1: load(tokenizer_dir, &format!("{prefix}.mlp.fc1.weight"), stream)?,
                fc2: load(tokenizer_dir, &format!("{prefix}.mlp.fc2.weight"), stream)?,
                mlp_scale: load(
                    tokenizer_dir,
                    &format!("{prefix}.mlp_layer_scale.scale"),
                    stream,
                )?,
            });
        }
        Ok(Self {
            initial: load_conv(
                tokenizer_dir,
                "encoder.encoder.layers.0.conv",
                1,
                1,
                true,
                stream,
            )?,
            stages,
            final_conv: load_conv(
                tokenizer_dir,
                "encoder.encoder.layers.14.conv",
                1,
                1,
                true,
                stream,
            )?,
            transformer,
            downsample: load_conv(
                tokenizer_dir,
                "encoder.downsample.conv",
                2,
                1,
                false,
                stream,
            )?,
            semantic: load_quantizer(
                tokenizer_dir,
                "encoder.quantizer.semantic_residual_vector_quantizer",
                1,
                stream,
            )?,
            acoustic: load_quantizer(
                tokenizer_dir,
                "encoder.quantizer.acoustic_residual_vector_quantizer",
                15,
                stream,
            )?,
            utility: SpeakerKernels::load(stream)?,
            stream: Arc::clone(stream),
        })
    }

    /// Encode 24-kHz mono waveform samples into 16 Qwen codec IDs per 80-ms frame.
    pub fn encode(
        &self,
        samples: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<Vec<i32>>> {
        ensure!(!samples.is_empty(), "reference waveform is empty");
        let input = samples
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let mut hidden = self.stream.clone_htod(&input)?;
        let mut length = samples.len();
        hidden = self.conv(&hidden, length, &self.initial, kernels)?;
        for stage in &self.stages {
            hidden = self.residual(&hidden, length, &stage.residual, kernels)?;
            self.utility.elu(&mut hidden);
            hidden = self.conv(&hidden, length, &stage.downsample, kernels)?;
            length = length.div_ceil(stage.downsample.stride);
        }
        self.utility.elu(&mut hidden);
        hidden = self.conv(&hidden, length, &self.final_conv, kernels)?;
        let hidden = self.transform(hidden, length, kernels)?;
        let hidden_f16 = {
            let mut converted = self.stream.alloc_zeros::<f16>(length * HIDDEN)?;
            kernels.ops.copy_f32_to_f16(
                &hidden,
                &mut converted.slice_mut(..),
                (length * HIDDEN) as u32,
            )?;
            converted
        };
        let mut channel_first = self.stream.alloc_zeros::<f16>(length * HIDDEN)?;
        kernels.ops.transpose_f16(
            &hidden_f16,
            &mut channel_first,
            length as u32,
            HIDDEN as u32,
        )?;
        let downsampled = self.conv(&channel_first, length, &self.downsample, kernels)?;
        let frames = length.div_ceil(2);
        let mut frame_major = self.stream.alloc_zeros::<f16>(frames * HIDDEN)?;
        kernels
            .ops
            .transpose_f16(&downsampled, &mut frame_major, HIDDEN as u32, frames as u32)?;
        let semantic = self.quantize(&frame_major, frames, &self.semantic, kernels)?;
        let acoustic = self.quantize(&frame_major, frames, &self.acoustic, kernels)?;
        let mut output = vec![vec![0i32; 16]; frames];
        for frame in 0..frames {
            output[frame][0] = semantic[0][frame];
            for group in 0..15 {
                output[frame][group + 1] = acoustic[group][frame];
            }
        }
        Ok(output)
    }

    fn residual(
        &self,
        input: &CudaSlice<f16>,
        length: usize,
        block: &ResidualBlock,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mut hidden = self.stream.alloc_zeros::<f16>(input.len())?;
        self.stream.memcpy_dtod(input, &mut hidden)?;
        self.utility.elu(&mut hidden);
        hidden = self.conv(&hidden, length, &block.first, kernels)?;
        self.utility.elu(&mut hidden);
        hidden = self.conv(&hidden, length, &block.second, kernels)?;
        let mut output = self.stream.alloc_zeros::<f16>(input.len())?;
        kernels
            .ops
            .add_f16(input, &hidden, &mut output, input.len() as u32)?;
        Ok(output)
    }

    fn conv(
        &self,
        input: &CudaSlice<f16>,
        input_len: usize,
        conv: &Conv1d,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        ensure!(
            input.len() == conv.in_channels * input_len,
            "codec encoder conv input has {} values, expected {}",
            input.len(),
            conv.in_channels * input_len
        );
        let output_len = input_len.div_ceil(conv.stride);
        let width = conv.in_channels * conv.kernel_size;
        let mut unfolded = self.stream.alloc_zeros::<f16>(output_len * width)?;
        self.utility.unfold_causal_stride(
            input,
            &mut unfolded,
            conv.in_channels as u32,
            input_len as u32,
            output_len as u32,
            conv.kernel_size as u32,
            conv.stride as u32,
            conv.dilation as u32,
        );
        let mut channel_last = self
            .stream
            .alloc_zeros::<f16>(output_len * conv.out_channels)?;
        kernels.gemm.matmul_f16(
            &unfolded,
            &conv.weight,
            &mut channel_last,
            output_len as u32,
            conv.out_channels as u32,
            width as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut channel_last,
            &conv.bias,
            output_len as u32,
            conv.out_channels as u32,
        )?;
        let mut channel_first = self
            .stream
            .alloc_zeros::<f16>(output_len * conv.out_channels)?;
        kernels.ops.transpose_f16(
            &channel_last,
            &mut channel_first,
            output_len as u32,
            conv.out_channels as u32,
        )?;
        Ok(channel_first)
    }

    fn transform(
        &self,
        channel_first: CudaSlice<f16>,
        length: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f32>> {
        let elements = length * HIDDEN;
        let mut channel_last = self.stream.alloc_zeros::<f16>(elements)?;
        kernels.ops.transpose_f16(
            &channel_first,
            &mut channel_last,
            HIDDEN as u32,
            length as u32,
        )?;
        let mut hidden = self.stream.alloc_zeros::<f32>(elements)?;
        kernels
            .ops
            .copy_f16_to_f32(&channel_last, &mut hidden, elements as u32)?;
        let mut norm = self.stream.alloc_zeros::<f16>(elements)?;
        let mut q = self.stream.alloc_zeros::<f16>(elements)?;
        let mut k = self.stream.alloc_zeros::<f16>(elements)?;
        let mut v = self.stream.alloc_zeros::<f16>(elements)?;
        let mut attention = self.stream.alloc_zeros::<f16>(elements)?;
        let mut delta = self.stream.alloc_zeros::<f16>(elements)?;
        let mut scaled = self.stream.alloc_zeros::<f16>(elements)?;
        let mut ff = self.stream.alloc_zeros::<f16>(length * FF)?;
        let mut activated = self.stream.alloc_zeros::<f16>(length * FF)?;
        for layer in &self.transformer {
            kernels.ops.layer_norm_f32in(
                &hidden,
                &layer.input_norm_weight,
                &layer.input_norm_bias,
                &mut norm,
                length as u32,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.q_proj,
                &mut q,
                length as u32,
                HIDDEN as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.k_proj,
                &mut k,
                length as u32,
                HIDDEN as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.v_proj,
                &mut v,
                length as u32,
                HIDDEN as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .rope_neox(&mut q, length as u32, HEADS, HEAD_DIM, 0, 10_000.0)?;
            kernels
                .ops
                .rope_neox(&mut k, length as u32, HEADS, HEAD_DIM, 0, 10_000.0)?;
            kernels.ops.mha_fused_scaled(
                &q,
                &k.slice(..),
                &v.slice(..),
                &mut attention,
                HEAD_DIM,
                HEADS,
                HEADS,
                length as u32,
                length as u32,
                0,
                250,
                1.0 / (HEAD_DIM as f32).sqrt(),
                0.0,
            )?;
            kernels.gemm.matmul_f16(
                &attention,
                &layer.o_proj,
                &mut delta,
                length as u32,
                HIDDEN as u32,
                HIDDEN as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.attention_scale,
                &mut scaled,
                elements as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, elements as u32)?;
            kernels.ops.layer_norm_f32in(
                &hidden,
                &layer.post_norm_weight,
                &layer.post_norm_bias,
                &mut norm,
                length as u32,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.fc1,
                &mut ff,
                length as u32,
                FF as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .gelu_erf_f16(&ff, &mut activated, (length * FF) as u32)?;
            kernels.gemm.matmul_f16(
                &activated,
                &layer.fc2,
                &mut delta,
                length as u32,
                HIDDEN as u32,
                FF as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.mlp_scale,
                &mut scaled,
                elements as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, elements as u32)?;
        }
        Ok(hidden)
    }

    fn quantize(
        &self,
        embeddings: &CudaSlice<f16>,
        frames: usize,
        quantizer: &Quantizer,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<Vec<i32>>> {
        let mut projected = self.stream.alloc_zeros::<f16>(frames * CODEBOOK_DIM)?;
        kernels.gemm.matmul_f16(
            embeddings,
            &quantizer.projection,
            &mut projected,
            frames as u32,
            CODEBOOK_DIM as u32,
            HIDDEN as u32,
        )?;
        let mut residual = self.stream.alloc_zeros::<f16>(projected.len())?;
        self.stream.memcpy_dtod(&projected, &mut residual)?;
        let mut result = Vec::with_capacity(quantizer.codebooks.len());
        for codebook in &quantizer.codebooks {
            let mut indices = self.stream.alloc_zeros::<i32>(frames)?;
            self.utility.nearest_codebook(
                &residual,
                codebook,
                &mut indices,
                frames as u32,
                CODEBOOK_SIZE as u32,
                CODEBOOK_DIM as u32,
            );
            self.utility.subtract_codebook(
                &mut residual,
                codebook,
                &indices,
                frames as u32,
                CODEBOOK_DIM as u32,
            );
            self.stream.synchronize()?;
            let mut host = vec![0i32; frames];
            self.stream.memcpy_dtoh(&indices, &mut host)?;
            result.push(host);
        }
        Ok(result)
    }
}

fn load_conv(
    model_dir: &Path,
    prefix: &str,
    stride: usize,
    dilation: usize,
    has_bias: bool,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv1d> {
    let weight = load_f16_tensor(model_dir, &format!("{prefix}.weight"))?;
    ensure!(
        weight.shape.len() == 3,
        "{prefix}.weight has shape {:?}",
        weight.shape
    );
    let bias = if has_bias {
        let bias = load_f16_tensor(model_dir, &format!("{prefix}.bias"))?;
        ensure!(
            bias.shape == [weight.shape[0]],
            "{prefix}.bias has shape {:?}",
            bias.shape
        );
        bias.values
    } else {
        vec![f16::ZERO; weight.shape[0]]
    };
    Ok(Conv1d {
        in_channels: weight.shape[1],
        out_channels: weight.shape[0],
        kernel_size: weight.shape[2],
        stride,
        dilation,
        weight: stream
            .clone_htod(&weight.values)
            .with_context(|| format!("could not upload {prefix}.weight"))?,
        bias: stream
            .clone_htod(&bias)
            .with_context(|| format!("could not upload {prefix}.bias"))?,
    })
}

fn load(model_dir: &Path, name: &str, stream: &Arc<CudaStream>) -> anyhow::Result<CudaSlice<f16>> {
    let tensor = load_f16_tensor(model_dir, name)?;
    stream
        .clone_htod(&tensor.values)
        .with_context(|| format!("could not upload {name}"))
}

fn load_quantizer(
    model_dir: &Path,
    prefix: &str,
    codebooks: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Quantizer> {
    let projection = load(model_dir, &format!("{prefix}.input_proj.weight"), stream)?;
    let mut loaded = Vec::with_capacity(codebooks);
    for layer in 0..codebooks {
        let codebook_prefix = format!("{prefix}.layers.{layer}.codebook");
        let sums = crate::load_f32_tensor(model_dir, &format!("{codebook_prefix}.embed_sum"))?;
        let usage = crate::load_f32_tensor(model_dir, &format!("{codebook_prefix}.cluster_usage"))?;
        ensure!(
            sums.shape == [CODEBOOK_SIZE, CODEBOOK_DIM],
            "{codebook_prefix}.embed_sum has shape {:?}",
            sums.shape
        );
        let normalized = sums
            .values
            .chunks_exact(CODEBOOK_DIM)
            .zip(&usage.values)
            .flat_map(|(row, usage)| {
                let divisor = usage.max(1e-5);
                row.iter().map(move |value| f16::from_f32(value / divisor))
            })
            .collect::<Vec<_>>();
        loaded.push(stream.clone_htod(&normalized)?);
    }
    Ok(Quantizer {
        projection,
        codebooks: loaded,
    })
}
