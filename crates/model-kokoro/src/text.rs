use crate::{KokoroBiLstm, KokoroCheckpoint};
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const CHANNELS: usize = 512;

struct Conv {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
}

struct ConvBlock {
    conv: Conv,
    norm_weight: CudaSlice<f16>,
    norm_bias: CudaSlice<f16>,
}

pub struct KokoroTextEncoder {
    embedding: Vec<f32>,
    blocks: Vec<ConvBlock>,
    lstm: KokoroBiLstm,
}

impl KokoroTextEncoder {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
        let (embedding_shape, embedding) =
            checkpoint.tensor_f32("text_encoder", "module.embedding.weight")?;
        ensure!(
            embedding_shape == [178, CHANNELS],
            "invalid Kokoro text embedding"
        );
        let mut blocks = Vec::new();
        for index in 0..3 {
            let prefix = format!("module.cnn.{index}");
            blocks.push(ConvBlock {
                conv: load_weight_norm_conv(
                    &checkpoint,
                    "text_encoder",
                    &format!("{prefix}.0"),
                    CHANNELS,
                    CHANNELS,
                    5,
                    stream,
                )?,
                norm_weight: load_vector(
                    &checkpoint,
                    "text_encoder",
                    &format!("{prefix}.1.gamma"),
                    CHANNELS,
                    stream,
                )?,
                norm_bias: load_vector(
                    &checkpoint,
                    "text_encoder",
                    &format!("{prefix}.1.beta"),
                    CHANNELS,
                    stream,
                )?,
            });
        }
        Ok(Self {
            embedding,
            blocks,
            lstm: KokoroBiLstm::load(
                &checkpoint,
                "text_encoder",
                "module.lstm",
                CHANNELS,
                CHANNELS / 2,
                stream,
            )?,
        })
    }

    /// Encode tokens and repeat each output according to predicted durations.
    /// Returns frame-major `[acoustic_frames, 512]`.
    pub fn encode_aligned(
        &self,
        ids: &[usize],
        durations: &[usize],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            !ids.is_empty() && ids.len() == durations.len(),
            "invalid Kokoro text/duration geometry"
        );
        ensure!(
            ids.iter().all(|id| *id * CHANNELS < self.embedding.len()),
            "Kokoro text token outside vocabulary"
        );
        let frames = ids.len();
        let mut rows = Vec::with_capacity(frames * CHANNELS);
        for id in ids {
            rows.extend_from_slice(&self.embedding[id * CHANNELS..(id + 1) * CHANNELS]);
        }
        let stream = Arc::clone(kernels.ops.stream());
        let mut rows_gpu =
            stream.clone_htod(&rows.iter().copied().map(f16::from_f32).collect::<Vec<_>>())?;
        let mut channels = stream.alloc_zeros::<f16>(rows.len())?;
        kernels
            .ops
            .transpose_f16(&rows_gpu, &mut channels, frames as u32, CHANNELS as u32)?;
        for block in &self.blocks {
            let mut convolved = stream.alloc_zeros::<f16>(rows.len())?;
            kernels.ops.conv1d_general_f16(
                &channels,
                &block.conv.weight,
                &block.conv.bias,
                &mut convolved,
                CHANNELS as u32,
                CHANNELS as u32,
                frames as u32,
                frames as u32,
                5,
                1,
                2,
                1,
            )?;
            kernels
                .ops
                .transpose_f16(&convolved, &mut rows_gpu, CHANNELS as u32, frames as u32)?;
            let mut rows_f32 = stream.alloc_zeros::<f32>(rows.len())?;
            kernels
                .ops
                .copy_f16_to_f32(&rows_gpu, &mut rows_f32, rows.len() as u32)?;
            let mut normalized = stream.alloc_zeros::<f16>(rows.len())?;
            kernels.ops.layer_norm_f32in(
                &rows_f32,
                &block.norm_weight,
                &block.norm_bias,
                &mut normalized,
                frames as u32,
                CHANNELS as u32,
                1e-5,
            )?;
            let mut activated = stream.alloc_zeros::<f16>(rows.len())?;
            kernels
                .ops
                .leaky_relu_f16(&normalized, &mut activated, rows.len() as u32, 0.2)?;
            rows_gpu = activated;
            channels = stream.alloc_zeros::<f16>(rows.len())?;
            kernels
                .ops
                .transpose_f16(&rows_gpu, &mut channels, frames as u32, CHANNELS as u32)?;
        }
        stream.synchronize()?;
        let mut encoded_input = vec![f16::ZERO; rows.len()];
        stream.memcpy_dtoh(&rows_gpu, &mut encoded_input)?;
        let encoded_input = encoded_input
            .into_iter()
            .map(f16::to_f32)
            .collect::<Vec<_>>();
        let encoded = self.lstm.forward(&encoded_input, frames, kernels)?;
        let mut aligned = Vec::with_capacity(durations.iter().sum::<usize>() * CHANNELS);
        for (frame, duration) in durations.iter().copied().enumerate() {
            for _ in 0..duration {
                aligned.extend_from_slice(&encoded[frame * CHANNELS..(frame + 1) * CHANNELS]);
            }
        }
        Ok(aligned)
    }
}

fn load_vector(
    checkpoint: &KokoroCheckpoint,
    group: &str,
    name: &str,
    size: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f16>> {
    let (shape, values) = checkpoint.tensor_f16(group, name)?;
    ensure!(shape == [size], "invalid {group}.{name} shape");
    Ok(stream.clone_htod(&values)?)
}

fn load_weight_norm_conv(
    checkpoint: &KokoroCheckpoint,
    group: &str,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv> {
    let (g_shape, g) = checkpoint.tensor_f32(group, &format!("{prefix}.weight_g"))?;
    let (v_shape, v) = checkpoint.tensor_f32(group, &format!("{prefix}.weight_v"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16(group, &format!("{prefix}.bias"))?;
    ensure!(g_shape == [output, 1, 1], "invalid {prefix}.weight_g");
    ensure!(
        v_shape == [output, input, kernel],
        "invalid {prefix}.weight_v"
    );
    ensure!(bias_shape == [output], "invalid {prefix}.bias");
    let width = input * kernel;
    let mut weight = Vec::with_capacity(v.len());
    for channel in 0..output {
        let row = &v[channel * width..(channel + 1) * width];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        let scale = g[channel] / norm.max(1e-12);
        weight.extend(row.iter().map(|value| f16::from_f32(value * scale)));
    }
    Ok(Conv {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
    })
}
