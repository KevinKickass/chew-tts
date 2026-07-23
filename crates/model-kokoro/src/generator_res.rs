use crate::KokoroCheckpoint;
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::sync::Arc;

struct Conv {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    kernel: usize,
}

struct AdaIn {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct Stage {
    conv1: Conv,
    conv2: Conv,
    norm1: AdaIn,
    norm2: AdaIn,
    alpha1: CudaSlice<f16>,
    alpha2: CudaSlice<f16>,
    dilation: usize,
}

/// AdaIN + Snake residual block used inside Kokoro's iSTFTNet generator.
pub struct KokoroGeneratorResBlock {
    stages: Vec<Stage>,
    channels: usize,
}

impl KokoroGeneratorResBlock {
    pub fn load(
        checkpoint: &KokoroCheckpoint,
        prefix: &str,
        channels: usize,
        kernel: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let mut stages = Vec::new();
        for index in 0..3 {
            stages.push(Stage {
                conv1: load_conv(
                    checkpoint,
                    &format!("{prefix}.convs1.{index}"),
                    channels,
                    kernel,
                    stream,
                )?,
                conv2: load_conv(
                    checkpoint,
                    &format!("{prefix}.convs2.{index}"),
                    channels,
                    kernel,
                    stream,
                )?,
                norm1: AdaIn::load(checkpoint, &format!("{prefix}.adain1.{index}"), channels)?,
                norm2: AdaIn::load(checkpoint, &format!("{prefix}.adain2.{index}"), channels)?,
                alpha1: load_alpha(
                    checkpoint,
                    &format!("{prefix}.alpha1.{index}"),
                    channels,
                    stream,
                )?,
                alpha2: load_alpha(
                    checkpoint,
                    &format!("{prefix}.alpha2.{index}"),
                    channels,
                    stream,
                )?,
                dilation: [1, 3, 5][index],
            });
        }
        Ok(Self { stages, channels })
    }

    /// Frame-major input/output `[frames, channels]`.
    pub fn forward(
        &self,
        input: &[f32],
        frames: usize,
        style: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            input.len() == frames * self.channels && style.len() == 128,
            "invalid Kokoro generator residual input"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let rows =
            stream.clone_htod(&input.iter().copied().map(f16::from_f32).collect::<Vec<_>>())?;
        let mut hidden = stream.alloc_zeros::<f16>(input.len())?;
        kernels
            .ops
            .transpose_f16(&rows, &mut hidden, frames as u32, self.channels as u32)?;
        for stage in &self.stages {
            let normalized = stage
                .norm1
                .forward(&hidden, frames, style, self.channels, kernels)?;
            let mut activated = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels.ops.snake_f16(
                &normalized,
                &stage.alpha1,
                &mut activated,
                self.channels as u32,
                frames as u32,
            )?;
            let first = run_conv(
                &activated,
                frames,
                self.channels,
                &stage.conv1,
                stage.dilation,
                kernels,
            )?;
            let normalized = stage
                .norm2
                .forward(&first, frames, style, self.channels, kernels)?;
            let mut activated = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels.ops.snake_f16(
                &normalized,
                &stage.alpha2,
                &mut activated,
                self.channels as u32,
                frames as u32,
            )?;
            let second = run_conv(&activated, frames, self.channels, &stage.conv2, 1, kernels)?;
            let mut residual = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels
                .ops
                .add_f16(&hidden, &second, &mut residual, hidden.len() as u32)?;
            hidden = residual;
        }
        let mut output = stream.alloc_zeros::<f16>(hidden.len())?;
        kernels
            .ops
            .transpose_f16(&hidden, &mut output, self.channels as u32, frames as u32)?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; output.len()];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }
}

impl AdaIn {
    fn load(checkpoint: &KokoroCheckpoint, prefix: &str, channels: usize) -> anyhow::Result<Self> {
        let (weight_shape, weight) =
            checkpoint.tensor_f32("decoder", &format!("{prefix}.fc.weight"))?;
        let (bias_shape, bias) = checkpoint.tensor_f32("decoder", &format!("{prefix}.fc.bias"))?;
        ensure!(
            weight_shape == [channels * 2, 128] && bias_shape == [channels * 2],
            "invalid Kokoro generator AdaIN {prefix}"
        );
        Ok(Self { weight, bias })
    }

    fn forward(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        style: &[f32],
        channels: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let affine = (0..channels * 2)
            .map(|output| {
                self.bias[output]
                    + style
                        .iter()
                        .enumerate()
                        .map(|(input, value)| self.weight[output * 128 + input] * value)
                        .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let stream = Arc::clone(kernels.ops.stream());
        let gamma = stream.clone_htod(
            &affine[..channels]
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let beta = stream.clone_htod(
            &affine[channels..]
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut output = stream.alloc_zeros::<f16>(channels * frames)?;
        kernels.ops.instance_norm_affine_f16(
            input,
            &gamma,
            &beta,
            &mut output,
            channels as u32,
            frames as u32,
            1e-5,
        )?;
        Ok(output)
    }
}

fn run_conv(
    input: &CudaSlice<f16>,
    frames: usize,
    channels: usize,
    conv: &Conv,
    dilation: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(channels * frames)?;
    kernels.ops.conv1d_general_f16(
        input,
        &conv.weight,
        &conv.bias,
        &mut output,
        channels as u32,
        channels as u32,
        frames as u32,
        frames as u32,
        conv.kernel as u32,
        1,
        ((conv.kernel * dilation - dilation) / 2) as u32,
        dilation as u32,
    )?;
    Ok(output)
}

fn load_conv(
    checkpoint: &KokoroCheckpoint,
    prefix: &str,
    channels: usize,
    kernel: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv> {
    let (g_shape, g) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_g"))?;
    let (v_shape, v) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_v"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16("decoder", &format!("{prefix}.bias"))?;
    ensure!(
        g_shape == [channels, 1, 1]
            && v_shape == [channels, channels, kernel]
            && bias_shape == [channels],
        "invalid Kokoro generator convolution {prefix}"
    );
    let width = channels * kernel;
    let mut weight = Vec::with_capacity(v.len());
    for channel in 0..channels {
        let row = &v[channel * width..(channel + 1) * width];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        let scale = g[channel] / norm.max(1e-12);
        weight.extend(row.iter().map(|value| f16::from_f32(value * scale)));
    }
    Ok(Conv {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
        kernel,
    })
}

fn load_alpha(
    checkpoint: &KokoroCheckpoint,
    name: &str,
    channels: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f16>> {
    let (shape, values) = checkpoint.tensor_f16("decoder", name)?;
    ensure!(
        shape == [1, channels, 1],
        "invalid Kokoro generator alpha {name}"
    );
    Ok(stream.clone_htod(&values)?)
}
