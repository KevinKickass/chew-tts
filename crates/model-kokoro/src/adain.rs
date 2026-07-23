use crate::KokoroCheckpoint;
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::sync::Arc;

struct Conv {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    input: usize,
    output: usize,
    kernel: usize,
}

struct AdaIn {
    weight: Vec<f32>,
    bias: Vec<f32>,
    channels: usize,
}

/// Kokoro/StyleTTS2 AdaIN residual block used by prosody and decoder paths.
pub struct KokoroAdaInResBlock {
    conv1: Conv,
    conv2: Conv,
    norm1: AdaIn,
    norm2: AdaIn,
    shortcut: Option<Conv>,
    pool_weight: Option<CudaSlice<f16>>,
    pool_bias: Option<CudaSlice<f16>>,
    input: usize,
    output: usize,
    upsample: bool,
}

impl KokoroAdaInResBlock {
    pub fn load(
        checkpoint: &KokoroCheckpoint,
        group: &str,
        prefix: &str,
        input: usize,
        output: usize,
        upsample: bool,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let shortcut = if input != output {
            Some(load_weight_norm_conv(
                checkpoint,
                group,
                &format!("{prefix}.conv1x1"),
                input,
                output,
                1,
                false,
                stream,
            )?)
        } else {
            None
        };
        let (pool_weight, pool_bias) = if upsample {
            let pool = load_weight_norm_conv(
                checkpoint,
                group,
                &format!("{prefix}.pool"),
                1,
                input,
                3,
                true,
                stream,
            )?;
            (Some(pool.weight), Some(pool.bias))
        } else {
            (None, None)
        };
        Ok(Self {
            conv1: load_weight_norm_conv(
                checkpoint,
                group,
                &format!("{prefix}.conv1"),
                input,
                output,
                3,
                true,
                stream,
            )?,
            conv2: load_weight_norm_conv(
                checkpoint,
                group,
                &format!("{prefix}.conv2"),
                output,
                output,
                3,
                true,
                stream,
            )?,
            norm1: AdaIn::load(checkpoint, group, &format!("{prefix}.norm1"), input)?,
            norm2: AdaIn::load(checkpoint, group, &format!("{prefix}.norm2"), output)?,
            shortcut,
            pool_weight,
            pool_bias,
            input,
            output,
            upsample,
        })
    }

    /// Frame-major input/output.
    pub fn forward(
        &self,
        input: &[f32],
        frames: usize,
        style: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            input.len() == frames * self.input && style.len() == 128,
            "invalid Kokoro AdaIN block input"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let rows =
            stream.clone_htod(&input.iter().copied().map(f16::from_f32).collect::<Vec<_>>())?;
        let mut channels = stream.alloc_zeros::<f16>(input.len())?;
        kernels
            .ops
            .transpose_f16(&rows, &mut channels, frames as u32, self.input as u32)?;
        let mut residual = self.norm1.forward(&channels, frames, style, kernels)?;
        let mut activated = stream.alloc_zeros::<f16>(residual.len())?;
        kernels
            .ops
            .leaky_relu_f16(&residual, &mut activated, residual.len() as u32, 0.2)?;
        let output_frames = if self.upsample { frames * 2 } else { frames };
        if self.upsample {
            let mut pooled = stream.alloc_zeros::<f16>(self.input * output_frames)?;
            kernels.ops.conv_transpose1d_depthwise_f16(
                &activated,
                self.pool_weight.as_ref().expect("upsample pool"),
                self.pool_bias.as_ref().expect("upsample pool bias"),
                &mut pooled,
                self.input as u32,
                frames as u32,
                output_frames as u32,
                3,
                2,
                1,
            )?;
            activated = pooled;
        }
        residual = run_conv(&activated, output_frames, &self.conv1, kernels)?;
        residual = self
            .norm2
            .forward(&residual, output_frames, style, kernels)?;
        let mut second_activation = stream.alloc_zeros::<f16>(residual.len())?;
        kernels.ops.leaky_relu_f16(
            &residual,
            &mut second_activation,
            residual.len() as u32,
            0.2,
        )?;
        residual = run_conv(&second_activation, output_frames, &self.conv2, kernels)?;

        let mut shortcut = if self.upsample {
            let mut repeated = stream.alloc_zeros::<f16>(self.input * output_frames)?;
            kernels.ops.repeat_interleave_f16(
                &channels,
                &mut repeated,
                self.input as u32,
                frames as u32,
                2,
            )?;
            repeated
        } else {
            channels
        };
        if let Some(projection) = &self.shortcut {
            shortcut = run_conv(&shortcut, output_frames, projection, kernels)?;
        }
        let mut sum = stream.alloc_zeros::<f16>(self.output * output_frames)?;
        kernels.ops.add_f16(
            &residual,
            &shortcut,
            &mut sum,
            (self.output * output_frames) as u32,
        )?;
        let mut scaled = stream.alloc_zeros::<f16>(sum.len())?;
        kernels.ops.scale_f16(
            &sum,
            &mut scaled,
            sum.len() as u32,
            std::f32::consts::FRAC_1_SQRT_2,
        )?;
        let mut output_rows = stream.alloc_zeros::<f16>(scaled.len())?;
        kernels.ops.transpose_f16(
            &scaled,
            &mut output_rows,
            self.output as u32,
            output_frames as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; output_rows.len()];
        stream.memcpy_dtoh(&output_rows, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }
}

impl AdaIn {
    fn load(
        checkpoint: &KokoroCheckpoint,
        group: &str,
        prefix: &str,
        channels: usize,
    ) -> anyhow::Result<Self> {
        let (weight_shape, weight) =
            checkpoint.tensor_f32(group, &format!("{prefix}.fc.weight"))?;
        let (bias_shape, bias) = checkpoint.tensor_f32(group, &format!("{prefix}.fc.bias"))?;
        ensure!(
            weight_shape == [channels * 2, 128] && bias_shape == [channels * 2],
            "invalid Kokoro AdaIN {group}.{prefix}"
        );
        Ok(Self {
            weight,
            bias,
            channels,
        })
    }

    fn forward(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        style: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let affine = (0..self.channels * 2)
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
            &affine[..self.channels]
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let beta = stream.clone_htod(
            &affine[self.channels..]
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut output = stream.alloc_zeros::<f16>(self.channels * frames)?;
        kernels.ops.instance_norm_affine_f16(
            input,
            &gamma,
            &beta,
            &mut output,
            self.channels as u32,
            frames as u32,
            1e-5,
        )?;
        Ok(output)
    }
}

fn run_conv(
    input: &CudaSlice<f16>,
    frames: usize,
    conv: &Conv,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(conv.output * frames)?;
    kernels.ops.conv1d_general_f16(
        input,
        &conv.weight,
        &conv.bias,
        &mut output,
        conv.input as u32,
        conv.output as u32,
        frames as u32,
        frames as u32,
        conv.kernel as u32,
        1,
        (conv.kernel / 2) as u32,
        1,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn load_weight_norm_conv(
    checkpoint: &KokoroCheckpoint,
    group: &str,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    bias: bool,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv> {
    let (g_shape, g) = checkpoint.tensor_f32(group, &format!("{prefix}.weight_g"))?;
    let (v_shape, v) = checkpoint.tensor_f32(group, &format!("{prefix}.weight_v"))?;
    ensure!(g_shape == [output, 1, 1], "invalid {prefix}.weight_g");
    ensure!(
        v_shape == [output, input, kernel],
        "invalid {prefix}.weight_v {v_shape:?}"
    );
    let bias_values = if bias {
        let (shape, values) = checkpoint.tensor_f16(group, &format!("{prefix}.bias"))?;
        ensure!(shape == [output], "invalid {prefix}.bias");
        values
    } else {
        vec![f16::ZERO; output]
    };
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
        bias: stream.clone_htod(&bias_values)?,
        input,
        output,
        kernel,
    })
}
