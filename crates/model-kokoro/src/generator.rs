use crate::{KokoroCheckpoint, KokoroGeneratorResBlock};
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::path::Path;
use std::sync::Arc;

struct Conv {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    input: usize,
    output: usize,
    kernel: usize,
}

struct ConvTranspose {
    phase_weights: Vec<CudaSlice<f16>>,
    phase_offsets: Vec<i32>,
    bias: CudaSlice<f16>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

pub struct KokoroGenerator {
    source_weight: Vec<f32>,
    source_bias: f32,
    ups: Vec<ConvTranspose>,
    noise_convs: Vec<(Conv, usize, usize)>,
    noise_res: Vec<KokoroGeneratorResBlock>,
    resblocks: Vec<KokoroGeneratorResBlock>,
    conv_post: Conv,
}

impl KokoroGenerator {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
        let (source_weight_shape, source_weight) =
            checkpoint.tensor_f32("decoder", "module.generator.m_source.l_linear.weight")?;
        let (source_bias_shape, source_bias) =
            checkpoint.tensor_f32("decoder", "module.generator.m_source.l_linear.bias")?;
        ensure!(
            source_weight_shape == [1, 9] && source_bias_shape == [1],
            "invalid Kokoro generator source"
        );
        let ups = vec![
            load_transpose(
                &checkpoint,
                "module.generator.ups.0",
                512,
                256,
                20,
                10,
                5,
                stream,
            )?,
            load_transpose(
                &checkpoint,
                "module.generator.ups.1",
                256,
                128,
                12,
                6,
                3,
                stream,
            )?,
        ];
        let noise_convs = vec![
            (
                load_plain_conv(
                    &checkpoint,
                    "module.generator.noise_convs.0",
                    22,
                    256,
                    12,
                    stream,
                )?,
                6,
                3,
            ),
            (
                load_plain_conv(
                    &checkpoint,
                    "module.generator.noise_convs.1",
                    22,
                    128,
                    1,
                    stream,
                )?,
                1,
                0,
            ),
        ];
        let noise_res = vec![
            KokoroGeneratorResBlock::load(
                &checkpoint,
                "module.generator.noise_res.0",
                256,
                7,
                stream,
            )?,
            KokoroGeneratorResBlock::load(
                &checkpoint,
                "module.generator.noise_res.1",
                128,
                11,
                stream,
            )?,
        ];
        let mut resblocks = Vec::new();
        for (stage, channels) in [256, 128].into_iter().enumerate() {
            for (slot, kernel) in [3, 7, 11].into_iter().enumerate() {
                resblocks.push(KokoroGeneratorResBlock::load(
                    &checkpoint,
                    &format!("module.generator.resblocks.{}", stage * 3 + slot),
                    channels,
                    kernel,
                    stream,
                )?);
            }
        }
        Ok(Self {
            source_weight,
            source_bias: source_bias[0],
            ups,
            noise_convs,
            noise_res,
            resblocks,
            conv_post: load_weight_norm_conv(
                &checkpoint,
                "module.generator.conv_post",
                128,
                22,
                7,
                stream,
            )?,
        })
    }

    /// Convert frame-major decoder latent and F0 to 24-kHz waveform.
    pub fn synthesize(
        &self,
        latent: &[f32],
        f0: &[f32],
        frames: usize,
        style: &[f32],
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            latent.len() == frames * 512 && f0.len() == frames && style.len() == 128,
            "invalid Kokoro generator input"
        );
        let source = harmonic_source(f0, 300, &self.source_weight, self.source_bias, seed);
        let source_spectrum = source_stft(&source);
        let source_frames = frames * 60 + 1;
        ensure!(
            source_spectrum.len() == source_frames * 22,
            "invalid Kokoro source STFT"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let latent_rows = stream.clone_htod(
            &latent
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut hidden = stream.alloc_zeros::<f16>(latent.len())?;
        kernels
            .ops
            .transpose_f16(&latent_rows, &mut hidden, frames as u32, 512)?;
        let source_gpu = stream.clone_htod(
            &source_spectrum
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut hidden_frames = frames;
        for stage in 0..2 {
            let mut activated = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels
                .ops
                .leaky_relu_f16(&hidden, &mut activated, hidden.len() as u32, 0.1)?;
            let (source_conv, stride, padding) = &self.noise_convs[stage];
            let source_len = (source_frames + 2 * padding - source_conv.kernel) / stride + 1;
            let source_branch = run_conv(
                &source_gpu,
                source_frames,
                source_conv,
                *stride,
                *padding,
                kernels,
            )?;
            let source_branch =
                self.noise_res[stage].forward_device(&source_branch, source_len, style, kernels)?;
            hidden = run_transpose(&activated, hidden_frames, &self.ups[stage], kernels)?;
            hidden_frames *= self.ups[stage].stride;
            if stage == 1 {
                hidden =
                    reflection_pad_left(&hidden, self.ups[stage].output, hidden_frames, kernels)?;
                hidden_frames += 1;
            }
            ensure!(
                hidden.len() == source_branch.len(),
                "Kokoro generator source/main mismatch"
            );
            let mut fused = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels
                .ops
                .add_f16(&hidden, &source_branch, &mut fused, hidden.len() as u32)?;
            let blocks = &self.resblocks[stage * 3..stage * 3 + 3];
            let first = blocks[0].forward_device(&fused, hidden_frames, style, kernels)?;
            let second = blocks[1].forward_device(&fused, hidden_frames, style, kernels)?;
            let third = blocks[2].forward_device(&fused, hidden_frames, style, kernels)?;
            let mut first_two = stream.alloc_zeros::<f16>(fused.len())?;
            kernels
                .ops
                .add_f16(&first, &second, &mut first_two, fused.len() as u32)?;
            let mut sum = stream.alloc_zeros::<f16>(fused.len())?;
            kernels
                .ops
                .add_f16(&first_two, &third, &mut sum, fused.len() as u32)?;
            hidden = stream.alloc_zeros::<f16>(fused.len())?;
            kernels
                .ops
                .scale_f16(&sum, &mut hidden, fused.len() as u32, 1.0 / 3.0)?;
        }
        let mut activated = stream.alloc_zeros::<f16>(hidden.len())?;
        kernels
            .ops
            .leaky_relu_f16(&hidden, &mut activated, hidden.len() as u32, 0.01)?;
        let spectrum = run_conv(&activated, hidden_frames, &self.conv_post, 1, 3, kernels)?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; spectrum.len()];
        stream.memcpy_dtoh(&spectrum, &mut host)?;
        let host = host.into_iter().map(f16::to_f32).collect::<Vec<_>>();
        Ok(output_istft(&host, hidden_frames))
    }
}

fn run_conv(
    input: &CudaSlice<f16>,
    input_frames: usize,
    conv: &Conv,
    stride: usize,
    padding: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let output_frames = (input_frames + 2 * padding - conv.kernel) / stride + 1;
    let stream = Arc::clone(kernels.ops.stream());
    let width = conv.input * conv.kernel;
    let mut unfolded = stream.alloc_zeros::<f16>(output_frames * width)?;
    kernels.ops.unfold_conv1d_f16(
        input,
        &mut unfolded,
        conv.input as u32,
        input_frames as u32,
        output_frames as u32,
        conv.kernel as u32,
        stride as u32,
        padding as u32,
        1,
    )?;
    let mut rows = stream.alloc_zeros::<f16>(output_frames * conv.output)?;
    kernels.gemm.matmul_f16(
        &unfolded,
        &conv.weight,
        &mut rows,
        output_frames as u32,
        conv.output as u32,
        width as u32,
    )?;
    kernels.ops.add_bias_f16_inplace(
        &mut rows,
        &conv.bias,
        output_frames as u32,
        conv.output as u32,
    )?;
    let mut output = stream.alloc_zeros::<f16>(conv.output * output_frames)?;
    kernels
        .ops
        .transpose_f16(&rows, &mut output, output_frames as u32, conv.output as u32)?;
    Ok(output)
}

fn run_transpose(
    input: &CudaSlice<f16>,
    input_frames: usize,
    conv: &ConvTranspose,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let output_frames = (input_frames - 1) * conv.stride - 2 * conv.padding + conv.kernel;
    ensure!(
        output_frames == input_frames * conv.stride,
        "unsupported Kokoro transposed convolution geometry"
    );
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(conv.output * output_frames)?;
    let width = conv.input * 2;
    let mut unfolded = stream.alloc_zeros::<f16>(input_frames * width)?;
    let mut rows = stream.alloc_zeros::<f16>(input_frames * conv.output)?;
    for (phase, (weight, offset)) in conv
        .phase_weights
        .iter()
        .zip(&conv.phase_offsets)
        .enumerate()
    {
        kernels.ops.unfold_adjacent_f16(
            input,
            &mut unfolded,
            conv.input as u32,
            input_frames as u32,
            *offset,
        )?;
        kernels.gemm.matmul_f16(
            &unfolded,
            weight,
            &mut rows,
            input_frames as u32,
            conv.output as u32,
            width as u32,
        )?;
        kernels.ops.scatter_conv_transpose_phase_f16(
            &rows,
            &conv.bias,
            &mut output,
            input_frames as u32,
            conv.output as u32,
            conv.stride as u32,
            phase as u32,
        )?;
    }
    Ok(output)
}

fn reflection_pad_left(
    input: &CudaSlice<f16>,
    channels: usize,
    frames: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut padded = stream.alloc_zeros::<f16>((frames + 1) * channels)?;
    kernels
        .ops
        .reflection_pad_left_f16(input, &mut padded, channels as u32, frames as u32)?;
    Ok(padded)
}

fn load_plain_conv(
    checkpoint: &KokoroCheckpoint,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv> {
    let (weight_shape, weight) = checkpoint.tensor_f16("decoder", &format!("{prefix}.weight"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16("decoder", &format!("{prefix}.bias"))?;
    ensure!(
        weight_shape == [output, input, kernel] && bias_shape == [output],
        "invalid Kokoro generator convolution {prefix}"
    );
    Ok(Conv {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
        input,
        output,
        kernel,
    })
}

fn load_weight_norm_conv(
    checkpoint: &KokoroCheckpoint,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv> {
    let (g_shape, g) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_g"))?;
    let (v_shape, v) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_v"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16("decoder", &format!("{prefix}.bias"))?;
    ensure!(
        g_shape == [output, 1, 1] && v_shape == [output, input, kernel] && bias_shape == [output],
        "invalid Kokoro weight-normalized convolution {prefix}"
    );
    let width = input * kernel;
    let mut weight = Vec::with_capacity(v.len());
    for channel in 0..output {
        let row = &v[channel * width..(channel + 1) * width];
        let scale = g[channel]
            / row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
        weight.extend(row.iter().map(|value| f16::from_f32(value * scale)));
    }
    Ok(Conv {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
        input,
        output,
        kernel,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_transpose(
    checkpoint: &KokoroCheckpoint,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<ConvTranspose> {
    let (g_shape, g) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_g"))?;
    let (v_shape, v) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_v"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16("decoder", &format!("{prefix}.bias"))?;
    ensure!(
        g_shape == [input, 1, 1] && v_shape == [input, output, kernel] && bias_shape == [output],
        "invalid Kokoro transposed convolution {prefix}"
    );
    let width = output * kernel;
    let mut weight = Vec::with_capacity(v.len());
    for channel in 0..input {
        let row = &v[channel * width..(channel + 1) * width];
        let scale = g[channel]
            / row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
        weight.extend(row.iter().map(|value| f16::from_f32(value * scale)));
    }
    ensure!(
        kernel == stride * 2 && kernel == stride + 2 * padding,
        "unsupported Kokoro transposed convolution geometry"
    );
    let mut phase_weights = Vec::with_capacity(stride);
    let mut phase_offsets = Vec::with_capacity(stride);
    for phase in 0..stride {
        let combined = phase + padding;
        let kernel_first = combined % stride;
        let kernel_second = kernel_first + stride;
        let carry = combined / stride;
        let mut phase_weight = Vec::with_capacity(output * input * 2);
        for output_channel in 0..output {
            for input_channel in 0..input {
                let base = (input_channel * output + output_channel) * kernel;
                // unfold_adjacent emits source q+carry-1 followed by q+carry.
                phase_weight.push(weight[base + kernel_second]);
                phase_weight.push(weight[base + kernel_first]);
            }
        }
        phase_weights.push(stream.clone_htod(&phase_weight)?);
        phase_offsets.push(carry as i32 - 1);
    }
    Ok(ConvTranspose {
        phase_weights,
        phase_offsets,
        bias: stream.clone_htod(&bias)?,
        input,
        output,
        kernel,
        stride,
        padding,
    })
}

fn harmonic_source(f0: &[f32], scale: usize, weight: &[f32], bias: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut phase = [0.0f32; 9];
    let offsets = std::array::from_fn::<_, 9, _>(|harmonic| {
        if harmonic == 0 {
            0.0
        } else {
            rng.uniform() * std::f32::consts::TAU
        }
    });
    let mut output = Vec::with_capacity(f0.len() * scale);
    for value in f0 {
        for _ in 0..scale {
            let voiced = *value > 10.0;
            let mut merged = bias;
            for harmonic in 0..9 {
                phase[harmonic] =
                    (phase[harmonic] + value * (harmonic + 1) as f32 / 24_000.0).fract();
                let sine =
                    0.1 * (std::f32::consts::TAU * phase[harmonic] + offsets[harmonic]).sin();
                let noise = rng.normal() * if voiced { 0.003 } else { 0.1 / 3.0 };
                merged += weight[harmonic] * (if voiced { sine } else { 0.0 } + noise);
            }
            output.push(merged.tanh());
        }
    }
    output
}

fn source_stft(source: &[f32]) -> Vec<f32> {
    const N: usize = 20;
    const HOP: usize = 5;
    let frames = source.len() / HOP + 1;
    let window = hann();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut magnitude = vec![0.0f32; 11 * frames];
    let mut phase = vec![0.0f32; 11 * frames];
    for frame in 0..frames {
        let mut buffer = vec![Complex32::default(); N];
        for index in 0..N {
            let position = reflect(frame * HOP + index, N / 2, source.len());
            buffer[index].re = source[position] * window[index];
        }
        fft.process(&mut buffer);
        for frequency in 0..11 {
            let value = buffer[frequency];
            magnitude[frequency * frames + frame] = value.norm();
            phase[frequency * frames + frame] = value.arg();
        }
    }
    magnitude.extend(phase);
    magnitude
}

fn output_istft(spectrum: &[f32], frames: usize) -> Vec<f32> {
    const N: usize = 20;
    const HOP: usize = 5;
    let full_len = N + HOP * (frames - 1);
    let mut output = vec![0.0f32; full_len];
    let mut envelope = vec![0.0f32; full_len];
    let window = hann();
    let mut planner = FftPlanner::<f32>::new();
    let inverse = planner.plan_fft_inverse(N);
    for frame in 0..frames {
        let mut buffer = vec![Complex32::default(); N];
        for frequency in 0..11 {
            let magnitude = spectrum[frequency * frames + frame].exp();
            let phase = spectrum[(11 + frequency) * frames + frame].sin();
            buffer[frequency] = Complex32::from_polar(magnitude, phase);
        }
        for frequency in 1..10 {
            buffer[N - frequency] = buffer[frequency].conj();
        }
        inverse.process(&mut buffer);
        for index in 0..N {
            let position = frame * HOP + index;
            output[position] += buffer[index].re / N as f32 * window[index];
            envelope[position] += window[index] * window[index];
        }
    }
    for (value, norm) in output.iter_mut().zip(envelope) {
        if norm > 1e-11 {
            *value /= norm;
        }
    }
    output[N / 2..output.len() - N / 2].to_vec()
}

fn hann() -> [f32; 20] {
    std::array::from_fn(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / 20.0).cos())
}

fn reflect(padded_position: usize, padding: usize, len: usize) -> usize {
    let position = padded_position as isize - padding as isize;
    if position < 0 {
        (-position) as usize
    } else if position >= len as isize {
        (2 * len as isize - position - 2) as usize
    } else {
        position as usize
    }
}

struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            spare: None,
        }
    }
    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) as f32 + 0.5) / (1u32 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().max(1e-7).ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}
