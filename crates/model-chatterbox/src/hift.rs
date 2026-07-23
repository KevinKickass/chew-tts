use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_safetensors::MappedSafetensors;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::path::Path;
use std::sync::Arc;

const MEL_BINS: usize = 80;
const F0_CHANNELS: usize = 512;

struct Conv1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
}

struct ConvTranspose1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

struct SnakeConv {
    alpha: CudaSlice<f16>,
    conv: Conv1d,
    dilation: usize,
}

struct HiFTResBlock {
    first: Vec<SnakeConv>,
    second: Vec<SnakeConv>,
    channels: usize,
}

/// Native Chatterbox HiFT neural-source-filter vocoder.
pub struct ChatterboxHiFT {
    f0: ChatterboxF0Predictor,
    source_weight: Vec<f32>,
    source_bias: f32,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<HiFTResBlock>,
    resblocks: Vec<HiFTResBlock>,
    conv_post: Conv1d,
}

/// Chatterbox HiFT's five-layer convolutional F0 predictor.
pub struct ChatterboxF0Predictor {
    layers: Vec<Conv1d>,
    classifier_weight: CudaSlice<f16>,
    classifier_bias: f32,
}

impl ChatterboxF0Predictor {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let mut layers = Vec::with_capacity(5);
        for (layer, index) in [0, 2, 4, 6, 8].into_iter().enumerate() {
            let prefix = format!("mel2wav.f0_predictor.condnet.{index}");
            let in_channels = if layer == 0 { MEL_BINS } else { F0_CHANNELS };
            layers.push(load_weight_norm_conv(
                &weights,
                &prefix,
                in_channels,
                F0_CHANNELS,
                3,
                stream,
            )?);
        }
        let (classifier_shape, classifier) =
            weights.tensor_f16("mel2wav.f0_predictor.classifier.weight")?;
        ensure!(
            classifier_shape == [1, F0_CHANNELS],
            "invalid F0 classifier"
        );
        let (bias_shape, bias) = weights.tensor_f32("mel2wav.f0_predictor.classifier.bias")?;
        ensure!(bias_shape == [1], "invalid F0 classifier bias");
        Ok(Self {
            layers,
            classifier_weight: stream.clone_htod(&classifier)?,
            classifier_bias: bias[0],
        })
    }

    /// Predict one absolute F0 value per frame from frame-major `[T, 80]` mel.
    pub fn predict(
        &self,
        mel: &[f32],
        frames: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(frames > 0, "F0 input must not be empty");
        ensure!(
            mel.len() == frames * MEL_BINS,
            "F0 input has {} values, expected {}",
            mel.len(),
            frames * MEL_BINS
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mel_f16 = mel.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let mel_rows = stream.clone_htod(&mel_f16)?;
        let mut hidden = stream.alloc_zeros::<f16>(mel.len())?;
        kernels
            .ops
            .transpose_f16(&mel_rows, &mut hidden, frames as u32, MEL_BINS as u32)?;
        let mut channels = MEL_BINS;
        for layer in &self.layers {
            ensure!(layer.in_channels == channels, "invalid F0 layer chain");
            let convolved = run_conv(&hidden, frames, layer, 1, 1, 1, kernels)?;
            let mut activated = stream.alloc_zeros::<f16>(layer.out_channels * frames)?;
            kernels.ops.elu_f16(
                &convolved,
                &mut activated,
                (layer.out_channels * frames) as u32,
            )?;
            hidden = activated;
            channels = layer.out_channels;
        }
        let mut rows = stream.alloc_zeros::<f16>(frames * F0_CHANNELS)?;
        kernels
            .ops
            .transpose_f16(&hidden, &mut rows, F0_CHANNELS as u32, frames as u32)?;
        let mut output = stream.alloc_zeros::<f16>(frames)?;
        kernels.gemm.matmul_f16(
            &rows,
            &self.classifier_weight,
            &mut output,
            frames as u32,
            1,
            F0_CHANNELS as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; frames];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host
            .into_iter()
            .map(|value| (value.to_f32() + self.classifier_bias).abs())
            .collect())
    }
}

impl ChatterboxHiFT {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let f0 = ChatterboxF0Predictor::load(model_dir, stream)?;
        let (source_weight_shape, source_weight) =
            weights.tensor_f32("mel2wav.m_source.l_linear.weight")?;
        let (source_bias_shape, source_bias) =
            weights.tensor_f32("mel2wav.m_source.l_linear.bias")?;
        ensure!(source_weight_shape == [1, 9], "invalid HiFT source weight");
        ensure!(source_bias_shape == [1], "invalid HiFT source bias");
        let conv_pre = load_weight_norm_conv(&weights, "mel2wav.conv_pre", 80, 512, 7, stream)?;
        let mut ups = Vec::new();
        for (index, (input, output, kernel, stride, padding)) in [
            (512, 256, 16, 8, 4),
            (256, 128, 11, 5, 3),
            (128, 64, 7, 3, 2),
        ]
        .into_iter()
        .enumerate()
        {
            ups.push(load_weight_norm_transpose(
                &weights,
                &format!("mel2wav.ups.{index}"),
                input,
                output,
                kernel,
                stride,
                padding,
                stream,
            )?);
        }
        let mut source_downs = Vec::new();
        for (index, (output, kernel, stride, padding)) in
            [(256, 30, 15, 7), (128, 6, 3, 1), (64, 1, 1, 0)]
                .into_iter()
                .enumerate()
        {
            source_downs.push(load_plain_conv(
                &weights,
                &format!("mel2wav.source_downs.{index}"),
                18,
                output,
                kernel,
                stream,
            )?);
            debug_assert_eq!(conv_output_len(121, kernel, stride, padding, 1), {
                (121 + 2 * padding - kernel) / stride + 1
            });
        }
        let source_resblocks = [(256, 7), (128, 7), (64, 11)]
            .into_iter()
            .enumerate()
            .map(|(index, (channels, kernel))| {
                HiFTResBlock::load(
                    &weights,
                    &format!("mel2wav.source_resblocks.{index}"),
                    channels,
                    kernel,
                    stream,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut resblocks = Vec::new();
        for (stage, channels) in [256, 128, 64].into_iter().enumerate() {
            for (slot, kernel) in [3, 7, 11].into_iter().enumerate() {
                resblocks.push(HiFTResBlock::load(
                    &weights,
                    &format!("mel2wav.resblocks.{}", stage * 3 + slot),
                    channels,
                    kernel,
                    stream,
                )?);
            }
        }
        let conv_post = load_weight_norm_conv(&weights, "mel2wav.conv_post", 64, 18, 7, stream)?;
        Ok(Self {
            f0,
            source_weight,
            source_bias: source_bias[0],
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
        })
    }

    /// Convert frame-major `[T, 80]` S3 mel into 24-kHz mono PCM.
    pub fn synthesize(
        &self,
        mel: &[f32],
        frames: usize,
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            mel.len() == frames * MEL_BINS && frames > 0,
            "invalid HiFT mel geometry"
        );
        let f0 = self.f0.predict(mel, frames, kernels)?;
        let source = harmonic_source(&f0, 480, &self.source_weight, self.source_bias, seed);
        let source_stft = stft_source(&source);
        ensure!(
            source_stft.len() == 18 * (frames * 120 + 1),
            "unexpected source STFT geometry"
        );

        let stream = Arc::clone(kernels.ops.stream());
        let mel_f16 = mel.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let mel_rows = stream.clone_htod(&mel_f16)?;
        let mut mel_channels = stream.alloc_zeros::<f16>(mel.len())?;
        kernels
            .ops
            .transpose_f16(&mel_rows, &mut mel_channels, frames as u32, MEL_BINS as u32)?;
        let mut hidden = run_conv(&mel_channels, frames, &self.conv_pre, 1, 3, 1, kernels)?;
        debug_device("conv_pre", &hidden, kernels)?;
        let source_gpu = stream.clone_htod(
            &source_stft
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let source_frames = frames * 120 + 1;
        let stage_geometry = [(15, 7), (3, 1), (1, 0)];

        for stage in 0..3 {
            let activated = leaky(
                &hidden,
                self.ups[stage].in_channels,
                frames_for_stage(frames, stage),
                kernels,
            )?;
            hidden = run_transpose(
                &activated,
                frames_for_stage(frames, stage),
                &self.ups[stage],
                kernels,
            )?;
            debug_device(&format!("ups.{stage}"), &hidden, kernels)?;
            let mut hidden_len = frames_for_stage(frames, stage + 1);
            if stage == 2 {
                hidden = reflection_pad_left(
                    &hidden,
                    self.ups[stage].out_channels,
                    hidden_len,
                    kernels,
                )?;
                hidden_len += 1;
            }
            let (stride, padding) = stage_geometry[stage];
            let mut source_branch = run_conv(
                &source_gpu,
                source_frames,
                &self.source_downs[stage],
                stride,
                padding,
                1,
                kernels,
            )?;
            source_branch =
                self.source_resblocks[stage].forward(source_branch, hidden_len, kernels)?;
            debug_device(
                &format!("source_resblocks.{stage}"),
                &source_branch,
                kernels,
            )?;
            ensure!(
                hidden.len() == source_branch.len(),
                "HiFT source/main shape mismatch at stage {stage}"
            );
            let mut fused = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels
                .ops
                .add_f16(&hidden, &source_branch, &mut fused, hidden.len() as u32)?;

            let mut sum: Option<CudaSlice<f16>> = None;
            for block in &self.resblocks[stage * 3..stage * 3 + 3] {
                let branch = block.forward(clone_device(&fused, kernels)?, hidden_len, kernels)?;
                if let Some(existing) = sum.take() {
                    let mut added = stream.alloc_zeros::<f16>(branch.len())?;
                    kernels
                        .ops
                        .add_f16(&existing, &branch, &mut added, branch.len() as u32)?;
                    sum = Some(added);
                } else {
                    sum = Some(branch);
                }
            }
            hidden = sum.context("missing HiFT residual branch")?;
            let mut host = vec![f16::ZERO; hidden.len()];
            stream.synchronize()?;
            stream.memcpy_dtoh(&hidden, &mut host)?;
            for value in &mut host {
                *value = f16::from_f32(value.to_f32() / 3.0);
            }
            hidden = stream.clone_htod(&host)?;
            debug_device(&format!("resblocks.{stage}"), &hidden, kernels)?;
        }
        let final_len = frames * 120 + 1;
        let mut activated = stream.alloc_zeros::<f16>(hidden.len())?;
        kernels
            .ops
            .leaky_relu_f16(&hidden, &mut activated, hidden.len() as u32, 0.01)?;
        let spectrum = run_conv(&activated, final_len, &self.conv_post, 1, 3, 1, kernels)?;
        debug_device("conv_post", &spectrum, kernels)?;
        stream.synchronize()?;
        let mut spectrum_host = vec![f16::ZERO; spectrum.len()];
        stream.memcpy_dtoh(&spectrum, &mut spectrum_host)?;
        let spectrum_host = spectrum_host
            .into_iter()
            .map(f16::to_f32)
            .collect::<Vec<_>>();
        Ok(istft_output(&spectrum_host, final_len)
            .into_iter()
            .map(|sample| sample.clamp(-0.99, 0.99))
            .collect())
    }
}

fn debug_device(
    label: &str,
    values: &CudaSlice<f16>,
    kernels: &mut GpuKernels,
) -> anyhow::Result<()> {
    if std::env::var_os("CHEW_TTS_DEBUG").is_none() {
        return Ok(());
    }
    let stream = Arc::clone(kernels.ops.stream());
    stream.synchronize()?;
    let mut host = vec![f16::ZERO; values.len()];
    stream.memcpy_dtoh(values, &mut host)?;
    let first = host
        .iter()
        .take(6)
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let sum_sq = host
        .iter()
        .map(|value| {
            let value = value.to_f32() as f64;
            value * value
        })
        .sum::<f64>();
    eprintln!(
        "{label}: n={} rms={:.6} first={first:?}",
        host.len(),
        (sum_sq / host.len() as f64).sqrt()
    );
    Ok(())
}

impl HiFTResBlock {
    fn load(
        weights: &MappedSafetensors,
        prefix: &str,
        channels: usize,
        kernel: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let mut first = Vec::new();
        let mut second = Vec::new();
        for index in 0..3 {
            for (target, group, dilation) in [
                (&mut first, "convs1", [1, 3, 5][index]),
                (&mut second, "convs2", 1),
            ] {
                let alpha_name = format!(
                    "{prefix}.activations{}.{}.alpha",
                    if group == "convs1" { 1 } else { 2 },
                    index
                );
                let (alpha_shape, alpha) = weights.tensor_f16(&alpha_name)?;
                ensure!(alpha_shape == [channels], "invalid {alpha_name}");
                target.push(SnakeConv {
                    alpha: stream.clone_htod(&alpha)?,
                    conv: load_weight_norm_conv(
                        weights,
                        &format!("{prefix}.{group}.{index}"),
                        channels,
                        channels,
                        kernel,
                        stream,
                    )?,
                    dilation,
                });
            }
        }
        Ok(Self {
            first,
            second,
            channels,
        })
    }

    fn forward(
        &self,
        mut hidden: CudaSlice<f16>,
        frames: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let stream = Arc::clone(kernels.ops.stream());
        for index in 0..3 {
            let mut activated = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels.ops.snake_f16(
                &hidden,
                &self.first[index].alpha,
                &mut activated,
                self.channels as u32,
                frames as u32,
            )?;
            let first = run_conv(
                &activated,
                frames,
                &self.first[index].conv,
                1,
                (self.first[index].conv.kernel * self.first[index].dilation
                    - self.first[index].dilation)
                    / 2,
                self.first[index].dilation,
                kernels,
            )?;
            let mut activated2 = stream.alloc_zeros::<f16>(first.len())?;
            kernels.ops.snake_f16(
                &first,
                &self.second[index].alpha,
                &mut activated2,
                self.channels as u32,
                frames as u32,
            )?;
            let second = run_conv(
                &activated2,
                frames,
                &self.second[index].conv,
                1,
                (self.second[index].conv.kernel - 1) / 2,
                1,
                kernels,
            )?;
            let mut residual = stream.alloc_zeros::<f16>(hidden.len())?;
            kernels
                .ops
                .add_f16(&hidden, &second, &mut residual, hidden.len() as u32)?;
            hidden = residual;
        }
        Ok(hidden)
    }
}

fn load_weight_norm_conv(
    weights: &MappedSafetensors,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv1d> {
    let g_name = format!("{prefix}.parametrizations.weight.original0");
    let v_name = format!("{prefix}.parametrizations.weight.original1");
    let bias_name = format!("{prefix}.bias");
    let (g_shape, g) = weights
        .tensor_f32(&g_name)
        .with_context(|| format!("could not load {g_name}"))?;
    let (v_shape, v) = weights
        .tensor_f32(&v_name)
        .with_context(|| format!("could not load {v_name}"))?;
    let (bias_shape, bias) = weights.tensor_f16(&bias_name)?;
    ensure!(g_shape == [out_channels, 1, 1], "invalid {g_name} shape");
    ensure!(
        v_shape == [out_channels, in_channels, kernel],
        "invalid {v_name} shape"
    );
    ensure!(bias_shape == [out_channels], "invalid {bias_name} shape");
    let width = in_channels * kernel;
    let mut normalized = Vec::with_capacity(v.len());
    for channel in 0..out_channels {
        let row = &v[channel * width..(channel + 1) * width];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        let scale = g[channel] / norm.max(1e-12);
        normalized.extend(row.iter().map(|value| f16::from_f32(value * scale)));
    }
    Ok(Conv1d {
        weight: stream.clone_htod(&normalized)?,
        bias: stream.clone_htod(&bias)?,
        in_channels,
        out_channels,
        kernel,
    })
}

fn load_plain_conv(
    weights: &MappedSafetensors,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv1d> {
    let (weight_shape, weight) = weights.tensor_f16(&format!("{prefix}.weight"))?;
    let (bias_shape, bias) = weights.tensor_f16(&format!("{prefix}.bias"))?;
    ensure!(
        weight_shape == [out_channels, in_channels, kernel],
        "invalid {prefix}.weight shape"
    );
    ensure!(bias_shape == [out_channels], "invalid {prefix}.bias shape");
    Ok(Conv1d {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
        in_channels,
        out_channels,
        kernel,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_weight_norm_transpose(
    weights: &MappedSafetensors,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<ConvTranspose1d> {
    let g_name = format!("{prefix}.parametrizations.weight.original0");
    let v_name = format!("{prefix}.parametrizations.weight.original1");
    let (g_shape, g) = weights.tensor_f32(&g_name)?;
    let (v_shape, v) = weights.tensor_f32(&v_name)?;
    let (bias_shape, bias) = weights.tensor_f16(&format!("{prefix}.bias"))?;
    ensure!(g_shape == [in_channels, 1, 1], "invalid {g_name} shape");
    ensure!(
        v_shape == [in_channels, out_channels, kernel],
        "invalid {v_name} shape"
    );
    ensure!(bias_shape == [out_channels], "invalid {prefix}.bias shape");
    let width = out_channels * kernel;
    let mut normalized = Vec::with_capacity(v.len());
    for channel in 0..in_channels {
        let row = &v[channel * width..(channel + 1) * width];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        let scale = g[channel] / norm.max(1e-12);
        normalized.extend(row.iter().map(|value| f16::from_f32(value * scale)));
    }
    Ok(ConvTranspose1d {
        weight: stream.clone_htod(&normalized)?,
        bias: stream.clone_htod(&bias)?,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
    })
}

fn conv_output_len(
    input_len: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (input_len + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}

fn run_conv(
    input: &CudaSlice<f16>,
    input_len: usize,
    conv: &Conv1d,
    stride: usize,
    padding: usize,
    dilation: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let output_len = conv_output_len(input_len, conv.kernel, stride, padding, dilation);
    let stream = Arc::clone(kernels.ops.stream());
    let width = conv.in_channels * conv.kernel;
    let mut unfolded = stream.alloc_zeros::<f16>(output_len * width)?;
    kernels.ops.unfold_conv1d_f16(
        input,
        &mut unfolded,
        conv.in_channels as u32,
        input_len as u32,
        output_len as u32,
        conv.kernel as u32,
        stride as u32,
        padding as u32,
        dilation as u32,
    )?;
    let mut rows = stream.alloc_zeros::<f16>(output_len * conv.out_channels)?;
    kernels.gemm.matmul_f16(
        &unfolded,
        &conv.weight,
        &mut rows,
        output_len as u32,
        conv.out_channels as u32,
        width as u32,
    )?;
    kernels.ops.add_bias_f16_inplace(
        &mut rows,
        &conv.bias,
        output_len as u32,
        conv.out_channels as u32,
    )?;
    let mut output = stream.alloc_zeros::<f16>(conv.out_channels * output_len)?;
    kernels.ops.transpose_f16(
        &rows,
        &mut output,
        output_len as u32,
        conv.out_channels as u32,
    )?;
    Ok(output)
}

fn run_transpose(
    input: &CudaSlice<f16>,
    input_len: usize,
    conv: &ConvTranspose1d,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let output_len = (input_len - 1) * conv.stride - 2 * conv.padding + conv.kernel;
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(conv.out_channels * output_len)?;
    kernels.ops.conv_transpose1d_general_f16(
        input,
        &conv.weight,
        &conv.bias,
        &mut output,
        conv.in_channels as u32,
        conv.out_channels as u32,
        input_len as u32,
        output_len as u32,
        conv.kernel as u32,
        conv.stride as u32,
        conv.padding as u32,
    )?;
    Ok(output)
}

fn leaky(
    input: &CudaSlice<f16>,
    channels: usize,
    frames: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    ensure!(
        input.len() == channels * frames,
        "invalid activation geometry"
    );
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(input.len())?;
    kernels
        .ops
        .leaky_relu_f16(input, &mut output, input.len() as u32, 0.1)?;
    Ok(output)
}

fn clone_device(
    input: &CudaSlice<f16>,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f16>(input.len())?;
    kernels
        .ops
        .copy_f16(input, &mut output.slice_mut(..), input.len() as u32)?;
    Ok(output)
}

fn reflection_pad_left(
    input: &CudaSlice<f16>,
    channels: usize,
    frames: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    ensure!(
        frames >= 2 && input.len() == channels * frames,
        "invalid reflection pad"
    );
    let stream = Arc::clone(kernels.ops.stream());
    stream.synchronize()?;
    let mut host = vec![f16::ZERO; input.len()];
    stream.memcpy_dtoh(input, &mut host)?;
    let mut padded = vec![f16::ZERO; channels * (frames + 1)];
    for channel in 0..channels {
        let source = &host[channel * frames..(channel + 1) * frames];
        let target = &mut padded[channel * (frames + 1)..(channel + 1) * (frames + 1)];
        target[0] = source[1];
        target[1..].copy_from_slice(source);
    }
    Ok(stream.clone_htod(&padded)?)
}

fn frames_for_stage(mel_frames: usize, stage: usize) -> usize {
    match stage {
        0 => mel_frames,
        1 => mel_frames * 8,
        2 => mel_frames * 40,
        3 => mel_frames * 120,
        _ => unreachable!(),
    }
}

fn harmonic_source(f0: &[f32], scale: usize, weight: &[f32], bias: f32, seed: u64) -> Vec<f32> {
    let mut rng = GaussianRng::new(seed);
    let phases = (0..9)
        .map(|harmonic| {
            if harmonic == 0 {
                0.0
            } else {
                (rng.uniform() * 2.0 - 1.0) * std::f32::consts::PI
            }
        })
        .collect::<Vec<_>>();
    let mut phase = [0.0f32; 9];
    let mut output = Vec::with_capacity(f0.len() * scale);
    for frame_f0 in f0 {
        for _ in 0..scale {
            let voiced = *frame_f0 > 10.0;
            let mut merged = bias;
            for harmonic in 0..9 {
                phase[harmonic] =
                    (phase[harmonic] + *frame_f0 * (harmonic + 1) as f32 / 24_000.0).fract();
                let sine = 0.1 * (std::f32::consts::TAU * phase[harmonic] + phases[harmonic]).sin();
                let noise_scale = if voiced { 0.003 } else { 0.1 / 3.0 };
                let value = if voiced { sine } else { 0.0 } + rng.normal() * noise_scale;
                merged += value * weight[harmonic];
            }
            output.push(merged.tanh());
        }
    }
    output
}

fn stft_source(source: &[f32]) -> Vec<f32> {
    const NFFT: usize = 16;
    const HOP: usize = 4;
    let padding = NFFT / 2;
    let padded_len = source.len() + 2 * padding;
    let frames = (padded_len - NFFT) / HOP + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(NFFT);
    let window = hann_window();
    let mut real = vec![0.0f32; 9 * frames];
    let mut imaginary = vec![0.0f32; 9 * frames];
    for frame in 0..frames {
        let mut buffer = vec![Complex32::default(); NFFT];
        for (index, value) in buffer.iter_mut().enumerate() {
            let padded_index = frame * HOP + index;
            let source_index =
                reflect_index(padded_index as isize - padding as isize, source.len());
            value.re = source[source_index] * window[index];
        }
        fft.process(&mut buffer);
        for frequency in 0..9 {
            real[frequency * frames + frame] = buffer[frequency].re;
            imaginary[frequency * frames + frame] = buffer[frequency].im;
        }
    }
    real.extend(imaginary);
    real
}

fn istft_output(spectrum: &[f32], frames: usize) -> Vec<f32> {
    const NFFT: usize = 16;
    const HOP: usize = 4;
    let centered_len = NFFT + HOP * (frames - 1);
    let mut waveform = vec![0.0f32; centered_len];
    let mut envelope = vec![0.0f32; centered_len];
    let window = hann_window();
    let mut planner = FftPlanner::<f32>::new();
    let inverse = planner.plan_fft_inverse(NFFT);
    for frame in 0..frames {
        let mut buffer = vec![Complex32::default(); NFFT];
        for frequency in 0..9 {
            let magnitude = spectrum[frequency * frames + frame].exp().min(100.0);
            let phase = spectrum[(9 + frequency) * frames + frame].sin();
            buffer[frequency] = Complex32::from_polar(magnitude, phase);
        }
        for frequency in 1..8 {
            buffer[NFFT - frequency] = buffer[frequency].conj();
        }
        inverse.process(&mut buffer);
        for index in 0..NFFT {
            let position = frame * HOP + index;
            waveform[position] += buffer[index].re / NFFT as f32 * window[index];
            envelope[position] += window[index] * window[index];
        }
    }
    for (sample, norm) in waveform.iter_mut().zip(envelope) {
        if norm > 1e-11 {
            *sample /= norm;
        }
    }
    waveform[8..waveform.len() - 8].to_vec()
}

fn hann_window() -> [f32; 16] {
    std::array::from_fn(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / 16.0).cos())
}

fn reflect_index(index: isize, len: usize) -> usize {
    if index < 0 {
        (-index) as usize
    } else if index >= len as isize {
        (2 * len as isize - index - 2) as usize
    } else {
        index as usize
    }
}

struct GaussianRng {
    state: u64,
    spare: Option<f32>,
}

impl GaussianRng {
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
