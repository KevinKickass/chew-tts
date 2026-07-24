use crate::VoxCpm2Config;
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::load_f32_tensor;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

struct Conv1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
}

struct ConvTranspose1d {
    phase_weights: Vec<CudaSlice<f16>>,
    bias: CudaSlice<f16>,
    input_channels: usize,
    output_channels: usize,
    stride: usize,
}

struct ResidualUnit {
    first_alpha: CudaSlice<f16>,
    first_conv: Conv1d,
    second_alpha: CudaSlice<f16>,
    second_conv: Conv1d,
}

struct DecoderStage {
    scale: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    alpha: CudaSlice<f16>,
    upsample: ConvTranspose1d,
    residuals: Vec<ResidualUnit>,
}

struct F32Conv1d {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
}

struct F32ResidualUnit {
    first_alpha: CudaSlice<f32>,
    first_conv: F32Conv1d,
    second_alpha: CudaSlice<f32>,
    second_conv: F32Conv1d,
}

struct EncoderBlock {
    residuals: Vec<F32ResidualUnit>,
    alpha: CudaSlice<f32>,
    downsample: F32Conv1d,
    stride: usize,
}

/// Native mixed-precision VoxCPM2 48-kHz AudioVAE decoder.
///
/// The official VAE runs in FP32. Chew folds weight normalization once while
/// loading and stores FP16 convolution weights; every convolution kernel
/// accumulates in FP32. No Python or pickle parser is involved at runtime.
pub struct VoxCpm2AudioDecoder {
    depthwise_stem: Conv1d,
    expand: Conv1d,
    stages: Vec<DecoderStage>,
    final_alpha: CudaSlice<f16>,
    head: Conv1d,
    latent_dim: usize,
    samples_per_latent: usize,
    stream: Arc<CudaStream>,
}

pub struct VoxCpm2AudioEncoder {
    stem: F32Conv1d,
    blocks: Vec<EncoderBlock>,
    mean: F32Conv1d,
    hop_length: usize,
    latent_dim: usize,
    stream: Arc<CudaStream>,
}

impl VoxCpm2AudioEncoder {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(
            model_dir.join("audiovae.safetensors").is_file(),
            "native VoxCPM2 requires converted audiovae.safetensors"
        );
        let mut channels = config.audio_vae_config.encoder_dim;
        let stem = load_conv_f32(model_dir, "encoder.block.0", 1, channels, 7, 1, 1, stream)?;
        let mut blocks = Vec::with_capacity(config.audio_vae_config.encoder_rates.len());
        for (block_index, &stride) in config.audio_vae_config.encoder_rates.iter().enumerate() {
            let prefix = format!("encoder.block.{}.block", block_index + 1);
            let mut residuals = Vec::with_capacity(3);
            for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
                let unit_prefix = format!("{prefix}.{unit}.block");
                residuals.push(F32ResidualUnit {
                    first_alpha: load_vector_f32(
                        model_dir,
                        &format!("{unit_prefix}.0.alpha"),
                        channels,
                        stream,
                    )?,
                    first_conv: load_conv_f32(
                        model_dir,
                        &format!("{unit_prefix}.1"),
                        channels,
                        channels,
                        7,
                        dilation,
                        channels,
                        stream,
                    )?,
                    second_alpha: load_vector_f32(
                        model_dir,
                        &format!("{unit_prefix}.2.alpha"),
                        channels,
                        stream,
                    )?,
                    second_conv: load_conv_f32(
                        model_dir,
                        &format!("{unit_prefix}.3"),
                        channels,
                        channels,
                        1,
                        1,
                        1,
                        stream,
                    )?,
                });
            }
            let alpha = load_vector_f32(model_dir, &format!("{prefix}.3.alpha"), channels, stream)?;
            let output_channels = channels * 2;
            let downsample = load_conv_f32(
                model_dir,
                &format!("{prefix}.4"),
                channels,
                output_channels,
                stride * 2,
                1,
                1,
                stream,
            )?;
            blocks.push(EncoderBlock {
                residuals,
                alpha,
                downsample,
                stride,
            });
            channels = output_channels;
        }
        let mean = load_conv_f32(
            model_dir,
            "encoder.fc_mu",
            channels,
            config.audio_vae_config.latent_dim,
            3,
            1,
            1,
            stream,
        )?;
        Ok(Self {
            stem,
            blocks,
            mean,
            hop_length: config.audio_vae_config.encoder_rates.iter().product(),
            latent_dim: config.audio_vae_config.latent_dim,
            stream: Arc::clone(stream),
        })
    }

    /// Encode 16-kHz mono PCM to frame-major 64-dimensional AudioVAE latents.
    pub fn encode(&self, audio: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        ensure!(!audio.is_empty(), "VoxCPM2 reference audio is empty");
        ensure!(
            audio.iter().all(|value| value.is_finite()),
            "VoxCPM2 reference audio contains non-finite samples"
        );
        let padded_len = audio.len().div_ceil(self.hop_length) * self.hop_length;
        let mut host = audio.to_vec();
        host.resize(padded_len, 0.0);
        let mut frames = padded_len;
        let input = self.stream.clone_htod(&host)?;
        let mut hidden = conv_forward_f32(&self.stream, &input, frames, &self.stem, kernels)?;
        for block in &self.blocks {
            for residual in &block.residuals {
                hidden = residual_forward_f32(&self.stream, hidden, frames, residual, kernels)?;
            }
            let mut activated = self.stream.alloc_zeros::<f32>(hidden.len())?;
            kernels.ops.snake_f32(
                &hidden,
                &block.alpha,
                &mut activated,
                block.downsample.input_channels as u32,
                frames as u32,
            )?;
            let output_frames = frames.div_ceil(block.stride);
            let mut downsampled = self
                .stream
                .alloc_zeros::<f32>(block.downsample.output_channels * output_frames)?;
            kernels.ops.conv1d_causal_stride_f32(
                &activated,
                &block.downsample.weight,
                &block.downsample.bias,
                &mut downsampled,
                block.downsample.input_channels as u32,
                block.downsample.output_channels as u32,
                frames as u32,
                output_frames as u32,
                block.downsample.kernel as u32,
                block.stride as u32,
                block.stride as u32,
                block.downsample.groups as u32,
            )?;
            hidden = downsampled;
            frames = output_frames;
        }
        hidden = conv_forward_f32(&self.stream, &hidden, frames, &self.mean, kernels)?;
        self.stream.synchronize()?;
        let mut channel_first = vec![0.0f32; self.latent_dim * frames];
        self.stream.memcpy_dtoh(&hidden, &mut channel_first)?;
        let mut frame_major = vec![0.0f32; channel_first.len()];
        for frame in 0..frames {
            for channel in 0..self.latent_dim {
                frame_major[frame * self.latent_dim + channel] =
                    channel_first[channel * frames + frame];
            }
        }
        Ok(frame_major)
    }
}

impl VoxCpm2AudioDecoder {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let vae_path = model_dir.join("audiovae.safetensors");
        ensure!(
            vae_path.is_file(),
            "native VoxCPM2 requires converted audiovae.safetensors"
        );
        let latent = config.audio_vae_config.latent_dim;
        let decoder_dim = config.audio_vae_config.decoder_dim;
        let depthwise_stem = load_conv(
            model_dir,
            "decoder.model.0",
            latent,
            latent,
            7,
            1,
            latent,
            stream,
        )?;
        let expand = load_conv(
            model_dir,
            "decoder.model.1",
            latent,
            decoder_dim,
            1,
            1,
            1,
            stream,
        )?;
        let mut stages = Vec::with_capacity(config.audio_vae_config.decoder_rates.len());
        for (stage_index, &stride) in config.audio_vae_config.decoder_rates.iter().enumerate() {
            let model_index = stage_index + 2;
            let input_channels = decoder_dim >> stage_index;
            let output_channels = input_channels / 2;
            let prefix = format!("decoder.model.{model_index}.block");
            let alpha = load_vector(
                model_dir,
                &format!("{prefix}.0.alpha"),
                input_channels,
                stream,
            )?;
            let upsample = load_transpose(
                model_dir,
                &format!("{prefix}.1"),
                input_channels,
                output_channels,
                stride * 2,
                stride,
                stream,
            )?;
            let conditioning = format!("decoder.sr_cond_model.{model_index}");
            let scale = load_condition(
                model_dir,
                &conditioning,
                "scale_embed",
                input_channels,
                stream,
            )?;
            let bias = load_condition(
                model_dir,
                &conditioning,
                "bias_embed",
                input_channels,
                stream,
            )?;
            let mut residuals = Vec::with_capacity(3);
            for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
                let unit_prefix = format!("{prefix}.{}.block", unit + 2);
                residuals.push(ResidualUnit {
                    first_alpha: load_vector(
                        model_dir,
                        &format!("{unit_prefix}.0.alpha"),
                        output_channels,
                        stream,
                    )?,
                    first_conv: load_conv(
                        model_dir,
                        &format!("{unit_prefix}.1"),
                        output_channels,
                        output_channels,
                        7,
                        dilation,
                        output_channels,
                        stream,
                    )?,
                    second_alpha: load_vector(
                        model_dir,
                        &format!("{unit_prefix}.2.alpha"),
                        output_channels,
                        stream,
                    )?,
                    second_conv: load_conv(
                        model_dir,
                        &format!("{unit_prefix}.3"),
                        output_channels,
                        output_channels,
                        1,
                        1,
                        1,
                        stream,
                    )?,
                });
            }
            stages.push(DecoderStage {
                scale,
                bias,
                alpha,
                upsample,
                residuals,
            });
        }
        let final_channels = decoder_dim >> config.audio_vae_config.decoder_rates.len();
        let final_alpha = load_vector(model_dir, "decoder.model.8.alpha", final_channels, stream)?;
        let head = load_conv(
            model_dir,
            "decoder.model.9",
            final_channels,
            1,
            7,
            1,
            1,
            stream,
        )?;
        let samples_per_latent = config.audio_vae_config.decoder_rates.iter().product();
        Ok(Self {
            depthwise_stem,
            expand,
            stages,
            final_alpha,
            head,
            latent_dim: latent,
            samples_per_latent,
            stream: Arc::clone(stream),
        })
    }

    /// Decode frame-major 64-dimensional latents to 48-kHz mono PCM.
    pub fn decode(&self, latents: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        ensure!(
            !latents.is_empty() && latents.len().is_multiple_of(self.latent_dim),
            "VoxCPM2 decoder needs complete latent frames"
        );
        ensure!(
            latents.iter().all(|value| value.is_finite()),
            "VoxCPM2 decoder received non-finite latents"
        );
        let mut frames = latents.len() / self.latent_dim;
        let host = latents
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let frame_major = self.stream.clone_htod(&host)?;
        let mut hidden = self.stream.alloc_zeros::<f16>(host.len())?;
        kernels.ops.transpose_f16(
            &frame_major,
            &mut hidden,
            frames as u32,
            self.latent_dim as u32,
        )?;
        hidden = self.conv(&hidden, frames, &self.depthwise_stem, kernels)?;
        hidden = self.conv(&hidden, frames, &self.expand, kernels)?;
        for stage in &self.stages {
            let mut conditioned = self.stream.alloc_zeros::<f16>(hidden.len())?;
            kernels.ops.channel_affine_f16(
                &hidden,
                &stage.scale,
                &stage.bias,
                &mut conditioned,
                stage.upsample.input_channels as u32,
                frames as u32,
            )?;
            let mut activated = self.stream.alloc_zeros::<f16>(conditioned.len())?;
            kernels.ops.snake_f16(
                &conditioned,
                &stage.alpha,
                &mut activated,
                stage.upsample.input_channels as u32,
                frames as u32,
            )?;
            hidden = self.conv_transpose(&activated, frames, &stage.upsample, kernels)?;
            frames *= stage.upsample.stride;
            for residual in &stage.residuals {
                hidden = self.residual(hidden, frames, residual, kernels)?;
            }
        }
        let mut activated = self.stream.alloc_zeros::<f16>(hidden.len())?;
        kernels.ops.snake_f16(
            &hidden,
            &self.final_alpha,
            &mut activated,
            self.head.input_channels as u32,
            frames as u32,
        )?;
        let mut waveform = self.conv(&activated, frames, &self.head, kernels)?;
        kernels.ops.tanh_f16_inplace(&mut waveform)?;
        ensure!(
            frames == latents.len() / self.latent_dim * self.samples_per_latent,
            "VoxCPM2 decoder output geometry disagrees"
        );
        self.stream.synchronize()?;
        let mut host = vec![f16::ZERO; frames];
        self.stream.memcpy_dtoh(&waveform, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    fn residual(
        &self,
        input: CudaSlice<f16>,
        frames: usize,
        unit: &ResidualUnit,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let channels = unit.first_conv.input_channels;
        let mut activated = self.stream.alloc_zeros::<f16>(input.len())?;
        kernels.ops.snake_f16(
            &input,
            &unit.first_alpha,
            &mut activated,
            channels as u32,
            frames as u32,
        )?;
        let hidden = self.conv(&activated, frames, &unit.first_conv, kernels)?;
        kernels.ops.snake_f16(
            &hidden,
            &unit.second_alpha,
            &mut activated,
            channels as u32,
            frames as u32,
        )?;
        let hidden = self.conv(&activated, frames, &unit.second_conv, kernels)?;
        let mut output = self.stream.alloc_zeros::<f16>(input.len())?;
        kernels
            .ops
            .add_f16(&input, &hidden, &mut output, input.len() as u32)?;
        Ok(output)
    }

    fn conv(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        conv: &Conv1d,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mut output = self
            .stream
            .alloc_zeros::<f16>(conv.output_channels * frames)?;
        if conv.groups == 1 && conv.output_channels >= 32 {
            conv1d_causal_gemm(
                input,
                &conv.weight,
                &conv.bias,
                &mut output,
                conv.input_channels as u32,
                conv.output_channels as u32,
                frames as u32,
                conv.kernel as u32,
                conv.dilation as u32,
                kernels,
            )?;
        } else {
            kernels.ops.conv1d_causal_f16(
                input,
                &conv.weight,
                &conv.bias,
                &mut output,
                conv.input_channels as u32,
                conv.output_channels as u32,
                frames as u32,
                conv.kernel as u32,
                conv.dilation as u32,
                conv.groups as u32,
            )?;
        }
        Ok(output)
    }

    fn conv_transpose(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        conv: &ConvTranspose1d,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mut output = self
            .stream
            .alloc_zeros::<f16>(conv.output_channels * frames * conv.stride)?;
        conv_transpose1d_causal_gemm(
            input,
            &conv.phase_weights,
            &conv.bias,
            &mut output,
            conv.input_channels as u32,
            conv.output_channels as u32,
            frames as u32,
            conv.stride as u32,
            kernels,
        )?;
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d_causal_gemm(
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    in_channels: u32,
    out_channels: u32,
    frames: u32,
    kernel_size: u32,
    dilation: u32,
    kernels: &mut GpuKernels,
) -> anyhow::Result<()> {
    let stream = Arc::clone(kernels.ops.stream());
    let width = in_channels * kernel_size;
    let mut unfolded = stream.alloc_zeros::<f16>((frames * width) as usize)?;
    kernels.ops.unfold_causal_f16(
        input,
        &mut unfolded,
        in_channels,
        frames,
        kernel_size,
        dilation,
    )?;
    let mut channel_last = stream.alloc_zeros::<f16>((frames * out_channels) as usize)?;
    kernels.gemm.matmul_f16(
        &unfolded,
        weight,
        &mut channel_last,
        frames,
        out_channels,
        width,
    )?;
    kernels
        .ops
        .add_bias_f16_inplace(&mut channel_last, bias, frames, out_channels)?;
    kernels
        .ops
        .transpose_f16(&channel_last, output, frames, out_channels)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_causal_gemm(
    input: &CudaSlice<f16>,
    phase_weights: &[CudaSlice<f16>],
    bias: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    in_channels: u32,
    out_channels: u32,
    input_len: u32,
    stride: u32,
    kernels: &mut GpuKernels,
) -> anyhow::Result<()> {
    ensure!(
        phase_weights.len() == stride as usize,
        "VoxCPM2 transposed convolution phase count disagrees"
    );
    let stream = Arc::clone(kernels.ops.stream());
    let width = in_channels * 2;
    let mut unfolded = stream.alloc_zeros::<f16>((input_len * width) as usize)?;
    kernels
        .ops
        .unfold_causal_f16(input, &mut unfolded, in_channels, input_len, 2, 1)?;
    let mut phase_output = stream.alloc_zeros::<f16>((input_len * out_channels) as usize)?;
    for (phase, weight) in phase_weights.iter().enumerate() {
        kernels.gemm.matmul_f16(
            &unfolded,
            weight,
            &mut phase_output,
            input_len,
            out_channels,
            width,
        )?;
        kernels.ops.scatter_conv_transpose_phase_f16(
            &phase_output,
            bias,
            output,
            input_len,
            out_channels,
            stride,
            phase as u32,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_conv(
    model_dir: &Path,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv1d> {
    let weight = load_weight_norm(model_dir, prefix, output, output, input / groups, kernel)?;
    let bias = load_bias(model_dir, prefix, output)?;
    Ok(Conv1d {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
        input_channels: input,
        output_channels: output,
        kernel,
        dilation,
        groups,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_conv_f32(
    model_dir: &Path,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<F32Conv1d> {
    let weight = load_weight_norm_f32(model_dir, prefix, output, output, input / groups, kernel)?;
    let bias = load_bias_f32(model_dir, prefix, output)?;
    Ok(F32Conv1d {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
        input_channels: input,
        output_channels: output,
        kernel,
        dilation,
        groups,
    })
}

fn load_transpose(
    model_dir: &Path,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<ConvTranspose1d> {
    let weight = load_weight_norm(model_dir, prefix, input, input, output, kernel)?;
    ensure!(
        kernel == stride * 2,
        "{prefix} requires a two-tap polyphase kernel"
    );
    let phase_weights = (0..stride)
        .map(|phase| {
            let mut packed = vec![f16::ZERO; output * input * 2];
            for output_channel in 0..output {
                for input_channel in 0..input {
                    let source = (input_channel * output + output_channel) * kernel;
                    let destination = (output_channel * input + input_channel) * 2;
                    packed[destination] = weight[source + phase + stride];
                    packed[destination + 1] = weight[source + phase];
                }
            }
            Ok(stream.clone_htod(&packed)?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let bias = load_bias(model_dir, prefix, output)?;
    Ok(ConvTranspose1d {
        phase_weights,
        bias: stream.clone_htod(&bias)?,
        input_channels: input,
        output_channels: output,
        stride,
    })
}

fn load_weight_norm(
    model_dir: &Path,
    prefix: &str,
    norm_channels: usize,
    dim0: usize,
    dim1: usize,
    kernel: usize,
) -> anyhow::Result<Vec<f16>> {
    Ok(
        load_weight_norm_f32(model_dir, prefix, norm_channels, dim0, dim1, kernel)?
            .into_iter()
            .map(f16::from_f32)
            .collect(),
    )
}

fn load_weight_norm_f32(
    model_dir: &Path,
    prefix: &str,
    norm_channels: usize,
    dim0: usize,
    dim1: usize,
    kernel: usize,
) -> anyhow::Result<Vec<f32>> {
    let g_name = format!("{prefix}.weight_g");
    let v_name = format!("{prefix}.weight_v");
    let g =
        load_f32_tensor(model_dir, &g_name).with_context(|| format!("could not load {g_name}"))?;
    let v =
        load_f32_tensor(model_dir, &v_name).with_context(|| format!("could not load {v_name}"))?;
    ensure!(
        g.values.len() == norm_channels && v.shape == [dim0, dim1, kernel],
        "{prefix} weight-normalization geometry disagrees"
    );
    let block = dim1 * kernel;
    let mut output = Vec::with_capacity(v.values.len());
    for (channel, values) in v.values.chunks_exact(block).enumerate() {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let scale = g.values[channel] / norm;
        output.extend(values.iter().map(|value| value * scale));
    }
    Ok(output)
}

fn load_bias(model_dir: &Path, prefix: &str, size: usize) -> anyhow::Result<Vec<f16>> {
    Ok(load_bias_f32(model_dir, prefix, size)?
        .into_iter()
        .map(f16::from_f32)
        .collect())
}

fn load_bias_f32(model_dir: &Path, prefix: &str, size: usize) -> anyhow::Result<Vec<f32>> {
    let name = format!("{prefix}.bias");
    let tensor =
        load_f32_tensor(model_dir, &name).with_context(|| format!("could not load {name}"))?;
    ensure!(tensor.shape == [size], "{name} has unexpected shape");
    Ok(tensor.values)
}

fn load_vector(
    model_dir: &Path,
    name: &str,
    size: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f16>> {
    let tensor =
        load_f32_tensor(model_dir, name).with_context(|| format!("could not load {name}"))?;
    ensure!(tensor.values.len() == size, "{name} has unexpected shape");
    let values = tensor
        .values
        .into_iter()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    Ok(stream.clone_htod(&values)?)
}

fn load_vector_f32(
    model_dir: &Path,
    name: &str,
    size: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f32>> {
    let tensor =
        load_f32_tensor(model_dir, name).with_context(|| format!("could not load {name}"))?;
    ensure!(tensor.values.len() == size, "{name} has unexpected shape");
    Ok(stream.clone_htod(&tensor.values)?)
}

fn load_condition(
    model_dir: &Path,
    prefix: &str,
    kind: &str,
    channels: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f16>> {
    let name = format!("{prefix}.{kind}.weight");
    let tensor =
        load_f32_tensor(model_dir, &name).with_context(|| format!("could not load {name}"))?;
    ensure!(tensor.shape == [4, channels], "{name} has unexpected shape");
    let start = 3 * channels;
    let values = tensor.values[start..start + channels]
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    Ok(stream.clone_htod(&values)?)
}

fn conv_forward_f32(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    frames: usize,
    conv: &F32Conv1d,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f32>> {
    let mut output = stream.alloc_zeros::<f32>(conv.output_channels * frames)?;
    kernels.ops.conv1d_causal_f32(
        input,
        &conv.weight,
        &conv.bias,
        &mut output,
        conv.input_channels as u32,
        conv.output_channels as u32,
        frames as u32,
        conv.kernel as u32,
        conv.dilation as u32,
        conv.groups as u32,
    )?;
    Ok(output)
}

fn residual_forward_f32(
    stream: &Arc<CudaStream>,
    mut input: CudaSlice<f32>,
    frames: usize,
    unit: &F32ResidualUnit,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f32>> {
    let channels = unit.first_conv.input_channels;
    let mut activated = stream.alloc_zeros::<f32>(input.len())?;
    kernels.ops.snake_f32(
        &input,
        &unit.first_alpha,
        &mut activated,
        channels as u32,
        frames as u32,
    )?;
    let hidden = conv_forward_f32(stream, &activated, frames, &unit.first_conv, kernels)?;
    kernels.ops.snake_f32(
        &hidden,
        &unit.second_alpha,
        &mut activated,
        channels as u32,
        frames as u32,
    )?;
    let hidden = conv_forward_f32(stream, &activated, frames, &unit.second_conv, kernels)?;
    let elements = input.len() as u32;
    kernels.ops.add_inplace_f32(&mut input, &hidden, elements)?;
    Ok(input)
}
