use crate::VibeVoiceConfig;
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::load_f16_tensor;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const PREFIX: &str = "model.acoustic_tokenizer.decoder";
const STAGE_DEPTHS: [usize; 7] = [8, 3, 3, 3, 3, 3, 3];

struct Conv1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    groups: usize,
}

struct ConvTranspose1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
}

struct DecoderBlock {
    norm: CudaSlice<f16>,
    mixer: Conv1d,
    gamma: CudaSlice<f16>,
    ffn_norm: CudaSlice<f16>,
    ffn_up: CudaSlice<f16>,
    ffn_up_bias: CudaSlice<f16>,
    ffn_down: CudaSlice<f16>,
    ffn_down_bias: CudaSlice<f16>,
    ffn_gamma: CudaSlice<f16>,
    channels: usize,
}

struct DecoderStage {
    upsample: Option<ConvTranspose1d>,
    blocks: Vec<DecoderBlock>,
}

/// Native causal acoustic decoder for VibeVoice-Realtime.
///
/// The checkpoint stores the codec in BF16. Convolution kernels currently
/// execute in FP16, while their accumulation and all normalization reductions
/// use FP32. This is the same mixed-precision boundary used by Chew's other
/// audio codecs and keeps the entire decoder resident on the GPU.
pub struct VibeVoiceAcousticDecoder {
    stem: Conv1d,
    stages: Vec<DecoderStage>,
    head: Conv1d,
    ratios: Vec<usize>,
    latent_dim: usize,
    norm_eps: f32,
    stream: Arc<CudaStream>,
}

impl VibeVoiceAcousticDecoder {
    pub fn load(
        model_dir: &Path,
        config: &VibeVoiceConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(
            config.acoustic_tokenizer_config.decoder_ratios.len() == 6,
            "VibeVoice decoder expects six upsampling ratios"
        );
        let stem = load_conv(
            model_dir,
            &format!("{PREFIX}.upsample_layers.0.0.conv.conv"),
            stream,
        )?;
        ensure!(
            stem.input_channels == config.acoustic_vae_dim && stem.output_channels == 2048,
            "unexpected VibeVoice decoder stem geometry"
        );
        let mut stages = Vec::with_capacity(STAGE_DEPTHS.len());
        for (stage, &depth) in STAGE_DEPTHS.iter().enumerate() {
            let channels = 2048 >> stage;
            let upsample = if stage == 0 {
                None
            } else {
                let prefix = format!("{PREFIX}.upsample_layers.{stage}.0.convtr.convtr");
                let conv = load_conv_transpose(
                    model_dir,
                    &prefix,
                    config.acoustic_tokenizer_config.decoder_ratios[stage - 1],
                    stream,
                )?;
                ensure!(
                    conv.input_channels == channels * 2 && conv.output_channels == channels,
                    "{prefix} has unexpected channels"
                );
                Some(conv)
            };
            let mut blocks = Vec::with_capacity(depth);
            for block in 0..depth {
                blocks.push(load_block(model_dir, stage, block, channels, stream)?);
            }
            stages.push(DecoderStage { upsample, blocks });
        }
        let head = load_conv(model_dir, &format!("{PREFIX}.head.conv.conv"), stream)?;
        ensure!(
            head.input_channels == 32 && head.output_channels == 1,
            "unexpected VibeVoice decoder head geometry"
        );
        Ok(Self {
            stem,
            stages,
            head,
            ratios: config.acoustic_tokenizer_config.decoder_ratios.clone(),
            latent_dim: config.acoustic_vae_dim,
            norm_eps: config.acoustic_tokenizer_config.layernorm_eps as f32,
            stream: Arc::clone(stream),
        })
    }

    /// Decode frame-major acoustic latents to 24-kHz mono samples.
    pub fn decode(&self, latents: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        ensure!(
            !latents.is_empty() && latents.len().is_multiple_of(self.latent_dim),
            "VibeVoice decoder needs complete acoustic latent frames"
        );
        ensure!(
            latents.iter().all(|value| value.is_finite()),
            "VibeVoice decoder received a non-finite latent"
        );
        let mut frames = latents.len() / self.latent_dim;
        let host = latents
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let frame_major = self.stream.clone_htod(&host)?;
        let mut channel_first = self.stream.alloc_zeros::<f16>(self.latent_dim * frames)?;
        kernels.ops.transpose_f16(
            &frame_major,
            &mut channel_first,
            frames as u32,
            self.latent_dim as u32,
        )?;
        channel_first = self.conv(&channel_first, frames, &self.stem, kernels)?;
        let mut hidden = self.to_frame_major(&channel_first, frames, 2048, kernels)?;

        for (stage_index, stage) in self.stages.iter().enumerate() {
            if let Some(upsample) = &stage.upsample {
                let channel_first =
                    self.to_channel_first(&hidden, frames, upsample.input_channels, kernels)?;
                let upsampled = self.conv_transpose(&channel_first, frames, upsample, kernels)?;
                frames *= self.ratios[stage_index - 1];
                hidden =
                    self.to_frame_major(&upsampled, frames, upsample.output_channels, kernels)?;
            }
            for block in &stage.blocks {
                hidden = self.block(hidden, frames, block, kernels)?;
            }
        }
        let channel_first = self.to_channel_first(&hidden, frames, 32, kernels)?;
        let waveform = self.conv(&channel_first, frames, &self.head, kernels)?;
        self.stream.synchronize()?;
        let mut host = vec![f16::ZERO; frames];
        self.stream.memcpy_dtoh(&waveform, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    fn block(
        &self,
        hidden: CudaSlice<f16>,
        frames: usize,
        block: &DecoderBlock,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let elements = frames * block.channels;
        let mut norm = self.stream.alloc_zeros::<f16>(elements)?;
        kernels.ops.rms_norm(
            &hidden,
            &block.norm,
            &mut norm,
            frames as u32,
            block.channels as u32,
            self.norm_eps,
        )?;
        let norm_cf = self.to_channel_first(&norm, frames, block.channels, kernels)?;
        let mixed_cf = self.conv(&norm_cf, frames, &block.mixer, kernels)?;
        let mixed = self.to_frame_major(&mixed_cf, frames, block.channels, kernels)?;
        let mut scaled = self.stream.alloc_zeros::<f16>(elements)?;
        kernels.ops.mul_f16_broadcast(
            &mixed,
            &block.gamma,
            &mut scaled,
            elements as u32,
            block.channels as u32,
        )?;
        let mut after_mixer = self.stream.alloc_zeros::<f16>(elements)?;
        kernels
            .ops
            .add_f16(&hidden, &scaled, &mut after_mixer, elements as u32)?;

        kernels.ops.rms_norm(
            &after_mixer,
            &block.ffn_norm,
            &mut norm,
            frames as u32,
            block.channels as u32,
            self.norm_eps,
        )?;
        let expanded = block.channels * 4;
        let mut up = self.stream.alloc_zeros::<f16>(frames * expanded)?;
        kernels.gemm.matmul_f16(
            &norm,
            &block.ffn_up,
            &mut up,
            frames as u32,
            expanded as u32,
            block.channels as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut up,
            &block.ffn_up_bias,
            frames as u32,
            expanded as u32,
        )?;
        let mut activated = self.stream.alloc_zeros::<f16>(up.len())?;
        kernels
            .ops
            .gelu_erf_f16(&up, &mut activated, up.len() as u32)?;
        let mut down = self.stream.alloc_zeros::<f16>(elements)?;
        kernels.gemm.matmul_f16(
            &activated,
            &block.ffn_down,
            &mut down,
            frames as u32,
            block.channels as u32,
            expanded as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut down,
            &block.ffn_down_bias,
            frames as u32,
            block.channels as u32,
        )?;
        kernels.ops.mul_f16_broadcast(
            &down,
            &block.ffn_gamma,
            &mut scaled,
            elements as u32,
            block.channels as u32,
        )?;
        let mut output = self.stream.alloc_zeros::<f16>(elements)?;
        kernels
            .ops
            .add_f16(&after_mixer, &scaled, &mut output, elements as u32)?;
        Ok(output)
    }

    fn conv(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        conv: &Conv1d,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        ensure!(
            input.len() == conv.input_channels * frames,
            "VibeVoice convolution input size mismatch"
        );
        let mut output = self
            .stream
            .alloc_zeros::<f16>(conv.output_channels * frames)?;
        kernels.ops.conv1d_causal_f16(
            input,
            &conv.weight,
            &conv.bias,
            &mut output,
            conv.input_channels as u32,
            conv.output_channels as u32,
            frames as u32,
            conv.kernel as u32,
            1,
            conv.groups as u32,
        )?;
        Ok(output)
    }

    fn conv_transpose(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        conv: &ConvTranspose1d,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        ensure!(
            input.len() == conv.input_channels * frames,
            "VibeVoice transposed convolution input size mismatch"
        );
        let mut output = self
            .stream
            .alloc_zeros::<f16>(conv.output_channels * frames * conv.stride)?;
        kernels.ops.conv_transpose1d_causal_f16(
            input,
            &conv.weight,
            &conv.bias,
            &mut output,
            conv.input_channels as u32,
            conv.output_channels as u32,
            frames as u32,
            conv.kernel as u32,
            conv.stride as u32,
        )?;
        Ok(output)
    }

    fn to_frame_major(
        &self,
        channel_first: &CudaSlice<f16>,
        frames: usize,
        channels: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mut output = self.stream.alloc_zeros::<f16>(frames * channels)?;
        kernels
            .ops
            .transpose_f16(channel_first, &mut output, channels as u32, frames as u32)?;
        Ok(output)
    }

    fn to_channel_first(
        &self,
        frame_major: &CudaSlice<f16>,
        frames: usize,
        channels: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mut output = self.stream.alloc_zeros::<f16>(frames * channels)?;
        kernels
            .ops
            .transpose_f16(frame_major, &mut output, frames as u32, channels as u32)?;
        Ok(output)
    }
}

fn load_block(
    model_dir: &Path,
    stage: usize,
    block: usize,
    channels: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<DecoderBlock> {
    let prefix = format!("{PREFIX}.stages.{stage}.{block}");
    let linear1 = load_f16_tensor(model_dir, &format!("{prefix}.ffn.linear1.weight"))?;
    let linear2 = load_f16_tensor(model_dir, &format!("{prefix}.ffn.linear2.weight"))?;
    ensure!(
        linear1.shape == [channels * 4, channels] && linear2.shape == [channels, channels * 4],
        "{prefix} FFN geometry is invalid"
    );
    Ok(DecoderBlock {
        norm: load_vector(
            model_dir,
            &format!("{prefix}.norm.weight"),
            channels,
            stream,
        )?,
        mixer: load_conv(model_dir, &format!("{prefix}.mixer.conv.conv.conv"), stream)?,
        gamma: load_vector(model_dir, &format!("{prefix}.gamma"), channels, stream)?,
        ffn_norm: load_vector(
            model_dir,
            &format!("{prefix}.ffn_norm.weight"),
            channels,
            stream,
        )?,
        ffn_up: stream.clone_htod(&linear1.values)?,
        ffn_up_bias: load_vector(
            model_dir,
            &format!("{prefix}.ffn.linear1.bias"),
            channels * 4,
            stream,
        )?,
        ffn_down: stream.clone_htod(&linear2.values)?,
        ffn_down_bias: load_vector(
            model_dir,
            &format!("{prefix}.ffn.linear2.bias"),
            channels,
            stream,
        )?,
        ffn_gamma: load_vector(model_dir, &format!("{prefix}.ffn_gamma"), channels, stream)?,
        channels,
    })
}

fn load_conv(model_dir: &Path, prefix: &str, stream: &Arc<CudaStream>) -> anyhow::Result<Conv1d> {
    let weight = load_f16_tensor(model_dir, &format!("{prefix}.weight"))
        .with_context(|| format!("could not load {prefix}.weight"))?;
    ensure!(weight.shape.len() == 3, "{prefix}.weight is not Conv1d");
    let bias = load_vector(
        model_dir,
        &format!("{prefix}.bias"),
        weight.shape[0],
        stream,
    )?;
    let groups = if weight.shape[1] == 1 && weight.shape[0] > 1 {
        weight.shape[0]
    } else {
        1
    };
    Ok(Conv1d {
        input_channels: weight.shape[1] * groups,
        output_channels: weight.shape[0],
        kernel: weight.shape[2],
        groups,
        weight: stream.clone_htod(&weight.values)?,
        bias,
    })
}

fn load_conv_transpose(
    model_dir: &Path,
    prefix: &str,
    stride: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<ConvTranspose1d> {
    let weight = load_f16_tensor(model_dir, &format!("{prefix}.weight"))
        .with_context(|| format!("could not load {prefix}.weight"))?;
    ensure!(
        weight.shape.len() == 3,
        "{prefix}.weight is not ConvTranspose1d"
    );
    let bias = load_vector(
        model_dir,
        &format!("{prefix}.bias"),
        weight.shape[1],
        stream,
    )?;
    ensure!(
        weight.shape[2] == stride * 2,
        "{prefix}.weight has unexpected kernel"
    );
    Ok(ConvTranspose1d {
        input_channels: weight.shape[0],
        output_channels: weight.shape[1],
        kernel: weight.shape[2],
        stride,
        weight: stream.clone_htod(&weight.values)?,
        bias,
    })
}

fn load_vector(
    model_dir: &Path,
    name: &str,
    length: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<f16>> {
    let tensor = load_f16_tensor(model_dir, name)?;
    ensure!(
        tensor.shape == [length],
        "{name} has shape {:?}",
        tensor.shape
    );
    Ok(stream.clone_htod(&tensor.values)?)
}
