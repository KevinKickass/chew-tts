use crate::load_f16_tensor;
use anyhow::{Context, ensure};
use chew_kernel::{GpuKernels, SpeakerKernels};
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

struct Conv1d {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
}

struct SeRes2Block {
    tdnn1: Conv1d,
    res2: Vec<Conv1d>,
    tdnn2: Conv1d,
    se1: Conv1d,
    se2: Conv1d,
}

pub struct SpeakerEncoder {
    initial: Conv1d,
    blocks: Vec<SeRes2Block>,
    mfa: Conv1d,
    asp_tdnn: Conv1d,
    asp_conv: Conv1d,
    fc: Conv1d,
    speaker_kernels: SpeakerKernels,
    stream: Arc<CudaStream>,
}

impl SpeakerEncoder {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let initial = load_conv(model_dir, "speaker_encoder.blocks.0.conv", 1, stream)?;
        let mut blocks = Vec::with_capacity(3);
        for block_index in 1..=3 {
            let prefix = format!("speaker_encoder.blocks.{block_index}");
            let mut res2 = Vec::with_capacity(7);
            for res2_index in 0..7 {
                res2.push(load_conv(
                    model_dir,
                    &format!("{prefix}.res2net_block.blocks.{res2_index}.conv"),
                    block_index + 1,
                    stream,
                )?);
            }
            blocks.push(SeRes2Block {
                tdnn1: load_conv(model_dir, &format!("{prefix}.tdnn1.conv"), 1, stream)?,
                res2,
                tdnn2: load_conv(model_dir, &format!("{prefix}.tdnn2.conv"), 1, stream)?,
                se1: load_conv(model_dir, &format!("{prefix}.se_block.conv1"), 1, stream)?,
                se2: load_conv(model_dir, &format!("{prefix}.se_block.conv2"), 1, stream)?,
            });
        }
        Ok(Self {
            initial,
            blocks,
            mfa: load_conv(model_dir, "speaker_encoder.mfa.conv", 1, stream)?,
            asp_tdnn: load_conv(model_dir, "speaker_encoder.asp.tdnn.conv", 1, stream)?,
            asp_conv: load_conv(model_dir, "speaker_encoder.asp.conv", 1, stream)?,
            fc: load_conv(model_dir, "speaker_encoder.fc", 1, stream)?,
            speaker_kernels: SpeakerKernels::load(stream)?,
            stream: Arc::clone(stream),
        })
    }

    /// Extract the 2048-value Qwen Base speaker embedding from a channel-last
    /// 128-bin log-mel spectrogram.
    pub fn encode_mel(
        &self,
        mel: &[f32],
        frames: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(frames > 0, "speaker mel has no frames");
        ensure!(
            mel.len() == frames * 128,
            "speaker mel has {} values, expected {}",
            mel.len(),
            frames * 128
        );
        let mut channel_first = vec![f16::ZERO; mel.len()];
        for frame in 0..frames {
            for channel in 0..128 {
                channel_first[channel * frames + frame] = f16::from_f32(mel[frame * 128 + channel]);
            }
        }
        let input = self.stream.clone_htod(&channel_first)?;
        let initial = self.conv_relu(&input, frames, &self.initial, kernels)?;
        let mut hidden = initial;
        let mut aggregated = self.stream.alloc_zeros::<f16>(1536 * frames)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let output = self.se_res2(&hidden, frames, block, kernels)?;
            self.speaker_kernels.append_channel_block(
                &output,
                &mut aggregated,
                512,
                1536,
                (index * 512) as u32,
                frames as u32,
            );
            hidden = output;
        }
        let hidden = self.conv_relu(&aggregated, frames, &self.mfa, kernels)?;

        let mut mean = self.stream.alloc_zeros::<f16>(1536)?;
        let mut stddev = self.stream.alloc_zeros::<f16>(1536)?;
        self.speaker_kernels.channel_stats(
            &hidden,
            None,
            &mut mean,
            &mut stddev,
            1536,
            frames as u32,
        );
        let mut context = self.stream.alloc_zeros::<f16>(4608 * frames)?;
        self.speaker_kernels.append_context(
            &hidden,
            &mean,
            &stddev,
            &mut context,
            1536,
            frames as u32,
        );
        let mut attention = self.conv_relu(&context, frames, &self.asp_tdnn, kernels)?;
        self.speaker_kernels.tanh(&mut attention);
        let mut attention = self.conv(&attention, frames, &self.asp_conv, kernels)?;
        self.speaker_kernels
            .softmax_channels(&mut attention, 1536, frames as u32);
        self.speaker_kernels.channel_stats(
            &hidden,
            Some(&attention),
            &mut mean,
            &mut stddev,
            1536,
            frames as u32,
        );
        let mut pooled = self.stream.alloc_zeros::<f16>(3072)?;
        self.speaker_kernels
            .append_channel_block(&mean, &mut pooled, 1536, 3072, 0, 1);
        self.speaker_kernels
            .append_channel_block(&stddev, &mut pooled, 1536, 3072, 1536, 1);
        let embedding = self.conv(&pooled, 1, &self.fc, kernels)?;
        self.stream.synchronize()?;
        let mut host = vec![f16::ZERO; 2048];
        self.stream.memcpy_dtoh(&embedding, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    fn se_res2(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        block: &SeRes2Block,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let hidden = self.conv_relu(input, frames, &block.tdnn1, kernels)?;
        let part_len = 64 * frames;
        let mut merged = self.stream.alloc_zeros::<f16>(512 * frames)?;
        let mut previous: Option<CudaSlice<f16>> = None;
        for part in 0..8 {
            let mut current = self.stream.alloc_zeros::<f16>(part_len)?;
            self.stream.memcpy_dtod(
                &hidden.slice(part * part_len..(part + 1) * part_len),
                &mut current,
            )?;
            let output = if part == 0 {
                current
            } else {
                let block_input = if let Some(previous) = previous.as_ref().filter(|_| part > 1) {
                    let mut sum = self.stream.alloc_zeros::<f16>(part_len)?;
                    kernels
                        .ops
                        .add_f16(&current, previous, &mut sum, part_len as u32)?;
                    sum
                } else {
                    current
                };
                self.conv_relu(&block_input, frames, &block.res2[part - 1], kernels)?
            };
            self.speaker_kernels.append_channel_block(
                &output,
                &mut merged,
                64,
                512,
                (part * 64) as u32,
                frames as u32,
            );
            previous = Some(output);
        }
        let hidden = self.conv_relu(&merged, frames, &block.tdnn2, kernels)?;
        let mut mean = self.stream.alloc_zeros::<f16>(512)?;
        self.speaker_kernels
            .channel_mean(&hidden, &mut mean, 512, frames as u32);
        let mut scale = self.conv_relu(&mean, 1, &block.se1, kernels)?;
        scale = self.conv(&scale, 1, &block.se2, kernels)?;
        self.speaker_kernels.sigmoid(&mut scale);
        let mut scaled = self.stream.alloc_zeros::<f16>(512 * frames)?;
        self.speaker_kernels
            .channel_scale(&hidden, &scale, &mut scaled, 512, frames as u32);
        let mut output = self.stream.alloc_zeros::<f16>(512 * frames)?;
        kernels
            .ops
            .add_f16(&scaled, input, &mut output, (512 * frames) as u32)?;
        Ok(output)
    }

    fn conv_relu(
        &self,
        input: &CudaSlice<f16>,
        frames: usize,
        conv: &Conv1d,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mut output = self.conv(input, frames, conv, kernels)?;
        self.speaker_kernels.relu(&mut output);
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
            input.len() == conv.in_channels * frames,
            "speaker conv input has {} values, expected {}",
            input.len(),
            conv.in_channels * frames
        );
        let width = conv.in_channels * conv.kernel_size;
        let mut unfolded = self.stream.alloc_zeros::<f16>(frames * width)?;
        self.speaker_kernels.unfold_reflect(
            input,
            &mut unfolded,
            conv.in_channels as u32,
            frames as u32,
            conv.kernel_size as u32,
            conv.dilation as u32,
        );
        let mut channel_last = self.stream.alloc_zeros::<f16>(frames * conv.out_channels)?;
        kernels.gemm.matmul_f16(
            &unfolded,
            &conv.weight,
            &mut channel_last,
            frames as u32,
            conv.out_channels as u32,
            width as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut channel_last,
            &conv.bias,
            frames as u32,
            conv.out_channels as u32,
        )?;
        let mut channel_first = self.stream.alloc_zeros::<f16>(frames * conv.out_channels)?;
        kernels.ops.transpose_f16(
            &channel_last,
            &mut channel_first,
            frames as u32,
            conv.out_channels as u32,
        )?;
        Ok(channel_first)
    }
}

fn load_conv(
    model_dir: &Path,
    prefix: &str,
    dilation: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv1d> {
    let weight = load_f16_tensor(model_dir, &format!("{prefix}.weight"))?;
    ensure!(
        weight.shape.len() == 3 && weight.shape[2] > 0,
        "{prefix}.weight has invalid shape {:?}",
        weight.shape
    );
    let bias = load_f16_tensor(model_dir, &format!("{prefix}.bias"))?;
    ensure!(
        bias.shape == [weight.shape[0]],
        "{prefix}.bias has shape {:?}",
        bias.shape
    );
    Ok(Conv1d {
        in_channels: weight.shape[1],
        out_channels: weight.shape[0],
        kernel_size: weight.shape[2],
        dilation,
        weight: stream
            .clone_htod(&weight.values)
            .with_context(|| format!("could not upload {prefix}.weight"))?,
        bias: stream
            .clone_htod(&bias.values)
            .with_context(|| format!("could not upload {prefix}.bias"))?,
    })
}
