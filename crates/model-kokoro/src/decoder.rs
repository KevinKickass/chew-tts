use crate::{KokoroAdaInResBlock, KokoroCheckpoint};
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

struct Conv {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

/// Kokoro decoder up to the iSTFTNet generator input.
pub struct KokoroDecoderFrontend {
    f0_conv: Conv,
    noise_conv: Conv,
    asr_projection: Conv,
    encode: KokoroAdaInResBlock,
    decode: Vec<KokoroAdaInResBlock>,
}

impl KokoroDecoderFrontend {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
        let mut decode = Vec::new();
        for index in 0..3 {
            decode.push(KokoroAdaInResBlock::load(
                &checkpoint,
                "decoder",
                &format!("module.decode.{index}"),
                1090,
                1024,
                false,
                stream,
            )?);
        }
        decode.push(KokoroAdaInResBlock::load(
            &checkpoint,
            "decoder",
            "module.decode.3",
            1090,
            512,
            true,
            stream,
        )?);
        Ok(Self {
            f0_conv: load_weight_norm_conv(
                &checkpoint,
                "module.F0_conv",
                1,
                1,
                3,
                2,
                1,
                true,
                stream,
            )?,
            noise_conv: load_weight_norm_conv(
                &checkpoint,
                "module.N_conv",
                1,
                1,
                3,
                2,
                1,
                true,
                stream,
            )?,
            asr_projection: load_weight_norm_conv(
                &checkpoint,
                "module.asr_res.0",
                512,
                64,
                1,
                1,
                0,
                true,
                stream,
            )?,
            encode: KokoroAdaInResBlock::load(
                &checkpoint,
                "decoder",
                "module.encode",
                514,
                1024,
                false,
                stream,
            )?,
            decode,
        })
    }

    /// Produce frame-major `[2*acoustic_frames, 512]` generator input.
    pub fn decode(
        &self,
        asr: &[f32],
        f0: &[f32],
        noise: &[f32],
        acoustic_frames: usize,
        style: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            asr.len() == acoustic_frames * 512
                && f0.len() == acoustic_frames * 2
                && noise.len() == acoustic_frames * 2
                && style.len() == 128,
            "invalid Kokoro decoder input geometry"
        );
        let f0_down = run_conv_host(f0, acoustic_frames * 2, &self.f0_conv, kernels)?;
        let noise_down = run_conv_host(noise, acoustic_frames * 2, &self.noise_conv, kernels)?;
        ensure!(
            f0_down.len() == acoustic_frames && noise_down.len() == acoustic_frames,
            "invalid Kokoro downsampled prosody"
        );
        let asr_res = run_conv_host(asr, acoustic_frames, &self.asr_projection, kernels)?;
        let mut encoded_input = Vec::with_capacity(acoustic_frames * 514);
        for frame in 0..acoustic_frames {
            encoded_input.extend_from_slice(&asr[frame * 512..(frame + 1) * 512]);
            encoded_input.push(f0_down[frame]);
            encoded_input.push(noise_down[frame]);
        }
        let mut hidden = self
            .encode
            .forward(&encoded_input, acoustic_frames, style, kernels)?;
        for block in &self.decode[..3] {
            let mut input = Vec::with_capacity(acoustic_frames * 1090);
            for frame in 0..acoustic_frames {
                input.extend_from_slice(&hidden[frame * 1024..(frame + 1) * 1024]);
                input.extend_from_slice(&asr_res[frame * 64..(frame + 1) * 64]);
                input.push(f0_down[frame]);
                input.push(noise_down[frame]);
            }
            hidden = block.forward(&input, acoustic_frames, style, kernels)?;
        }
        let mut input = Vec::with_capacity(acoustic_frames * 1090);
        for frame in 0..acoustic_frames {
            input.extend_from_slice(&hidden[frame * 1024..(frame + 1) * 1024]);
            input.extend_from_slice(&asr_res[frame * 64..(frame + 1) * 64]);
            input.push(f0_down[frame]);
            input.push(noise_down[frame]);
        }
        self.decode[3].forward(&input, acoustic_frames, style, kernels)
    }
}

fn run_conv_host(
    input: &[f32],
    frames: usize,
    conv: &Conv,
    kernels: &mut GpuKernels,
) -> anyhow::Result<Vec<f32>> {
    ensure!(
        input.len() == frames * conv.input,
        "invalid Kokoro convolution input"
    );
    let output_frames = (frames + 2 * conv.padding - (conv.kernel - 1) - 1) / conv.stride + 1;
    let stream = Arc::clone(kernels.ops.stream());
    let rows = stream.clone_htod(&input.iter().copied().map(f16::from_f32).collect::<Vec<_>>())?;
    let mut channels = stream.alloc_zeros::<f16>(input.len())?;
    kernels
        .ops
        .transpose_f16(&rows, &mut channels, frames as u32, conv.input as u32)?;
    let mut output = stream.alloc_zeros::<f16>(output_frames * conv.output)?;
    kernels.ops.conv1d_general_f16(
        &channels,
        &conv.weight,
        &conv.bias,
        &mut output,
        conv.input as u32,
        conv.output as u32,
        frames as u32,
        output_frames as u32,
        conv.kernel as u32,
        conv.stride as u32,
        conv.padding as u32,
        1,
    )?;
    let mut output_rows = stream.alloc_zeros::<f16>(output.len())?;
    kernels.ops.transpose_f16(
        &output,
        &mut output_rows,
        conv.output as u32,
        output_frames as u32,
    )?;
    stream.synchronize()?;
    let mut host = vec![f16::ZERO; output_rows.len()];
    stream.memcpy_dtoh(&output_rows, &mut host)?;
    Ok(host.into_iter().map(f16::to_f32).collect())
}

#[allow(clippy::too_many_arguments)]
fn load_weight_norm_conv(
    checkpoint: &KokoroCheckpoint,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    bias: bool,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Conv> {
    let (g_shape, g) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_g"))?;
    let (v_shape, v) = checkpoint.tensor_f32("decoder", &format!("{prefix}.weight_v"))?;
    ensure!(
        g_shape == [output, 1, 1] && v_shape == [output, input, kernel],
        "invalid Kokoro decoder convolution {prefix}"
    );
    let bias_values = if bias {
        let (shape, values) = checkpoint.tensor_f16("decoder", &format!("{prefix}.bias"))?;
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
        stride,
        padding,
    })
}
