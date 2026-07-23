use crate::{KokoroAdaInResBlock, KokoroBiLstm, KokoroCheckpoint};
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

struct Projection {
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
}

pub struct KokoroF0Noise {
    shared: KokoroBiLstm,
    f0_blocks: Vec<KokoroAdaInResBlock>,
    noise_blocks: Vec<KokoroAdaInResBlock>,
    f0_projection: Projection,
    noise_projection: Projection,
}

impl KokoroF0Noise {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
        let load_branch = |name: &str| -> anyhow::Result<Vec<KokoroAdaInResBlock>> {
            Ok(vec![
                KokoroAdaInResBlock::load(
                    &checkpoint,
                    "predictor",
                    &format!("module.{name}.0"),
                    512,
                    512,
                    false,
                    stream,
                )?,
                KokoroAdaInResBlock::load(
                    &checkpoint,
                    "predictor",
                    &format!("module.{name}.1"),
                    512,
                    256,
                    true,
                    stream,
                )?,
                KokoroAdaInResBlock::load(
                    &checkpoint,
                    "predictor",
                    &format!("module.{name}.2"),
                    256,
                    256,
                    false,
                    stream,
                )?,
            ])
        };
        Ok(Self {
            shared: KokoroBiLstm::load(
                &checkpoint,
                "predictor",
                "module.shared",
                640,
                256,
                stream,
            )?,
            f0_blocks: load_branch("F0")?,
            noise_blocks: load_branch("N")?,
            f0_projection: load_projection(&checkpoint, "module.F0_proj", 256, stream)?,
            noise_projection: load_projection(&checkpoint, "module.N_proj", 256, stream)?,
        })
    }

    pub fn predict(
        &self,
        aligned: &[f32],
        frames: usize,
        style: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            aligned.len() == frames * 640 && style.len() == 128,
            "invalid Kokoro F0/noise input"
        );
        let shared = self.shared.forward(aligned, frames, kernels)?;
        let f0 = self.run_branch(
            &shared,
            frames,
            style,
            &self.f0_blocks,
            &self.f0_projection,
            kernels,
        )?;
        let noise = self.run_branch(
            &shared,
            frames,
            style,
            &self.noise_blocks,
            &self.noise_projection,
            kernels,
        )?;
        Ok((f0, noise))
    }

    fn run_branch(
        &self,
        shared: &[f32],
        frames: usize,
        style: &[f32],
        blocks: &[KokoroAdaInResBlock],
        projection: &Projection,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let mut hidden = blocks[0].forward(shared, frames, style, kernels)?;
        hidden = blocks[1].forward(&hidden, frames, style, kernels)?;
        hidden = blocks[2].forward(&hidden, frames * 2, style, kernels)?;
        let output_frames = frames * 2;
        let stream = Arc::clone(kernels.ops.stream());
        let rows = stream.clone_htod(
            &hidden
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut channels = stream.alloc_zeros::<f16>(hidden.len())?;
        kernels
            .ops
            .transpose_f16(&rows, &mut channels, output_frames as u32, 256)?;
        let mut output = stream.alloc_zeros::<f16>(output_frames)?;
        kernels.ops.conv1d_general_f16(
            &channels,
            &projection.weight,
            &projection.bias,
            &mut output,
            256,
            1,
            output_frames as u32,
            output_frames as u32,
            1,
            1,
            0,
            1,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; output_frames];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }
}

fn load_projection(
    checkpoint: &KokoroCheckpoint,
    prefix: &str,
    input: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<Projection> {
    let (weight_shape, weight) = checkpoint.tensor_f16("predictor", &format!("{prefix}.weight"))?;
    let (bias_shape, bias) = checkpoint.tensor_f16("predictor", &format!("{prefix}.bias"))?;
    ensure!(
        weight_shape == [1, input, 1] && bias_shape == [1],
        "invalid Kokoro projection {prefix}"
    );
    Ok(Projection {
        weight: stream.clone_htod(&weight)?,
        bias: stream.clone_htod(&bias)?,
    })
}
