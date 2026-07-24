use crate::VibeVoiceConfig;
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, QwenDType, load_bf16_tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

struct DiffusionLayer {
    norm: CudaSlice<bf16>,
    modulation: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
}

/// Native VibeVoice velocity-prediction head. The initial correctness path
/// evaluates one CFG branch at a time; batching both branches is a later
/// launch-count optimization and does not change the math.
pub struct VibeVoiceDiffusionHead {
    noisy_projection: CudaSlice<bf16>,
    condition_projection: CudaSlice<bf16>,
    timestep_in: CudaSlice<bf16>,
    timestep_out: CudaSlice<bf16>,
    layers: Vec<DiffusionLayer>,
    final_modulation: CudaSlice<bf16>,
    final_projection: CudaSlice<bf16>,
    final_norm_ones: CudaSlice<bf16>,
    hidden: usize,
    intermediate: usize,
    latent: usize,
    norm_eps: f32,
}

impl VibeVoiceDiffusionHead {
    pub fn load(
        model_dir: &Path,
        config: &VibeVoiceConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let cfg = &config.diffusion_head_config;
        let hidden = cfg.hidden_size;
        let intermediate = (hidden as f64 * cfg.head_ffn_ratio) as usize;
        let latent = cfg.latent_size;
        let prefix = "model.prediction_head";
        let load = |name: &str, expected: &[usize]| {
            load_weight(model_dir, &format!("{prefix}.{name}"), expected, stream)
        };
        let mut layers = Vec::with_capacity(cfg.head_layers);
        for layer in 0..cfg.head_layers {
            let prefix = format!("layers.{layer}");
            layers.push(DiffusionLayer {
                norm: load(&format!("{prefix}.norm.weight"), &[hidden])?,
                modulation: load(
                    &format!("{prefix}.adaLN_modulation.1.weight"),
                    &[hidden * 3, hidden],
                )?,
                gate: load(
                    &format!("{prefix}.ffn.gate_proj.weight"),
                    &[intermediate, hidden],
                )?,
                up: load(
                    &format!("{prefix}.ffn.up_proj.weight"),
                    &[intermediate, hidden],
                )?,
                down: load(
                    &format!("{prefix}.ffn.down_proj.weight"),
                    &[hidden, intermediate],
                )?,
            });
        }
        Ok(Self {
            noisy_projection: load("noisy_images_proj.weight", &[hidden, latent])?,
            condition_projection: load("cond_proj.weight", &[hidden, hidden])?,
            timestep_in: load("t_embedder.mlp.0.weight", &[hidden, 256])?,
            timestep_out: load("t_embedder.mlp.2.weight", &[hidden, hidden])?,
            layers,
            final_modulation: load(
                "final_layer.adaLN_modulation.1.weight",
                &[hidden * 2, hidden],
            )?,
            final_projection: load("final_layer.linear.weight", &[latent, hidden])?,
            final_norm_ones: stream.clone_htod(&vec![bf16::ONE; hidden])?,
            hidden,
            intermediate,
            latent,
            norm_eps: cfg.rms_norm_eps as f32,
        })
    }

    pub fn forward(
        &self,
        noisy_latent: &[f32],
        timestep: f32,
        condition: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            noisy_latent.len() == self.latent,
            "VibeVoice noisy latent has {} values, expected {}",
            noisy_latent.len(),
            self.latent
        );
        ensure!(
            condition.len() == self.hidden,
            "VibeVoice diffusion condition has {} values, expected {}",
            condition.len(),
            self.hidden
        );
        let stream = Arc::clone(kernels.ops.stream());
        let noisy = stream.clone_htod(
            &noisy_latent
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let condition = stream.clone_htod(
            &condition
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut x = stream.alloc_zeros::<bf16>(self.hidden)?;
        let mut projected_condition = stream.alloc_zeros::<bf16>(self.hidden)?;
        let mut timestep_hidden = stream.alloc_zeros::<bf16>(self.hidden)?;
        let mut timestep_activated = stream.alloc_zeros::<bf16>(self.hidden)?;
        let mut c = stream.alloc_zeros::<bf16>(self.hidden)?;
        let timestep_embedding = stream.clone_htod(&timestep_embedding(timestep))?;

        Bf16::gemv(
            kernels,
            &noisy,
            &self.noisy_projection,
            &mut x,
            self.hidden as u32,
            self.latent as u32,
        )?;
        Bf16::gemv(
            kernels,
            &condition,
            &self.condition_projection,
            &mut projected_condition,
            self.hidden as u32,
            self.hidden as u32,
        )?;
        Bf16::gemv(
            kernels,
            &timestep_embedding,
            &self.timestep_in,
            &mut timestep_hidden,
            self.hidden as u32,
            256,
        )?;
        kernels.ops.silu_act_bf16(
            &timestep_hidden,
            &mut timestep_activated,
            self.hidden as u32,
        )?;
        Bf16::gemv(
            kernels,
            &timestep_activated,
            &self.timestep_out,
            &mut timestep_hidden,
            self.hidden as u32,
            self.hidden as u32,
        )?;
        kernels.ops.add_bf16(
            &projected_condition,
            &timestep_hidden,
            &mut c,
            self.hidden as u32,
        )?;

        let mut c_activated = stream.alloc_zeros::<bf16>(self.hidden)?;
        kernels
            .ops
            .silu_act_bf16(&c, &mut c_activated, self.hidden as u32)?;
        let mut x_f32 = stream.alloc_zeros::<f32>(self.hidden)?;
        let mut norm = stream.alloc_zeros::<bf16>(self.hidden)?;
        let mut modulation = stream.alloc_zeros::<bf16>(self.hidden * 3)?;
        let mut modulated = stream.alloc_zeros::<bf16>(self.hidden)?;
        let mut gate = stream.alloc_zeros::<bf16>(self.intermediate)?;
        let mut up = stream.alloc_zeros::<bf16>(self.intermediate)?;
        let mut activated = stream.alloc_zeros::<bf16>(self.intermediate)?;
        let mut delta = stream.alloc_zeros::<bf16>(self.hidden)?;

        for layer in &self.layers {
            kernels
                .ops
                .copy_bf16_to_f32(&x, &mut x_f32, self.hidden as u32)?;
            kernels.ops.rms_norm_f32in_bf16(
                &x_f32,
                &layer.norm,
                &mut norm,
                1,
                self.hidden as u32,
                self.norm_eps,
            )?;
            Bf16::gemv(
                kernels,
                &c_activated,
                &layer.modulation,
                &mut modulation,
                (self.hidden * 3) as u32,
                self.hidden as u32,
            )?;
            kernels.ops.modulate_bf16_views(
                &norm,
                &modulation.slice(..self.hidden),
                &modulation.slice(self.hidden..self.hidden * 2),
                &mut modulated,
                self.hidden as u32,
            )?;
            Bf16::gemv(
                kernels,
                &modulated,
                &layer.gate,
                &mut gate,
                self.intermediate as u32,
                self.hidden as u32,
            )?;
            Bf16::gemv(
                kernels,
                &modulated,
                &layer.up,
                &mut up,
                self.intermediate as u32,
                self.hidden as u32,
            )?;
            kernels
                .ops
                .silu_bf16(&gate, &up, &mut activated, self.intermediate as u32)?;
            Bf16::gemv(
                kernels,
                &activated,
                &layer.down,
                &mut delta,
                self.hidden as u32,
                self.intermediate as u32,
            )?;
            kernels.ops.gated_residual_bf16_view(
                &mut x,
                &modulation.slice(self.hidden * 2..),
                &delta,
                self.hidden as u32,
            )?;
        }

        kernels
            .ops
            .copy_bf16_to_f32(&x, &mut x_f32, self.hidden as u32)?;
        kernels.ops.rms_norm_f32in_bf16(
            &x_f32,
            &self.final_norm_ones,
            &mut norm,
            1,
            self.hidden as u32,
            self.norm_eps,
        )?;
        let mut final_modulation = stream.alloc_zeros::<bf16>(self.hidden * 2)?;
        Bf16::gemv(
            kernels,
            &c_activated,
            &self.final_modulation,
            &mut final_modulation,
            (self.hidden * 2) as u32,
            self.hidden as u32,
        )?;
        kernels.ops.modulate_bf16_views(
            &norm,
            &final_modulation.slice(..self.hidden),
            &final_modulation.slice(self.hidden..),
            &mut modulated,
            self.hidden as u32,
        )?;
        let mut output = stream.alloc_zeros::<bf16>(self.latent)?;
        Bf16::gemv(
            kernels,
            &modulated,
            &self.final_projection,
            &mut output,
            self.latent as u32,
            self.hidden as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![bf16::ZERO; self.latent];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(bf16::to_f32).collect())
    }
}

fn load_weight(
    model_dir: &Path,
    name: &str,
    expected: &[usize],
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<bf16>> {
    let tensor =
        load_bf16_tensor(model_dir, name).with_context(|| format!("could not load {name}"))?;
    ensure!(
        tensor.shape == expected,
        "{name} has shape {:?}, expected {expected:?}",
        tensor.shape
    );
    Ok(stream.clone_htod(&tensor.values)?)
}

fn timestep_embedding(timestep: f32) -> Vec<bf16> {
    let half = 128;
    let mut output = Vec::with_capacity(256);
    for index in 0..half {
        let frequency = (-10_000.0_f32.ln() * index as f32 / half as f32).exp();
        output.push(bf16::from_f32((timestep * frequency).cos()));
    }
    for index in 0..half {
        let frequency = (-10_000.0_f32.ln() * index as f32 / half as f32).exp();
        output.push(bf16::from_f32((timestep * frequency).sin()));
    }
    output
}
