use crate::projections::Bf16Linear;
use crate::{VoxCpm2Config, VoxCpm2TransformerBackbones};
use anyhow::ensure;
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, QwenDType};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct VoxCpm2FlowDecoder {
    input: Bf16Linear,
    condition: Bf16Linear,
    time_1: Bf16Linear,
    time_2: Bf16Linear,
    delta_1: Bf16Linear,
    delta_2: Bf16Linear,
    output: Bf16Linear,
    patch_size: usize,
    feature_dim: usize,
    hidden: usize,
    cfg: f32,
    stream: Arc<CudaStream>,
    time_cache: Mutex<HashMap<u32, Vec<f32>>>,
}

impl VoxCpm2FlowDecoder {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let prefix = "feat_decoder.estimator";
        let feature = config.feat_dim;
        let hidden = config.dit_config.hidden_dim;
        Ok(Self {
            input: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.in_proj"),
                feature,
                hidden,
                true,
                stream,
            )?,
            condition: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.cond_proj"),
                feature,
                hidden,
                true,
                stream,
            )?,
            time_1: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.time_mlp.linear_1"),
                hidden,
                hidden,
                true,
                stream,
            )?,
            time_2: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.time_mlp.linear_2"),
                hidden,
                hidden,
                true,
                stream,
            )?,
            delta_1: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.delta_time_mlp.linear_1"),
                hidden,
                hidden,
                true,
                stream,
            )?,
            delta_2: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.delta_time_mlp.linear_2"),
                hidden,
                hidden,
                true,
                stream,
            )?,
            output: Bf16Linear::load(
                model_dir,
                &format!("{prefix}.out_proj"),
                hidden,
                feature,
                true,
                stream,
            )?,
            patch_size: config.patch_size,
            feature_dim: feature,
            hidden,
            cfg: config.dit_config.cfm_config.inference_cfg_rate as f32,
            stream: Arc::clone(stream),
            time_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Generate one four-frame acoustic patch with VoxCPM2's 10-step Euler
    /// flow and CFG-Zero* guidance.
    pub fn generate_patch(
        &self,
        backbones: &VoxCpm2TransformerBackbones,
        mu: &[f32],
        condition: &[f32],
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            mu.len() == self.hidden * 2,
            "VoxCPM2 DiT expects two conditioning tokens"
        );
        ensure!(
            condition.len() == self.patch_size * self.feature_dim,
            "VoxCPM2 DiT condition patch has invalid geometry"
        );
        let mut rng = GaussianRng::new(seed);
        let mut x = (0..self.patch_size * self.feature_dim)
            .map(|_| rng.normal())
            .collect::<Vec<_>>();
        let steps = 10usize;
        let t_span = (0..=steps)
            .map(|index| {
                let t = 1.0 - index as f32 / steps as f32;
                2.0 * t + (FRAC_PI_2 * t).cos() - 1.0
            })
            .collect::<Vec<_>>();
        let projected_condition =
            self.linear_host(&self.condition, condition, self.patch_size, kernels)?;
        for step in 1..=steps {
            let dt = t_span[step - 1] - t_span[step];
            if step == 1 {
                continue;
            }
            let time = self.time_condition(t_span[step - 1], kernels)?;
            let (positive, negative) =
                self.estimate_pair(backbones, &x, mu, &time, &projected_condition, kernels)?;
            let dot = positive
                .iter()
                .zip(&negative)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let norm = negative.iter().map(|value| value * value).sum::<f32>() + 1e-8;
            let scale = dot / norm;
            for ((value, positive), negative) in x.iter_mut().zip(positive).zip(negative) {
                let guided = negative * scale + self.cfg * (positive - negative * scale);
                *value -= dt * guided;
            }
        }
        ensure!(
            x.iter().all(|value| value.is_finite()),
            "VoxCPM2 flow produced non-finite latents"
        );
        Ok(x)
    }

    fn estimate_pair(
        &self,
        backbones: &VoxCpm2TransformerBackbones,
        x: &[f32],
        mu: &[f32],
        time: &[f32],
        projected_condition: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let x = self.linear_host(&self.input, x, self.patch_size, kernels)?;
        let tokens = 2 + 1 + self.patch_size * 2;
        let mut sequence = Vec::with_capacity(tokens * self.hidden * 2);
        sequence.extend_from_slice(mu);
        sequence.extend_from_slice(time);
        sequence.extend_from_slice(projected_condition);
        sequence.extend_from_slice(&x);
        sequence.resize(sequence.len() + mu.len(), 0.0);
        sequence.extend_from_slice(time);
        sequence.extend_from_slice(projected_condition);
        sequence.extend_from_slice(&x);
        let hidden = backbones.dit_forward_batched(&sequence, tokens, 2, kernels)?;
        let first_tail = (tokens - self.patch_size) * self.hidden;
        let second_tail = (tokens * 2 - self.patch_size) * self.hidden;
        let mut tails = Vec::with_capacity(self.patch_size * self.hidden * 2);
        tails.extend_from_slice(&hidden[first_tail..tokens * self.hidden]);
        tails.extend_from_slice(&hidden[second_tail..tokens * 2 * self.hidden]);
        let output = self.linear_host(&self.output, &tails, self.patch_size * 2, kernels)?;
        let split = self.patch_size * self.feature_dim;
        Ok((output[..split].to_vec(), output[split..].to_vec()))
    }

    fn time_condition(&self, timestep: f32, kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let key = timestep.to_bits();
        if let Some(cached) = self
            .time_cache
            .lock()
            .expect("VoxCPM2 time cache poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }
        let time = self.time_embedding(timestep);
        let delta = self.time_embedding(0.0);
        let time = self.mlp(&self.time_1, &self.time_2, &time, kernels)?;
        let delta = self.mlp(&self.delta_1, &self.delta_2, &delta, kernels)?;
        let combined = time
            .into_iter()
            .zip(delta)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        self.time_cache
            .lock()
            .expect("VoxCPM2 time cache poisoned")
            .insert(key, combined.clone());
        Ok(combined)
    }

    fn mlp(
        &self,
        first: &Bf16Linear,
        second: &Bf16Linear,
        input: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let hidden = self.linear_host(first, input, 1, kernels)?;
        let activated = hidden
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        self.linear_host(second, &activated, 1, kernels)
    }

    fn linear_host(
        &self,
        linear: &Bf16Linear,
        input: &[f32],
        rows: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            input.len() == rows * linear.input_size(),
            "DiT linear input geometry disagrees"
        );
        let host = input
            .iter()
            .copied()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let device = self.stream.clone_htod(&host)?;
        let output = linear.forward_native(&device, rows, kernels)?;
        self.stream.synchronize()?;
        let mut host = vec![Bf16::zero(); rows * linear.output_size()];
        self.stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(Bf16::to_f32).collect())
    }

    fn time_embedding(&self, value: f32) -> Vec<f32> {
        let half = self.hidden / 2;
        let denominator = (half - 1) as f32;
        let scale = 10_000.0f32.ln() / denominator;
        let mut output = Vec::with_capacity(self.hidden);
        for index in 0..half {
            output.push((1000.0 * value * (-scale * index as f32).exp()).sin());
        }
        for index in 0..half {
            output.push((1000.0 * value * (-scale * index as f32).exp()).cos());
        }
        output
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

    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.unit().max(1e-7).ln()).sqrt();
        let angle = std::f32::consts::TAU * self.unit();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn unit(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) as f32 + 0.5) / (1u32 << 24) as f32
    }
}
