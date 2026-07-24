use crate::DiffusionHeadConfig;
use anyhow::{bail, ensure};

const SOLVER_ORDER: usize = 2;
const MAX_BETA: f32 = 0.999;

/// Second-order DPM-Solver++ schedule used by the official realtime model.
///
/// The scheduler intentionally stays on the CPU: it moves only one 64-value
/// acoustic latent per diffusion step, so a CUDA implementation would add
/// launch overhead without removing a meaningful transfer.
pub struct VibeVoiceScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    model_outputs: [Option<Vec<f32>>; SOLVER_ORDER],
    step_index: Option<usize>,
    lower_order_steps: usize,
}

impl VibeVoiceScheduler {
    pub fn new(config: &DiffusionHeadConfig, inference_steps: usize) -> anyhow::Result<Self> {
        ensure!(config.diffusion_type == "ddpm", "only DDPM is supported");
        ensure!(
            config.ddpm_beta_schedule == "cosine",
            "only the cosine DDPM schedule is supported"
        );
        ensure!(
            config.prediction_type == "v_prediction",
            "only velocity prediction is supported"
        );
        ensure!(config.ddpm_num_steps > 0, "DDPM needs training steps");
        ensure!(inference_steps > 0, "DDPM needs inference steps");

        let alphas = cumulative_alphas(config.ddpm_num_steps);
        let base_sigmas = alphas
            .iter()
            .map(|alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect::<Vec<_>>();
        let timesteps = linspace_timesteps(config.ddpm_num_steps, inference_steps);
        let mut sigmas = timesteps
            .iter()
            .map(|&step| base_sigmas[step])
            .collect::<Vec<_>>();
        sigmas.push(0.0);
        Ok(Self {
            timesteps,
            sigmas,
            model_outputs: [None, None],
            step_index: None,
            lower_order_steps: 0,
        })
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn reset(&mut self) {
        self.model_outputs = [None, None];
        self.step_index = None;
        self.lower_order_steps = 0;
    }

    pub fn step(
        &mut self,
        model_output: &[f32],
        timestep: usize,
        sample: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            model_output.len() == sample.len(),
            "scheduler model/sample size mismatch"
        );
        if self.step_index.is_none() {
            self.step_index = Some(
                self.timesteps
                    .iter()
                    .position(|&value| value == timestep)
                    .unwrap_or(self.timesteps.len() - 1),
            );
        }
        let index = self.step_index.expect("initialized above");
        ensure!(
            index < self.timesteps.len(),
            "scheduler is already complete"
        );

        let converted = self.convert_model_output(model_output, sample, index);
        self.model_outputs[0] = self.model_outputs[1].take();
        self.model_outputs[1] = Some(converted);

        let final_step = index + 1 == self.timesteps.len();
        let penultimate_low_order = index + 2 == self.timesteps.len() && self.timesteps.len() < 15;
        let result = if self.lower_order_steps < 1 || final_step {
            self.first_order(sample, index)?
        } else if SOLVER_ORDER == 2 || self.lower_order_steps < 2 || penultimate_low_order {
            self.second_order(sample, index)?
        } else {
            bail!("unsupported third-order scheduler path")
        };
        self.lower_order_steps = (self.lower_order_steps + 1).min(SOLVER_ORDER);
        self.step_index = Some(index + 1);
        Ok(result)
    }

    fn convert_model_output(&self, output: &[f32], sample: &[f32], index: usize) -> Vec<f32> {
        let (alpha, sigma) = alpha_sigma(self.sigmas[index]);
        output
            .iter()
            .zip(sample)
            .map(|(&velocity, &value)| alpha * value - sigma * velocity)
            .collect()
    }

    fn first_order(&self, sample: &[f32], index: usize) -> anyhow::Result<Vec<f32>> {
        let model = self.model_outputs[1]
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("first-order scheduler output is missing"))?;
        let (alpha_t, sigma_t) = alpha_sigma(self.sigmas[index + 1]);
        let (alpha_s, sigma_s) = alpha_sigma(self.sigmas[index]);
        let h = log_snr(alpha_t, sigma_t) - log_snr(alpha_s, sigma_s);
        let sample_scale = sigma_t / sigma_s;
        let model_scale = -alpha_t * ((-h).exp() - 1.0);
        Ok(sample
            .iter()
            .zip(model)
            .map(|(&value, &prediction)| sample_scale * value + model_scale * prediction)
            .collect())
    }

    fn second_order(&self, sample: &[f32], index: usize) -> anyhow::Result<Vec<f32>> {
        ensure!(index > 0, "second-order scheduler needs a previous step");
        let previous = self.model_outputs[0]
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("previous scheduler output is missing"))?;
        let current = self.model_outputs[1]
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("current scheduler output is missing"))?;
        ensure!(
            previous.len() == sample.len() && current.len() == sample.len(),
            "second-order scheduler size mismatch"
        );
        let (alpha_t, sigma_t) = alpha_sigma(self.sigmas[index + 1]);
        let (alpha_s0, sigma_s0) = alpha_sigma(self.sigmas[index]);
        let (alpha_s1, sigma_s1) = alpha_sigma(self.sigmas[index - 1]);
        let lambda_t = log_snr(alpha_t, sigma_t);
        let lambda_s0 = log_snr(alpha_s0, sigma_s0);
        let lambda_s1 = log_snr(alpha_s1, sigma_s1);
        let h = lambda_t - lambda_s0;
        let r0 = (lambda_s0 - lambda_s1) / h;
        let sample_scale = sigma_t / sigma_s0;
        let model_scale = -alpha_t * ((-h).exp() - 1.0);
        Ok(sample
            .iter()
            .zip(current)
            .zip(previous)
            .map(|((&value, &d0), &old)| {
                let d1 = (d0 - old) / r0;
                sample_scale * value + model_scale * d0 + 0.5 * model_scale * d1
            })
            .collect())
    }
}

fn alpha_bar_cosine(t: f32) -> f32 {
    let value = ((t + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2).cos();
    value * value
}

fn cumulative_alphas(steps: usize) -> Vec<f32> {
    let mut cumulative = 1.0;
    (0..steps)
        .map(|index| {
            let t1 = index as f32 / steps as f32;
            let t2 = (index + 1) as f32 / steps as f32;
            let beta = (1.0 - alpha_bar_cosine(t2) / alpha_bar_cosine(t1)).min(MAX_BETA);
            cumulative *= 1.0 - beta;
            cumulative
        })
        .collect()
}

fn linspace_timesteps(training_steps: usize, inference_steps: usize) -> Vec<usize> {
    let mut forward = Vec::with_capacity(inference_steps + 1);
    for index in 0..=inference_steps {
        let value = (training_steps - 1) as f64 * index as f64 / inference_steps as f64;
        forward.push(value.round() as usize);
    }
    (1..=inference_steps)
        .rev()
        .map(|index| forward[index])
        .collect()
}

fn alpha_sigma(sigma: f32) -> (f32, f32) {
    let alpha = 1.0 / (sigma * sigma + 1.0).sqrt();
    (alpha, sigma * alpha)
}

fn log_snr(alpha: f32, sigma: f32) -> f32 {
    alpha.ln() - sigma.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VibeVoiceConfig;

    fn config() -> VibeVoiceConfig {
        serde_json::from_str(include_str!("../../../tests/data/vibevoice-config.json")).unwrap()
    }

    #[test]
    fn official_twenty_step_schedule_matches_reference() {
        let scheduler =
            VibeVoiceScheduler::new(&config().diffusion_head_config, 20).expect("scheduler");
        assert_eq!(
            scheduler.timesteps(),
            &[
                999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250,
                200, 150, 100, 50
            ]
        );
        assert_eq!(scheduler.sigmas.len(), 21);
        assert_eq!(scheduler.sigmas[20], 0.0);
    }

    #[test]
    fn complete_schedule_stays_finite() {
        let mut scheduler =
            VibeVoiceScheduler::new(&config().diffusion_head_config, 20).expect("scheduler");
        let mut sample = vec![0.25; 64];
        for timestep in scheduler.timesteps().to_vec() {
            let velocity = vec![0.01; 64];
            sample = scheduler.step(&velocity, timestep, &sample).expect("step");
            assert!(sample.iter().all(|value| value.is_finite()));
        }
    }
}
