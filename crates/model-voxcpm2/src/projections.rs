use crate::VoxCpm2Config;
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, QwenDType, load_bf16_tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use std::path::Path;
use std::sync::Arc;

pub(crate) struct Bf16Linear {
    weight: CudaSlice<Bf16>,
    bias: Option<CudaSlice<Bf16>>,
    input: usize,
    output: usize,
}

pub struct VoxCpm2Projections {
    enc_to_lm: Bf16Linear,
    lm_to_dit: Bf16Linear,
    res_to_dit: Bf16Linear,
    fusion_concat: Bf16Linear,
    fsq_in: Bf16Linear,
    fsq_out: Bf16Linear,
    stop_proj: Bf16Linear,
    stop_head: Bf16Linear,
    fsq_scale: f32,
    stream: Arc<CudaStream>,
}

pub struct VoxCpm2ProjectionOutputs {
    pub enc_to_lm: Vec<f32>,
    pub lm_to_dit: Vec<f32>,
    pub res_to_dit: Vec<f32>,
    pub fusion_concat: Vec<f32>,
    pub fsq: Vec<f32>,
    pub stop_logits: Vec<f32>,
}

impl Bf16Linear {
    pub(crate) fn load(
        model_dir: &Path,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = load_bf16_tensor(model_dir, &weight_name)
            .with_context(|| format!("could not load {weight_name}"))?;
        ensure!(
            weight.shape == [output, input],
            "{weight_name} has shape {:?}, expected [{output}, {input}]",
            weight.shape
        );
        let weight = stream.clone_htod(&weight.values)?;
        let bias = bias
            .then(|| {
                let name = format!("{prefix}.bias");
                let tensor = load_bf16_tensor(model_dir, &name)
                    .with_context(|| format!("could not load {name}"))?;
                ensure!(
                    tensor.shape == [output],
                    "{name} has shape {:?}, expected [{output}]",
                    tensor.shape
                );
                Ok::<_, anyhow::Error>(stream.clone_htod(&tensor.values)?)
            })
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            input,
            output,
        })
    }

    pub(crate) fn input_size(&self) -> usize {
        self.input
    }

    pub(crate) fn output_size(&self) -> usize {
        self.output
    }

    pub(crate) fn forward_native(
        &self,
        input: &CudaSlice<Bf16>,
        rows: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<Bf16>> {
        ensure!(
            input.len() >= rows * self.input,
            "linear input is smaller than its declared geometry"
        );
        let stream = kernels.ops.stream();
        let mut output = stream.alloc_zeros::<Bf16>(rows * self.output)?;
        if rows == 1 {
            Bf16::gemv(
                kernels,
                input,
                &self.weight,
                &mut output,
                self.output as u32,
                self.input as u32,
            )?;
        } else {
            Bf16::matmul(
                kernels,
                input,
                &self.weight,
                &mut output,
                rows as u32,
                self.output as u32,
                self.input as u32,
            )?;
        }
        if let Some(bias) = &self.bias {
            Bf16::add_bias(kernels, &mut output, bias, rows as u32, self.output as u32)?;
        }
        Ok(output)
    }
}

impl VoxCpm2Projections {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let lm = config.lm_config.hidden_size;
        let local = config.encoder_config.hidden_dim;
        let dit = config.dit_config.hidden_dim;
        let fsq = config.scalar_quantization_latent_dim;
        Ok(Self {
            enc_to_lm: Bf16Linear::load(model_dir, "enc_to_lm_proj", local, lm, true, stream)?,
            lm_to_dit: Bf16Linear::load(model_dir, "lm_to_dit_proj", lm, dit, true, stream)?,
            res_to_dit: Bf16Linear::load(model_dir, "res_to_dit_proj", lm, dit, true, stream)?,
            fusion_concat: Bf16Linear::load(
                model_dir,
                "fusion_concat_proj",
                lm * 2,
                lm,
                true,
                stream,
            )?,
            fsq_in: Bf16Linear::load(model_dir, "fsq_layer.in_proj", lm, fsq, true, stream)?,
            fsq_out: Bf16Linear::load(model_dir, "fsq_layer.out_proj", fsq, lm, true, stream)?,
            stop_proj: Bf16Linear::load(model_dir, "stop_proj", lm, lm, true, stream)?,
            stop_head: Bf16Linear::load(model_dir, "stop_head", lm, 2, false, stream)?,
            fsq_scale: config.scalar_quantization_scale as f32,
            stream: Arc::clone(stream),
        })
    }

    pub fn smoke(&self, kernels: &mut GpuKernels) -> anyhow::Result<VoxCpm2ProjectionOutputs> {
        let lm_input = deterministic_input(self.lm_to_dit.input, 0.007);
        let local_input = deterministic_input(self.enc_to_lm.input, 0.009);
        let fusion_input = deterministic_input(self.fusion_concat.input, 0.005);
        let enc_to_lm = self.forward_host(&self.enc_to_lm, &local_input, 1, kernels)?;
        let lm_to_dit = self.forward_host(&self.lm_to_dit, &lm_input, 1, kernels)?;
        let res_to_dit = self.forward_host(&self.res_to_dit, &lm_input, 1, kernels)?;
        let fusion_concat = self.forward_host(&self.fusion_concat, &fusion_input, 1, kernels)?;
        let fsq = self.fsq_host(&lm_input, kernels)?;
        let stop_logits = self.stop_host(&lm_input, kernels)?;
        ensure!(
            enc_to_lm
                .iter()
                .chain(&lm_to_dit)
                .chain(&res_to_dit)
                .chain(&fusion_concat)
                .chain(&fsq)
                .chain(&stop_logits)
                .all(|value| value.is_finite()),
            "VoxCPM2 projections produced non-finite output"
        );
        Ok(VoxCpm2ProjectionOutputs {
            enc_to_lm,
            lm_to_dit,
            res_to_dit,
            fusion_concat,
            fsq,
            stop_logits,
        })
    }

    fn forward_host(
        &self,
        linear: &Bf16Linear,
        input: &[f32],
        rows: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let input = input
            .iter()
            .copied()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let input = self.stream.clone_htod(&input)?;
        let output = linear.forward_native(&input, rows, kernels)?;
        self.download(&output)
    }

    pub(crate) fn encode_local(
        &self,
        input: &[f32],
        rows: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        self.forward_host(&self.enc_to_lm, input, rows, kernels)
    }

    pub(crate) fn flow_condition(
        &self,
        lm: &[f32],
        residual: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let mut output = self.forward_host(&self.lm_to_dit, lm, 1, kernels)?;
        output.extend(self.forward_host(&self.res_to_dit, residual, 1, kernels)?);
        Ok(output)
    }

    pub(crate) fn fuse(
        &self,
        input: &[f32],
        rows: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        self.forward_host(&self.fusion_concat, input, rows, kernels)
    }

    pub(crate) fn quantize(
        &self,
        input: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        self.fsq_host(input, kernels)
    }

    pub(crate) fn should_stop(
        &self,
        input: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<bool> {
        let logits = self.stop_host(input, kernels)?;
        Ok(logits[1] > logits[0])
    }

    fn fsq_host(&self, input: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let input = input
            .iter()
            .copied()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let input = self.stream.clone_htod(&input)?;
        let mut latent = self.fsq_in.forward_native(&input, 1, kernels)?;
        kernels
            .ops
            .fsq_quantize_bf16(&mut latent, self.fsq_in.output as u32, self.fsq_scale)?;
        let output = self.fsq_out.forward_native(&latent, 1, kernels)?;
        self.download(&output)
    }

    fn stop_host(&self, input: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let input = input
            .iter()
            .copied()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let input = self.stream.clone_htod(&input)?;
        let hidden = self.stop_proj.forward_native(&input, 1, kernels)?;
        let mut activated = self.stream.alloc_zeros::<Bf16>(hidden.len())?;
        Bf16::silu_act(kernels, &hidden, &mut activated, hidden.len() as u32)?;
        // Release the larger temporary before the classifier allocates output.
        drop(hidden);
        let output = self.stop_head.forward_native(&activated, 1, kernels)?;
        self.download(&output)
    }

    fn download(&self, output: &CudaSlice<Bf16>) -> anyhow::Result<Vec<f32>> {
        self.stream.synchronize()?;
        let mut host = vec![Bf16::zero(); output.len()];
        self.stream.memcpy_dtoh(output, &mut host)?;
        Ok(host.into_iter().map(Bf16::to_f32).collect())
    }
}

fn deterministic_input(size: usize, step: f32) -> Vec<f32> {
    (0..size)
        .map(|index| ((index as f32 + 1.0) * step).sin() * 0.125)
        .collect()
}
