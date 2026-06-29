use crate::config::ModelConfig;
use crate::forward::{ScratchBuffers, gemm_q, project_last_logits};
use crate::weights::{
    LoadError, QuantWeight, try_upload_and_dequant_any, upload_and_dequant_any,
    upload_quantized_any,
};
use chew_gguf::{GgufFile, TensorInfo};
use chew_kernel::{DequantKernels, GpuKernels, KernelError};
use chew_vram::VramAllocator;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MambaLayerLayout {
    pub attn_norm: TensorInfo,
    pub ssm_a: TensorInfo,
    pub ssm_conv1d_bias: TensorInfo,
    pub ssm_conv1d_weight: TensorInfo,
    pub ssm_d: TensorInfo,
    pub ssm_dt_bias: TensorInfo,
    pub ssm_dt_weight: TensorInfo,
    pub ssm_in_weight: TensorInfo,
    pub ssm_out_weight: TensorInfo,
    pub ssm_x_weight: TensorInfo,
}

#[derive(Debug, Clone)]
pub struct MambaLayout {
    pub n_layers: u32,
    pub model_dim: u32,
    pub inner_dim: u32,
    pub state_dim: u32,
    pub conv_kernel: u32,
    pub dt_rank: u32,
    pub vocab_size: u32,
}

pub struct MambaLayerWeights {
    pub attn_norm: CudaSlice<half::f16>,
    pub ssm_in: QuantWeight,
    pub ssm_conv1d: QuantWeight,
    pub ssm_conv1d_bias: CudaSlice<half::f16>,
    pub ssm_x: QuantWeight,
    pub ssm_dt: QuantWeight,
    pub ssm_dt_bias: CudaSlice<half::f16>,
    pub ssm_a: CudaSlice<half::f16>,
    pub ssm_d: CudaSlice<half::f16>,
    pub ssm_out: QuantWeight,
    pub ssm_dt_norm: Option<CudaSlice<half::f16>>,
    pub ssm_b_norm: Option<CudaSlice<half::f16>>,
    pub ssm_c_norm: Option<CudaSlice<half::f16>>,
}

pub struct MambaModelWeights {
    pub token_embd: CudaSlice<half::f16>,
    pub output_norm: CudaSlice<half::f16>,
    pub output: QuantWeight,
    pub layers: Vec<MambaLayerWeights>,
}

pub struct MambaRuntimeState {
    pub layers: Vec<MambaRefState>,
}

impl MambaRuntimeState {
    pub fn new(weights: &MambaModelWeights) -> Result<Self, LoadError> {
        let mut layers = Vec::with_capacity(weights.layers.len());
        for (i, layer) in weights.layers.iter().enumerate() {
            let inner_dim = layer.ssm_d.len();
            if inner_dim == 0 {
                return Err(LoadError::MissingTensor(format!(
                    "mamba layer {i} has zero inner_dim"
                )));
            }
            let a_len = layer.ssm_a.len();
            if a_len % inner_dim != 0 {
                return Err(LoadError::MissingTensor(format!(
                    "mamba layer {i} ssm_a length {a_len} is not divisible by inner_dim {inner_dim}"
                )));
            }
            let state_dim = a_len / inner_dim;
            let conv_elems = layer.ssm_conv1d.n_elements as usize;
            if conv_elems % inner_dim != 0 {
                return Err(LoadError::MissingTensor(format!(
                    "mamba layer {i} ssm_conv1d elements {conv_elems} not divisible by inner_dim {inner_dim}"
                )));
            }
            let conv_kernel = conv_elems / inner_dim;
            layers.push(MambaRefState::new(inner_dim, state_dim, conv_kernel));
        }
        Ok(Self { layers })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.conv.fill(0.0);
            layer.ssm.fill(0.0);
        }
    }
}

#[derive(Debug, Clone)]
pub struct MambaRefLayerWeights<'a> {
    /// [model_dim, 2 * inner_dim]
    pub ssm_in: &'a [f32],
    /// [conv_kernel, inner_dim]
    pub ssm_conv1d: &'a [f32],
    /// [inner_dim]
    pub ssm_conv1d_bias: &'a [f32],
    /// [inner_dim, dt_rank + 2 * state_dim]
    pub ssm_x: &'a [f32],
    /// [dt_rank, inner_dim]
    pub ssm_dt: &'a [f32],
    /// [inner_dim]
    pub ssm_dt_bias: &'a [f32],
    /// [inner_dim, state_dim]
    pub ssm_a: &'a [f32],
    /// [inner_dim]
    pub ssm_d: &'a [f32],
    /// [inner_dim, model_dim]
    pub ssm_out: &'a [f32],
    pub ssm_dt_norm: Option<&'a [f32]>,
    pub ssm_b_norm: Option<&'a [f32]>,
    pub ssm_c_norm: Option<&'a [f32]>,
    pub model_dim: usize,
    pub inner_dim: usize,
    pub state_dim: usize,
    pub conv_kernel: usize,
    pub dt_rank: usize,
}

#[derive(Debug, Clone)]
pub struct MambaRefState {
    /// Rolling short-convolution cache: [(conv_kernel - 1), inner_dim]
    pub conv: Vec<f32>,
    /// Selective-scan state: [inner_dim, state_dim]
    pub ssm: Vec<f32>,
}

impl MambaRefState {
    pub fn new(inner_dim: usize, state_dim: usize, conv_kernel: usize) -> Self {
        Self {
            conv: vec![0.0; (conv_kernel.saturating_sub(1)) * inner_dim],
            ssm: vec![0.0; inner_dim * state_dim],
        }
    }
}

impl MambaLayout {
    pub fn inspect(gguf: &GgufFile, config: &ModelConfig) -> Result<Self, LoadError> {
        let layer0 = MambaLayerLayout::load(gguf, 0)?;

        let inner_dim = expect_shape2(&layer0.ssm_a, "blk.0.ssm_a")?.1;
        let state_dim = expect_shape2(&layer0.ssm_a, "blk.0.ssm_a")?.0;
        let conv = expect_shape2(&layer0.ssm_conv1d_weight, "blk.0.ssm_conv1d.weight")?;
        let conv_kernel = conv.0;
        let conv_inner = conv.1;
        let dt = expect_shape2(&layer0.ssm_dt_weight, "blk.0.ssm_dt.weight")?;
        let dt_rank = dt.0;
        let dt_inner = dt.1;
        let x = expect_shape2(&layer0.ssm_x_weight, "blk.0.ssm_x.weight")?;
        let x_inner = x.0;
        let x_proj = x.1;
        let expected_x_proj = dt_rank + 2 * state_dim;

        if conv_inner != inner_dim {
            return Err(LoadError::MissingTensor(format!(
                "mamba layout mismatch: conv inner dim {conv_inner} != ssm inner dim {inner_dim}"
            )));
        }
        if dt_inner != inner_dim {
            return Err(LoadError::MissingTensor(format!(
                "mamba layout mismatch: dt inner dim {dt_inner} != ssm inner dim {inner_dim}"
            )));
        }
        if x_inner != inner_dim {
            return Err(LoadError::MissingTensor(format!(
                "mamba layout mismatch: x-proj inner dim {x_inner} != ssm inner dim {inner_dim}"
            )));
        }
        if x_proj != expected_x_proj {
            return Err(LoadError::MissingTensor(format!(
                "mamba layout mismatch: x-proj width {x_proj} != dt_rank + 2*state_dim ({expected_x_proj})"
            )));
        }

        let token_embd = gguf
            .find_tensor("token_embd.weight")
            .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
        let vocab_size = token_embd
            .shape
            .get(1)
            .copied()
            .ok_or_else(|| LoadError::MissingTensor("token_embd.weight shape".into()))?
            as u32;

        Ok(Self {
            n_layers: config.n_layers,
            model_dim: config.dim,
            inner_dim,
            state_dim,
            conv_kernel,
            dt_rank,
            vocab_size,
        })
    }
}

impl MambaModelWeights {
    pub fn load(
        gguf: &GgufFile,
        config: &ModelConfig,
        alloc: &VramAllocator,
        dequant: &DequantKernels,
        gpu_idx: usize,
    ) -> Result<Self, LoadError> {
        let layout = MambaLayout::inspect(gguf, config)?;
        let token_embd = upload_and_dequant_any(
            gguf,
            &[
                "token_embd.weight",
                "backbone.embedding.weight",
                "backbone.embeddings.weight",
            ],
            alloc,
            dequant,
            gpu_idx,
        )?;
        let output_norm = upload_and_dequant_any(
            gguf,
            &[
                "output_norm.weight",
                "backbone.norm_f.weight",
                "model.norm_f.weight",
            ],
            alloc,
            dequant,
            gpu_idx,
        )?;
        let output = upload_quantized_any(
            gguf,
            &[
                "output.weight",
                "lm_head.weight",
                "output.out_proj.weight",
                "token_embd.weight",
            ],
            alloc,
            gpu_idx,
        )?;

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers {
            let pfx = format!("blk.{i}");
            let hf = format!("backbone.layers.{i}.mixer");
            let layer = MambaLayerWeights {
                attn_norm: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.attn_norm.weight"),
                        &format!("backbone.layers.{i}.norm.weight"),
                        &format!("model.layers.{i}.norm.weight"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_in: upload_quantized_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_in.weight"),
                        &format!("{hf}.in_proj.weight"),
                        &format!("model.layers.{i}.in_proj.weight"),
                    ],
                    alloc,
                    gpu_idx,
                )?,
                ssm_conv1d: upload_quantized_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_conv1d.weight"),
                        &format!("{hf}.conv1d.weight"),
                        &format!("model.layers.{i}.conv1d.weight"),
                    ],
                    alloc,
                    gpu_idx,
                )?,
                ssm_conv1d_bias: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_conv1d.bias"),
                        &format!("{hf}.conv1d.bias"),
                        &format!("model.layers.{i}.conv1d.bias"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_x: upload_quantized_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_x.weight"),
                        &format!("{hf}.x_proj.weight"),
                        &format!("model.layers.{i}.x_proj.weight"),
                    ],
                    alloc,
                    gpu_idx,
                )?,
                ssm_dt: upload_quantized_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_dt.weight"),
                        &format!("{hf}.dt_proj.weight"),
                        &format!("model.layers.{i}.dt_proj.weight"),
                    ],
                    alloc,
                    gpu_idx,
                )?,
                ssm_dt_bias: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_dt.bias"),
                        &format!("{hf}.dt_proj.bias"),
                        &format!("model.layers.{i}.dt_proj.bias"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_a: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_a"),
                        &format!("{hf}.A_log"),
                        &format!("model.layers.{i}.A_log"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_d: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_d"),
                        &format!("{hf}.D"),
                        &format!("model.layers.{i}.D"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_out: upload_quantized_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_out.weight"),
                        &format!("{hf}.out_proj.weight"),
                        &format!("model.layers.{i}.out_proj.weight"),
                    ],
                    alloc,
                    gpu_idx,
                )?,
                ssm_dt_norm: try_upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_dt_norm.weight"),
                        &format!("{hf}.dt_layernorm.weight"),
                        &format!("model.layers.{i}.mamba.dt_layernorm.weight"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_b_norm: try_upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_b_norm.weight"),
                        &format!("{hf}.b_layernorm.weight"),
                        &format!("model.layers.{i}.mamba.b_layernorm.weight"),
                        &format!("model.layers.{i}.mamba.B_layernorm.weight"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ssm_c_norm: try_upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("{pfx}.ssm_c_norm.weight"),
                        &format!("{hf}.c_layernorm.weight"),
                        &format!("model.layers.{i}.mamba.c_layernorm.weight"),
                        &format!("model.layers.{i}.mamba.C_layernorm.weight"),
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
            };
            validate_loaded_layer(&layout, &layer, i as usize)?;
            layers.push(layer);
        }

        Ok(Self {
            token_embd,
            output_norm,
            output,
            layers,
        })
    }
}

/// Functional Mamba forward path.
///
/// This is a correctness-first implementation:
/// - linear projections stay on GPU via quantized GEMM
/// - selective scan/state update is executed via the CPU reference path
/// - recurrent state is preserved across calls via `runtime`
pub fn forward(
    hidden: &mut CudaSlice<f32>,
    weights: &MambaModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    runtime: &mut MambaRuntimeState,
) -> Result<(), KernelError> {
    if runtime.layers.len() != weights.layers.len() {
        return Err(KernelError::Launch(format!(
            "mamba runtime state mismatch: {} state layers for {} weight layers",
            runtime.layers.len(),
            weights.layers.len()
        )));
    }
    if seq_len == 0 {
        return Ok(());
    }

    let stream = Arc::clone(kernels.ops.stream());
    let seq = seq_len as usize;
    let dim = config.dim as usize;

    for (i, layer) in weights.layers.iter().enumerate() {
        let state = &mut runtime.layers[i];
        let inner_dim = layer.ssm_d.len();
        if inner_dim == 0 {
            return Err(KernelError::Launch(format!(
                "mamba layer {i} has zero inner_dim"
            )));
        }
        let state_dim = layer.ssm_a.len() / inner_dim;
        let conv_kernel = (layer.ssm_conv1d.n_elements as usize) / inner_dim;
        let dt_rank = (layer.ssm_dt.n_elements as usize) / inner_dim;

        kernels.ops.rms_norm_f32in(
            hidden,
            &layer.attn_norm,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;

        let mut xz_gpu = alloc_f16(&stream, seq * 2 * inner_dim)?;
        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.ssm_in,
            &mut xz_gpu,
            seq_len,
            (2 * inner_dim) as u32,
            config.dim,
        )?;
        let xz = dtoh_f16_to_f32(&stream, &xz_gpu)?;

        let mut x = vec![0.0f32; seq * inner_dim];
        let mut z = vec![0.0f32; seq * inner_dim];
        for t in 0..seq {
            let src = &xz[t * 2 * inner_dim..(t + 1) * 2 * inner_dim];
            x[t * inner_dim..(t + 1) * inner_dim].copy_from_slice(&src[..inner_dim]);
            z[t * inner_dim..(t + 1) * inner_dim].copy_from_slice(&src[inner_dim..]);
        }

        let conv_w = quant_to_host_f32(kernels, &layer.ssm_conv1d)?;
        let conv_b = dtoh_f16_to_f32(&stream, &layer.ssm_conv1d_bias)?;
        let x = conv1d_ref(
            &mut state.conv,
            &x,
            &conv_w,
            &conv_b,
            seq,
            inner_dim,
            conv_kernel,
        )
        .map_err(load_to_kernel)?;

        let x_gpu = htod_f32_to_f16(&stream, &x)?;
        let mut x_dbc_gpu = alloc_f16(&stream, seq * (dt_rank + 2 * state_dim))?;
        gemm_q(
            kernels,
            &x_gpu,
            &layer.ssm_x,
            &mut x_dbc_gpu,
            seq_len,
            (dt_rank + 2 * state_dim) as u32,
            inner_dim as u32,
        )?;
        let x_dbc = dtoh_f16_to_f32(&stream, &x_dbc_gpu)?;

        let mut dt_small = vec![0.0f32; seq * dt_rank];
        let mut b = vec![0.0f32; seq * state_dim];
        let mut c = vec![0.0f32; seq * state_dim];
        for t in 0..seq {
            let src = &x_dbc[t * (dt_rank + 2 * state_dim)..(t + 1) * (dt_rank + 2 * state_dim)];
            dt_small[t * dt_rank..(t + 1) * dt_rank].copy_from_slice(&src[..dt_rank]);
            b[t * state_dim..(t + 1) * state_dim]
                .copy_from_slice(&src[dt_rank..dt_rank + state_dim]);
            c[t * state_dim..(t + 1) * state_dim].copy_from_slice(&src[dt_rank + state_dim..]);
        }

        if let Some(w) = &layer.ssm_dt_norm {
            let ww = dtoh_f16_to_f32(&stream, w)?;
            rms_norm_rows(&mut dt_small, &ww, seq, dt_rank).map_err(load_to_kernel)?;
        }
        if let Some(w) = &layer.ssm_b_norm {
            let ww = dtoh_f16_to_f32(&stream, w)?;
            rms_norm_rows(&mut b, &ww, seq, state_dim).map_err(load_to_kernel)?;
        }
        if let Some(w) = &layer.ssm_c_norm {
            let ww = dtoh_f16_to_f32(&stream, w)?;
            rms_norm_rows(&mut c, &ww, seq, state_dim).map_err(load_to_kernel)?;
        }

        let dt_small_gpu = htod_f32_to_f16(&stream, &dt_small)?;
        let mut dt_gpu = alloc_f16(&stream, seq * inner_dim)?;
        gemm_q(
            kernels,
            &dt_small_gpu,
            &layer.ssm_dt,
            &mut dt_gpu,
            seq_len,
            inner_dim as u32,
            dt_rank as u32,
        )?;
        let dt = dtoh_f16_to_f32(&stream, &dt_gpu)?;

        let dt_bias = dtoh_f16_to_f32(&stream, &layer.ssm_dt_bias)?;
        let a = dtoh_f16_to_f32(&stream, &layer.ssm_a)?;
        let d = dtoh_f16_to_f32(&stream, &layer.ssm_d)?;

        let y = selective_scan_ref(
            &mut state.ssm,
            &x,
            Some(&z),
            &dt,
            &dt_bias,
            &a,
            &b,
            &c,
            &d,
            seq,
            inner_dim,
            state_dim,
        )
        .map_err(load_to_kernel)?;

        let y_gpu = htod_f32_to_f16(&stream, &y)?;
        let mut out_gpu = alloc_f16(&stream, seq * dim)?;
        gemm_q(
            kernels,
            &y_gpu,
            &layer.ssm_out,
            &mut out_gpu,
            seq_len,
            config.dim,
            inner_dim as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &out_gpu, seq_len * config.dim)?;
    }

    kernels.ops.rms_norm_f32in(
        hidden,
        &weights.output_norm,
        &mut scratch.norm_out,
        seq_len,
        config.dim,
        config.rms_norm_eps,
    )?;
    project_last_logits(
        kernels,
        &stream,
        &scratch.norm_out,
        &mut scratch.attn_out,
        &weights.output,
        &mut scratch.logits,
        seq_len,
        config.vocab_size,
        config.dim,
    )?;
    if let Some(cap) = config.logit_softcap {
        kernels
            .ops
            .logit_softcap_inplace(&mut scratch.logits, config.vocab_size, cap)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct MambaRefStepParams<'a> {
    /// Input/skip stream u: [inner_dim]
    pub u: &'a [f32],
    /// Gate stream z: [inner_dim]
    pub z: Option<&'a [f32]>,
    /// Delta pre-activation (before bias + softplus): [inner_dim]
    pub dt: &'a [f32],
    /// Delta bias: [inner_dim]
    pub dt_bias: &'a [f32],
    /// State transition: [inner_dim, state_dim]
    pub a: &'a [f32],
    /// Input-to-state projection for current token: [state_dim]
    pub b: &'a [f32],
    /// State-to-output projection for current token: [state_dim]
    pub c: &'a [f32],
    /// Skip parameter: [inner_dim]
    pub d: &'a [f32],
    pub inner_dim: usize,
    pub state_dim: usize,
}

/// Reference single-token selective scan step for Mamba.
///
/// This mirrors the fallback path in the official `mamba_simple.py`:
/// `dt = softplus(dt + dt_bias)`,
/// `dA = exp(dt * A)`,
/// `dB = dt * B`,
/// `state = state * dA + u * dB`,
/// `y = <state, C> + D * u`,
/// optional gate `y *= silu(z)`.
pub fn selective_scan_step_ref(
    state: &mut [f32],
    p: &MambaRefStepParams<'_>,
) -> Result<Vec<f32>, LoadError> {
    if p.u.len() != p.inner_dim
        || p.dt.len() != p.inner_dim
        || p.dt_bias.len() != p.inner_dim
        || p.d.len() != p.inner_dim
        || p.a.len() != p.inner_dim * p.state_dim
        || p.b.len() != p.state_dim
        || p.c.len() != p.state_dim
        || state.len() != p.inner_dim * p.state_dim
        || p.z.is_some_and(|z| z.len() != p.inner_dim)
    {
        return Err(LoadError::MissingTensor(
            "mamba selective_scan_step_ref received inconsistent shapes".into(),
        ));
    }

    let mut y = vec![0.0f32; p.inner_dim];
    for d in 0..p.inner_dim {
        let dt = softplus(p.dt[d] + p.dt_bias[d]);
        let mut acc = p.d[d] * p.u[d];
        for n in 0..p.state_dim {
            let idx = d * p.state_dim + n;
            let d_a = (dt * p.a[idx]).exp();
            let d_b = dt * p.b[n];
            state[idx] = state[idx] * d_a + p.u[d] * d_b;
            acc += state[idx] * p.c[n];
        }
        y[d] = match p.z {
            Some(z) => acc * silu(z[d]),
            None => acc,
        };
    }
    Ok(y)
}

/// Reference sequence scan for Mamba with token-varying dt/B/C.
///
/// Shapes:
/// - `u`, `dt`, `z`: [seq_len, inner_dim]
/// - `b`, `c`: [seq_len, state_dim]
/// - `state`: [inner_dim, state_dim]
/// Returns `out`: [seq_len, inner_dim]
pub fn selective_scan_ref(
    state: &mut [f32],
    u: &[f32],
    z: Option<&[f32]>,
    dt: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    seq_len: usize,
    inner_dim: usize,
    state_dim: usize,
) -> Result<Vec<f32>, LoadError> {
    if u.len() != seq_len * inner_dim
        || dt.len() != seq_len * inner_dim
        || z.is_some_and(|zv| zv.len() != seq_len * inner_dim)
        || b.len() != seq_len * state_dim
        || c.len() != seq_len * state_dim
    {
        return Err(LoadError::MissingTensor(
            "mamba selective_scan_ref received inconsistent sequence shapes".into(),
        ));
    }

    let mut out = vec![0.0f32; seq_len * inner_dim];
    for t in 0..seq_len {
        let step = MambaRefStepParams {
            u: &u[t * inner_dim..(t + 1) * inner_dim],
            z: z.map(|zv| &zv[t * inner_dim..(t + 1) * inner_dim]),
            dt: &dt[t * inner_dim..(t + 1) * inner_dim],
            dt_bias,
            a,
            b: &b[t * state_dim..(t + 1) * state_dim],
            c: &c[t * state_dim..(t + 1) * state_dim],
            d,
            inner_dim,
            state_dim,
        };
        let y = selective_scan_step_ref(state, &step)?;
        out[t * inner_dim..(t + 1) * inner_dim].copy_from_slice(&y);
    }
    Ok(out)
}

/// Reference depthwise short convolution used by classic Mamba-1.
///
/// Shapes:
/// - `state`: [(conv_kernel - 1), inner_dim]
/// - `x`: [seq_len, inner_dim]
/// - `kernel`: [conv_kernel, inner_dim]
/// - `bias`: [inner_dim]
/// Returns activated output [seq_len, inner_dim] and updates `state`.
pub fn conv1d_ref(
    state: &mut [f32],
    x: &[f32],
    kernel: &[f32],
    bias: &[f32],
    seq_len: usize,
    inner_dim: usize,
    conv_kernel: usize,
) -> Result<Vec<f32>, LoadError> {
    let hist = conv_kernel.saturating_sub(1);
    if state.len() != hist * inner_dim
        || x.len() != seq_len * inner_dim
        || kernel.len() != conv_kernel * inner_dim
        || bias.len() != inner_dim
    {
        return Err(LoadError::MissingTensor(
            "mamba conv1d_ref received inconsistent shapes".into(),
        ));
    }

    let mut stacked = Vec::with_capacity((hist + seq_len) * inner_dim);
    stacked.extend_from_slice(state);
    stacked.extend_from_slice(x);

    let mut out = vec![0.0f32; seq_len * inner_dim];
    for t in 0..seq_len {
        for c in 0..inner_dim {
            let mut acc = bias[c];
            for k in 0..conv_kernel {
                let src_idx = (t + k) * inner_dim + c;
                // GGUF/ggml stores 2D tensors with dim0 as the contiguous axis.
                // For conv weight shape [conv_kernel, inner_dim], index is
                // kernel[k + c * conv_kernel].
                let ker_idx = k + c * conv_kernel;
                acc += stacked[src_idx] * kernel[ker_idx];
            }
            out[t * inner_dim + c] = silu(acc);
        }
    }

    if hist > 0 {
        let start = seq_len * inner_dim;
        let end = start + hist * inner_dim;
        state.copy_from_slice(&stacked[start..end]);
    }

    Ok(out)
}

/// Reference full classic Mamba-1 layer.
///
/// Input/output shape is [seq_len, model_dim]. `state` is updated in place.
pub fn mamba_layer_ref(
    input: &[f32],
    state: &mut MambaRefState,
    weights: &MambaRefLayerWeights<'_>,
    seq_len: usize,
) -> Result<Vec<f32>, LoadError> {
    if input.len() != seq_len * weights.model_dim {
        return Err(LoadError::MissingTensor(format!(
            "mamba_layer_ref expected input {} elements, got {}",
            seq_len * weights.model_dim,
            input.len()
        )));
    }

    validate_ref_layer_weights(weights)?;

    let xz = matmul_row_major(
        input,
        weights.ssm_in,
        seq_len,
        weights.model_dim,
        2 * weights.inner_dim,
    )?;
    let mut x = vec![0.0f32; seq_len * weights.inner_dim];
    let mut z = vec![0.0f32; seq_len * weights.inner_dim];
    for t in 0..seq_len {
        let src = &xz[t * 2 * weights.inner_dim..(t + 1) * 2 * weights.inner_dim];
        x[t * weights.inner_dim..(t + 1) * weights.inner_dim]
            .copy_from_slice(&src[..weights.inner_dim]);
        z[t * weights.inner_dim..(t + 1) * weights.inner_dim]
            .copy_from_slice(&src[weights.inner_dim..]);
    }

    let x = conv1d_ref(
        &mut state.conv,
        &x,
        weights.ssm_conv1d,
        weights.ssm_conv1d_bias,
        seq_len,
        weights.inner_dim,
        weights.conv_kernel,
    )?;

    let x_dbc = matmul_row_major(
        &x,
        weights.ssm_x,
        seq_len,
        weights.inner_dim,
        weights.dt_rank + 2 * weights.state_dim,
    )?;
    let mut dt_small = vec![0.0f32; seq_len * weights.dt_rank];
    let mut b = vec![0.0f32; seq_len * weights.state_dim];
    let mut c = vec![0.0f32; seq_len * weights.state_dim];
    for t in 0..seq_len {
        let src = &x_dbc[t * (weights.dt_rank + 2 * weights.state_dim)
            ..(t + 1) * (weights.dt_rank + 2 * weights.state_dim)];
        dt_small[t * weights.dt_rank..(t + 1) * weights.dt_rank]
            .copy_from_slice(&src[..weights.dt_rank]);
        b[t * weights.state_dim..(t + 1) * weights.state_dim]
            .copy_from_slice(&src[weights.dt_rank..weights.dt_rank + weights.state_dim]);
        c[t * weights.state_dim..(t + 1) * weights.state_dim]
            .copy_from_slice(&src[weights.dt_rank + weights.state_dim..]);
    }

    if let Some(w) = weights.ssm_dt_norm {
        rms_norm_rows(&mut dt_small, w, seq_len, weights.dt_rank)?;
    }
    if let Some(w) = weights.ssm_b_norm {
        rms_norm_rows(&mut b, w, seq_len, weights.state_dim)?;
    }
    if let Some(w) = weights.ssm_c_norm {
        rms_norm_rows(&mut c, w, seq_len, weights.state_dim)?;
    }

    let dt = matmul_row_major(
        &dt_small,
        weights.ssm_dt,
        seq_len,
        weights.dt_rank,
        weights.inner_dim,
    )?;

    let y = selective_scan_ref(
        &mut state.ssm,
        &x,
        Some(&z),
        &dt,
        weights.ssm_dt_bias,
        weights.ssm_a,
        &b,
        &c,
        weights.ssm_d,
        seq_len,
        weights.inner_dim,
        weights.state_dim,
    )?;

    matmul_row_major(
        &y,
        weights.ssm_out,
        seq_len,
        weights.inner_dim,
        weights.model_dim,
    )
}

impl MambaLayerLayout {
    pub fn load(gguf: &GgufFile, layer: u32) -> Result<Self, LoadError> {
        let pfx = format!("blk.{layer}");
        Ok(Self {
            attn_norm: tensor(gguf, &format!("{pfx}.attn_norm.weight"))?,
            ssm_a: tensor(gguf, &format!("{pfx}.ssm_a"))?,
            ssm_conv1d_bias: tensor(gguf, &format!("{pfx}.ssm_conv1d.bias"))?,
            ssm_conv1d_weight: tensor(gguf, &format!("{pfx}.ssm_conv1d.weight"))?,
            ssm_d: tensor(gguf, &format!("{pfx}.ssm_d"))?,
            ssm_dt_bias: tensor(gguf, &format!("{pfx}.ssm_dt.bias"))?,
            ssm_dt_weight: tensor(gguf, &format!("{pfx}.ssm_dt.weight"))?,
            ssm_in_weight: tensor(gguf, &format!("{pfx}.ssm_in.weight"))?,
            ssm_out_weight: tensor(gguf, &format!("{pfx}.ssm_out.weight"))?,
            ssm_x_weight: tensor(gguf, &format!("{pfx}.ssm_x.weight"))?,
        })
    }
}

fn tensor(gguf: &GgufFile, name: &str) -> Result<TensorInfo, LoadError> {
    gguf.find_tensor(name)
        .cloned()
        .ok_or_else(|| LoadError::MissingTensor(name.to_string()))
}

fn expect_shape2(t: &TensorInfo, name: &str) -> Result<(u32, u32), LoadError> {
    if t.shape.len() != 2 {
        return Err(LoadError::MissingTensor(format!(
            "{name} expected rank-2 tensor, got shape {:?}",
            t.shape
        )));
    }
    Ok((t.shape[0] as u32, t.shape[1] as u32))
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn load_to_kernel(e: LoadError) -> KernelError {
    KernelError::Launch(e.to_string())
}

fn alloc_f16(stream: &Arc<CudaStream>, n: usize) -> Result<CudaSlice<half::f16>, KernelError> {
    stream
        .alloc_zeros::<half::f16>(n)
        .map_err(|e| KernelError::Launch(e.to_string()))
}

fn dtoh_f16_to_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<half::f16>,
) -> Result<Vec<f32>, KernelError> {
    let mut host = vec![half::f16::ZERO; src.len()];
    stream
        .memcpy_dtoh(src, &mut host)
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    Ok(host.into_iter().map(|x| x.to_f32()).collect())
}

fn htod_f32_to_f16(
    stream: &Arc<CudaStream>,
    src: &[f32],
) -> Result<CudaSlice<half::f16>, KernelError> {
    let host: Vec<half::f16> = src.iter().copied().map(half::f16::from_f32).collect();
    let mut dst = stream
        .alloc_zeros::<half::f16>(host.len())
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    stream
        .memcpy_htod(&host, &mut dst)
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    Ok(dst)
}

fn quant_to_host_f32(kernels: &mut GpuKernels, w: &QuantWeight) -> Result<Vec<f32>, KernelError> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut deq = stream
        .alloc_zeros::<half::f16>(w.n_elements as usize)
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    kernels
        .dequant
        .dequant(&w.data, &mut deq, w.n_elements, w.quant_type)?;
    dtoh_f16_to_f32(&stream, &deq)
}

fn validate_ref_layer_weights(weights: &MambaRefLayerWeights<'_>) -> Result<(), LoadError> {
    let expected = [
        (
            "ssm_in",
            weights.ssm_in.len(),
            weights.model_dim * 2 * weights.inner_dim,
        ),
        (
            "ssm_conv1d",
            weights.ssm_conv1d.len(),
            weights.conv_kernel * weights.inner_dim,
        ),
        (
            "ssm_conv1d_bias",
            weights.ssm_conv1d_bias.len(),
            weights.inner_dim,
        ),
        (
            "ssm_x",
            weights.ssm_x.len(),
            weights.inner_dim * (weights.dt_rank + 2 * weights.state_dim),
        ),
        (
            "ssm_dt",
            weights.ssm_dt.len(),
            weights.dt_rank * weights.inner_dim,
        ),
        ("ssm_dt_bias", weights.ssm_dt_bias.len(), weights.inner_dim),
        (
            "ssm_a",
            weights.ssm_a.len(),
            weights.inner_dim * weights.state_dim,
        ),
        ("ssm_d", weights.ssm_d.len(), weights.inner_dim),
        (
            "ssm_out",
            weights.ssm_out.len(),
            weights.inner_dim * weights.model_dim,
        ),
    ];
    for (name, got, want) in expected {
        if got != want {
            return Err(LoadError::MissingTensor(format!(
                "mamba ref tensor {name} expected {want} elements, got {got}"
            )));
        }
    }
    if let Some(w) = weights.ssm_dt_norm
        && w.len() != weights.dt_rank
    {
        return Err(LoadError::MissingTensor(format!(
            "mamba ref tensor ssm_dt_norm expected {} elements, got {}",
            weights.dt_rank,
            w.len()
        )));
    }
    if let Some(w) = weights.ssm_b_norm
        && w.len() != weights.state_dim
    {
        return Err(LoadError::MissingTensor(format!(
            "mamba ref tensor ssm_b_norm expected {} elements, got {}",
            weights.state_dim,
            w.len()
        )));
    }
    if let Some(w) = weights.ssm_c_norm
        && w.len() != weights.state_dim
    {
        return Err(LoadError::MissingTensor(format!(
            "mamba ref tensor ssm_c_norm expected {} elements, got {}",
            weights.state_dim,
            w.len()
        )));
    }
    Ok(())
}

fn matmul_row_major(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, LoadError> {
    if input.len() != rows * in_dim || weight.len() != in_dim * out_dim {
        return Err(LoadError::MissingTensor(
            "mamba matmul_row_major received inconsistent shapes".into(),
        ));
    }
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let x = &input[r * in_dim..(r + 1) * in_dim];
        let y = &mut out[r * out_dim..(r + 1) * out_dim];
        for i in 0..in_dim {
            let xi = x[i];
            let wrow = &weight[i * out_dim..(i + 1) * out_dim];
            for o in 0..out_dim {
                y[o] += xi * wrow[o];
            }
        }
    }
    Ok(out)
}

fn rms_norm_rows(x: &mut [f32], weight: &[f32], rows: usize, dim: usize) -> Result<(), LoadError> {
    if x.len() != rows * dim || weight.len() != dim {
        return Err(LoadError::MissingTensor(
            "mamba rms_norm_rows received inconsistent shapes".into(),
        ));
    }
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean_sq + 1e-5).sqrt();
        for i in 0..dim {
            row[i] = row[i] * inv * weight[i];
        }
    }
    Ok(())
}

fn validate_loaded_layer(
    layout: &MambaLayout,
    layer: &MambaLayerWeights,
    idx: usize,
) -> Result<(), LoadError> {
    let inner = layout.inner_dim as usize;
    let state = layout.state_dim as usize;
    let conv = layout.conv_kernel as usize;
    let dt_rank = layout.dt_rank as usize;

    expect_quant_shape(
        &layer.ssm_in,
        "ssm_in",
        idx,
        &[layout.model_dim as usize, 2 * inner],
    )?;
    expect_quant_shape(&layer.ssm_conv1d, "ssm_conv1d", idx, &[conv, inner])?;
    expect_f16_len(&layer.ssm_conv1d_bias, "ssm_conv1d.bias", idx, inner)?;
    expect_quant_shape(&layer.ssm_x, "ssm_x", idx, &[inner, dt_rank + 2 * state])?;
    expect_quant_shape(&layer.ssm_dt, "ssm_dt", idx, &[dt_rank, inner])?;
    expect_f16_len(&layer.ssm_dt_bias, "ssm_dt.bias", idx, inner)?;
    expect_f16_len(&layer.ssm_a, "ssm_a", idx, inner * state)?;
    expect_f16_len(&layer.ssm_d, "ssm_d", idx, inner)?;
    expect_quant_shape(
        &layer.ssm_out,
        "ssm_out",
        idx,
        &[inner, layout.model_dim as usize],
    )?;

    Ok(())
}

fn expect_quant_shape(
    w: &QuantWeight,
    name: &str,
    layer: usize,
    dims: &[usize],
) -> Result<(), LoadError> {
    let expected: usize = dims.iter().product();
    if w.n_elements as usize != expected {
        return Err(LoadError::MissingTensor(format!(
            "mamba layer {layer} tensor {name} expected {} elements for shape {:?}, got {}",
            expected, dims, w.n_elements
        )));
    }
    Ok(())
}

fn expect_f16_len(
    w: &CudaSlice<half::f16>,
    name: &str,
    layer: usize,
    expected: usize,
) -> Result<(), LoadError> {
    let got = w.len();
    if got != expected {
        return Err(LoadError::MissingTensor(format!(
            "mamba layer {layer} tensor {name} expected {} elements, got {}",
            expected, got
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MambaRefLayerWeights, MambaRefState, MambaRefStepParams, conv1d_ref, mamba_layer_ref,
        selective_scan_ref, selective_scan_step_ref,
    };

    #[test]
    fn selective_scan_step_ref_updates_state_and_output() {
        let inner_dim = 2usize;
        let state_dim = 2usize;
        let mut state = vec![0.0; inner_dim * state_dim];
        let params = MambaRefStepParams {
            u: &[1.0, 2.0],
            z: Some(&[0.5, -1.0]),
            dt: &[0.0, 0.25],
            dt_bias: &[0.0, 0.0],
            a: &[-1.0, -2.0, -1.5, -0.5],
            b: &[0.25, -0.5],
            c: &[1.0, 0.5],
            d: &[0.1, 0.2],
            inner_dim,
            state_dim,
        };
        let y = selective_scan_step_ref(&mut state, &params).unwrap();
        assert_eq!(y.len(), inner_dim);
        assert!(state.iter().any(|v| *v != 0.0));
        assert!(y[0].is_finite() && y[1].is_finite());
    }

    #[test]
    fn selective_scan_ref_runs_multiple_steps() {
        let seq_len = 3usize;
        let inner_dim = 2usize;
        let state_dim = 2usize;
        let mut state = vec![0.0; inner_dim * state_dim];
        let out = selective_scan_ref(
            &mut state,
            &[1.0, 2.0, 0.5, -1.0, 0.25, 0.75],
            None,
            &[0.1, 0.2, 0.3, 0.4, -0.2, 0.5],
            &[0.0, 0.0],
            &[-1.0, -2.0, -1.5, -0.5],
            &[0.25, -0.5, 0.5, 0.25, -0.75, 0.1],
            &[1.0, 0.5, -0.5, 1.5, 0.25, -1.0],
            &[0.1, 0.2],
            seq_len,
            inner_dim,
            state_dim,
        )
        .unwrap();
        assert_eq!(out.len(), seq_len * inner_dim);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(state.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conv1d_ref_updates_state_and_outputs() {
        let mut state = vec![0.0; 2];
        let out = conv1d_ref(
            &mut state,
            &[1.0, 2.0, 3.0, 4.0],
            &[0.5, 0.25, 1.0, -0.5],
            &[0.1, -0.2],
            2,
            2,
            2,
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(state, vec![3.0, 4.0]);
    }

    #[test]
    fn mamba_layer_ref_runs_end_to_end() {
        let weights = MambaRefLayerWeights {
            ssm_in: &[1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.25, 0.5],
            ssm_conv1d: &[0.5, -0.25, 1.0, 0.75],
            ssm_conv1d_bias: &[0.1, -0.2],
            ssm_x: &[0.5, -0.1, 0.2, -0.2, 0.25, 0.1],
            ssm_dt: &[0.4, -0.1],
            ssm_dt_bias: &[0.0, 0.1],
            ssm_a: &[-1.0, -0.5],
            ssm_d: &[0.2, 0.3],
            ssm_out: &[0.5, -0.25, 0.75, 0.4],
            ssm_dt_norm: None,
            ssm_b_norm: None,
            ssm_c_norm: None,
            model_dim: 2,
            inner_dim: 2,
            state_dim: 1,
            conv_kernel: 2,
            dt_rank: 1,
        };
        let mut state = MambaRefState::new(2, 1, 2);
        let out = mamba_layer_ref(&[1.0, 0.5, -0.25, 2.0], &mut state, &weights, 2).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(state.conv.iter().all(|v| v.is_finite()));
        assert!(state.ssm.iter().all(|v| v.is_finite()));
    }
}
