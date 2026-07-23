use crate::{ATTENTION_HEADS, HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_safetensors::MappedSafetensors;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

pub struct ChatterboxT3Layer {
    input_norm: CudaSlice<f16>,
    q_proj: CudaSlice<f16>,
    k_proj: CudaSlice<f16>,
    v_proj: CudaSlice<f16>,
    o_proj: CudaSlice<f16>,
    post_attention_norm: CudaSlice<f16>,
    gate_proj: CudaSlice<f16>,
    up_proj: CudaSlice<f16>,
    down_proj: CudaSlice<f16>,
}

pub struct ChatterboxT3Transformer {
    layers: Vec<ChatterboxT3Layer>,
    final_norm: CudaSlice<f16>,
}

struct ChatterboxT3Scratch {
    norm: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    attention: CudaSlice<f16>,
    attention_out: CudaSlice<f16>,
    gate: CudaSlice<f16>,
    up: CudaSlice<f16>,
    activation: CudaSlice<f16>,
    mlp_out: CudaSlice<f16>,
}

impl ChatterboxT3Scratch {
    fn allocate(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        Ok(Self {
            norm: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            q: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            k: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            v: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            attention: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            attention_out: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
            gate: stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?,
            up: stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?,
            activation: stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?,
            mlp_out: stream.alloc_zeros::<f16>(HIDDEN_SIZE)?,
        })
    }
}

impl ChatterboxT3Layer {
    pub fn load(
        model_dir: &Path,
        layer_index: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(
            layer_index < crate::LAYERS,
            "Chatterbox T3 layer {layer_index} is outside 0..{}",
            crate::LAYERS
        );
        let path = model_dir.join("t3_mtl23ls_v3.safetensors");
        let weights = MappedSafetensors::open(&path)?;
        let prefix = format!("tfmr.layers.{layer_index}");
        let load = |suffix: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let (shape, values) = weights
                .tensor_f16(&name)
                .with_context(|| format!("could not load Chatterbox T3 {name}"))?;
            ensure!(
                shape == expected,
                "Chatterbox T3 {name} has shape {shape:?}, expected {expected:?}"
            );
            stream
                .clone_htod(&values)
                .with_context(|| format!("could not upload Chatterbox T3 {name}"))
        };
        Ok(Self {
            input_norm: load("input_layernorm.weight", &[HIDDEN_SIZE])?,
            q_proj: load("self_attn.q_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            k_proj: load("self_attn.k_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            v_proj: load("self_attn.v_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            o_proj: load("self_attn.o_proj.weight", &[HIDDEN_SIZE, HIDDEN_SIZE])?,
            post_attention_norm: load("post_attention_layernorm.weight", &[HIDDEN_SIZE])?,
            gate_proj: load("mlp.gate_proj.weight", &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            up_proj: load("mlp.up_proj.weight", &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            down_proj: load("mlp.down_proj.weight", &[HIDDEN_SIZE, INTERMEDIATE_SIZE])?,
        })
    }

    /// Validate one complete native T3 decoder layer. At position zero the
    /// Llama-3 scaled RoPE factors are all identity, making this an exact
    /// correctness target before the cached multi-token path is added.
    pub fn forward_first_token(
        &self,
        hidden_host: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            hidden_host.len() == HIDDEN_SIZE,
            "Chatterbox T3 hidden input has {} values, expected {HIDDEN_SIZE}",
            hidden_host.len()
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = ChatterboxT3Scratch::allocate(&stream)?;
        self.forward_first_token_device(&mut hidden, &mut scratch, kernels)?;
        stream.synchronize()?;
        let mut output = vec![0.0; HIDDEN_SIZE];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
    }

    fn forward_first_token_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        scratch: &mut ChatterboxT3Scratch,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        kernels.ops.rms_norm_f32in(
            hidden,
            &self.input_norm,
            &mut scratch.norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        for (weight, output) in [
            (&self.q_proj, &mut scratch.q),
            (&self.k_proj, &mut scratch.k),
            (&self.v_proj, &mut scratch.v),
        ] {
            kernels.gemm.matmul_f16(
                &scratch.norm,
                weight,
                output,
                1,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        // RoPE at position zero is an identity transform.
        kernels.ops.mha_fused(
            &scratch.q,
            &scratch.k.slice(..),
            &scratch.v.slice(..),
            &mut scratch.attention,
            HEAD_DIM as u32,
            ATTENTION_HEADS as u32,
            ATTENTION_HEADS as u32,
            1,
            1,
            0,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.attention,
            &self.o_proj,
            &mut scratch.attention_out,
            1,
            HIDDEN_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.attention_out, HIDDEN_SIZE as u32)?;

        kernels.ops.rms_norm_f32in(
            hidden,
            &self.post_attention_norm,
            &mut scratch.norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.norm,
            &self.gate_proj,
            &mut scratch.gate,
            1,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.norm,
            &self.up_proj,
            &mut scratch.up,
            1,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.ops.silu(
            &scratch.gate,
            &scratch.up,
            &mut scratch.activation,
            INTERMEDIATE_SIZE as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.activation,
            &self.down_proj,
            &mut scratch.mlp_out,
            1,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.mlp_out, HIDDEN_SIZE as u32)?;
        Ok(())
    }
}

impl ChatterboxT3Transformer {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(crate::LAYERS);
        for layer_index in 0..crate::LAYERS {
            layers.push(
                ChatterboxT3Layer::load(model_dir, layer_index, stream)
                    .with_context(|| format!("could not load Chatterbox T3 layer {layer_index}"))?,
            );
        }
        let path = model_dir.join("t3_mtl23ls_v3.safetensors");
        let weights = MappedSafetensors::open(path)?;
        let (shape, values) = weights.tensor_f16("tfmr.norm.weight")?;
        ensure!(
            shape == [HIDDEN_SIZE],
            "Chatterbox T3 final norm has shape {shape:?}, expected [{HIDDEN_SIZE}]"
        );
        let final_norm = stream.clone_htod(&values)?;
        Ok(Self { layers, final_norm })
    }

    pub fn forward_first_token(
        &self,
        hidden_host: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            hidden_host.len() == HIDDEN_SIZE,
            "Chatterbox T3 hidden input has {} values, expected {HIDDEN_SIZE}",
            hidden_host.len()
        );
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = ChatterboxT3Scratch::allocate(&stream)?;
        for layer in &self.layers {
            layer.forward_first_token_device(&mut hidden, &mut scratch, kernels)?;
        }
        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.final_norm,
            &mut scratch.norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        stream.synchronize()?;
        let mut output = vec![f16::ZERO; HIDDEN_SIZE];
        stream.memcpy_dtoh(&scratch.norm, &mut output)?;
        Ok(output.into_iter().map(f16::to_f32).collect())
    }
}
