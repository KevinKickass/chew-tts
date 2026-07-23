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
        let mut norm = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        let mut q = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        let mut k = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        let mut v = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        let mut attention = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        let mut attention_out = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;
        let mut gate = stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?;
        let mut up = stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?;
        let mut activation = stream.alloc_zeros::<f16>(INTERMEDIATE_SIZE)?;
        let mut mlp_out = stream.alloc_zeros::<f16>(HIDDEN_SIZE)?;

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.input_norm,
            &mut norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        for (weight, output) in [
            (&self.q_proj, &mut q),
            (&self.k_proj, &mut k),
            (&self.v_proj, &mut v),
        ] {
            kernels.gemm.matmul_f16(
                &norm,
                weight,
                output,
                1,
                HIDDEN_SIZE as u32,
                HIDDEN_SIZE as u32,
            )?;
        }
        // RoPE at position zero is an identity transform.
        kernels.ops.mha_fused(
            &q,
            &k.slice(..),
            &v.slice(..),
            &mut attention,
            HEAD_DIM as u32,
            ATTENTION_HEADS as u32,
            ATTENTION_HEADS as u32,
            1,
            1,
            0,
        )?;
        kernels.gemm.matmul_f16(
            &attention,
            &self.o_proj,
            &mut attention_out,
            1,
            HIDDEN_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &attention_out, HIDDEN_SIZE as u32)?;

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.post_attention_norm,
            &mut norm,
            1,
            HIDDEN_SIZE as u32,
            1e-5,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.gate_proj,
            &mut gate,
            1,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.up_proj,
            &mut up,
            1,
            INTERMEDIATE_SIZE as u32,
            HIDDEN_SIZE as u32,
        )?;
        kernels
            .ops
            .silu(&gate, &up, &mut activation, INTERMEDIATE_SIZE as u32)?;
        kernels.gemm.matmul_f16(
            &activation,
            &self.down_proj,
            &mut mlp_out,
            1,
            HIDDEN_SIZE as u32,
            INTERMEDIATE_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &mlp_out, HIDDEN_SIZE as u32)?;
        stream.synchronize()?;
        let mut output = vec![0.0; HIDDEN_SIZE];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
    }
}
