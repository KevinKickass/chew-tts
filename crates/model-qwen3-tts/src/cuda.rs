use crate::{TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

/// One native CUDA Qwen3-TTS talker decoder layer.
///
/// This intentionally owns unquantized f16 weights first. It gives us a small,
/// exact correctness target before adding the full model, KV cache, CUDA graphs,
/// and quantized weight formats.
pub struct TalkerDecoderLayer {
    input_norm: CudaSlice<f16>,
    q_proj: CudaSlice<f16>,
    k_proj: CudaSlice<f16>,
    v_proj: CudaSlice<f16>,
    q_norm: CudaSlice<f16>,
    k_norm: CudaSlice<f16>,
    o_proj: CudaSlice<f16>,
    post_attention_norm: CudaSlice<f16>,
    gate_proj: CudaSlice<f16>,
    up_proj: CudaSlice<f16>,
    down_proj: CudaSlice<f16>,
}

impl TalkerDecoderLayer {
    pub fn load(
        model_dir: impl AsRef<Path>,
        layer_index: usize,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let prefix = format!("talker.model.layers.{layer_index}");
        let load = |suffix: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let tensor = load_f16_tensor(model_dir.as_ref(), &name)
                .with_context(|| format!("could not load {name}"))?;
            ensure!(
                tensor.shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                tensor.shape
            );
            stream
                .clone_htod(&tensor.values)
                .with_context(|| format!("could not upload {name}"))
        };

        let hidden = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        Ok(Self {
            input_norm: load("input_layernorm.weight", &[hidden])?,
            q_proj: load("self_attn.q_proj.weight", &[q_dim, hidden])?,
            k_proj: load("self_attn.k_proj.weight", &[kv_dim, hidden])?,
            v_proj: load("self_attn.v_proj.weight", &[kv_dim, hidden])?,
            q_norm: load("self_attn.q_norm.weight", &[config.head_dim])?,
            k_norm: load("self_attn.k_norm.weight", &[config.head_dim])?,
            o_proj: load("self_attn.o_proj.weight", &[hidden, q_dim])?,
            post_attention_norm: load("post_attention_layernorm.weight", &[hidden])?,
            gate_proj: load("mlp.gate_proj.weight", &[intermediate, hidden])?,
            up_proj: load("mlp.up_proj.weight", &[intermediate, hidden])?,
            down_proj: load("mlp.down_proj.weight", &[hidden, intermediate])?,
        })
    }

    /// Run a single-token decoder step without a prior KV cache.
    ///
    /// Position zero makes MRoPE identical to regular NeoX RoPE. A later
    /// multi-token golden test covers positions, causality, and cache growth.
    pub fn forward_first_token(
        &self,
        hidden_host: &[f32],
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let hidden_dim = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        ensure!(
            hidden_host.len() == hidden_dim,
            "hidden input has {} values, expected {hidden_dim}",
            hidden_host.len()
        );

        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut norm = stream.alloc_zeros::<f16>(hidden_dim)?;
        let mut q = stream.alloc_zeros::<f16>(q_dim)?;
        let mut k = stream.alloc_zeros::<f16>(kv_dim)?;
        let mut v = stream.alloc_zeros::<f16>(kv_dim)?;
        let mut attention = stream.alloc_zeros::<f16>(q_dim)?;
        let mut attention_out = stream.alloc_zeros::<f16>(hidden_dim)?;
        let mut gate = stream.alloc_zeros::<f16>(intermediate)?;
        let mut up = stream.alloc_zeros::<f16>(intermediate)?;
        let mut activation = stream.alloc_zeros::<f16>(intermediate)?;
        let mut mlp_out = stream.alloc_zeros::<f16>(hidden_dim)?;

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.input_norm,
            &mut norm,
            1,
            hidden_dim as u32,
            config.rms_norm_eps as f32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.q_proj,
            &mut q,
            1,
            q_dim as u32,
            hidden_dim as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.k_proj,
            &mut k,
            1,
            kv_dim as u32,
            hidden_dim as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.v_proj,
            &mut v,
            1,
            kv_dim as u32,
            hidden_dim as u32,
        )?;

        // The CUDA RMSNorm kernel permits the same allocation as input/output,
        // matching Qwen's per-head Q/K normalization.
        unsafe {
            let q_in = &q as *const CudaSlice<f16>;
            let q_out = &mut q as *mut CudaSlice<f16>;
            kernels.ops.rms_norm(
                &*q_in,
                &self.q_norm,
                &mut *q_out,
                config.num_attention_heads as u32,
                config.head_dim as u32,
                config.rms_norm_eps as f32,
            )?;
            let k_in = &k as *const CudaSlice<f16>;
            let k_out = &mut k as *mut CudaSlice<f16>;
            kernels.ops.rms_norm(
                &*k_in,
                &self.k_norm,
                &mut *k_out,
                config.num_key_value_heads as u32,
                config.head_dim as u32,
                config.rms_norm_eps as f32,
            )?;
        }

        kernels.ops.rope_neox(
            &mut q,
            1,
            config.num_attention_heads as u32,
            config.head_dim as u32,
            0,
            config.rope_theta as f32,
        )?;
        kernels.ops.rope_neox(
            &mut k,
            1,
            config.num_key_value_heads as u32,
            config.head_dim as u32,
            0,
            config.rope_theta as f32,
        )?;
        kernels.ops.mha_fused(
            &q,
            &k.slice(..),
            &v.slice(..),
            &mut attention,
            config.head_dim as u32,
            config.num_attention_heads as u32,
            config.num_key_value_heads as u32,
            1,
            1,
            0,
        )?;
        kernels.gemm.matmul_f16(
            &attention,
            &self.o_proj,
            &mut attention_out,
            1,
            hidden_dim as u32,
            q_dim as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &attention_out, hidden_dim as u32)?;

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.post_attention_norm,
            &mut norm,
            1,
            hidden_dim as u32,
            config.rms_norm_eps as f32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.gate_proj,
            &mut gate,
            1,
            intermediate as u32,
            hidden_dim as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.up_proj,
            &mut up,
            1,
            intermediate as u32,
            hidden_dim as u32,
        )?;
        kernels
            .ops
            .silu(&gate, &up, &mut activation, intermediate as u32)?;
        kernels.gemm.matmul_f16(
            &activation,
            &self.down_proj,
            &mut mlp_out,
            1,
            hidden_dim as u32,
            intermediate as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &mlp_out, hidden_dim as u32)?;
        stream.synchronize()?;

        let mut output = vec![0.0f32; hidden_dim];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
    }
}
