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

/// K/V state for one talker layer.
pub struct TalkerLayerKvCache {
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    position: usize,
    max_seq_len: usize,
    kv_dim: usize,
}

impl TalkerLayerKvCache {
    pub fn allocate(
        max_seq_len: usize,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(max_seq_len > 0, "KV cache capacity must be non-zero");
        let kv_dim = config.num_key_value_heads * config.head_dim;
        Ok(Self {
            k: stream.alloc_zeros::<f16>(max_seq_len * kv_dim)?,
            v: stream.alloc_zeros::<f16>(max_seq_len * kv_dim)?,
            position: 0,
            max_seq_len,
            kv_dim,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }
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
        self.forward_prefill(hidden_host, 1, config, kernels)
    }

    /// Run a causal prompt prefill starting at position zero.
    pub fn forward_prefill(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let mut cache = TalkerLayerKvCache::allocate(seq_len, config, &stream)?;
        self.forward_cached(hidden_host, seq_len, config, kernels, &mut cache)
    }

    /// Run one or more consecutive tokens and append their K/V state.
    pub fn forward_cached(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
        cache: &mut TalkerLayerKvCache,
    ) -> anyhow::Result<Vec<f32>> {
        let hidden_dim = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            cache.kv_dim == kv_dim,
            "KV cache geometry does not match the model"
        );
        ensure!(
            cache.position + seq_len <= cache.max_seq_len,
            "KV cache capacity {} exceeded by position {} + {seq_len}",
            cache.max_seq_len,
            cache.position
        );
        ensure!(
            hidden_host.len() == seq_len * hidden_dim,
            "hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * hidden_dim
        );
        let rows = u32::try_from(seq_len).context("sequence length exceeds CUDA limits")?;
        let position = u32::try_from(cache.position).context("KV position exceeds CUDA limits")?;
        let total_kv_len =
            u32::try_from(cache.position + seq_len).context("KV length exceeds CUDA limits")?;

        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut norm = stream.alloc_zeros::<f16>(seq_len * hidden_dim)?;
        let mut q = stream.alloc_zeros::<f16>(seq_len * q_dim)?;
        let mut k = stream.alloc_zeros::<f16>(seq_len * kv_dim)?;
        let mut v = stream.alloc_zeros::<f16>(seq_len * kv_dim)?;
        let mut attention = stream.alloc_zeros::<f16>(seq_len * q_dim)?;
        let mut attention_out = stream.alloc_zeros::<f16>(seq_len * hidden_dim)?;
        let mut gate = stream.alloc_zeros::<f16>(seq_len * intermediate)?;
        let mut up = stream.alloc_zeros::<f16>(seq_len * intermediate)?;
        let mut activation = stream.alloc_zeros::<f16>(seq_len * intermediate)?;
        let mut mlp_out = stream.alloc_zeros::<f16>(seq_len * hidden_dim)?;

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.input_norm,
            &mut norm,
            rows,
            hidden_dim as u32,
            config.rms_norm_eps as f32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.q_proj,
            &mut q,
            rows,
            q_dim as u32,
            hidden_dim as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.k_proj,
            &mut k,
            rows,
            kv_dim as u32,
            hidden_dim as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.v_proj,
            &mut v,
            rows,
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
                rows * config.num_attention_heads as u32,
                config.head_dim as u32,
                config.rms_norm_eps as f32,
            )?;
            let k_in = &k as *const CudaSlice<f16>;
            let k_out = &mut k as *mut CudaSlice<f16>;
            kernels.ops.rms_norm(
                &*k_in,
                &self.k_norm,
                &mut *k_out,
                rows * config.num_key_value_heads as u32,
                config.head_dim as u32,
                config.rms_norm_eps as f32,
            )?;
        }

        kernels.ops.rope_neox(
            &mut q,
            rows,
            config.num_attention_heads as u32,
            config.head_dim as u32,
            position,
            config.rope_theta as f32,
        )?;
        kernels.ops.rope_neox(
            &mut k,
            rows,
            config.num_key_value_heads as u32,
            config.head_dim as u32,
            position,
            config.rope_theta as f32,
        )?;
        let cache_offset = cache.position * kv_dim;
        let cache_end = cache_offset + seq_len * kv_dim;
        {
            let mut destination = cache.k.slice_mut(cache_offset..cache_end);
            kernels
                .ops
                .copy_f16(&k, &mut destination, rows * kv_dim as u32)?;
        }
        {
            let mut destination = cache.v.slice_mut(cache_offset..cache_end);
            kernels
                .ops
                .copy_f16(&v, &mut destination, rows * kv_dim as u32)?;
        }
        kernels.ops.mha_fused(
            &q,
            &cache.k.slice(..cache_end),
            &cache.v.slice(..cache_end),
            &mut attention,
            config.head_dim as u32,
            config.num_attention_heads as u32,
            config.num_key_value_heads as u32,
            rows,
            total_kv_len,
            position,
        )?;
        kernels.gemm.matmul_f16(
            &attention,
            &self.o_proj,
            &mut attention_out,
            rows,
            hidden_dim as u32,
            q_dim as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &attention_out, rows * hidden_dim as u32)?;

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.post_attention_norm,
            &mut norm,
            rows,
            hidden_dim as u32,
            config.rms_norm_eps as f32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.gate_proj,
            &mut gate,
            rows,
            intermediate as u32,
            hidden_dim as u32,
        )?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.up_proj,
            &mut up,
            rows,
            intermediate as u32,
            hidden_dim as u32,
        )?;
        kernels
            .ops
            .silu(&gate, &up, &mut activation, rows * intermediate as u32)?;
        kernels.gemm.matmul_f16(
            &activation,
            &self.down_proj,
            &mut mlp_out,
            rows,
            hidden_dim as u32,
            intermediate as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &mlp_out, rows * hidden_dim as u32)?;
        stream.synchronize()?;

        let mut output = vec![0.0f32; seq_len * hidden_dim];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        cache.position += seq_len;
        Ok(output)
    }
}
