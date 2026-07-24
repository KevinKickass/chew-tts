use crate::{QwenDType, TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

mod device;
mod stack;

pub use stack::{TalkerGenerationSession, TalkerTransformer};

/// One native CUDA Qwen3-TTS talker decoder layer.
///
/// This intentionally owns unquantized f16 weights first. It gives us a small,
/// exact correctness target before adding the full model, KV cache, CUDA graphs,
/// and quantized weight formats.
pub struct TalkerDecoderLayer<T: QwenDType = f16> {
    input_norm: CudaSlice<T>,
    q_proj: CudaSlice<T>,
    k_proj: CudaSlice<T>,
    v_proj: CudaSlice<T>,
    q_bias: Option<CudaSlice<T>>,
    k_bias: Option<CudaSlice<T>>,
    v_bias: Option<CudaSlice<T>>,
    q_norm: Option<CudaSlice<f16>>,
    k_norm: Option<CudaSlice<f16>>,
    o_proj: CudaSlice<T>,
    post_attention_norm: CudaSlice<T>,
    gate_proj: CudaSlice<T>,
    up_proj: CudaSlice<T>,
    down_proj: CudaSlice<T>,
}

/// K/V state for one talker layer.
pub struct TalkerLayerKvCache {
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    position: usize,
    max_seq_len: usize,
    kv_dim: usize,
}

/// Reusable device buffers shared by every talker layer.
pub struct TalkerLayerScratch<T: QwenDType = f16> {
    max_tokens: usize,
    pub(crate) norm: CudaSlice<T>,
    q_native: CudaSlice<T>,
    k_native: CudaSlice<T>,
    v_native: CudaSlice<T>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    attention: CudaSlice<f16>,
    attention_native: CudaSlice<T>,
    attention_out: CudaSlice<T>,
    gate: CudaSlice<T>,
    up: CudaSlice<T>,
    activation: CudaSlice<T>,
    mlp_out: CudaSlice<T>,
}

impl<T: QwenDType> TalkerLayerScratch<T> {
    pub fn allocate(
        max_tokens: usize,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(max_tokens > 0, "scratch capacity must be non-zero");
        let hidden = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        Ok(Self {
            max_tokens,
            norm: stream.alloc_zeros::<T>(max_tokens * hidden)?,
            q_native: stream.alloc_zeros::<T>(max_tokens * q_dim)?,
            k_native: stream.alloc_zeros::<T>(max_tokens * kv_dim)?,
            v_native: stream.alloc_zeros::<T>(max_tokens * kv_dim)?,
            q: stream.alloc_zeros::<f16>(max_tokens * q_dim)?,
            k: stream.alloc_zeros::<f16>(max_tokens * kv_dim)?,
            v: stream.alloc_zeros::<f16>(max_tokens * kv_dim)?,
            attention: stream.alloc_zeros::<f16>(max_tokens * q_dim)?,
            attention_native: stream.alloc_zeros::<T>(max_tokens * q_dim)?,
            attention_out: stream.alloc_zeros::<T>(max_tokens * hidden)?,
            gate: stream.alloc_zeros::<T>(max_tokens * intermediate)?,
            up: stream.alloc_zeros::<T>(max_tokens * intermediate)?,
            activation: stream.alloc_zeros::<T>(max_tokens * intermediate)?,
            mlp_out: stream.alloc_zeros::<T>(max_tokens * hidden)?,
        })
    }
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

impl<T: QwenDType> TalkerDecoderLayer<T> {
    pub fn load(
        model_dir: impl AsRef<Path>,
        layer_index: usize,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        Self::load_from_prefix(
            model_dir,
            &format!("talker.model.layers.{layer_index}"),
            config,
            stream,
        )
    }

    pub(super) fn load_from_prefix(
        model_dir: impl AsRef<Path>,
        prefix: &str,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        Self::load_architecture(model_dir, prefix, config, stream, true)
    }

    /// Load a standard Qwen2 decoder layer. Qwen2 does not have Q/K RMSNorm,
    /// but otherwise uses the same GQA, RoPE, and SwiGLU path.
    pub fn load_qwen2_from_prefix(
        model_dir: impl AsRef<Path>,
        prefix: &str,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        Self::load_architecture(model_dir, prefix, config, stream, false)
    }

    fn load_architecture(
        model_dir: impl AsRef<Path>,
        prefix: &str,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
        qk_norm: bool,
    ) -> anyhow::Result<Self> {
        let load = |suffix: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<T>> {
            let name = format!("{prefix}.{suffix}");
            let (shape, tensor) = T::load(model_dir.as_ref(), &name, stream)?;
            ensure!(
                shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                shape
            );
            Ok(tensor)
        };
        let load_f16 = |suffix: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let tensor = load_f16_tensor(model_dir.as_ref(), &name)
                .with_context(|| format!("could not load {name}"))?;
            ensure!(
                tensor.shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                tensor.shape
            );
            Ok(stream.clone_htod(&tensor.values)?)
        };

        let hidden = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let load_attention_bias = |suffix: &str, expected: &[usize]| {
            let name = format!("{prefix}.{suffix}");
            let (shape, tensor) = T::load(model_dir.as_ref(), &name, stream)?;
            ensure!(
                shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                shape
            );
            Ok::<_, anyhow::Error>(tensor)
        };
        Ok(Self {
            input_norm: load("input_layernorm.weight", &[hidden])?,
            q_proj: load("self_attn.q_proj.weight", &[q_dim, hidden])?,
            k_proj: load("self_attn.k_proj.weight", &[kv_dim, hidden])?,
            v_proj: load("self_attn.v_proj.weight", &[kv_dim, hidden])?,
            q_bias: (!qk_norm)
                .then(|| load_attention_bias("self_attn.q_proj.bias", &[q_dim]))
                .transpose()?,
            k_bias: (!qk_norm)
                .then(|| load_attention_bias("self_attn.k_proj.bias", &[kv_dim]))
                .transpose()?,
            v_bias: (!qk_norm)
                .then(|| load_attention_bias("self_attn.v_proj.bias", &[kv_dim]))
                .transpose()?,
            q_norm: qk_norm
                .then(|| load_f16("self_attn.q_norm.weight", &[config.head_dim]))
                .transpose()?,
            k_norm: qk_norm
                .then(|| load_f16("self_attn.k_norm.weight", &[config.head_dim]))
                .transpose()?,
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
        let kv_dim = config.num_key_value_heads * config.head_dim;
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
        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = TalkerLayerScratch::<T>::allocate(seq_len, config, &stream)?;
        self.forward_cached_device(&mut hidden, seq_len, config, kernels, cache, &mut scratch)?;
        stream.synchronize()?;

        let mut output = vec![0.0f32; seq_len * hidden_dim];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        cache.position += seq_len;
        Ok(output)
    }
}
