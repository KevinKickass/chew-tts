use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::weights::{ModelWeights, QuantWeight};
use chew_kernel::{GpuKernels, KernelError};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;
use tracing::info;

/// Scratch buffers for the forward pass — reused across calls.
///
/// Hidden state (residual stream) stays f32 to prevent precision loss
/// from compounding over 32+ transformer layers.
/// All other intermediate buffers are f16 for VRAM efficiency.
/// KV cache is f16.
pub struct ScratchBuffers {
    /// After attention norm / FFN norm: [seq_len, dim] (f16)
    pub norm_out: CudaSlice<half::f16>,
    /// Q projection result: [seq_len, n_heads, head_dim] (f16)
    pub q: CudaSlice<half::f16>,
    /// K projection result: [seq_len, n_kv_heads, head_dim] (f16)
    pub k: CudaSlice<half::f16>,
    /// V projection result: [seq_len, n_kv_heads, head_dim] (f16)
    pub v: CudaSlice<half::f16>,
    /// MHA output: [seq_len, n_heads, head_dim] = [seq_len, dim] (f16)
    pub attn_mha_out: CudaSlice<half::f16>,
    /// After output projection: [seq_len, dim] (f16)
    pub attn_out: CudaSlice<half::f16>,
    /// Residual workspace: [seq_len, dim] (f32 — matches hidden)
    pub residual: CudaSlice<f32>,
    /// FFN gate result: [seq_len, ff_dim] (f16)
    pub ffn_gate_out: CudaSlice<half::f16>,
    /// FFN up result: [seq_len, ff_dim] (f16)
    pub ffn_up_out: CudaSlice<half::f16>,
    /// FFN SiLU result: [seq_len, ff_dim] (f16)
    pub ffn_silu_out: CudaSlice<half::f16>,
    /// FFN down / residual: [seq_len, dim] (f16)
    pub ffn_out: CudaSlice<half::f16>,
    /// Logits: [seq_len, vocab_size] (f16)
    pub logits: CudaSlice<half::f16>,
}

impl ScratchBuffers {
    pub fn alloc(
        config: &ModelConfig,
        max_batch_seq: u32,
        max_kv_len: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let s = max_batch_seq as usize;
        let d = config.dim as usize;
        let ff = config.ff_dim as usize;
        let nh = config.n_heads as usize;
        let nkv = config.n_kv_heads as usize;
        let hd = config.head_dim as usize;
        let v = config.vocab_size as usize;
        let _kv = max_kv_len as usize;

        Ok(Self {
            norm_out: stream.alloc_zeros::<half::f16>(s * d)?,
            q: stream.alloc_zeros::<half::f16>(s * nh * hd)?,
            k: stream.alloc_zeros::<half::f16>(s * nkv * hd)?,
            v: stream.alloc_zeros::<half::f16>(s * nkv * hd)?,
            attn_mha_out: stream.alloc_zeros::<half::f16>(s * d)?,
            attn_out: stream.alloc_zeros::<half::f16>(s * d)?,
            residual: stream.alloc_zeros::<f32>(s * d)?,
            ffn_gate_out: stream.alloc_zeros::<half::f16>(s * ff)?,
            ffn_up_out: stream.alloc_zeros::<half::f16>(s * ff)?,
            ffn_silu_out: stream.alloc_zeros::<half::f16>(s * ff)?,
            ffn_out: stream.alloc_zeros::<half::f16>(s * d)?,
            logits: stream.alloc_zeros::<half::f16>(s * v)?,
        })
    }
}

/// Helper: dequant-GEMM for a quantized weight.
/// C = A @ W^T where W is quantized.
///
/// For M=1 (decode), uses fused GEMV kernel (reads quantized data directly,
/// ~3x less memory bandwidth). Falls back to dequant+cuBLAS for unsupported types.
///
/// All buffers are f16.
fn gemm_q(
    kernels: &mut GpuKernels,
    a: &CudaSlice<half::f16>,
    w: &QuantWeight,
    c: &mut CudaSlice<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), KernelError> {
    // Fused GEMV for decode (M=1): uses pre-quantized Q8_1 input
    if m == 1 {
        let used = kernels.gemv.gemv(&w.data, a, c, n, k, w.quant_type)?;
        if used {
            return Ok(());
        }
        // Fall through to dequant+cuBLAS for unsupported quant types
    }

    kernels.gemm.matmul_dequant(
        a,
        &w.data,
        w.quant_type,
        w.n_elements,
        c,
        m, n, k,
        &kernels.dequant,
    )
}

/// Run the transformer forward pass.
///
/// `hidden` is the input embedding in f32: [seq_len, dim].
/// After return, `scratch.logits` contains the output logits in f16.
pub fn forward(
    hidden: &mut CudaSlice<f32>,
    weights: &ModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;
    let n_elems = seq_len * config.dim;

    let stream_ref = Arc::clone(kernels.ops.stream());

    // Profiling: disabled for production (sync barriers kill performance)
    let profile = false;
    let mut t_norm = 0u128;
    let mut t_gemm = 0u128;
    let mut t_rope = 0u128;
    let mut t_kv = 0u128;
    let mut t_mha = 0u128;
    let mut t_silu = 0u128;
    let mut t_add = 0u128;
    macro_rules! timed {
        ($accum:ident, $body:expr) => {{
            if profile { let _ = stream_ref.synchronize(); }
            let _t0 = std::time::Instant::now();
            let _r = $body;
            if profile { let _ = stream_ref.synchronize(); $accum += _t0.elapsed().as_micros(); }
            _r
        }};
    }

    // Debug: optionally limit layers via env var
    let max_layers = std::env::var("CHEW_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(config.n_layers as usize);
    let n_layers = max_layers.min(config.n_layers as usize);
    if n_layers < config.n_layers as usize {
        info!(n_layers, total = config.n_layers, "DEBUG: running limited layers");
    }

    // Layer 0: separate RMSNorm (no previous FFN residual to fuse with)
    if n_layers > 0 {
        timed!(t_norm, kernels.ops.rms_norm_f32in(
            hidden, &weights.layers[0].attn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        ))?;
    }

    for layer_idx in 0..n_layers {
        let layer = &weights.layers[layer_idx];

        // 2. QKV projections — quantize norm_out once, reuse for all 3
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }
        timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
            seq_len, config.n_heads * config.head_dim, config.dim))?;
        timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
            seq_len, config.n_kv_heads * config.head_dim, config.dim))?;
        timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
            seq_len, config.n_kv_heads * config.head_dim, config.dim))?;

        // 3. RoPE on Q and K — fused into one launch
        timed!(t_rope, {
            // Use k_cache temporarily for both Q and K RoPE
            kernels.ops.rope(&mut scratch.q, seq_len, config.n_heads, config.head_dim, pos, config.rope_theta)?;
            kernels.ops.rope(&mut scratch.k, seq_len, config.n_kv_heads, config.head_dim, pos, config.rope_theta)
        })?;

        // 4. Write K, V into KV cache
        let kv_elems = seq_len * config.n_kv_heads * config.head_dim;
        {
            let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
            kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
        }
        {
            let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
            kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
        }

        // 5. Fused Multi-Head Attention with GQA
        timed!(t_mha, {
            let k_full = kv_cache.k_full(layer_idx, total_kv_len);
            let v_full = kv_cache.v_full(layer_idx, total_kv_len);
            kernels.ops.mha_fused(
                &scratch.q, &k_full, &v_full, &mut scratch.attn_mha_out,
                config.head_dim, config.n_heads, config.n_kv_heads,
                seq_len, total_kv_len, pos,
            )
        })?;

        // 6. Output projection — quantize mha_out for GEMV
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.attn_mha_out, config.n_heads * config.head_dim)?;
        }
        timed!(t_gemm, gemm_q(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
            seq_len, config.dim, config.n_heads * config.head_dim))?;

        // 7+8. Fused: hidden += attn_out, then RMSNorm → norm_out
        timed!(t_add, kernels.ops.fused_add_rmsnorm(
            hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        ))?;

        // 9. Gate + Up projections — quantize norm_out once for both
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }
        timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
            seq_len, config.ff_dim, config.dim))?;
        timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
            seq_len, config.ff_dim, config.dim))?;

        // 10. SiLU(gate) * up (f16)
        timed!(t_silu, kernels.ops.silu(
            &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
            seq_len * config.ff_dim,
        ))?;

        // 11. Down projection — quantize silu_out for GEMV
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        }
        timed!(t_gemm, gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
            seq_len, config.dim, config.ff_dim))?;

        // 12. Residual + next layer's attn_norm (fused if not last layer)
        if layer_idx + 1 < n_layers {
            // Fused: hidden += ffn_out, then RMSNorm with next layer's attn_norm
            timed!(t_add, kernels.ops.fused_add_rmsnorm(
                hidden, &scratch.ffn_out,
                &weights.layers[layer_idx + 1].attn_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            ))?;
        } else {
            // Last layer: just add, norm happens after the loop
            timed!(t_add, kernels.ops.add_inplace_f32_f16(hidden, &scratch.ffn_out, n_elems))?;
        }
    }

    // Print profile results
    if profile {
        let total = t_norm + t_gemm + t_rope + t_kv + t_mha + t_silu + t_add;
        info!(
            gemv_us = t_gemm, mha_us = t_mha, norm_us = t_norm,
            rope_us = t_rope, kv_us = t_kv, silu_us = t_silu, add_us = t_add,
            total_us = total, kv_len = total_kv_len,
            "PROFILE decode step"
        );
    }

    // === Final norm + logits ===

    // Final RMSNorm: f32 hidden → f16 norm_out
    kernels.ops.rms_norm_f32in(
        hidden, &weights.output_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    // Output logits — dequant on-the-fly, f16
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }
    gemm_q(kernels, &scratch.norm_out, &weights.output, &mut scratch.logits,
        seq_len, config.vocab_size, config.dim)?;

    kv_cache.advance(seq_len);

    Ok(())
}
