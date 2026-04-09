use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::weights::{ModelWeights, QuantWeight, StreamingWeights, LayerWeights,
    HostLayerLayout, StreamingLayerNorms, TensorSlot};
use crate::WeightStorage;
use chew_kernel::{GpuKernels, KernelError};
use cudarc::driver::{CudaSlice, CudaStream};
use cudarc::driver::sys;
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
    // Per-layer embedding scratch buffers (Gemma 4)
    /// After inp_gate projection + GELU: [seq_len, epl] (f16)
    pub pe_gate_out: Option<CudaSlice<half::f16>>,
    /// After proj projection: [seq_len, dim] (f16)
    pub pe_proj_out: Option<CudaSlice<half::f16>>,
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
        let hd = config.max_head_dim as usize;
        let v = config.vocab_size as usize;
        let _kv = max_kv_len as usize;

        // Per-layer embedding buffers (Gemma 4 only)
        let (pe_gate_out, pe_proj_out) = if let Some(epl) = config.embd_per_layer {
            let epl = epl as usize;
            (
                Some(stream.alloc_zeros::<half::f16>(s * epl)?),
                Some(stream.alloc_zeros::<half::f16>(s * d)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            norm_out: stream.alloc_zeros::<half::f16>(s * d)?,
            q: stream.alloc_zeros::<half::f16>(s * nh * hd)?,
            k: stream.alloc_zeros::<half::f16>(s * nkv * hd)?,
            v: stream.alloc_zeros::<half::f16>(s * nkv * hd)?,
            attn_mha_out: stream.alloc_zeros::<half::f16>(s * nh * hd)?,
            attn_out: stream.alloc_zeros::<half::f16>(s * d)?,
            residual: stream.alloc_zeros::<f32>(s * d)?,
            ffn_gate_out: stream.alloc_zeros::<half::f16>(s * ff)?,
            ffn_up_out: stream.alloc_zeros::<half::f16>(s * ff)?,
            ffn_silu_out: stream.alloc_zeros::<half::f16>(s * ff)?,
            ffn_out: stream.alloc_zeros::<half::f16>(s * d)?,
            logits: stream.alloc_zeros::<half::f16>(s * v)?,
            pe_gate_out,
            pe_proj_out,
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
pub(crate) fn gemm_q(
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

    let profile = std::env::var("CHEW_PROFILE").is_ok() && seq_len == 1;
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
        if seq_len == 1 {
            // Fused: RMSNorm + Q8_1 quantize in one kernel
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.rms_norm_f32in_q8(
                hidden, &weights.layers[0].attn_norm, &mut scratch.norm_out,
                x_q8, seq_len, config.dim, config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.rms_norm_f32in(
                hidden, &weights.layers[0].attn_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
        }
    }

    for layer_idx in 0..n_layers {
        let layer = &weights.layers[layer_idx];

        // 2. QKV projections — for decode, x_q8 already contains quantized norm_out
        //    (from fused rms_norm_f32in_q8 or fused_add_rmsnorm_q8)
        if seq_len == 1 && layer_idx > 0 {
            // Layer 0 was already fused above; layers 1+ were fused at end of previous layer
            // x_q8 is already set — no quantize_input needed
        } else if seq_len > 1 {
            // Prefill path: no GEMV quantization needed
        }
        // Q projection (separate) + K+V projection (fused dual GEMV for decode)
        timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
            seq_len, config.n_heads * config.head_dim, config.dim))?;
        if seq_len == 1 && layer.attn_k.quant_type == layer.attn_v.quant_type {
            let nk = config.n_kv_heads * config.head_dim;
            let used = timed!(t_gemm, kernels.gemv.gemv_dual(
                &layer.attn_k.data, &layer.attn_v.data,
                &mut scratch.k, &mut scratch.v,
                nk, config.dim, layer.attn_k.quant_type,
            ))?;
            if !used {
                timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
                    seq_len, config.n_kv_heads * config.head_dim, config.dim))?;
                timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
                    seq_len, config.n_kv_heads * config.head_dim, config.dim))?;
            }
        } else {
            timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
                seq_len, config.n_kv_heads * config.head_dim, config.dim))?;
            timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
                seq_len, config.n_kv_heads * config.head_dim, config.dim))?;
        }

        // 3. RoPE on Q and K
        timed!(t_rope, {
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

        // 6. Output projection — quantize mha_out for GEMV (reuse existing quantize_input)
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.attn_mha_out, config.n_heads * config.head_dim)?;
        }
        // Note: quantize_input is cheap (~1µs) and hard to fuse with MHA output write
        timed!(t_gemm, gemm_q(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
            seq_len, config.dim, config.n_heads * config.head_dim))?;

        // 7+8. Fused: hidden += attn_out, then RMSNorm → norm_out
        if seq_len == 1 {
            // Fused: add + RMSNorm + Q8_1 quantize in one kernel
            let x_q8 = kernels.gemv.x_q8_mut();
            timed!(t_add, kernels.ops.fused_add_rmsnorm_q8(
                hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
                x_q8, seq_len, config.dim, config.rms_norm_eps,
            ))?;
        } else {
            timed!(t_add, kernels.ops.fused_add_rmsnorm(
                hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            ))?;
        }

        // 9. Gate + Up projections — fused dual GEMV for decode, separate for prefill
        if seq_len == 1 && layer.ffn_gate.quant_type == layer.ffn_up.quant_type {
            let used = timed!(t_gemm, kernels.gemv.gemv_dual(
                &layer.ffn_gate.data, &layer.ffn_up.data,
                &mut scratch.ffn_gate_out, &mut scratch.ffn_up_out,
                config.ff_dim, config.dim, layer.ffn_gate.quant_type,
            ))?;
            if !used {
                // Fallback for non-Q4_K types
                timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
                    seq_len, config.ff_dim, config.dim))?;
                timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
                    seq_len, config.ff_dim, config.dim))?;
            }
        } else {
            timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
                seq_len, config.ff_dim, config.dim))?;
            timed!(t_gemm, gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
                seq_len, config.ff_dim, config.dim))?;
        }

        // 10. SiLU(gate) * up (f16)
        timed!(t_silu, kernels.ops.silu(
            &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
            seq_len * config.ff_dim,
        ))?;

        // 11. Quantize for down projection GEMV
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        }
        timed!(t_gemm, gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
            seq_len, config.dim, config.ff_dim))?;

        // 12. Residual + next layer's attn_norm (fused if not last layer)
        if layer_idx + 1 < n_layers {
            if seq_len == 1 {
                // Fused: add + RMSNorm + Q8_1 quantize for next layer's QKV
                let x_q8 = kernels.gemv.x_q8_mut();
                timed!(t_add, kernels.ops.fused_add_rmsnorm_q8(
                    hidden, &scratch.ffn_out,
                    &weights.layers[layer_idx + 1].attn_norm, &mut scratch.norm_out,
                    x_q8, seq_len, config.dim, config.rms_norm_eps,
                ))?;
            } else {
                // Fused: hidden += ffn_out, then RMSNorm with next layer's attn_norm
                timed!(t_add, kernels.ops.fused_add_rmsnorm(
                    hidden, &scratch.ffn_out,
                    &weights.layers[layer_idx + 1].attn_norm, &mut scratch.norm_out,
                    seq_len, config.dim, config.rms_norm_eps,
                ))?;
            }
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

/// Run the Gemma 4 transformer forward pass.
///
/// Key differences from Llama:
/// - Variable head_dim per layer (512 for full attn, 256 for SWA)
/// - QK norms after Q/K projection
/// - V norm (no weight) after V projection
/// - GELU instead of SiLU
/// - Post-attention and post-FFN norms (applied before residual add)
/// - Per-layer embedding (skipped for initial implementation)
/// - Layer output scale
/// - Shared KV layers
/// - RoPE NeoX (interleaved first/second half)
/// - Input embedding scaled by sqrt(dim)
/// - Logit softcapping (applied after logit projection, in engine)
/// Pre-computed per-layer token embeddings for the current batch.
/// Shape: [seq_len, n_layers * epl] in f16 — computed once per forward call.
/// At each layer l, the relevant slice is columns [l*epl : (l+1)*epl].
pub struct PerLayerEmbeddings {
    /// The full dequantized embedding data: [seq_len, n_layers * epl] in f16
    pub data: CudaSlice<half::f16>,
    /// Embedding dimension per layer
    pub epl: u32,
    /// Row width (n_layers * epl)
    pub row_width: u32,
    /// Number of tokens
    pub seq_len: u32,
}

impl PerLayerEmbeddings {
    /// Get a view for a specific layer's embeddings: [seq_len, epl] starting at offset.
    /// Returns (offset_in_elements, epl) for manual slicing since views need contiguous data.
    /// NOTE: The data is NOT contiguous per-layer (it's interleaved across layers in each row).
    /// We need to either:
    /// a) Restructure the data to be per-layer contiguous, or
    /// b) Use a strided copy/gather kernel
    ///
    /// Since each token's data is [l0_epl, l1_epl, ..., lN_epl], to get layer l's data
    /// we need elements at positions [tok_i * row_width + l * epl .. + (l+1)*epl] for each token.
    pub fn layer_offset(&self, layer_idx: usize) -> usize {
        layer_idx * self.epl as usize
    }
}

pub fn forward_gemma4(
    hidden: &mut CudaSlice<f32>,
    weights: &ModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    pe: Option<&PerLayerEmbeddings>,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;

    let stream_ref = Arc::clone(kernels.ops.stream());
    let _ = &stream_ref;

    let max_layers = std::env::var("CHEW_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(config.n_layers as usize);
    let n_layers = max_layers.min(config.n_layers as usize);

    for layer_idx in 0..n_layers {
        let layer = &weights.layers[layer_idx];
        let hd = config.layer_head_dim(layer_idx);
        let has_kv = config.has_kv(layer_idx);
        let rope_theta = config.layer_rope_theta(layer_idx);

        // Debug helper: check f16 buffer for NaN
        let dbg_layer = layer_idx <= 1 && seq_len > 1;
        macro_rules! check_nan {
            ($buf:expr, $n:expr, $label:expr) => {
                if dbg_layer {
                    let cnt = ($n as usize).min($buf.len()).min(256);
                    let mut d = vec![half::f16::ZERO; cnt];
                    let _ = stream_ref.memcpy_dtoh(&$buf.slice(0..cnt), &mut d);
                    let nan = d.iter().any(|x| x.to_f32().is_nan());
                    let mx = d.iter().map(|x| x.to_f32().abs()).filter(|x| x.is_finite()).fold(0f32, f32::max);
                    if nan { tracing::error!(layer=layer_idx, mx, "NaN in {}", $label); }
                    else { tracing::info!(layer=layer_idx, mx, "OK {}", $label); }
                }
            };
        }

        // 1. Attention norm: f32 hidden → f16 norm_out
        kernels.ops.rms_norm_f32in(
            hidden, &layer.attn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
        check_nan!(scratch.norm_out, seq_len * config.dim, "after_attn_norm");

        // DEEP DEBUG: dump values at key points in layer 0 for last position
        let deep_dbg = (layer_idx == 0 || layer_idx == 5) && seq_len > 1;
        if deep_dbg {
            let last_off = ((seq_len - 1) * config.dim) as usize;
            let mut d = vec![half::f16::ZERO; 8];
            let _ = stream_ref.memcpy_dtoh(&scratch.norm_out.slice(last_off..last_off+8), &mut d);
            let vals: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
            tracing::info!(?vals, "L0_NORM_OUT last pos [0:8]");
        }

        // 2. QKV projections
        let q_dim = config.n_heads * hd;
        let kv_dim = config.n_kv_heads * hd;

        // Quantize norm_out for GEMV (decode path, M=1)
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }

        gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
            seq_len, q_dim, config.dim)?;
        check_nan!(scratch.q, seq_len * q_dim, "after_Q_proj");
        if deep_dbg {
            let last_off = ((seq_len - 1) * q_dim) as usize;
            let mut d = vec![half::f16::ZERO; 8];
            let _ = stream_ref.memcpy_dtoh(&scratch.q.slice(last_off..last_off+8), &mut d);
            let vals: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
            tracing::info!(?vals, "L0_Q_PROJ last pos [0:8]");
        }
        gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
            seq_len, kv_dim, config.dim)?;
        check_nan!(scratch.k, seq_len * kv_dim, "after_K_proj");
        gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
            seq_len, kv_dim, config.dim)?;
        check_nan!(scratch.v, seq_len * kv_dim, "after_V_proj");

        // 3. QK norms (RMSNorm with learned weights)
        // The rms_norm kernel requires different src and dst pointers.
        // We use unsafe to allow in-place operation since it's element-wise.
        if let Some(ref q_norm) = layer.attn_q_norm {
            // Q norm: per-head norm, weight shape [head_dim]
            // Q is [seq_len, n_heads, head_dim] — treat as (seq_len * n_heads) rows of head_dim
            // SAFETY: rms_norm reads x[row,i] then writes out[row,i] — same buffer is safe.
            let src_ptr = &scratch.q as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.q as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm(
                    &*src_ptr, q_norm, &mut *dst_ptr,
                    seq_len * config.n_heads, hd, config.rms_norm_eps,
                )?;
            }
        }
        if let Some(ref k_norm) = layer.attn_k_norm {
            let src_ptr = &scratch.k as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.k as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm(
                    &*src_ptr, k_norm, &mut *dst_ptr,
                    seq_len * config.n_kv_heads, hd, config.rms_norm_eps,
                )?;
            }
        }

        // V norm (no weight — just normalize by RMS)
        {
            let src_ptr = &scratch.v as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.v as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm_no_weight(
                    &*src_ptr, &mut *dst_ptr,
                    seq_len * config.n_kv_heads, hd, config.rms_norm_eps,
                )?;
            }
        }

        check_nan!(scratch.q, seq_len * q_dim, "after_QK_norm");
        check_nan!(scratch.k, seq_len * kv_dim, "after_K_norm");
        check_nan!(scratch.v, seq_len * kv_dim, "after_V_norm");

        if deep_dbg {
            let last_off = ((seq_len - 1) * q_dim) as usize;
            let mut d = vec![half::f16::ZERO; 8];
            let _ = stream_ref.memcpy_dtoh(&scratch.q.slice(last_off..last_off+8), &mut d);
            let vals: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
            tracing::info!(?vals, "L0_Q_NORM last pos [0:8]");
        }

        // 4. RoPE NeoX on Q and K
        // Full attention layers use frequency factors (only rotate 128/512 dims).
        // SWA layers use standard RoPE (all 256 dims rotated).
        let is_swa = config.is_swa(layer_idx);
        if !is_swa {
            if let Some(ref ff) = weights.rope_freq_factors {
                kernels.ops.rope_neox_freqs(&mut scratch.q, ff, seq_len, config.n_heads, hd, pos, rope_theta)?;
                kernels.ops.rope_neox_freqs(&mut scratch.k, ff, seq_len, config.n_kv_heads, hd, pos, rope_theta)?;
            } else {
                kernels.ops.rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
                kernels.ops.rope_neox(&mut scratch.k, seq_len, config.n_kv_heads, hd, pos, rope_theta)?;
            }
        } else {
            kernels.ops.rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
            kernels.ops.rope_neox(&mut scratch.k, seq_len, config.n_kv_heads, hd, pos, rope_theta)?;
        }
        check_nan!(scratch.q, seq_len * q_dim, "after_RoPE_Q");
        check_nan!(scratch.k, seq_len * kv_dim, "after_RoPE_K");

        if deep_dbg {
            let last_off = ((seq_len - 1) * q_dim) as usize;
            let mut d = vec![half::f16::ZERO; 8];
            let _ = stream_ref.memcpy_dtoh(&scratch.q.slice(last_off..last_off+8), &mut d);
            let vals: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
            tracing::info!(?vals, "L0_Q_ROPE last pos [0:8]");
        }

        // 5. Write K, V into KV cache
        // For KV-owning layers: write to own cache slot.
        // For shared layers: skip K/V write — reuse earlier layer's cache.
        let kv_source = config.kv_source_layer(layer_idx);
        if has_kv {
            let kv_elems = seq_len * kv_dim;
            {
                let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
                kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
            }
            {
                let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
                kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
            }
        }

        // 6. Multi-Head Attention
        // Shared layers reuse KV from kv_source_layer.
        {
            let k_full = kv_cache.k_full(kv_source, total_kv_len);
            let v_full = kv_cache.v_full(kv_source, total_kv_len);

            // Gemma 4: attention_scale = 1.0 (Q and K are already RMS-normed)
            kernels.ops.mha_fused_scaled(
                &scratch.q, &k_full, &v_full, &mut scratch.attn_mha_out,
                hd, config.n_heads, config.n_kv_heads,
                seq_len, total_kv_len, pos,
                config.attention_scale,
            )?;
        }
        check_nan!(scratch.attn_mha_out, seq_len * q_dim, "after_MHA");

        // 7. Output projection
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.attn_mha_out, q_dim)?;
        }
        gemm_q(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
            seq_len, config.dim, q_dim)?;

        check_nan!(scratch.attn_out, seq_len * config.dim, "after_output_proj");

        // 8. Post-attention norm: attn_out = rmsnorm(attn_out) * weight
        if let Some(ref pan) = layer.post_attention_norm {
            kernels.ops.post_norm_add(
                hidden, &scratch.attn_out, pan, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
        } else {
            // No post-norm: just add to residual
            kernels.ops.add_inplace_f32_f16(hidden, &scratch.attn_out, seq_len * config.dim)?;
        }

        // 9. FFN norm
        kernels.ops.rms_norm_f32in(
            hidden, &layer.ffn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;

        // 10. Gate + Up projections
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }
        gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
            seq_len, config.ff_dim, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
            seq_len, config.ff_dim, config.dim)?;

        // 11. GELU(gate) * up (Gemma 4 uses GELU instead of SiLU)
        kernels.ops.gelu(
            &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
            seq_len * config.ff_dim,
        )?;

        // 12. Down projection
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        }
        gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
            seq_len, config.dim, config.ff_dim)?;

        check_nan!(scratch.ffn_out, seq_len * config.dim, "after_ffn_down");

        // 13. Post-FFN norm
        if let Some(ref pfn) = layer.post_ffw_norm {
            kernels.ops.post_norm_add(
                hidden, &scratch.ffn_out, pfn, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.add_inplace_f32_f16(hidden, &scratch.ffn_out, seq_len * config.dim)?;
        }

        // Check hidden state after post-FFN norm
        if dbg_layer {
            let mut d = vec![0.0f32; 8];
            let _ = stream_ref.memcpy_dtoh(&hidden.slice(0..8), &mut d);
            let nan = d.iter().any(|x| x.is_nan());
            let mx = d.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            tracing::info!(layer=layer_idx, nan, mx, "hidden after post-ffn-norm+scale");
        }

        // DEBUG: check for NaN after each layer
        if seq_len > 1 {
            let n = (seq_len * config.dim) as usize;
            let mut dbg = vec![0.0f32; n.min(8)];
            let _ = stream_ref.memcpy_dtoh(&hidden.slice(0..dbg.len()), &mut dbg);
            let has_nan = dbg.iter().any(|x| x.is_nan());
            let maxv = dbg.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            tracing::info!(layer = layer_idx, has_nan, maxv, first4 = ?&dbg[..4], "layer check");
        }

        // 14. Per-layer embedding (Gemma 4)
        // Steps: hidden @ inp_gate → GELU → mul(layer_embd) → proj → rmsnorm → residual add
        if let (Some(inp_gate), Some(proj), Some(post_norm), Some(pe_data), Some(epl)) = (
            &layer.inp_gate, &layer.proj, &layer.post_norm, pe, config.embd_per_layer,
        ) {
            let pe_gate = scratch.pe_gate_out.as_mut().expect("pe_gate_out not allocated");
            let pe_proj = scratch.pe_proj_out.as_mut().expect("pe_proj_out not allocated");

            // 14a. Convert hidden (f32) → f16 norm_out for matmul input
            let n_elems_pe = (seq_len * config.dim) as usize;
            {
                let mut norm_view = scratch.norm_out.slice_mut(0..n_elems_pe);
                kernels.ops.copy_f32_to_f16(hidden, &mut norm_view, seq_len * config.dim)?;
            }

            // 14b. hidden_f16 @ inp_gate^T → pe_gate [seq_len, epl]
            if seq_len == 1 {
                kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
            }
            gemm_q(kernels, &scratch.norm_out, inp_gate, pe_gate,
                seq_len, epl, config.dim)?;

            // 14c. GELU(pe_gate) in-place
            {
                let src_ptr = pe_gate as *const CudaSlice<half::f16>;
                let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
                unsafe {
                    kernels.ops.gelu_act(&*src_ptr, &mut *dst_ptr, seq_len * epl)?;
                }
            }

            // 14d. Strided multiply with per-layer token embedding
            // pe_data.data: [seq_len, row_width] where row_width = n_layers * epl
            // For layer l, we need columns [l*epl : (l+1)*epl] for each token.
            // pe_strided_mul handles the strided access in one kernel launch.
            {
                let src_ptr = pe_gate as *const CudaSlice<half::f16>;
                let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
                let layer_off = (layer_idx as u32) * epl;
                unsafe {
                    kernels.ops.pe_strided_mul(
                        &*src_ptr, &pe_data.data, &mut *dst_ptr,
                        epl, pe_data.row_width, layer_off, seq_len,
                    )?;
                }
            }

            // DEBUG: dump pe_gate after strided multiply for layer 0
            if layer_idx == 0 && seq_len > 1 {
                let last_off = ((seq_len - 1) * epl) as usize;
                let mut d = vec![half::f16::ZERO; 8];
                let _ = stream_ref.memcpy_dtoh(&pe_gate.slice(last_off..last_off+8), &mut d);
                let vals: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
                tracing::info!(?vals, "PE_GATE_MUL L0 last pos [0:8]");
                // Also dump pe_data values at layer 0 for last token
                let pe_off = (9 * pe_data.row_width) as usize; // last token, start of row
                let mut pd = vec![half::f16::ZERO; 8];
                let _ = stream_ref.memcpy_dtoh(&pe_data.data.slice(pe_off..pe_off+8), &mut pd);
                let pvals: Vec<f32> = pd.iter().map(|x| x.to_f32()).collect();
                tracing::info!(?pvals, "PE_DATA L0 last token [0:8]");
            }

            // 14e. pe_gate @ proj^T → pe_proj [seq_len, dim]
            if seq_len == 1 {
                kernels.gemv.quantize_input(pe_gate, epl)?;
            }
            gemm_q(kernels, pe_gate, proj, pe_proj,
                seq_len, config.dim, epl)?;

            // 14f. RMSNorm + residual add: hidden += rmsnorm(pe_proj) * post_norm
            kernels.ops.post_norm_add(
                hidden, pe_proj, post_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
        }

        // 15. Layer output scale
        if let Some(scale) = layer.layer_output_scale {
            if (scale - 1.0).abs() > 1e-6 {
                kernels.ops.scale_f32_inplace(hidden, seq_len * config.dim, scale)?;
            }
        }

        // Debug: dump last position hidden state after specific layers
        if (layer_idx == 0 || layer_idx == 5 || layer_idx == 23 || layer_idx == 41) && seq_len > 1 {
            let last_off = ((seq_len - 1) * config.dim) as usize;
            let mut dbg = vec![0.0f32; 8];
            let _ = stream_ref.memcpy_dtoh(&hidden.slice(last_off..last_off+8), &mut dbg);
            tracing::info!(?dbg, layer = layer_idx, "LAYER_DONE last pos hidden[0:8]");
        }
    }

    // === Final norm + logits ===
    kernels.ops.rms_norm_f32in(
        hidden, &weights.output_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    // DEBUG: check norm_out for NaN
    {
        let mut dbg = vec![half::f16::ZERO; 8];
        let _ = stream_ref.memcpy_dtoh(&scratch.norm_out.slice(0..8), &mut dbg);
        let has_nan = dbg.iter().any(|x| x.to_f32().is_nan());
        let maxv = dbg.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
        tracing::info!(has_nan, maxv, "final_norm_out check");
    }

    // Output logits
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }
    gemm_q(kernels, &scratch.norm_out, &weights.output, &mut scratch.logits,
        seq_len, config.vocab_size, config.dim)?;

    // DEBUG: check logits for NaN (both first and last position)
    {
        let mut dbg0 = vec![half::f16::ZERO; 8];
        let _ = stream_ref.memcpy_dtoh(&scratch.logits.slice(0..8), &mut dbg0);
        let nan0 = dbg0.iter().any(|x| x.to_f32().is_nan());
        let max0 = dbg0.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);

        let last_off = ((seq_len - 1) * config.vocab_size) as usize;
        let end_off = last_off + config.vocab_size as usize;
        // Check: does the buffer have enough space?
        let buf_len = scratch.logits.len();
        let mut dbg_last = vec![half::f16::ZERO; 8];
        if end_off <= buf_len {
            let _ = stream_ref.memcpy_dtoh(&scratch.logits.slice(last_off..last_off+8), &mut dbg_last);
        } else {
            tracing::error!(last_off, end_off, buf_len, "LOGIT BUFFER OVERFLOW!");
        }
        let nan_last = dbg_last.iter().any(|x| x.to_f32().is_nan());
        let max_last = dbg_last.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);

        tracing::info!(nan0, max0, nan_last, max_last, seq_len, vocab=config.vocab_size, last_off, "logits check");
    }

    // Logit softcapping: tanh(logits / cap) * cap
    // The kernel supports different src/dst, but we want in-place.
    // We'll modify the CUDA kernel to support in-place by checking pointers,
    // or just make it write to the same buffer (CUDA allows this for element-wise).
    // Actually, the borrow checker issue is Rust-side. The kernel reads idx then writes idx,
    // so same-buffer is safe. Use unsafe to work around the borrow checker.
    if let Some(cap) = config.logit_softcap {
        let n_logits = seq_len * config.vocab_size;
        kernels.ops.logit_softcap_inplace(&mut scratch.logits, n_logits, cap)?;
    }

    kv_cache.advance(seq_len);

    Ok(())
}

// =============================================================
// Streaming forward pass — layers partially on host RAM
// =============================================================

/// Upload a streamed layer's weight data from host buffer into a GPU shell.
fn upload_layer_to_shell_direct(
    host_data: &[u8],
    layout: &HostLayerLayout,
    shell: &mut LayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    let base = layout.offset;

    fn upload_slot(host_data: &[u8], base: usize, slot: &TensorSlot, qw: &mut crate::weights::QuantWeight, stream: &Arc<CudaStream>) -> Result<(), String> {
        let src = &host_data[base + slot.off..base + slot.off + slot.size];
        stream.memcpy_htod(src, &mut qw.data)
            .map_err(|e| e.to_string())?;
        qw.quant_type = slot.quant_type;
        qw.n_elements = slot.n_elements;
        Ok(())
    }

    upload_slot(host_data, base, &layout.attn_q, &mut shell.attn_q, stream)?;
    upload_slot(host_data, base, &layout.attn_k, &mut shell.attn_k, stream)?;
    upload_slot(host_data, base, &layout.attn_v, &mut shell.attn_v, stream)?;
    upload_slot(host_data, base, &layout.attn_output, &mut shell.attn_output, stream)?;
    upload_slot(host_data, base, &layout.ffn_gate, &mut shell.ffn_gate, stream)?;
    upload_slot(host_data, base, &layout.ffn_up, &mut shell.ffn_up, stream)?;
    upload_slot(host_data, base, &layout.ffn_down, &mut shell.ffn_down, stream)?;

    if let Some(ref slot) = layout.inp_gate {
        if let Some(ref mut qw) = shell.inp_gate {
            upload_slot(host_data, base, slot, qw, stream)?;
        }
    }
    if let Some(ref slot) = layout.proj {
        if let Some(ref mut qw) = shell.proj {
            upload_slot(host_data, base, slot, qw, stream)?;
        }
    }
    shell.layer_output_scale = layout.layer_output_scale;

    Ok(())
}

/// Copy norm weights from always-resident norms into a shell.
fn copy_norms_to_shell_direct(
    norms: &StreamingLayerNorms,
    shell: &mut LayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    stream.memcpy_dtod(&norms.attn_norm, &mut shell.attn_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.ffn_norm, &mut shell.ffn_norm).map_err(|e| e.to_string())?;

    if let (Some(src), Some(dst)) = (&norms.attn_q_norm, &mut shell.attn_q_norm) {
        stream.memcpy_dtod(src, dst).map_err(|e| e.to_string())?;
    }
    if let (Some(src), Some(dst)) = (&norms.attn_k_norm, &mut shell.attn_k_norm) {
        stream.memcpy_dtod(src, dst).map_err(|e| e.to_string())?;
    }
    if let (Some(src), Some(dst)) = (&norms.post_attention_norm, &mut shell.post_attention_norm) {
        stream.memcpy_dtod(src, dst).map_err(|e| e.to_string())?;
    }
    if let (Some(src), Some(dst)) = (&norms.post_ffw_norm, &mut shell.post_ffw_norm) {
        stream.memcpy_dtod(src, dst).map_err(|e| e.to_string())?;
    }
    if let (Some(src), Some(dst)) = (&norms.post_norm, &mut shell.post_norm) {
        stream.memcpy_dtod(src, dst).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Streaming forward pass: resident layers from GPU, streamed layers DMA'd on demand.
///
/// This function handles both Llama-style and Gemma4-style models by checking
/// config flags. It processes layers one at a time, uploading streamed layer
/// weights from host RAM into a double-buffered GPU shell before execution.
pub fn forward_streaming(
    hidden: &mut CudaSlice<f32>,
    weights: &mut WeightStorage,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    pe: Option<&PerLayerEmbeddings>,
    stream: &Arc<CudaStream>,
) -> Result<(), KernelError> {
    let sw = match weights {
        WeightStorage::Streaming(s) => s,
        _ => panic!("forward_streaming called with non-Streaming weights"),
    };

    let is_gemma4 = config.is_gemma4();
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;
    let n_elems = seq_len * config.dim;

    let max_layers = std::env::var("CHEW_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(config.n_layers as usize);
    let n_layers = max_layers.min(config.n_layers as usize);
    let n_resident = sw.n_resident;

    // For Llama-style: layer 0 separate RMSNorm
    if !is_gemma4 && n_layers > 0 {
        let norm = &sw.layer_norms[0].attn_norm;
        if seq_len == 1 {
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.rms_norm_f32in_q8(
                hidden, norm, &mut scratch.norm_out,
                x_q8, seq_len, config.dim, config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.rms_norm_f32in(
                hidden, norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
        }
    }

    // Track which shell is "active" for double buffering
    let mut use_shell_a = true;

    for layer_idx in 0..n_layers {
        // Get layer weights: either resident or streamed via shell
        let layer: &LayerWeights = if layer_idx < n_resident {
            &sw.resident_layers[layer_idx]
        } else {
            // Upload streamed layer into the active shell
            let streamed_idx = layer_idx - n_resident;
            let layout = &sw.host_layer_offsets[streamed_idx];
            let norms = &sw.layer_norms[layer_idx];

            {
                let shell = if use_shell_a { &mut sw.shell_a } else { &mut sw.shell_b };

                // DMA: host → GPU (synchronous for now, async double-buffer later)
                upload_layer_to_shell_direct(
                    &sw.host_layer_data, layout, shell, stream,
                ).map_err(|e| KernelError::Launch(format!("streaming upload: {e}")))?;
                copy_norms_to_shell_direct(norms, shell, stream)
                    .map_err(|e| KernelError::Launch(format!("norm copy: {e}")))?;
            }

            // Ensure DMA is complete before compute
            stream.synchronize()
                .map_err(|e| KernelError::Launch(format!("stream sync: {e}")))?;

            use_shell_a = !use_shell_a;

            if use_shell_a { &sw.shell_b } else { &sw.shell_a }
        };

        // Get next layer's norm for fused operations (Llama path)
        let next_attn_norm: Option<&CudaSlice<half::f16>> = if layer_idx + 1 < n_layers {
            Some(&sw.layer_norms[layer_idx + 1].attn_norm)
        } else {
            None
        };

        if is_gemma4 {
            forward_streaming_layer_gemma4(
                hidden, layer, config, kernels, kv_cache, scratch,
                seq_len, layer_idx, pos, total_kv_len, pe, sw,
            )?;
        } else {
            forward_streaming_layer_llama(
                hidden, layer, config, kernels, kv_cache, scratch,
                seq_len, layer_idx, n_layers, next_attn_norm, n_elems,
            )?;
        }
    }

    // === Final norm + logits ===
    kernels.ops.rms_norm_f32in(
        hidden, &sw.output_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }
    gemm_q(kernels, &scratch.norm_out, &sw.output, &mut scratch.logits,
        seq_len, config.vocab_size, config.dim)?;

    // Logit softcapping (Gemma4)
    if let Some(cap) = config.logit_softcap {
        let n_logits = seq_len * config.vocab_size;
        kernels.ops.logit_softcap_inplace(&mut scratch.logits, n_logits, cap)?;
    }

    kv_cache.advance(seq_len);
    Ok(())
}

/// Process a single Llama-style layer in the streaming forward pass.
fn forward_streaming_layer_llama(
    hidden: &mut CudaSlice<f32>,
    layer: &LayerWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    layer_idx: usize,
    _n_layers: usize,
    next_attn_norm: Option<&CudaSlice<half::f16>>,
    n_elems: u32,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;

    // QKV projections (norm_out already computed by previous layer's fused op)
    if seq_len == 1 && layer_idx > 0 {
        // x_q8 already set from previous fused op
    }
    gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
        seq_len, config.n_heads * config.head_dim, config.dim)?;
    if seq_len == 1 && layer.attn_k.quant_type == layer.attn_v.quant_type {
        let nk = config.n_kv_heads * config.head_dim;
        let used = kernels.gemv.gemv_dual(
            &layer.attn_k.data, &layer.attn_v.data,
            &mut scratch.k, &mut scratch.v,
            nk, config.dim, layer.attn_k.quant_type,
        )?;
        if !used {
            gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
                seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
            gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
                seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
        }
    } else {
        gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
            seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
            seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
    }

    // RoPE
    kernels.ops.rope(&mut scratch.q, seq_len, config.n_heads, config.head_dim, pos, config.rope_theta)?;
    kernels.ops.rope(&mut scratch.k, seq_len, config.n_kv_heads, config.head_dim, pos, config.rope_theta)?;

    // Write KV cache
    let kv_elems = seq_len * config.n_kv_heads * config.head_dim;
    {
        let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
        kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
    }
    {
        let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
        kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
    }

    // MHA
    {
        let k_full = kv_cache.k_full(layer_idx, total_kv_len);
        let v_full = kv_cache.v_full(layer_idx, total_kv_len);
        kernels.ops.mha_fused(
            &scratch.q, &k_full, &v_full, &mut scratch.attn_mha_out,
            config.head_dim, config.n_heads, config.n_kv_heads,
            seq_len, total_kv_len, pos,
        )?;
    }

    // Output projection
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.attn_mha_out, config.n_heads * config.head_dim)?;
    }
    gemm_q(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
        seq_len, config.dim, config.n_heads * config.head_dim)?;

    // Fused add + RMSNorm for FFN
    if seq_len == 1 {
        let x_q8 = kernels.gemv.x_q8_mut();
        kernels.ops.fused_add_rmsnorm_q8(
            hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
            x_q8, seq_len, config.dim, config.rms_norm_eps,
        )?;
    } else {
        kernels.ops.fused_add_rmsnorm(
            hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
    }

    // FFN: gate + up
    if seq_len == 1 && layer.ffn_gate.quant_type == layer.ffn_up.quant_type {
        let used = kernels.gemv.gemv_dual(
            &layer.ffn_gate.data, &layer.ffn_up.data,
            &mut scratch.ffn_gate_out, &mut scratch.ffn_up_out,
            config.ff_dim, config.dim, layer.ffn_gate.quant_type,
        )?;
        if !used {
            gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
                seq_len, config.ff_dim, config.dim)?;
            gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
                seq_len, config.ff_dim, config.dim)?;
        }
    } else {
        gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
            seq_len, config.ff_dim, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
            seq_len, config.ff_dim, config.dim)?;
    }

    // SiLU(gate) * up
    kernels.ops.silu(
        &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
        seq_len * config.ff_dim,
    )?;

    // Down projection
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
    }
    gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
        seq_len, config.dim, config.ff_dim)?;

    // Residual + next layer's attn_norm
    if let Some(next_norm) = next_attn_norm {
        if seq_len == 1 {
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.fused_add_rmsnorm_q8(
                hidden, &scratch.ffn_out, next_norm, &mut scratch.norm_out,
                x_q8, seq_len, config.dim, config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.fused_add_rmsnorm(
                hidden, &scratch.ffn_out, next_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
        }
    } else {
        // Last layer
        kernels.ops.add_inplace_f32_f16(hidden, &scratch.ffn_out, n_elems)?;
    }

    Ok(())
}

/// Process a single Gemma4-style layer in the streaming forward pass.
fn forward_streaming_layer_gemma4(
    hidden: &mut CudaSlice<f32>,
    layer: &LayerWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    layer_idx: usize,
    pos: u32,
    total_kv_len: u32,
    pe: Option<&PerLayerEmbeddings>,
    sw: &StreamingWeights,
) -> Result<(), KernelError> {
    let hd = config.layer_head_dim(layer_idx);
    let has_kv = config.has_kv(layer_idx);
    let rope_theta = config.layer_rope_theta(layer_idx);

    // 1. Attention norm
    kernels.ops.rms_norm_f32in(
        hidden, &layer.attn_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    // 2. QKV projections
    let q_dim = config.n_heads * hd;
    let kv_dim = config.n_kv_heads * hd;

    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }

    gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
        seq_len, q_dim, config.dim)?;
    gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
        seq_len, kv_dim, config.dim)?;
    gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
        seq_len, kv_dim, config.dim)?;

    // 3. QK norms
    if let Some(ref q_norm) = layer.attn_q_norm {
        let src_ptr = &scratch.q as *const CudaSlice<half::f16>;
        let dst_ptr = &mut scratch.q as *mut CudaSlice<half::f16>;
        unsafe {
            kernels.ops.rms_norm(
                &*src_ptr, q_norm, &mut *dst_ptr,
                seq_len * config.n_heads, hd, config.rms_norm_eps,
            )?;
        }
    }
    if let Some(ref k_norm) = layer.attn_k_norm {
        let src_ptr = &scratch.k as *const CudaSlice<half::f16>;
        let dst_ptr = &mut scratch.k as *mut CudaSlice<half::f16>;
        unsafe {
            kernels.ops.rms_norm(
                &*src_ptr, k_norm, &mut *dst_ptr,
                seq_len * config.n_kv_heads, hd, config.rms_norm_eps,
            )?;
        }
    }

    // V norm (no weight)
    {
        let src_ptr = &scratch.v as *const CudaSlice<half::f16>;
        let dst_ptr = &mut scratch.v as *mut CudaSlice<half::f16>;
        unsafe {
            kernels.ops.rms_norm_no_weight(
                &*src_ptr, &mut *dst_ptr,
                seq_len * config.n_kv_heads, hd, config.rms_norm_eps,
            )?;
        }
    }

    // 4. RoPE NeoX
    let is_swa = config.is_swa(layer_idx);
    if !is_swa {
        if let Some(ref ff) = sw.rope_freq_factors {
            kernels.ops.rope_neox_freqs(&mut scratch.q, ff, seq_len, config.n_heads, hd, pos, rope_theta)?;
            kernels.ops.rope_neox_freqs(&mut scratch.k, ff, seq_len, config.n_kv_heads, hd, pos, rope_theta)?;
        } else {
            kernels.ops.rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
            kernels.ops.rope_neox(&mut scratch.k, seq_len, config.n_kv_heads, hd, pos, rope_theta)?;
        }
    } else {
        kernels.ops.rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
        kernels.ops.rope_neox(&mut scratch.k, seq_len, config.n_kv_heads, hd, pos, rope_theta)?;
    }

    // 5. Write KV cache
    let kv_source = config.kv_source_layer(layer_idx);
    if has_kv {
        let kv_elems = seq_len * kv_dim;
        {
            let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
            kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
        }
        {
            let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
            kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
        }
    }

    // 6. MHA
    {
        let k_full = kv_cache.k_full(kv_source, total_kv_len);
        let v_full = kv_cache.v_full(kv_source, total_kv_len);
        kernels.ops.mha_fused_scaled(
            &scratch.q, &k_full, &v_full, &mut scratch.attn_mha_out,
            hd, config.n_heads, config.n_kv_heads,
            seq_len, total_kv_len, pos,
            config.attention_scale,
        )?;
    }

    // 7. Output projection
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.attn_mha_out, q_dim)?;
    }
    gemm_q(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
        seq_len, config.dim, q_dim)?;

    // 8. Post-attention norm + residual
    if let Some(ref pan) = layer.post_attention_norm {
        kernels.ops.post_norm_add(
            hidden, &scratch.attn_out, pan, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
    } else {
        kernels.ops.add_inplace_f32_f16(hidden, &scratch.attn_out, seq_len * config.dim)?;
    }

    // 9. FFN norm
    kernels.ops.rms_norm_f32in(
        hidden, &layer.ffn_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    // 10. Gate + Up
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }
    gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
        seq_len, config.ff_dim, config.dim)?;
    gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
        seq_len, config.ff_dim, config.dim)?;

    // 11. GELU(gate) * up
    kernels.ops.gelu(
        &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
        seq_len * config.ff_dim,
    )?;

    // 12. Down projection
    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
    }
    gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
        seq_len, config.dim, config.ff_dim)?;

    // 13. Post-FFN norm + residual
    if let Some(ref pfn) = layer.post_ffw_norm {
        kernels.ops.post_norm_add(
            hidden, &scratch.ffn_out, pfn, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
    } else {
        kernels.ops.add_inplace_f32_f16(hidden, &scratch.ffn_out, seq_len * config.dim)?;
    }

    // 14. Per-layer embedding (Gemma 4)
    if let (Some(inp_gate), Some(proj), Some(post_norm), Some(pe_data), Some(epl)) = (
        &layer.inp_gate, &layer.proj, &layer.post_norm, pe, config.embd_per_layer,
    ) {
        let pe_gate = scratch.pe_gate_out.as_mut().expect("pe_gate_out not allocated");
        let pe_proj = scratch.pe_proj_out.as_mut().expect("pe_proj_out not allocated");

        let n_elems_pe = (seq_len * config.dim) as usize;
        {
            let mut norm_view = scratch.norm_out.slice_mut(0..n_elems_pe);
            kernels.ops.copy_f32_to_f16(hidden, &mut norm_view, seq_len * config.dim)?;
        }

        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }
        gemm_q(kernels, &scratch.norm_out, inp_gate, pe_gate,
            seq_len, epl, config.dim)?;

        {
            let src_ptr = pe_gate as *const CudaSlice<half::f16>;
            let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.gelu_act(&*src_ptr, &mut *dst_ptr, seq_len * epl)?;
            }
        }

        {
            let src_ptr = pe_gate as *const CudaSlice<half::f16>;
            let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
            let layer_off = (layer_idx as u32) * epl;
            unsafe {
                kernels.ops.pe_strided_mul(
                    &*src_ptr, &pe_data.data, &mut *dst_ptr,
                    epl, pe_data.row_width, layer_off, seq_len,
                )?;
            }
        }

        if seq_len == 1 {
            kernels.gemv.quantize_input(pe_gate, epl)?;
        }
        gemm_q(kernels, pe_gate, proj, pe_proj,
            seq_len, config.dim, epl)?;

        kernels.ops.post_norm_add(
            hidden, pe_proj, post_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
    }

    // 15. Layer output scale
    if let Some(scale) = layer.layer_output_scale {
        if (scale - 1.0).abs() > 1e-6 {
            kernels.ops.scale_f32_inplace(hidden, seq_len * config.dim, scale)?;
        }
    }

    Ok(())
}

// =============================================================
// CUDA Graph capture and replay for decode (seq_len=1)
// =============================================================

/// Captured CUDA graph for the decode forward pass.
///
/// The graph captures all GPU operations from the forward pass
/// (layers 0..N + final norm + logit projection + argmax).
/// Dynamic per-step values (pos, kv_len, kv_offset) are stored
/// in a device buffer that kernels read from, so the graph can
/// be replayed without re-capture.
pub struct DecodeGraph {
    /// The instantiated executable graph.
    instance: sys::CUgraphExec,
    /// Device buffer for dynamic decode parameters.
    /// Layout: [pos, kv_len, kv_offset]
    pub decode_params: CudaSlice<i32>,
    /// Host-side copy of params (kept alive for async memcpy safety).
    params_host: [i32; 3],
    /// Max KV length this graph was captured for (shared memory ceiling).
    _max_kv_len: u32,
    /// Raw stream handle for graph launch.
    raw_stream: sys::CUstream,
}

// SAFETY: CUgraph and CUgraphExec handles are GPU resources
// bound to a specific device context. We only access them from
// the same thread that created them (engine is single-threaded).
unsafe impl Send for DecodeGraph {}

impl DecodeGraph {
    /// Capture a decode forward pass into a CUDA graph.
    ///
    /// Runs the forward pass once (during capture, kernels are recorded but not executed),
    /// then instantiates the graph for fast replay.
    ///
    /// `pos` is the current position for this first captured step.
    pub fn capture(
        hidden: &mut CudaSlice<f32>,
        weights: &ModelWeights,
        config: &ModelConfig,
        kernels: &mut GpuKernels,
        kv_cache: &mut KvCache,
        scratch: &mut ScratchBuffers,
        pos: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, KernelError> {
        let raw_stream = stream.cu_stream;
        let max_kv_len = kv_cache.max_seq();

        // Allocate persistent decode_params on device: [pos, kv_len, kv_offset]
        let kv_stride = kv_cache.kv_stride();
        let total_kv_len = pos + 1; // decode: seq_len=1
        let kv_offset = pos * kv_stride;
        let params_host = [pos as i32, total_kv_len as i32, kv_offset as i32];
        let mut decode_params = stream.alloc_zeros::<i32>(3)
            .map_err(|e| KernelError::Launch(e.to_string()))?;
        stream.memcpy_htod(&params_host, &mut decode_params)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        // Synchronize before capture to ensure param upload is complete
        stream.synchronize().map_err(|e| KernelError::Launch(e.to_string()))?;

        // Begin stream capture
        unsafe {
            cudarc::driver::result::stream::begin_capture(
                raw_stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            ).map_err(|e| KernelError::Launch(format!("begin_capture: {e}")))?;
        }

        // Run the graph-compatible forward pass (this records, not executes)
        let result = forward_for_graph(
            hidden, weights, config, kernels, kv_cache, scratch,
            &decode_params, max_kv_len,
        );

        // End capture regardless of forward result
        let graph = unsafe {
            cudarc::driver::result::stream::end_capture(raw_stream)
                .map_err(|e| KernelError::Launch(format!("end_capture: {e}")))?
        };

        // Check forward result after ending capture
        result?;

        // Instantiate the graph
        let instance = unsafe {
            cudarc::driver::result::graph::instantiate(
                graph,
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            ).map_err(|e| KernelError::Launch(format!("graph instantiate: {e}")))?
        };

        // Now actually execute the first step by launching the graph
        unsafe {
            cudarc::driver::result::graph::launch(instance, raw_stream)
                .map_err(|e| KernelError::Launch(format!("graph launch: {e}")))?;
        }

        // Advance KV cache (the graph executed the forward pass)
        kv_cache.advance(1);

        info!(pos, max_kv_len, "CUDA graph captured and launched");

        Ok(Self {
            instance,
            decode_params,
            params_host: params_host,
            _max_kv_len: max_kv_len,
            raw_stream,
        })
    }

    /// Update decode parameters and replay the captured graph.
    ///
    /// This is the hot path — a single memcpy + a single graph launch
    /// replaces ~300 individual kernel launches.
    pub fn replay(
        &mut self,
        pos: u32,
        kv_stride: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<(), KernelError> {
        let total_kv_len = pos + 1;
        let kv_offset = pos * kv_stride;
        // Update persistent host buffer (stays alive for async memcpy)
        self.params_host = [pos as i32, total_kv_len as i32, kv_offset as i32];

        // Upload new parameters (async — will complete before graph kernels read them
        // because they're on the same stream)
        stream.memcpy_htod(&self.params_host, &mut self.decode_params)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        // Replay the graph
        unsafe {
            cudarc::driver::result::graph::launch(self.instance, self.raw_stream)
                .map_err(|e| KernelError::Launch(format!("graph replay: {e}")))?;
        }

        Ok(())
    }
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::result::graph::exec_destroy(self.instance);
            // graph was freed by AUTO_FREE_ON_LAUNCH on first launch
        }
    }
}

/// Forward pass using graph-compatible kernels.
///
/// Uses `decode_params` device buffer for pos/kv_len/kv_offset instead of
/// scalar kernel arguments. Uses base KV cache pointers with offset writes
/// instead of pre-sliced views.
///
/// Only supports seq_len=1 (decode mode).
fn forward_for_graph(
    hidden: &mut CudaSlice<f32>,
    weights: &ModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    decode_params: &CudaSlice<i32>,
    max_kv_len: u32,
) -> Result<(), KernelError> {
    let seq_len = 1u32;
    let n_elems = config.dim;

    let n_layers = config.n_layers as usize;

    // Layer 0: initial RMSNorm + Q8_1 quantize (fused)
    {
        let x_q8 = kernels.gemv.x_q8_mut();
        kernels.ops.rms_norm_f32in_q8(
            hidden, &weights.layers[0].attn_norm, &mut scratch.norm_out,
            x_q8, seq_len, config.dim, config.rms_norm_eps,
        )?;
    }

    for layer_idx in 0..n_layers {
        let layer = &weights.layers[layer_idx];

        // QKV: Q separate + K+V dual (fused, x_q8 already set)
        gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
            seq_len, config.n_heads * config.head_dim, config.dim)?;
        if layer.attn_k.quant_type == layer.attn_v.quant_type {
            let nk = config.n_kv_heads * config.head_dim;
            let _ = kernels.gemv.gemv_dual(
                &layer.attn_k.data, &layer.attn_v.data,
                &mut scratch.k, &mut scratch.v,
                nk, config.dim, layer.attn_k.quant_type,
            )?;
        } else {
            gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
                seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
            gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
                seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
        }

        // RoPE (graph-compatible)
        kernels.ops.rope_graph(
            &mut scratch.q, decode_params, seq_len, config.n_heads, config.head_dim, config.rope_theta,
        )?;
        kernels.ops.rope_graph(
            &mut scratch.k, decode_params, seq_len, config.n_kv_heads, config.head_dim, config.rope_theta,
        )?;

        // KV cache write (graph-compatible offset-based copy)
        let kv_elems = seq_len * config.n_kv_heads * config.head_dim;
        {
            let k_base = kv_cache.k_base_mut(layer_idx);
            kernels.ops.copy_f16_with_offset(&scratch.k, k_base, decode_params, kv_elems)?;
        }
        {
            let v_base = kv_cache.v_base_mut(layer_idx);
            kernels.ops.copy_f16_with_offset(&scratch.v, v_base, decode_params, kv_elems)?;
        }

        // MHA (tiled, graph-compatible, fixed smem)
        {
            let k_base = kv_cache.k_base(layer_idx);
            let v_base = kv_cache.v_base(layer_idx);
            kernels.ops.mha_fused_graph(
                &scratch.q, k_base, v_base, &mut scratch.attn_mha_out,
                decode_params,
                config.head_dim, config.n_heads, config.n_kv_heads,
                seq_len, max_kv_len,
            )?;
        }

        // Output projection
        kernels.gemv.quantize_input(&scratch.attn_mha_out, config.n_heads * config.head_dim)?;
        gemm_q(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
            seq_len, config.dim, config.n_heads * config.head_dim)?;

        // Fused add + RMSNorm + Q8_1 quantize
        {
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.fused_add_rmsnorm_q8(
                hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
                x_q8, seq_len, config.dim, config.rms_norm_eps,
            )?;
        }

        // Gate+Up (dual fused, x_q8 already set)
        if layer.ffn_gate.quant_type == layer.ffn_up.quant_type {
            let _ = kernels.gemv.gemv_dual(
                &layer.ffn_gate.data, &layer.ffn_up.data,
                &mut scratch.ffn_gate_out, &mut scratch.ffn_up_out,
                config.ff_dim, config.dim, layer.ffn_gate.quant_type,
            )?;
        } else {
            gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
                seq_len, config.ff_dim, config.dim)?;
            gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
                seq_len, config.ff_dim, config.dim)?;
        }

        // SiLU
        kernels.ops.silu(
            &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
            seq_len * config.ff_dim,
        )?;

        // Down projection
        kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
            seq_len, config.dim, config.ff_dim)?;

        // Residual + next layer's fused norm+Q8
        if layer_idx + 1 < n_layers {
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.fused_add_rmsnorm_q8(
                hidden, &scratch.ffn_out,
                &weights.layers[layer_idx + 1].attn_norm, &mut scratch.norm_out,
                x_q8, seq_len, config.dim, config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.add_inplace_f32_f16(hidden, &scratch.ffn_out, n_elems)?;
        }
    }

    // Final norm + logits
    kernels.ops.rms_norm_f32in(
        hidden, &weights.output_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    gemm_q(kernels, &scratch.norm_out, &weights.output, &mut scratch.logits,
        seq_len, config.vocab_size, config.dim)?;

    // NOTE: kv_cache.advance() is NOT called here — the caller handles it
    // (for capture: done in DecodeGraph::capture; for replay: done in the decode loop)

    Ok(())
}
