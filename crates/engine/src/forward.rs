use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::weights::{ModelWeights, QuantWeight};
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

        // 9. Gate + Up projections — for decode, x_q8 already has quantized norm_out
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

    // Layer 0: initial RMSNorm
    kernels.ops.rms_norm_f32in(
        hidden, &weights.layers[0].attn_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    for layer_idx in 0..n_layers {
        let layer = &weights.layers[layer_idx];

        // QKV projections
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
            seq_len, config.n_heads * config.head_dim, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
            seq_len, config.n_kv_heads * config.head_dim, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.attn_v, &mut scratch.v,
            seq_len, config.n_kv_heads * config.head_dim, config.dim)?;

        // RoPE on Q and K — using graph-compatible kernel (reads pos from device memory)
        kernels.ops.rope_graph(
            &mut scratch.q, decode_params, seq_len, config.n_heads, config.head_dim, config.rope_theta,
        )?;
        kernels.ops.rope_graph(
            &mut scratch.k, decode_params, seq_len, config.n_kv_heads, config.head_dim, config.rope_theta,
        )?;

        // Write K, V into KV cache using offset-based copy (graph-compatible)
        let kv_elems = seq_len * config.n_kv_heads * config.head_dim;
        {
            let k_base = kv_cache.k_base_mut(layer_idx);
            kernels.ops.copy_f16_with_offset(&scratch.k, k_base, decode_params, kv_elems)?;
        }
        {
            let v_base = kv_cache.v_base_mut(layer_idx);
            kernels.ops.copy_f16_with_offset(&scratch.v, v_base, decode_params, kv_elems)?;
        }

        // MHA — using graph-compatible kernel (reads kv_len from device memory, uses base pointers)
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

        // Fused add + RMSNorm
        kernels.ops.fused_add_rmsnorm(
            hidden, &scratch.attn_out, &layer.ffn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;

        // Gate + Up projections
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
            seq_len, config.ff_dim, config.dim)?;
        gemm_q(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
            seq_len, config.ff_dim, config.dim)?;

        // SiLU
        kernels.ops.silu(
            &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
            seq_len * config.ff_dim,
        )?;

        // Down projection
        kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        gemm_q(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
            seq_len, config.dim, config.ff_dim)?;

        // Residual + next layer's norm (or just add for last layer)
        if layer_idx + 1 < n_layers {
            kernels.ops.fused_add_rmsnorm(
                hidden, &scratch.ffn_out,
                &weights.layers[layer_idx + 1].attn_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
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
