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
#[derive(Default)]
struct StreamingDemandQueue {
    items: Vec<usize>,
}

impl StreamingDemandQueue {
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn enqueue(&mut self, streamed_idx: usize) {
        if !self.items.contains(&streamed_idx) {
            self.items.push(streamed_idx);
        }
    }

    fn remove(&mut self, streamed_idx: usize) {
        self.items.retain(|&idx| idx != streamed_idx);
    }

    fn clear_and_fill_from(&mut self, start_streamed_idx: usize, streamed_total: usize) {
        self.items.clear();
        for idx in start_streamed_idx..streamed_total {
            self.items.push(idx);
        }
    }

    fn future_without(&self, streamed_idx: usize) -> Vec<usize> {
        self.items.iter().copied().filter(|&idx| idx != streamed_idx).collect()
    }

    fn iter_window(&self, limit: usize) -> impl Iterator<Item = usize> + '_ {
        self.items.iter().copied().take(limit)
    }
}

#[derive(Default)]
struct StreamingSchedulerStats {
    prefetch_hits: u64,
    prefetch_misses: u64,
    slot_reuse_hits: u64,
    forced_evictions: u64,
    waits_on_ready: u64,
    queue_depth_max: usize,
}

struct StreamingPrefetchScheduler<'a> {
    sw: &'a mut StreamingWeights,
    prefetch_window: usize,
    tick: u64,
    queue: StreamingDemandQueue,
    stats: StreamingSchedulerStats,
    profile: bool,
}

impl<'a> StreamingPrefetchScheduler<'a> {
    fn new(sw: &'a mut StreamingWeights, prefetch_window: usize, profile: bool) -> Self {
        Self {
            sw,
            prefetch_window,
            tick: 0,
            queue: StreamingDemandQueue::new(),
            stats: StreamingSchedulerStats::default(),
            profile,
        }
    }

    fn shell_ref(&self, slot: usize) -> &LayerWeights {
        if slot == 0 { &self.sw.shell_a } else { &self.sw.shell_b }
    }

    fn shell_mut(&mut self, slot: usize) -> &mut LayerWeights {
        if slot == 0 { &mut self.sw.shell_a } else { &mut self.sw.shell_b }
    }

    fn initialize_queue(&mut self, streamed_total: usize) {
        self.queue.clear_and_fill_from(0, streamed_total);
        self.stats.queue_depth_max = self.stats.queue_depth_max.max(self.queue.items.len());
    }

    fn has_layer(&self, streamed_idx: usize) -> bool {
        self.sw.shell_slots.iter().any(|slot| {
            slot.loaded == Some(streamed_idx) || slot.in_flight == Some(streamed_idx)
        })
    }

    fn next_use_distance(&self, loaded: Option<usize>, future: &[usize]) -> usize {
        match loaded {
            None => usize::MAX,
            Some(idx) => future.iter().position(|&v| v == idx).unwrap_or(usize::MAX),
        }
    }

    fn select_shell_slot(&mut self, next_streamed_idx: usize, future: &[usize]) -> usize {
        for slot in 0..self.sw.shell_slots.len() {
            let meta = self.sw.shell_slots[slot];
            if meta.loaded == Some(next_streamed_idx) || meta.in_flight == Some(next_streamed_idx) {
                self.stats.slot_reuse_hits += 1;
                return slot;
            }
        }

        for slot in 0..self.sw.shell_slots.len() {
            let meta = self.sw.shell_slots[slot];
            if !meta.locked && meta.loaded.is_none() && meta.in_flight.is_none() {
                return slot;
            }
        }

        let mut best_slot = None;
        let mut best_distance = 0usize;
        let mut best_last_used = u64::MAX;
        for slot in 0..self.sw.shell_slots.len() {
            let meta = self.sw.shell_slots[slot];
            if meta.locked {
                continue;
            }
            let dist = self.next_use_distance(meta.loaded, future);
            if best_slot.is_none()
                || dist > best_distance
                || (dist == best_distance && meta.last_used_tick < best_last_used) {
                best_slot = Some(slot);
                best_distance = dist;
                best_last_used = meta.last_used_tick;
            }
        }

        if best_slot.is_some() {
            self.stats.forced_evictions += 1;
        }
        best_slot.unwrap_or(0)
    }

    fn ensure_prefetch(&mut self, streamed_idx: usize, future: &[usize]) -> Result<(), KernelError> {
        if self.has_layer(streamed_idx) {
            self.stats.prefetch_hits += 1;
            return Ok(());
        }

        self.stats.prefetch_misses += 1;
        let slot = self.select_shell_slot(streamed_idx, future);
        let layout = &self.sw.host_layer_offsets[streamed_idx] as *const _;
        let norms = &self.sw.layer_norms[self.sw.n_resident + streamed_idx] as *const _;

        if let Some(ready) = self.sw.shell_ready[slot].take() {
            self.sw.dma_stream.wait(&ready)
                .map_err(|e| KernelError::Launch(format!("dma wait: {e}")))?;
        }

        {
            let dma_stream = Arc::clone(&self.sw.dma_stream);
            let host_data = self.sw.host_layer_data.as_slice() as *const [u8];
            let shell = self.shell_mut(slot);
            let host_data = unsafe { &*host_data };
            let layout = unsafe { &*layout };
            let norms = unsafe { &*norms };
            upload_layer_to_shell_direct(host_data, layout, shell, &dma_stream)
                .map_err(|e| KernelError::Launch(format!("streaming upload: {e}")))?;
            copy_norms_to_shell_direct(norms, shell, &dma_stream)
                .map_err(|e| KernelError::Launch(format!("norm copy: {e}")))?;
        }

        let ev = self.sw.dma_stream.record_event(None)
            .map_err(|e| KernelError::Launch(format!("dma record event: {e}")))?;
        self.sw.shell_ready[slot] = Some(ev);
        self.sw.shell_slots[slot].loaded = None;
        self.sw.shell_slots[slot].in_flight = Some(streamed_idx);
        self.sw.shell_slots[slot].locked = false;
        Ok(())
    }

    fn drain_prefetch_queue(&mut self) -> Result<(), KernelError> {
        self.stats.queue_depth_max = self.stats.queue_depth_max.max(self.queue.items.len());
        let queued: Vec<usize> = self.queue.iter_window(self.prefetch_window).collect();
        for streamed_idx in queued {
            let future = self.queue.future_without(streamed_idx);
            self.ensure_prefetch(streamed_idx, &future)?;
        }
        Ok(())
    }

    fn warm_initial_window(&mut self, streamed_total: usize) -> Result<(), KernelError> {
        if streamed_total == 0 {
            return Ok(());
        }
        self.initialize_queue(streamed_total);
        self.drain_prefetch_queue()
    }

    fn acquire(&mut self, streamed_idx: usize, _streamed_total: usize, stream: &Arc<CudaStream>) -> Result<usize, KernelError> {
        self.tick += 1;
        self.queue.enqueue(streamed_idx);
        self.drain_prefetch_queue()?;

        let slot = self.sw.shell_slots.iter().position(|meta| {
            meta.loaded == Some(streamed_idx) || meta.in_flight == Some(streamed_idx)
        }).expect("prefetched shell for streamed layer not found");

        if let Some(ready) = &self.sw.shell_ready[slot] {
            self.stats.waits_on_ready += 1;
            stream.wait(ready)
                .map_err(|e| KernelError::Launch(format!("stream wait: {e}")))?;
        }
        self.sw.shell_slots[slot].loaded = Some(streamed_idx);
        self.sw.shell_slots[slot].in_flight = None;
        self.sw.shell_slots[slot].locked = true;
        self.sw.shell_slots[slot].last_used_tick = self.tick;
        self.queue.remove(streamed_idx);

        self.drain_prefetch_queue()?;

        Ok(slot)
    }

    fn release(&mut self, slot: usize) {
        self.sw.shell_slots[slot].locked = false;
        self.sw.shell_slots[slot].last_used_tick = self.tick;
    }

    fn log_stats(&self) {
        if !self.profile {
            return;
        }

        let total_prefetch = self.stats.prefetch_hits + self.stats.prefetch_misses;
        let reuse_ratio = if total_prefetch > 0 {
            self.stats.slot_reuse_hits as f64 / total_prefetch as f64
        } else {
            0.0
        };
        let hit_ratio = if total_prefetch > 0 {
            self.stats.prefetch_hits as f64 / total_prefetch as f64
        } else {
            0.0
        };
        let wait_ratio = if total_prefetch > 0 {
            self.stats.waits_on_ready as f64 / total_prefetch as f64
        } else {
            0.0
        };

        info!(
            prefetch_hits = self.stats.prefetch_hits,
            prefetch_misses = self.stats.prefetch_misses,
            slot_reuse_hits = self.stats.slot_reuse_hits,
            forced_evictions = self.stats.forced_evictions,
            waits_on_ready = self.stats.waits_on_ready,
            queue_depth_max = self.stats.queue_depth_max,
            reuse_ratio = reuse_ratio,
            hit_ratio = hit_ratio,
            wait_ratio = wait_ratio,
            "streaming scheduler stats"
        );
    }
}

pub fn forward_streaming(
    hidden: &mut CudaSlice<f32>,
    weights: &mut WeightStorage,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    pe: Option<&crate::arch::gemma4_common::PerLayerEmbeddings>,
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

    let streamed_total = n_layers.saturating_sub(n_resident);
    let profile_streaming = std::env::var("CHEW_PROFILE").is_ok() && seq_len == 1;
    let mut scheduler = StreamingPrefetchScheduler::new(sw, 2, profile_streaming);
    scheduler.warm_initial_window(streamed_total)?;

    for layer_idx in 0..n_layers {
        // Get layer weights: either resident or streamed via shell
        let streamed_slot = if layer_idx < n_resident {
            None
        } else {
            let streamed_idx = layer_idx - n_resident;
            Some(scheduler.acquire(streamed_idx, streamed_total, stream)?)
        };
        let layer: &LayerWeights = if layer_idx < n_resident {
            &scheduler.sw.resident_layers[layer_idx]
        } else {
            scheduler.shell_ref(streamed_slot.expect("streamed slot missing"))
        };

        // Get next layer's norm for fused operations (Llama path)
        let next_attn_norm_ptr: Option<*const CudaSlice<half::f16>> = if layer_idx + 1 < n_layers {
            Some(&scheduler.sw.layer_norms[layer_idx + 1].attn_norm as *const _)
        } else {
            None
        };
        let next_attn_norm: Option<&CudaSlice<half::f16>> = next_attn_norm_ptr.map(|ptr| unsafe { &*ptr });

        if is_gemma4 {
            crate::arch::streaming_dense::forward_layer_gemma4(
                hidden, layer, config, kernels, kv_cache, scratch,
                seq_len, layer_idx, pos, total_kv_len, pe, scheduler.sw,
            )?;
        } else {
            crate::arch::streaming_dense::forward_layer_llama(
                hidden, layer, config, kernels, kv_cache, scratch,
                seq_len, layer_idx, next_attn_norm, n_elems,
            )?;
        }

        if let Some(slot) = streamed_slot {
            scheduler.release(slot);
        }
    }

    scheduler.log_stats();

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
