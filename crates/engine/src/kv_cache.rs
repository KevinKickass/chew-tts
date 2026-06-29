use crate::config::ModelConfig;
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut};
use std::sync::Arc;

/// KV cache for all layers — pre-allocated for max context length.
///
/// Layout per layer: [max_seq, n_kv_heads, head_dim] for both K and V.
/// For Gemma 4: head_dim varies per layer and shared-KV layers skip allocation.
pub struct KvCache {
    /// One (K, V) pair per KV-owning layer
    layers: Vec<KvLayerCache>,
    /// Current sequence position (how many tokens have been cached)
    pos: u32,
    max_seq: u32,
    n_kv_heads: u32,
    /// Head dim per layer (for variable head_dim models)
    head_dims: Vec<u32>,
    /// Number of layers that own KV caches
    n_kv_layers: u32,
    /// Default head_dim (for backward compat)
    head_dim: u32,
}

struct KvLayerCache {
    k: CudaSlice<half::f16>,
    v: CudaSlice<half::f16>,
    /// Stride per position: n_kv_heads * head_dim_for_this_layer
    stride: u32,
}

impl KvCache {
    /// Pre-allocate KV cache on GPU.
    pub fn alloc(
        config: &ModelConfig,
        max_seq: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        // Allocate KV cache for ALL layers (even shared ones — they compute their own K/V)
        let n_kv_layers = config.n_layers;
        let mut layers = Vec::with_capacity(n_kv_layers as usize);
        let mut total_bytes: u64 = 0;

        for i in 0..n_kv_layers {
            let hd = config.layer_head_dim(i as usize);
            let stride = config.layer_kv_heads(i as usize) * hd;
            let kv_size = (max_seq as usize) * (stride as usize);

            let k = stream.alloc_zeros::<half::f16>(kv_size)?;
            let v = stream.alloc_zeros::<half::f16>(kv_size)?;
            total_bytes += 2 * (kv_size as u64) * 2; // 2 buffers * f16
            layers.push(KvLayerCache { k, v, stride });
        }

        let total_mb = total_bytes / (1024 * 1024);
        tracing::info!(
            kv_layers = n_kv_layers,
            total_layers = config.n_layers,
            max_seq,
            kv_heads = config.n_kv_heads,
            total_mb,
            "KV cache allocated"
        );

        Ok(Self {
            layers,
            pos: 0,
            max_seq,
            n_kv_heads: config.n_kv_heads,
            head_dims: config.head_dims.clone(),
            n_kv_layers,
            head_dim: config.head_dim,
        })
    }

    pub fn pos(&self) -> u32 {
        self.pos
    }

    pub fn advance(&mut self, n_tokens: u32) {
        self.pos += n_tokens;
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Set the write position to an absolute value. Used for prefix-KV diffusion:
    /// prefill the prompt once (pos advances to P), then reset to P before each
    /// denoising step so the canvas K/V are rewritten while the prompt K/V stay.
    pub fn set_pos(&mut self, p: u32) {
        self.pos = p;
    }

    /// Resolve a layer index to its KV cache index.
    /// For Gemma 4: shared layers map to the last KV-owning layer.
    fn kv_idx(&self, layer: usize) -> usize {
        if (layer as u32) < self.n_kv_layers {
            layer
        } else {
            (self.n_kv_layers - 1) as usize
        }
    }

    /// Get mutable K slice for a layer at the current write position.
    /// Returns a view into [pos..pos+seq_len, n_kv_heads, head_dim].
    pub fn k_mut(&mut self, layer: usize, seq_len: u32) -> CudaViewMut<'_, half::f16> {
        let idx = self.kv_idx(layer);
        let stride = self.layers[idx].stride as usize;
        let start = (self.pos as usize) * stride;
        let end = start + (seq_len as usize) * stride;
        self.layers[idx].k.slice_mut(start..end)
    }

    /// Get mutable V slice for a layer at the current write position.
    pub fn v_mut(&mut self, layer: usize, seq_len: u32) -> CudaViewMut<'_, half::f16> {
        let idx = self.kv_idx(layer);
        let stride = self.layers[idx].stride as usize;
        let start = (self.pos as usize) * stride;
        let end = start + (seq_len as usize) * stride;
        self.layers[idx].v.slice_mut(start..end)
    }

    /// Get full K cache for a layer [0..pos+seq_len].
    pub fn k_full(&self, layer: usize, total_len: u32) -> CudaView<'_, half::f16> {
        let idx = self.kv_idx(layer);
        let stride = self.layers[idx].stride as usize;
        let end = (total_len as usize) * stride;
        self.layers[idx].k.slice(0..end)
    }

    /// Get full V cache for a layer [0..pos+seq_len].
    pub fn v_full(&self, layer: usize, total_len: u32) -> CudaView<'_, half::f16> {
        let idx = self.kv_idx(layer);
        let stride = self.layers[idx].stride as usize;
        let end = (total_len as usize) * stride;
        self.layers[idx].v.slice(0..end)
    }

    /// Get the full K cache buffer for a layer (base pointer, for CUDA Graph mode).
    pub fn k_base(&self, layer: usize) -> &CudaSlice<half::f16> {
        let idx = self.kv_idx(layer);
        &self.layers[idx].k
    }

    /// Get the full V cache buffer for a layer (base pointer, for CUDA Graph mode).
    pub fn v_base(&self, layer: usize) -> &CudaSlice<half::f16> {
        let idx = self.kv_idx(layer);
        &self.layers[idx].v
    }

    /// Get the full K cache buffer for a layer (mutable, for CUDA Graph offset writes).
    pub fn k_base_mut(&mut self, layer: usize) -> &mut CudaSlice<half::f16> {
        let idx = self.kv_idx(layer);
        &mut self.layers[idx].k
    }

    /// Get the full V cache buffer for a layer (mutable, for CUDA Graph offset writes).
    pub fn v_base_mut(&mut self, layer: usize) -> &mut CudaSlice<half::f16> {
        let idx = self.kv_idx(layer);
        &mut self.layers[idx].v
    }

    /// Max sequence length this cache was allocated for.
    pub fn max_seq(&self) -> u32 {
        self.max_seq
    }

    /// KV stride in elements for a given layer: n_kv_heads * head_dim
    pub fn kv_stride_for_layer(&self, layer: usize) -> u32 {
        let idx = self.kv_idx(layer);
        self.layers[idx].stride
    }

    /// KV stride in elements (default layer 0 for backward compat).
    pub fn kv_stride(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
}
