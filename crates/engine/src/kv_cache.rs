use crate::config::ModelConfig;
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut};
use std::sync::Arc;

/// KV cache for all layers — pre-allocated for max context length.
///
/// Layout per layer: [max_seq, n_kv_heads, head_dim] for both K and V.
pub struct KvCache {
    /// One (K, V) pair per layer
    layers: Vec<KvLayerCache>,
    /// Current sequence position (how many tokens have been cached)
    pos: u32,
    max_seq: u32,
    n_kv_heads: u32,
    head_dim: u32,
}

struct KvLayerCache {
    k: CudaSlice<half::f16>,
    v: CudaSlice<half::f16>,
}

impl KvCache {
    /// Pre-allocate KV cache on GPU.
    pub fn alloc(
        config: &ModelConfig,
        max_seq: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let kv_size = (max_seq as usize) * (config.n_kv_heads as usize) * (config.head_dim as usize);

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for _ in 0..config.n_layers {
            let k = stream.alloc_zeros::<half::f16>(kv_size)?;
            let v = stream.alloc_zeros::<half::f16>(kv_size)?;
            layers.push(KvLayerCache { k, v });
        }

        let total_mb = (config.n_layers as u64)
            * 2
            * (kv_size as u64)
            * 2  // f16 = 2 bytes
            / (1024 * 1024);
        tracing::info!(
            layers = config.n_layers,
            max_seq,
            kv_heads = config.n_kv_heads,
            head_dim = config.head_dim,
            total_mb,
            "KV cache allocated"
        );

        Ok(Self {
            layers,
            pos: 0,
            max_seq,
            n_kv_heads: config.n_kv_heads,
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

    /// Get mutable K slice for a layer at the current write position.
    /// Returns a view into [pos..pos+seq_len, n_kv_heads, head_dim].
    pub fn k_mut(&mut self, layer: usize, seq_len: u32) -> CudaViewMut<'_, half::f16> {
        let stride = (self.n_kv_heads * self.head_dim) as usize;
        let start = (self.pos as usize) * stride;
        let end = start + (seq_len as usize) * stride;
        self.layers[layer].k.slice_mut(start..end)
    }

    /// Get mutable V slice for a layer at the current write position.
    pub fn v_mut(&mut self, layer: usize, seq_len: u32) -> CudaViewMut<'_, half::f16> {
        let stride = (self.n_kv_heads * self.head_dim) as usize;
        let start = (self.pos as usize) * stride;
        let end = start + (seq_len as usize) * stride;
        self.layers[layer].v.slice_mut(start..end)
    }

    /// Get full K cache for a layer [0..pos+seq_len].
    pub fn k_full(&self, layer: usize, total_len: u32) -> CudaView<'_, half::f16> {
        let stride = (self.n_kv_heads * self.head_dim) as usize;
        let end = (total_len as usize) * stride;
        self.layers[layer].k.slice(0..end)
    }

    /// Get full V cache for a layer [0..pos+seq_len].
    pub fn v_full(&self, layer: usize, total_len: u32) -> CudaView<'_, half::f16> {
        let stride = (self.n_kv_heads * self.head_dim) as usize;
        let end = (total_len as usize) * stride;
        self.layers[layer].v.slice(0..end)
    }
}
