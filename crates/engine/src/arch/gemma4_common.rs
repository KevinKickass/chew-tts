use cudarc::driver::CudaSlice;

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
    /// Get a view offset for a specific layer's embeddings.
    /// NOTE: The data is not contiguous per-layer; consumers must use strided access.
    pub fn layer_offset(&self, layer_idx: usize) -> usize {
        layer_idx * self.epl as usize
    }
}
