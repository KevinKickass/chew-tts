use chew_gguf::{GgufHeader, GgufError};

/// Model hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub dim: u32,
    pub head_dim: u32,
    pub ff_dim: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    pub fn from_gguf(header: &GgufHeader) -> Result<Self, GgufError> {
        let arch = header
            .architecture()
            .ok_or_else(|| GgufError::TensorNotFound("general.architecture".into()))?
            .to_string();

        let n_layers = header.block_count().unwrap_or(32);
        let n_heads = header.head_count().unwrap_or(32);
        let n_kv_heads = header.head_count_kv().unwrap_or(n_heads);
        let dim = header.embedding_length().unwrap_or(4096);
        let head_dim = dim / n_heads;
        let ff_dim = header.feed_forward_length().unwrap_or(dim * 4);
        let vocab_size = header.vocab_size().unwrap_or(262144);
        let context_length = header.context_length().unwrap_or(4096);
        let rope_theta = header.rope_freq_base().unwrap_or(10000.0);
        let rms_norm_eps = header.rms_norm_eps().unwrap_or(1e-5);

        Ok(Self {
            arch,
            n_layers,
            n_heads,
            n_kv_heads,
            dim,
            head_dim,
            ff_dim,
            vocab_size,
            context_length,
            rope_theta,
            rms_norm_eps,
        })
    }

    /// Whether this model uses Grouped-Query Attention.
    pub fn is_gqa(&self) -> bool {
        self.n_kv_heads < self.n_heads
    }

    /// KV dimension = n_kv_heads * head_dim.
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
}
