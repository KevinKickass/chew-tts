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
    // Gemma 4 specific
    pub sliding_window: Option<u32>,
    pub swa_layers: Vec<bool>,       // per-layer: true = SWA, false = full attention
    pub n_kv_shared_layers: u32,     // layers from end that share KV cache
    pub attention_scale: f32,        // 1.0 for Gemma 4, 1/sqrt(d) for Llama
    pub logit_softcap: Option<f32>,  // final logit softcapping
    pub rope_theta_swa: Option<f32>, // separate rope base for SWA layers
    /// Per-layer head_dim: full attn layers have 512, SWA layers have 256
    pub head_dims: Vec<u32>,
    /// Per-layer embedding dimension (Gemma 4: 256)
    pub embd_per_layer: Option<u32>,
    /// Max head_dim across all layers (for scratch buffer sizing)
    pub max_head_dim: u32,
    /// Number of layers that own KV caches (rest reuse from earlier layers)
    pub n_kv_layers: u32,
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

        // Gemma 4 specific
        let is_gemma4 = arch == "gemma4";
        let sliding_window = header.get_u32(&format!("{arch}.attention.sliding_window")).ok();
        let n_kv_shared_layers = header.get_u32(&format!("{arch}.attention.shared_kv_layers")).unwrap_or(0);
        // Gemma 4 uses attention_scale=1.0 because Q and K are already
        // RMS-normalized per-head before attention. llama.cpp: f_attention_scale = 1.0
        // For non-Gemma4 models, standard 1/sqrt(head_dim).
        let attention_scale = if is_gemma4 { 1.0 } else { 1.0 / (head_dim as f32).sqrt() };
        let logit_softcap = header.get_f32(&format!("{arch}.final_logit_softcapping")).ok();
        let rope_theta_swa = header.get_f32(&format!("{arch}.rope.freq_base_swa")).ok();
        let embd_per_layer = header.get_u32(&format!("{arch}.embedding_length_per_layer_input")).ok();

        // SWA layer pattern (if present) — bool array in GGUF
        let swa_layers = if let Ok(pattern) = header.get_bool_array(&format!("{arch}.attention.sliding_window_pattern")) {
            pattern
        } else if let Ok(pattern) = header.get_u32_array(&format!("{arch}.attention.sliding_window_pattern")) {
            pattern.iter().map(|&v| v != 0).collect()
        } else {
            vec![false; n_layers as usize]
        };

        // Per-layer head_dim: read from GGUF key_length, then compute per layer
        // For Gemma 4: full attention layers use key_length (512), SWA uses head_dim (256 = dim/n_heads)
        let full_attn_head_dim = header.get_u32(&format!("{arch}.attention.key_length")).unwrap_or(head_dim);
        let swa_head_dim = header.get_u32(&format!("{arch}.attention.key_length_swa")).unwrap_or(dim / n_heads);

        let head_dims: Vec<u32> = if is_gemma4 {
            // Determine head_dim per layer from SWA pattern
            // SWA layers: n_heads * swa_hd = q_dim. From tensors: SWA q is [2560, 2048], so hd=256
            // Full layers: n_heads * full_hd = q_dim. From tensors: Full q is [2560, 4096], so hd=512
            swa_layers.iter().map(|&is_swa| {
                if is_swa { swa_head_dim } else { full_attn_head_dim }
            }).collect()
        } else {
            vec![head_dim; n_layers as usize]
        };

        let max_head_dim = *head_dims.iter().max().unwrap_or(&head_dim);

        if is_gemma4 {
            let swa_count = swa_layers.iter().filter(|&&x| x).count();
            let full_count = swa_layers.len() - swa_count;
            tracing::info!(swa_count, full_count, swa_head_dim, full_attn_head_dim, max_head_dim, "Gemma4 layer config");
        }

        // Number of layers that own their own KV cache
        // shared_kv_layers layers at the end reuse KV from layer (n_layers - shared_kv_layers - 1)
        let n_kv_layers = n_layers - n_kv_shared_layers;

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
            sliding_window,
            swa_layers,
            n_kv_shared_layers,
            attention_scale,
            logit_softcap,
            rope_theta_swa,
            head_dims,
            embd_per_layer,
            max_head_dim,
            n_kv_layers,
        })
    }

    /// Whether this model uses Grouped-Query Attention.
    pub fn is_gqa(&self) -> bool {
        self.n_kv_heads < self.n_heads
    }

    /// KV dimension = n_kv_heads * head_dim (for a specific layer).
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// Whether this is a Gemma 4 architecture.
    pub fn is_gemma4(&self) -> bool {
        self.arch == "gemma4"
    }

    /// Head dim for a specific layer.
    pub fn layer_head_dim(&self, layer: usize) -> u32 {
        self.head_dims.get(layer).copied().unwrap_or(self.head_dim)
    }

    /// Whether a layer is SWA (sliding window attention).
    pub fn is_swa(&self, layer: usize) -> bool {
        self.swa_layers.get(layer).copied().unwrap_or(false)
    }

    /// Whether a layer has its own KV cache (vs sharing with an earlier layer).
    pub fn has_kv(&self, layer: usize) -> bool {
        // Layers 0..(n_layers - n_kv_shared_layers) have their own KV.
        // The rest reuse the last KV-owning layer's cache.
        (layer as u32) < self.n_kv_layers
    }

    /// Get the KV cache source layer for a given layer.
    /// For layers that own their KV, returns the layer itself.
    /// For shared layers, returns the matching (SWA/FULL) KV-owning layer.
    /// SWA shared layers use last SWA owning layer, FULL shared use last FULL owning.
    pub fn kv_source_layer(&self, layer: usize) -> usize {
        if self.has_kv(layer) {
            layer
        } else {
            // Find last KV-owning layer with same SWA/FULL type
            let want_swa = self.is_swa(layer);
            let kv_end = self.n_kv_layers as usize;
            for i in (0..kv_end).rev() {
                if self.is_swa(i) == want_swa {
                    return i;
                }
            }
            // Fallback: last KV-owning layer
            kv_end - 1
        }
    }

    /// RoPE theta for a specific layer.
    pub fn layer_rope_theta(&self, layer: usize) -> f32 {
        if self.is_swa(layer) {
            self.rope_theta_swa.unwrap_or(self.rope_theta)
        } else {
            self.rope_theta
        }
    }
}
