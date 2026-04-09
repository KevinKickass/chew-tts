use crate::config::ModelConfig;
use chew_gguf::GgufFile;

/// Exact VRAM budget — computed from model config before any allocation.
///
/// Mirrors the allocation logic in `weights.rs` exactly:
/// - token_embd → f16 (for embed_tokens kernel)
/// - output_norm → f16 (small)
/// - per-layer norms → f16 (small)
/// - all large matrices → stay quantized (disk size)
/// - output.weight → quantized, OR tied embeddings → quantized copy of token_embd
/// - dequant scratch: one f16 buffer sized for the largest weight matrix
#[derive(Debug, Clone)]
pub struct VramPlan {
    /// Weights on GPU (bytes) — steady state after loading
    pub weights_bytes: u64,
    /// Peak loading overhead — temporary quantized bytes during upload_and_dequant
    pub loading_peak_bytes: u64,
    /// KV cache (bytes)
    pub kv_cache_bytes: u64,
    /// Scratch buffers for forward pass (bytes)
    pub scratch_bytes: u64,
    /// Dequant scratch — one max-weight-matrix as f16 (bytes)
    pub dequant_scratch_bytes: u64,
    /// cuBLAS workspace estimate (bytes)
    pub cublas_bytes: u64,
    /// Total steady-state VRAM (bytes)
    pub total_bytes: u64,
    /// Peak VRAM during loading (bytes) — this is what must fit
    pub peak_bytes: u64,
    /// Whether output uses tied embeddings
    pub tied_embeddings: bool,

    /// Chosen context length
    pub context_length: u32,
    /// Chosen max prefill batch
    pub max_batch: u32,
}

impl VramPlan {
    /// Compute exact VRAM requirements from model config and GGUF tensor info.
    ///
    /// Matches the loading logic in `weights.rs` precisely:
    /// - Norms: dequantized to f16 on GPU, quantized source freed
    /// - token_embd: dequantized to f16, quantized source freed
    /// - Everything else: stays quantized
    pub fn compute(
        config: &ModelConfig,
        gguf: &GgufFile,
        context_length: u32,
        max_batch: u32,
    ) -> Self {
        let mut weights_f16_bytes: u64 = 0;   // permanent f16 tensors
        let mut weights_quant_bytes: u64 = 0;  // permanent quantized tensors
        let mut max_dequant_temp: u64 = 0;     // peak temp during upload_and_dequant
        let mut max_weight_elements: u64 = 0;  // for dequant scratch sizing

        // Categorize each tensor exactly as weights.rs loads them
        for t in &gguf.tensors {
            let name = &t.name;
            let is_norm = name.contains("norm");
            let is_embd = name == "token_embd.weight";

            if is_norm || is_embd {
                // upload_and_dequant: f16 stays, quantized bytes temporary
                let f16_size = t.n_elements() * 2;
                let quant_size = t.data_size();
                weights_f16_bytes += f16_size;
                // During loading, both exist simultaneously
                if quant_size > max_dequant_temp {
                    max_dequant_temp = quant_size;
                }
            } else {
                // upload_quantized: stays as quantized bytes
                let quant_size = t.data_size();
                weights_quant_bytes += quant_size;
                // Track largest for dequant scratch
                if t.n_elements() > max_weight_elements {
                    max_weight_elements = t.n_elements();
                }
            }
        }

        // Tied embeddings: if no output.weight, token_embd is loaded AGAIN as quantized
        let tied_embeddings = gguf.find_tensor("output.weight").is_none();
        if tied_embeddings {
            if let Some(embd) = gguf.find_tensor("token_embd.weight") {
                weights_quant_bytes += embd.data_size();
                if embd.n_elements() > max_weight_elements {
                    max_weight_elements = embd.n_elements();
                }
            }
        }

        let weights_bytes = weights_f16_bytes + weights_quant_bytes;

        // Dequant scratch: capped at 128 MB (64M f16 elements).
        // Larger matrices (e.g. output [128256, 4096]) are processed in chunks.
        let max_scratch_elements: u64 = 64 * 1024 * 1024; // matches MAX_SCRATCH_ELEMENTS in gemm.rs
        let dequant_scratch_bytes = max_weight_elements.min(max_scratch_elements) * 2;

        // KV cache: 2 (K+V) * per KV-owning layer: max_seq * n_kv_heads * head_dim * 2 bytes (f16)
        let kv_cache_bytes: u64 = (0..config.n_kv_layers as usize).map(|i| {
            let hd = config.layer_head_dim(i) as u64;
            2 * context_length as u64 * config.n_kv_heads as u64 * hd * 2
        }).sum();

        // Scratch buffers for forward pass:
        // Most buffers are f16 (2 bytes), residual stays f32 (4 bytes).
        let s = max_batch as u64;
        let d = config.dim as u64;
        let ff = config.ff_dim as u64;
        let nh = config.n_heads as u64;
        let nkv = config.n_kv_heads as u64;
        let hd = config.max_head_dim as u64;
        let v = config.vocab_size as u64;

        let pe_elements = if let Some(epl) = config.embd_per_layer {
            let epl = epl as u64;
            s * epl + s * d  // pe_gate_out + pe_proj_out
        } else {
            0
        };
        let scratch_f16_elements = s * d       // norm_out
            + s * nh * hd                     // q
            + s * nkv * hd                    // k
            + s * nkv * hd                    // v
            + s * d                           // attn_mha_out
            + s * d                           // attn_out
            + s * ff                          // ffn_gate_out
            + s * ff                          // ffn_up_out
            + s * ff                          // ffn_silu_out
            + s * d                           // ffn_out
            + s * v                           // logits
            + pe_elements;                    // per-layer embedding (Gemma 4)
        let scratch_f32_elements = s * d;     // residual (f32)
        let scratch_bytes = scratch_f16_elements * 2 + scratch_f32_elements * 4;

        // cuBLAS workspace: ~32 MB typical
        let cublas_bytes = 32 * 1024 * 1024;

        // Steady-state total (after loading completes)
        let total_bytes = weights_bytes
            + dequant_scratch_bytes
            + kv_cache_bytes
            + scratch_bytes
            + cublas_bytes;

        // Peak during loading: steady state + temporary quantized bytes for largest dequant
        let peak_bytes = total_bytes + max_dequant_temp;

        Self {
            weights_bytes,
            loading_peak_bytes: max_dequant_temp,
            kv_cache_bytes,
            scratch_bytes,
            dequant_scratch_bytes,
            cublas_bytes,
            total_bytes,
            peak_bytes,
            tied_embeddings,
            context_length,
            max_batch,
        }
    }

    /// Find the best config that fits in available VRAM.
    ///
    /// Uses **peak_bytes** (not total_bytes) because loading temporarily needs more.
    /// Tries the requested context_length first, then halves it until it fits.
    /// Returns None if even context_length=256 doesn't fit.
    pub fn fit(
        config: &ModelConfig,
        gguf: &GgufFile,
        desired_context: u32,
        available_bytes: u64,
    ) -> Option<Self> {
        let mut ctx = desired_context;
        // 256 MB headroom for CUDA driver, display, misc
        let headroom = 256 * 1024 * 1024;

        while ctx >= 256 {
            let batch = ctx.min(2048);
            let plan = Self::compute(config, gguf, ctx, batch);

            if plan.peak_bytes + headroom <= available_bytes {
                return Some(plan);
            }
            ctx /= 2;
        }

        None
    }

    pub fn weights_mb(&self) -> u64 { self.weights_bytes / (1024 * 1024) }
    pub fn kv_cache_mb(&self) -> u64 { self.kv_cache_bytes / (1024 * 1024) }
    pub fn scratch_mb(&self) -> u64 { self.scratch_bytes / (1024 * 1024) }
    pub fn dequant_scratch_mb(&self) -> u64 { self.dequant_scratch_bytes / (1024 * 1024) }
    pub fn total_mb(&self) -> u64 { self.total_bytes / (1024 * 1024) }
    pub fn peak_mb(&self) -> u64 { self.peak_bytes / (1024 * 1024) }

    /// Print a clear VRAM budget report.
    pub fn print_report(&self, available_mb: Option<u64>) {
        println!("╔══════════════════════════════════════════╗");
        println!("║          VRAM BUDGET REPORT              ║");
        println!("╠══════════════════════════════════════════╣");
        println!("║  Weights (quant+f16):  {:>6} MB          ║", self.weights_mb());
        println!("║  Dequant scratch:      {:>6} MB          ║", self.dequant_scratch_mb());
        println!("║  KV cache ({}k ctx): {:>6} MB          ║",
            self.context_length / 1024, self.kv_cache_mb());
        println!("║  Forward scratch (b={}):  {:>4} MB          ║",
            self.max_batch, self.scratch_mb());
        println!("║  cuBLAS workspace:     {:>6} MB          ║",
            self.cublas_bytes / (1024 * 1024));
        println!("╠══════════════════════════════════════════╣");
        println!("║  Steady-state:         {:>6} MB          ║", self.total_mb());
        println!("║  Peak (during load):   {:>6} MB          ║", self.peak_mb());
        if self.tied_embeddings {
            println!("║  (tied embeddings: output = token_embd)  ║");
        }
        println!("╠══════════════════════════════════════════╣");
        if let Some(avail) = available_mb {
            let fits = self.peak_bytes + 256 * 1024 * 1024 <= avail * 1024 * 1024;
            let remaining = if fits {
                avail as i64 - self.peak_mb() as i64 - 256
            } else {
                avail as i64 - self.peak_mb() as i64 - 256
            };
            println!("║  GPU free:             {:>6} MB          ║", avail);
            println!("║  Headroom (driver):      256 MB          ║");
            if fits {
                println!("║  Remaining:            {:>6} MB          ║", remaining);
                println!("║                                          ║");
                println!("║  >>> FITS - GO <<<                       ║");
            } else {
                println!("║  Shortfall:            {:>6} MB          ║", -remaining);
                println!("║                                          ║");
                println!("║  >>> DOES NOT FIT <<<                    ║");
            }
        }
        println!("╚══════════════════════════════════════════╝");
    }
}
