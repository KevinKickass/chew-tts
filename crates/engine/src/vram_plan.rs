use crate::config::ModelConfig;
use chew_gguf::GgufFile;

/// Plan for streaming weights that don't fit in VRAM.
#[derive(Debug, Clone)]
pub struct StreamingPlan {
    /// Layers permanently in VRAM
    pub n_resident: u32,
    /// Layers that must be streamed from RAM
    pub n_streamed: u32,
    /// Average bytes per layer (quantized weights only, no norms)
    pub per_layer_bytes: u64,
    /// Max bytes for any single layer (for DMA slot sizing)
    pub max_layer_bytes: u64,
    /// Fixed VRAM usage (embeddings, norms, KV, scratch, etc.)
    pub fixed_bytes: u64,
    /// DMA double-buffer size (2 × max_layer_bytes)
    pub dma_slot_bytes: u64,
    /// KV cache bytes
    pub kv_cache_bytes: u64,
    /// Context length for streaming mode
    pub context_length: u32,
    /// Total layers in model
    pub total_layers: u32,
}

impl StreamingPlan {
    pub fn print_report(&self, available_mb: u64) {
        let res_mb = self.n_resident as u64 * self.per_layer_bytes / (1024 * 1024);
        let stream_mb = self.n_streamed as u64 * self.per_layer_bytes / (1024 * 1024);
        println!("╔══════════════════════════════════════════╗");
        println!("║       STREAMING VRAM BUDGET               ║");
        println!("╠══════════════════════════════════════════╣");
        println!(
            "║  Resident layers: {:>3}/{:>3} ({:>5} MB)      ║",
            self.n_resident, self.total_layers, res_mb
        );
        println!(
            "║  Streamed layers: {:>3}     ({:>5} MB host)  ║",
            self.n_streamed, stream_mb
        );
        println!(
            "║  DMA slots (2×):        {:>5} MB          ║",
            self.dma_slot_bytes / (1024 * 1024)
        );
        println!(
            "║  Fixed (KV+scratch+emb): {:>5} MB          ║",
            self.fixed_bytes / (1024 * 1024)
        );
        println!(
            "║  KV cache ({}k ctx):    {:>5} MB          ║",
            self.context_length / 1024,
            self.kv_cache_bytes / (1024 * 1024)
        );
        println!("╠══════════════════════════════════════════╣");
        println!(
            "║  GPU free:              {:>5} MB          ║",
            available_mb
        );
        if self.n_streamed == 0 {
            println!("║  >>> ALL LAYERS FIT — FULL SPEED <<<     ║");
        } else {
            let est_tps = if self.n_streamed <= 4 {
                "~40-50"
            } else if self.n_streamed <= 16 {
                "~15-25"
            } else {
                "~5-10"
            };
            println!("║  >>> STREAMING MODE — est. {} tok/s <<<  ║", est_tps);
        }
        println!("╚══════════════════════════════════════════╝");
    }
}

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
        let mut weights_f16_bytes: u64 = 0; // permanent f16 tensors
        let mut weights_quant_bytes: u64 = 0; // permanent quantized tensors
        let mut max_dequant_temp: u64 = 0; // peak temp during upload_and_dequant
        let mut max_weight_elements: u64 = 0; // for dequant scratch sizing

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
        let kv_cache_bytes: u64 = (0..config.n_kv_layers as usize)
            .map(|i| {
                let hd = config.layer_head_dim(i) as u64;
                2 * context_length as u64 * config.n_kv_heads as u64 * hd * 2
            })
            .sum();

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
            s * epl + s * d // pe_gate_out + pe_proj_out
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
            + v                               // logits (last token only)
            + pe_elements; // per-layer embedding (Gemma 4)
        let scratch_f32_elements = s * d; // residual (f32)
        let scratch_bytes = scratch_f16_elements * 2 + scratch_f32_elements * 4;

        // cuBLAS workspace: ~32 MB typical
        let cublas_bytes = 32 * 1024 * 1024;

        // Steady-state total (after loading completes)
        let total_bytes =
            weights_bytes + dequant_scratch_bytes + kv_cache_bytes + scratch_bytes + cublas_bytes;

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
    /// Returns None if even context_length=256 doesn't fit (caller should try streaming).
    pub fn fit(
        config: &ModelConfig,
        gguf: &GgufFile,
        desired_context: u32,
        available_bytes: u64,
    ) -> Option<Self> {
        Self::fit_with_batch(config, gguf, desired_context, None, available_bytes)
    }

    pub fn fit_with_batch(
        config: &ModelConfig,
        gguf: &GgufFile,
        desired_context: u32,
        desired_batch: Option<u32>,
        available_bytes: u64,
    ) -> Option<Self> {
        let mut ctx = desired_context;
        let headroom = 256 * 1024 * 1024;

        while ctx >= 256 {
            let batch_cap = desired_batch.unwrap_or(ctx).min(ctx).max(1);
            let mut lo = 1u32;
            let mut hi = batch_cap;
            let mut best = None;

            while lo <= hi {
                let mid = lo + (hi - lo) / 2;
                let plan = Self::compute(config, gguf, ctx, mid);
                if plan.peak_bytes + headroom <= available_bytes {
                    best = Some(plan);
                    lo = mid.saturating_add(1);
                } else if mid == 1 {
                    hi = 0;
                } else {
                    hi = mid - 1;
                }
            }

            if let Some(plan) = best {
                return Some(plan);
            }
            ctx /= 2;
        }

        None
    }

    /// Try streaming mode when normal loading doesn't fit.
    pub fn fit_streaming(
        config: &ModelConfig,
        gguf: &GgufFile,
        desired_context: u32,
        available_bytes: u64,
    ) -> Option<StreamingPlan> {
        Self::fit_streaming_with_batch(config, gguf, desired_context, None, available_bytes)
    }

    pub fn fit_streaming_with_batch(
        config: &ModelConfig,
        gguf: &GgufFile,
        desired_context: u32,
        desired_batch: Option<u32>,
        available_bytes: u64,
    ) -> Option<StreamingPlan> {
        // Try with desired context, then halve
        let mut ctx = desired_context.min(8192); // cap streaming context
        while ctx >= 256 {
            if let Some(plan) =
                Self::compute_streaming(config, gguf, ctx, desired_batch, available_bytes)
            {
                return Some(plan);
            }
            ctx /= 2;
        }
        None
    }

    /// Compute a streaming plan: how many layers fit in VRAM permanently?
    /// Returns (n_resident_layers, per_layer_bytes, streaming_plan) or None if even
    /// the fixed overhead doesn't fit.
    pub fn compute_streaming(
        config: &ModelConfig,
        gguf: &GgufFile,
        desired_context: u32,
        desired_batch: Option<u32>,
        available_bytes: u64,
    ) -> Option<StreamingPlan> {
        // DiffusionGemma needs extra free VRAM at run time for the canvas logit
        // buffer + SC probs (each ~canvas*vocab f16) plus fragmentation slack;
        // reserve more so fewer layers go resident and the run fits.
        let headroom: u64 = if config.is_diffusion() {
            768 * 1024 * 1024
        } else {
            256 * 1024 * 1024
        };

        // Calculate per-layer weight size: total model bytes minus global tensors, divided by layers.
        // This is more reliable than summing individual tensor data_size() which can have
        // rounding/padding issues.
        let total_model_bytes: u64 = gguf.tensors.iter().map(|t| t.data_size()).sum();
        let global_tensor_bytes: u64 = gguf
            .tensors
            .iter()
            .filter(|t| !t.name.starts_with("blk."))
            .map(|t| t.data_size())
            .sum();
        let all_norm_bytes: u64 = gguf
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("blk.") && t.name.contains("norm"))
            .map(|t| t.data_size())
            .sum();
        let layer_weight_total = total_model_bytes - global_tensor_bytes - all_norm_bytes;
        let avg_layer_bytes = layer_weight_total / config.n_layers as u64;
        let max_layer_bytes = avg_layer_bytes + avg_layer_bytes / 10; // 10% margin

        // Fixed VRAM: embeddings + output + norms + KV cache + scratch
        let mut fixed_bytes: u64 = 0;

        // Token embeddings — keep as f16 for embed_tokens kernel
        // TODO: could keep as quantized + gather_dequant to save ~900MB
        if let Some(t) = gguf.find_tensor("token_embd.weight") {
            fixed_bytes += t.n_elements() * 2; // f16 for now
        }
        // Output projection (quantized)
        if let Some(t) = gguf.find_tensor("output.weight") {
            fixed_bytes += t.data_size();
        } else if let Some(t) = gguf.find_tensor("token_embd.weight") {
            fixed_bytes += t.data_size(); // tied embeddings
        }
        // All norms (tiny, f16)
        for t in &gguf.tensors {
            if t.name.contains("norm") {
                fixed_bytes += t.n_elements() * 2;
            }
        }

        // KV cache
        let ctx = desired_context.min(4096); // streaming mode: cap context
        let kv_bytes: u64 = (0..config.n_layers as usize)
            .map(|i| {
                let hd = config.layer_head_dim(i) as u64;
                2 * ctx as u64 * config.n_kv_heads as u64 * hd * 2
            })
            .sum();
        fixed_bytes += kv_bytes;

        // Scratch + cuBLAS + dequant
        let scratch_bytes = {
            let s = desired_batch.unwrap_or(ctx.min(2048)).min(ctx).max(1) as u64;
            let d = config.dim as u64;
            let ff = config.ff_dim as u64;
            let nh = config.n_heads as u64;
            let nkv = config.n_kv_heads as u64;
            let hd = config.max_head_dim as u64;
            let v = config.vocab_size as u64;
            (s * d + s * nh * hd + 2 * s * nkv * hd + s * d + s * d + 3 * s * ff + s * d + v) * 2
                + s * d * 4
        };
        fixed_bytes += scratch_bytes;
        fixed_bytes += 32 * 1024 * 1024; // cuBLAS
        fixed_bytes += 128 * 1024 * 1024; // dequant scratch

        // 2 DMA slots
        let dma_slot_bytes = 2 * max_layer_bytes;

        let total_fixed = fixed_bytes + dma_slot_bytes + headroom;
        tracing::info!(
            fixed_mb = fixed_bytes / (1024 * 1024),
            dma_mb = dma_slot_bytes / (1024 * 1024),
            headroom_mb = headroom / (1024 * 1024),
            total_fixed_mb = total_fixed / (1024 * 1024),
            available_mb = available_bytes / (1024 * 1024),
            avg_layer_mb = avg_layer_bytes / (1024 * 1024),
            max_layer_mb = max_layer_bytes / (1024 * 1024),
            "streaming budget check"
        );
        if total_fixed >= available_bytes {
            return None;
        }

        let available_for_layers = available_bytes - total_fixed;
        // Reserve extra for 2 shell LayerWeights (allocated as f16 upper bounds, ~2× layer size)
        let shell_overhead = 2 * avg_layer_bytes * 2; // f16 shells are ~2× quantized size
        let safe_available = if available_for_layers > shell_overhead {
            available_for_layers - shell_overhead
        } else {
            0
        };
        let n_resident = (safe_available / avg_layer_bytes).min(config.n_layers as u64) as u32;
        let n_streamed = config.n_layers - n_resident;

        Some(StreamingPlan {
            n_resident,
            n_streamed,
            per_layer_bytes: avg_layer_bytes,
            max_layer_bytes,
            fixed_bytes,
            dma_slot_bytes,
            kv_cache_bytes: kv_bytes,
            context_length: ctx,
            total_layers: config.n_layers,
        })
    }

    pub fn weights_mb(&self) -> u64 {
        self.weights_bytes / (1024 * 1024)
    }
    pub fn kv_cache_mb(&self) -> u64 {
        self.kv_cache_bytes / (1024 * 1024)
    }
    pub fn scratch_mb(&self) -> u64 {
        self.scratch_bytes / (1024 * 1024)
    }
    pub fn dequant_scratch_mb(&self) -> u64 {
        self.dequant_scratch_bytes / (1024 * 1024)
    }
    pub fn total_mb(&self) -> u64 {
        self.total_bytes / (1024 * 1024)
    }
    pub fn peak_mb(&self) -> u64 {
        self.peak_bytes / (1024 * 1024)
    }

    /// Print a clear VRAM budget report.
    pub fn print_report(&self, available_mb: Option<u64>) {
        println!("╔══════════════════════════════════════════╗");
        println!("║          VRAM BUDGET REPORT              ║");
        println!("╠══════════════════════════════════════════╣");
        println!(
            "║  Weights (quant+f16):  {:>6} MB          ║",
            self.weights_mb()
        );
        println!(
            "║  Dequant scratch:      {:>6} MB          ║",
            self.dequant_scratch_mb()
        );
        println!(
            "║  KV cache ({}k ctx): {:>6} MB          ║",
            self.context_length / 1024,
            self.kv_cache_mb()
        );
        println!(
            "║  Forward scratch (b={}):  {:>4} MB          ║",
            self.max_batch,
            self.scratch_mb()
        );
        println!(
            "║  cuBLAS workspace:     {:>6} MB          ║",
            self.cublas_bytes / (1024 * 1024)
        );
        println!("╠══════════════════════════════════════════╣");
        println!(
            "║  Steady-state:         {:>6} MB          ║",
            self.total_mb()
        );
        println!(
            "║  Peak (during load):   {:>6} MB          ║",
            self.peak_mb()
        );
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
