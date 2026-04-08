pub mod config;
pub mod forward;
pub mod kv_cache;
pub mod sample;
pub mod vram_plan;
pub mod weights;

use config::ModelConfig;
use forward::ScratchBuffers;
use kv_cache::KvCache;
use vram_plan::VramPlan;
use weights::{LoadError, ModelWeights};

use chew_gguf::GgufFile;
use chew_kernel::GpuKernels;
use chew_vram::VramAllocator;
use cudarc::driver::CudaStream;
use std::path::Path;
use std::sync::Arc;
use tracing::info;

/// The Chew inference engine.
///
/// Owns the model weights, KV cache, GPU kernels, and scratch memory.
/// Call `generate()` to produce tokens.
pub struct ChewEngine {
    pub config: ModelConfig,
    weights: ModelWeights,
    kernels: GpuKernels,
    kv_cache: KvCache,
    scratch: ScratchBuffers,
    stream: Arc<CudaStream>,
    gpu_idx: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("load: {0}")]
    Load(#[from] LoadError),
    #[error("kernel: {0}")]
    Kernel(#[from] chew_kernel::KernelError),
    #[error("driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("gguf: {0}")]
    Gguf(#[from] chew_gguf::GgufError),
    #[error("vram: {0}")]
    Vram(#[from] chew_vram::VramError),
}

impl ChewEngine {
    /// Load a GGUF model and prepare for inference.
    ///
    /// Computes exact VRAM budget before allocating anything.
    /// If max_context is None, auto-selects the largest context that fits.
    pub fn load(
        model_path: impl AsRef<Path>,
        alloc: &VramAllocator,
        gpu_idx: usize,
        max_context: Option<u32>,
    ) -> Result<Self, EngineError> {
        let path = model_path.as_ref();
        info!(path = %path.display(), "loading model");

        // 1. Parse GGUF
        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf.header)?;

        info!(
            arch = %config.arch,
            layers = config.n_layers,
            dim = config.dim,
            heads = config.n_heads,
            kv_heads = config.n_kv_heads,
            ff = config.ff_dim,
            vocab = config.vocab_size,
            "model config"
        );

        let stream = Arc::clone(alloc.stream(gpu_idx));

        // 2. Compute VRAM budget BEFORE allocating anything
        let free_bytes = {
            let (free, total) = stream.context().mem_get_info()
                .map_err(EngineError::Driver)?;
            info!(free_mb = free / (1024*1024), total_mb = total / (1024*1024), "GPU VRAM");
            free as u64
        };

        let desired_ctx = max_context.unwrap_or(config.context_length.min(32768));
        let plan = VramPlan::fit(&config, &gguf, desired_ctx, free_bytes)
            .ok_or_else(|| EngineError::Load(LoadError::MissingTensor(
                format!("not enough VRAM: need model + 256 ctx, have {} MB free",
                    free_bytes / (1024 * 1024))
            )))?;

        // Print clear VRAM report
        plan.print_report(Some(free_bytes / (1024 * 1024)));

        let max_seq = plan.context_length;
        let max_batch = plan.max_batch;

        info!(
            weights_mb = plan.weights_mb(),
            kv_mb = plan.kv_cache_mb(),
            scratch_mb = plan.scratch_mb(),
            dequant_scratch_mb = plan.dequant_scratch_mb(),
            total_mb = plan.total_mb(),
            peak_mb = plan.peak_mb(),
            free_mb = free_bytes / (1024 * 1024),
            context = max_seq,
            max_batch,
            tied_embeddings = plan.tied_embeddings,
            "VRAM plan"
        );

        if max_seq < desired_ctx {
            info!(
                desired = desired_ctx,
                actual = max_seq,
                "reduced context length to fit VRAM"
            );
        }

        // 3. Load GPU kernels — compute max weight matrix size for dequant scratch
        let max_weight_elems = {
            let d = config.dim as usize;
            let nh = config.n_heads as usize;
            let hd = config.head_dim as usize;
            let ff = config.ff_dim as usize;
            let v = config.vocab_size as usize;
            // Largest matrices: ffn_gate/up [ff_dim, dim], output [vocab, dim]
            [nh * hd * d, ff * d, v * d].into_iter().max().unwrap()
        };
        // max_k for GEMV Q8_1 buffer: largest K dimension in any weight matrix
        let max_k = config.ff_dim.max(config.dim) as usize;
        let kernels = GpuKernels::load(&stream, max_weight_elems, max_k)?;

        // 4. Load + dequantize weights
        let weights =
            ModelWeights::load(&gguf, &config, alloc, &kernels.dequant, gpu_idx)?;

        // 5. Allocate KV cache
        let kv_cache = KvCache::alloc(&config, max_seq, &stream)?;

        // 6. Allocate scratch buffers
        let scratch = ScratchBuffers::alloc(&config, max_batch, max_seq, &stream)?;

        info!(context = max_seq, max_batch, "engine ready");

        Ok(Self {
            config,
            weights,
            kernels,
            kv_cache,
            scratch,
            stream,
            gpu_idx,
        })
    }

    /// Generate tokens from input token IDs.
    ///
    /// Returns generated token IDs (excluding the input).
    pub fn generate(
        &mut self,
        input_tokens: &[u32],
        max_new_tokens: u32,
        params: &sample::SampleParams,
        eos_token: u32,
    ) -> Result<Vec<u32>, EngineError> {
        self.kv_cache.reset();

        let mut all_tokens: Vec<u32> = input_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();

        // Prefill: process all input tokens at once
        let prefill_len = input_tokens.len() as u32;
        let token_ids_i32: Vec<i32> = input_tokens.iter().map(|&t| t as i32).collect();
        let mut token_ids_gpu = self.stream
            .alloc_zeros::<i32>(prefill_len as usize)
            .map_err(EngineError::Driver)?;
        self.stream
            .memcpy_htod(&token_ids_i32, &mut token_ids_gpu)
            .map_err(EngineError::Driver)?;

        // Embed tokens → f32 hidden state (avoids f16 precision loss over 32 layers)
        let mut hidden = self.stream
            .alloc_zeros::<f32>((prefill_len * self.config.dim) as usize)
            .map_err(EngineError::Driver)?;
        self.kernels.ops.embed_tokens_f32(
            &self.weights.token_embd,
            &token_ids_gpu,
            &mut hidden,
            prefill_len,
            self.config.dim,
        )?;

        // Forward pass on prefill
        forward::forward(
            &mut hidden,
            &self.weights,
            &self.config,
            &mut self.kernels,
            &mut self.kv_cache,
            &mut self.scratch,
            prefill_len,
        )?;

        // Sample first token from logits of last position
        // Logits are f16 — download and convert to f32 for sampling
        let vocab = self.config.vocab_size as usize;
        let mut logits_f16 = vec![half::f16::ZERO; vocab];
        let mut logits_f32 = vec![0.0f32; vocab];
        let last_logit_offset = ((prefill_len - 1) * self.config.vocab_size) as usize;
        let logit_view = self.scratch.logits.slice(
            last_logit_offset..last_logit_offset + vocab,
        );
        self.stream
            .memcpy_dtoh(&logit_view, &mut logits_f16)
            .map_err(EngineError::Driver)?;
        for (i, &v) in logits_f16.iter().enumerate() {
            logits_f32[i] = v.to_f32();
        }

        // Debug: check logits sanity
        let nan_count = logits_f32.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_f32.iter().filter(|x| x.is_infinite()).count();
        let zero_count = logits_f32.iter().filter(|x| **x == 0.0).count();
        let max_logit = logits_f32.iter().cloned().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        let min_logit = logits_f32.iter().cloned().filter(|x| x.is_finite()).fold(f32::INFINITY, f32::min);
        info!(nan_count, inf_count, zero_count, max_logit, min_logit, vocab = logits_f32.len(), "prefill logits");

        // Debug: show top 10 tokens and known-good token logits
        {
            let mut indexed: Vec<(usize, f32)> = logits_f32.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top10: Vec<(usize, f32)> = indexed.into_iter().take(10).collect();
            info!(?top10, "top 10 logit tokens");
            // Check where reference top tokens rank in chew's output
            // Reference with chat template: 60704='Paris'(24.82), 791='The'(24.25), 1131='...'(19.55)
            if logits_f32.len() > 60704 {
                info!(
                    tok_60704_paris = logits_f32[60704],
                    tok_791_the = logits_f32[791],
                    tok_1131_dots = logits_f32[1131],
                    tok_334_stars = logits_f32[334],
                    "reference token logits"
                );
                // Dump a few logit ranges to see if there's a pattern
                let first5: Vec<f32> = logits_f32[0..5].to_vec();
                let mid5: Vec<f32> = logits_f32[60700..60710].to_vec();
                info!(?first5, ?mid5, "logit samples");
            }
        }

        let mut next_token = sample::sample_token(&mut logits_f32, params, &all_tokens);
        generated.push(next_token);
        all_tokens.push(next_token);

        if next_token == eos_token {
            return Ok(generated);
        }

        // Decode loop: one token at a time
        let pos_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        // Pre-allocate decode buffers ONCE (no per-token GPU allocs!)
        let mut tok_gpu = self.stream
            .alloc_zeros::<i32>(1)
            .map_err(EngineError::Driver)?;
        let mut decode_hidden = self.stream
            .alloc_zeros::<f32>(self.config.dim as usize)
            .map_err(EngineError::Driver)?;

        for _step in 1..max_new_tokens {
            let last = *all_tokens.last().unwrap();
            let tok_i32 = [last as i32];
            self.stream
                .memcpy_htod(&tok_i32, &mut tok_gpu)
                .map_err(EngineError::Driver)?;

            self.kernels.ops.embed_tokens_f32(
                &self.weights.token_embd,
                &tok_gpu,
                &mut decode_hidden,
                1,
                self.config.dim,
            )?;

            forward::forward(
                &mut decode_hidden,
                &self.weights,
                &self.config,
                &mut self.kernels,
                &mut self.kv_cache,
                &mut self.scratch,
                1,
            )?;

            // Greedy: GPU argmax (4 bytes). Non-greedy: CPU sampling with heap-based top-K.
            if params.temperature == 0.0 {
                self.kernels.ops.argmax_f16(
                    &self.scratch.logits, &mut tok_gpu, self.config.vocab_size,
                )?;
                let mut result = [0i32];
                self.stream.memcpy_dtoh(&tok_gpu, &mut result)
                    .map_err(EngineError::Driver)?;
                next_token = result[0] as u32;
            } else {
                self.stream.memcpy_dtoh(
                    &self.scratch.logits.slice(0..vocab), &mut logits_f16,
                ).map_err(EngineError::Driver)?;
                for (i, &v) in logits_f16.iter().enumerate() {
                    logits_f32[i] = v.to_f32();
                }
                next_token = sample::sample_token(&mut logits_f32, params, &all_tokens);
            }
            generated.push(next_token);
            all_tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
        }

        Ok(generated)
    }

    /// Reset the KV cache (start a new conversation).
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}
