pub mod arch;
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
use weights::{LoadError, ModelWeights, StreamingWeights};

use chew_gguf::GgufFile;
use chew_kernel::GpuKernels;
use chew_vram::VramAllocator;
use cudarc::driver::CudaStream;
use std::path::Path;
use std::sync::Arc;
use tracing::info;

/// Weight storage mode: all-GPU or streaming from host RAM.
pub enum WeightStorage {
    /// All layers fit in VRAM — fastest path.
    Normal(ModelWeights),
    /// Some layers streamed from host RAM — double-buffered DMA.
    Streaming(StreamingWeights),
    /// Gemma 4 MoE — own layer structs + streaming.
    Moe(arch::gemma4_moe::MoeModelWeights),
}

impl WeightStorage {
    pub fn token_embd(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        match self {
            WeightStorage::Normal(w) => &w.token_embd,
            WeightStorage::Streaming(w) => &w.token_embd,
            WeightStorage::Moe(w) => &w.token_embd,
        }
    }

    pub fn output_norm(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        match self {
            WeightStorage::Normal(w) => &w.output_norm,
            WeightStorage::Streaming(w) => &w.output_norm,
            WeightStorage::Moe(w) => &w.output_norm,
        }
    }

    pub fn output(&self) -> &weights::QuantWeight {
        match self {
            WeightStorage::Normal(w) => &w.output,
            WeightStorage::Streaming(w) => &w.output,
            WeightStorage::Moe(w) => &w.output,
        }
    }

    pub fn per_layer_token_embd(&self) -> Option<&weights::QuantWeight> {
        match self {
            WeightStorage::Normal(w) => w.per_layer_token_embd.as_ref(),
            WeightStorage::Streaming(w) => w.per_layer_token_embd.as_ref(),
            WeightStorage::Moe(w) => w.per_layer_token_embd.as_ref(),
        }
    }

    pub fn per_layer_model_proj(&self) -> Option<&weights::QuantWeight> {
        match self {
            WeightStorage::Normal(w) => w.per_layer_model_proj.as_ref(),
            WeightStorage::Streaming(w) => w.per_layer_model_proj.as_ref(),
            WeightStorage::Moe(w) => w.per_layer_model_proj.as_ref(),
        }
    }

    pub fn per_layer_proj_norm(&self) -> Option<&cudarc::driver::CudaSlice<half::f16>> {
        match self {
            WeightStorage::Normal(w) => w.per_layer_proj_norm.as_ref(),
            WeightStorage::Streaming(w) => w.per_layer_proj_norm.as_ref(),
            WeightStorage::Moe(w) => w.per_layer_proj_norm.as_ref(),
        }
    }

    pub fn rope_freq_factors(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        match self {
            WeightStorage::Normal(w) => w.rope_freq_factors.as_ref(),
            WeightStorage::Streaming(w) => w.rope_freq_factors.as_ref(),
            WeightStorage::Moe(w) => w.rope_freq_factors.as_ref(),
        }
    }

    pub fn is_streaming(&self) -> bool {
        matches!(self, WeightStorage::Streaming(_) | WeightStorage::Moe(_))
    }
}

/// The Chew inference engine.
///
/// Owns the model weights, KV cache, GPU kernels, and scratch memory.
/// Call `generate()` to produce tokens.
pub struct ChewEngine {
    pub config: ModelConfig,
    weights: WeightStorage,
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
            max_head_dim = config.max_head_dim,
            n_kv_layers = config.n_kv_layers,
            logit_softcap = ?config.logit_softcap,
            sliding_window = ?config.sliding_window,
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

        // Try normal loading first (min 2k context), then streaming fallback
        let min_useful_ctx = 2048u32;
        let streaming_plan = if VramPlan::fit(&config, &gguf, desired_ctx, free_bytes)
            .filter(|p| p.context_length >= min_useful_ctx)
            .is_none() {
            // Normal doesn't fit — try streaming
            let sp = VramPlan::fit_streaming(&config, &gguf, desired_ctx, free_bytes);
            if let Some(ref plan) = sp {
                plan.print_report(free_bytes / (1024 * 1024));
                info!(
                    resident = plan.n_resident, streamed = plan.n_streamed,
                    layer_mb = plan.per_layer_bytes / (1024*1024),
                    "streaming mode: {} of {} layers in VRAM",
                    plan.n_resident, plan.total_layers,
                );
            }
            sp
        } else {
            None
        };

        let plan = VramPlan::fit(&config, &gguf, desired_ctx, free_bytes)
            .or_else(|| {
                // In streaming mode, create a minimal plan for just the resident parts
                streaming_plan.as_ref().map(|sp| {
                    VramPlan::compute(&config, &gguf, sp.context_length, sp.context_length.min(512))
                })
            })
            .ok_or_else(|| EngineError::Load(LoadError::MissingTensor(
                format!("not enough VRAM even for streaming: need fixed overhead, have {} MB free",
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
            let hd = config.max_head_dim as usize;
            let ff = config.ff_dim as usize;
            let v = config.vocab_size as usize;
            // Largest matrices: ffn_gate/up [ff_dim, dim], output [vocab, dim]
            // For Gemma 4: per_layer_token_embd can be huge but stays quantized
            let mut max = [nh * hd * d, ff * d, v * d].into_iter().max().unwrap();
            // MoE: expert tensors can be huge
            if config.is_moe() {
                let exp_ff = config.expert_ff_dim as usize;
                let n_exp = config.n_experts as usize;
                // gate_up_exps: [dim, expert_ff*2, n_experts] — we dequant one expert at a time
                max = max.max(d * exp_ff * 2);
                // down_exps: [expert_ff, dim, n_experts]
                max = max.max(exp_ff * d);
                // Full 3D tensor for GPU allocation sizing
                let _ = n_exp; // used in VRAM plan, not here
            }
            // per_layer_model_proj: [n_embd_per_layer * n_layers, dim]
            if let Some(epl) = config.embd_per_layer {
                let proj_size = (epl as usize) * (config.n_layers as usize) * d;
                max = max.max(proj_size);
            }
            max
        };
        // max_k for GEMV Q8_1 buffer: largest K dimension in any weight matrix
        let max_k = config.ff_dim.max(config.dim) as usize;
        let kernels = GpuKernels::load(&stream, max_weight_elems, max_k)?;

        // 4. Load + dequantize weights (normal, streaming, or MoE)
        let weights = if config.is_moe() {
            // MoE always uses streaming (expert weights are huge)
            let sp = streaming_plan.as_ref().ok_or_else(|| {
                EngineError::Load(LoadError::MissingTensor(
                    "MoE model requires streaming mode but VRAM plan failed".into()
                ))
            })?;
            info!("loading MoE in STREAMING mode: {} resident, {} streamed",
                sp.n_resident, sp.n_streamed);
            WeightStorage::Moe(
                arch::gemma4_moe::MoeModelWeights::load(&gguf, &config, sp, alloc, &kernels.dequant, gpu_idx)?
            )
        } else if let Some(ref sp) = streaming_plan {
            info!("loading in STREAMING mode: {} resident, {} streamed",
                sp.n_resident, sp.n_streamed);
            WeightStorage::Streaming(
                StreamingWeights::load(&gguf, &config, sp, alloc, &kernels.dequant, gpu_idx)?
            )
        } else {
            WeightStorage::Normal(
                ModelWeights::load(&gguf, &config, alloc, &kernels.dequant, gpu_idx)?
            )
        };

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
            self.weights.token_embd(),
            &token_ids_gpu,
            &mut hidden,
            prefill_len,
            self.config.dim,
        )?;

        // Scale embeddings by sqrt(dim) for Gemma 4
        if self.config.is_gemma4() {
            let scale = (self.config.dim as f32).sqrt();
            self.kernels.ops.scale_f32_inplace(&mut hidden, prefill_len * self.config.dim, scale)?;
        }

        let debug_logits = std::env::var("CHEW_DEBUG_LOGITS").is_ok();
        let debug_decode = std::env::var("CHEW_DEBUG_DECODE").is_ok();

        // Debug: dump hidden state for last position after embedding+scale
        if self.config.is_gemma4() && debug_decode {
            let last_off = ((prefill_len - 1) * self.config.dim) as usize;
            let mut dbg = vec![0.0f32; 8];
            self.stream.memcpy_dtoh(&hidden.slice(last_off..last_off+8), &mut dbg)
                .map_err(EngineError::Driver)?;
            info!(?dbg, pos = prefill_len - 1, "EMBED last pos first 8 values");
        }

        // Compute per-layer token embeddings (Gemma 4 only)
        let pe = if self.config.is_gemma4() {
            self.compute_per_layer_embeddings(&token_ids_gpu, &hidden, prefill_len)?
        } else {
            None
        };

        // Forward pass on prefill
        self.run_forward(&mut hidden, pe.as_ref(), prefill_len)?;

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

        if debug_logits {
            let nan_count = logits_f32.iter().filter(|x| x.is_nan()).count();
            let inf_count = logits_f32.iter().filter(|x| x.is_infinite()).count();
            let zero_count = logits_f32.iter().filter(|x| **x == 0.0).count();
            let max_logit = logits_f32.iter().cloned().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
            let min_logit = logits_f32.iter().cloned().filter(|x| x.is_finite()).fold(f32::INFINITY, f32::min);
            info!(nan_count, inf_count, zero_count, max_logit, min_logit, vocab = logits_f32.len(), "prefill logits");

            let mut indexed: Vec<(usize, f32)> = logits_f32.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top10: Vec<(usize, f32)> = indexed.into_iter().take(10).collect();
            info!(?top10, "top 10 logit tokens");
            if logits_f32.len() > 60704 {
                info!(
                    tok_60704_paris = logits_f32[60704],
                    tok_791_the = logits_f32[791],
                    tok_1131_dots = logits_f32[1131],
                    tok_334_stars = logits_f32[334],
                    "reference token logits"
                );
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
        // Pre-allocate decode buffers ONCE (no per-token GPU allocs!)
        let mut tok_gpu = self.stream
            .alloc_zeros::<i32>(1)
            .map_err(EngineError::Driver)?;
        let mut decode_hidden = self.stream
            .alloc_zeros::<f32>(self.config.dim as usize)
            .map_err(EngineError::Driver)?;

        // CUDA Graph: capture on first decode step, replay on subsequent steps.
        // Eliminates ~300 cuLaunchKernel calls per token (~1-2ms host overhead).
        // Works for both greedy and non-greedy: graph captures forward pass only,
        // sampling (argmax or CPU top-k) happens outside the graph.
        // Disable CUDA graph for Gemma 4 (different forward pass, variable head_dim)
        // Disable CUDA graph for streaming mode (weight data changes between layers)
        let use_graph = std::env::var("CHEW_NO_GRAPH").is_err()
            && max_new_tokens > 20
            && !self.config.is_gemma4()
            && !self.weights.is_streaming();
        let mut decode_graph: Option<forward::DecodeGraph> = None;

        let greedy = params.temperature == 0.0;
        let profile = std::env::var("CHEW_PROFILE").is_ok();

        if greedy {
            // ZERO-SYNC GREEDY PATH:
            // All tokens stay on GPU. argmax→tok_gpu→embed_tokens chain has no host roundtrip.
            // GPU token buffer stores all generated token IDs. Batch download at end.
            let mut token_buf_gpu = self.stream
                .alloc_zeros::<i32>(max_new_tokens as usize)
                .map_err(EngineError::Driver)?;
            let mut n_generated = 0u32;

            // GPU timing: sync before loop, measure total
            self.stream.synchronize().map_err(EngineError::Driver)?;
            let decode_t0 = std::time::Instant::now();

            for _step in 1..max_new_tokens {
                if _step == 1 {
                    // First decode: upload token from host
                    let last = *all_tokens.last().unwrap();
                    let tok_i32 = [last as i32];
                    self.stream.memcpy_htod(&tok_i32, &mut tok_gpu)
                        .map_err(EngineError::Driver)?;
                }
                // else: tok_gpu already has argmax from previous step

                self.kernels.ops.embed_tokens_f32(
                    self.weights.token_embd(), &tok_gpu, &mut decode_hidden, 1, self.config.dim,
                )?;

                // Scale embeddings by sqrt(dim) for Gemma 4
                if self.config.is_gemma4() {
                    let scale = (self.config.dim as f32).sqrt();
                    self.kernels.ops.scale_f32_inplace(&mut decode_hidden, self.config.dim, scale)?;
                }

                if _step == 1 && debug_decode {
                    let mut tok_host = [0i32];
                    self.stream.memcpy_dtoh(&tok_gpu, &mut tok_host).ok();
                    let mut emb = vec![0.0f32; 4];
                    self.stream.memcpy_dtoh(&decode_hidden.slice(0..4), &mut emb).ok();
                    info!(?tok_host, ?emb, "decode embed check");
                }

                if use_graph {
                    if let Some(ref mut dg) = decode_graph {
                        let pos = self.kv_cache.pos();
                        dg.replay(pos, self.kv_cache.kv_stride(), &self.stream)?;
                        self.kv_cache.advance(1);
                    } else {
                        let pos = self.kv_cache.pos();
                        match &self.weights {
                            WeightStorage::Normal(w) => {
                                decode_graph = Some(forward::DecodeGraph::capture(
                                    &mut decode_hidden, w, &self.config,
                                    &mut self.kernels, &mut self.kv_cache, &mut self.scratch,
                                    pos, &self.stream,
                                )?);
                            }
                            WeightStorage::Streaming(_) | WeightStorage::Moe(_) => {
                                unreachable!("CUDA graph disabled for streaming/MoE mode");
                            }
                        }
                    }
                } else {
                    let pe_t0 = if profile && self.config.is_gemma4() {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let decode_pe = if self.config.is_gemma4() {
                        self.compute_per_layer_embeddings(&tok_gpu, &decode_hidden, 1)?
                    } else {
                        None
                    };
                    if let Some(t0) = pe_t0 {
                        info!(compute_pe_us = t0.elapsed().as_micros(), "PROFILE decode compute_pe");
                    }
                    self.run_forward(&mut decode_hidden, decode_pe.as_ref(), 1)?;
                }

                if _step == 1 && debug_logits {
                    let vocab = self.config.vocab_size as usize;
                    let mut logits_f16 = vec![half::f16::ZERO; vocab];
                    let mut logits_f32 = vec![0.0f32; vocab];
                    self.stream
                        .memcpy_dtoh(&self.scratch.logits.slice(0..vocab), &mut logits_f16)
                        .map_err(EngineError::Driver)?;
                    for (i, &v) in logits_f16.iter().enumerate() {
                        logits_f32[i] = v.to_f32();
                    }
                    let mut indexed: Vec<(usize, f32)> = logits_f32.iter().cloned().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top10: Vec<(usize, f32)> = indexed.into_iter().take(10).collect();
                    info!(?top10, "decode step1 top 10 logit tokens");
                }

                // GPU argmax → tok_gpu (stays on GPU)
                self.kernels.ops.argmax_f16(
                    &self.scratch.logits, &mut tok_gpu, self.config.vocab_size,
                )?;

                // Copy token to GPU buffer (1 int, async, no sync)
                // Use a tiny device-to-device copy
                {
                    let mut dst = token_buf_gpu.slice_mut(n_generated as usize..(n_generated + 1) as usize);
                    self.stream.memcpy_dtod(&tok_gpu, &mut dst)
                        .map_err(EngineError::Driver)?;
                }
                n_generated += 1;

                // NO HOST SYNC — entire loop runs without host-GPU synchronization
            }

            // Sync and measure decode time
            self.stream.synchronize().map_err(EngineError::Driver)?;
            let decode_elapsed = decode_t0.elapsed();
            if n_generated > 0 {
                let ms_per_tok = decode_elapsed.as_secs_f64() * 1000.0 / n_generated as f64;
                let tps = n_generated as f64 / decode_elapsed.as_secs_f64();
                info!(n_generated, ms_per_tok = format!("{:.2}", ms_per_tok), decode_tps = format!("{:.1}", tps), graph = use_graph, "decode timing");
            }

            // Batch download ALL tokens at end (one sync for all)
            if n_generated > 0 {
                let mut host_tokens = vec![0i32; n_generated as usize];
                let src = token_buf_gpu.slice(0..n_generated as usize);
                self.stream.memcpy_dtoh(&src, &mut host_tokens)
                    .map_err(EngineError::Driver)?;
                for &t in &host_tokens {
                    let tok = t as u32;
                    generated.push(tok);
                    if tok == eos_token { break; } // truncate at EOS
                }
            }
        } else {
            // Non-greedy: CPU sampling, per-step sync (slower but supports top-k/top-p)
            for _step in 1..max_new_tokens {
                let last = *all_tokens.last().unwrap();
                let tok_i32 = [last as i32];
                self.stream.memcpy_htod(&tok_i32, &mut tok_gpu)
                    .map_err(EngineError::Driver)?;

                self.kernels.ops.embed_tokens_f32(
                    self.weights.token_embd(), &tok_gpu, &mut decode_hidden, 1, self.config.dim,
                )?;

                // Scale embeddings by sqrt(dim) for Gemma 4
                if self.config.is_gemma4() {
                    let scale = (self.config.dim as f32).sqrt();
                    self.kernels.ops.scale_f32_inplace(&mut decode_hidden, self.config.dim, scale)?;
                }

                // Debug: check decode embedding (sampling path)
                {
                    let mut tok_host = [0i32];
                    self.stream.memcpy_dtoh(&tok_gpu, &mut tok_host).ok();
                    let mut emb = vec![0.0f32; 4];
                    self.stream.memcpy_dtoh(&decode_hidden.slice(0..4), &mut emb).ok();
                    info!(?tok_host, ?emb, "decode embed check (sampling)");
                }

                {
                    let decode_pe = if self.config.is_gemma4() {
                        self.compute_per_layer_embeddings(&tok_gpu, &decode_hidden, 1)?
                    } else {
                        None
                    };
                    self.run_forward(&mut decode_hidden, decode_pe.as_ref(), 1)?;
                }

                self.stream.memcpy_dtoh(
                    &self.scratch.logits.slice(0..vocab), &mut logits_f16,
                ).map_err(EngineError::Driver)?;
                for (i, &v) in logits_f16.iter().enumerate() {
                    logits_f32[i] = v.to_f32();
                }
                next_token = sample::sample_token(&mut logits_f32, params, &all_tokens);
                generated.push(next_token);
                all_tokens.push(next_token);
                if next_token == eos_token { break; }
            }
        }

        Ok(generated)
    }

    /// Run the forward pass, dispatching based on weight storage type.
    fn run_forward(
        &mut self,
        hidden: &mut cudarc::driver::CudaSlice<f32>,
        pe: Option<&forward::PerLayerEmbeddings>,
        seq_len: u32,
    ) -> Result<(), EngineError> {
        // Dispatch by weight storage type
        // Streaming and MoE need mutable access to weights (for shell uploads)
        if matches!(self.weights, WeightStorage::Streaming(_)) {
            forward::forward_streaming(
                hidden, &mut self.weights, &self.config,
                &mut self.kernels, &mut self.kv_cache, &mut self.scratch,
                seq_len, pe, &self.stream,
            )?;
        } else if let WeightStorage::Moe(ref mut moe_w) = self.weights {
            arch::gemma4_moe::forward_moe_streaming(
                hidden, moe_w, &self.config,
                &mut self.kernels, &mut self.kv_cache, &mut self.scratch,
                seq_len, pe, &self.stream,
            )?;
        } else if let WeightStorage::Normal(ref w) = self.weights {
            if self.config.is_gemma4() {
                arch::gemma4_dense::forward(
                    hidden, w, &self.config,
                    &mut self.kernels, &mut self.kv_cache, &mut self.scratch,
                    seq_len, pe,
                )?;
            } else {
                arch::llama::forward(
                    hidden, w, &self.config,
                    &mut self.kernels, &mut self.kv_cache, &mut self.scratch,
                    seq_len,
                )?;
            }
        }
        Ok(())
    }

    /// Compute per-layer token embeddings for a set of token IDs.
    ///
    /// Full llama.cpp flow:
    /// 1. Gather + dequant per_layer_token_embd → tok_embd [n_tokens, n_layers*epl]
    /// 2. Scale tok_embd by sqrt(epl)
    /// 3. Convert hidden_f32 to f16, project through per_layer_model_proj^T → proj [n_tokens, n_layers*epl]
    /// 4. Scale proj by 1/sqrt(dim)
    /// 5. RMS-norm proj rows (reshaped as [n_tokens*n_layers, epl]) with per_layer_proj_norm
    /// 6. result = (tok_embd_scaled + proj_normed) / sqrt(2)
    fn compute_per_layer_embeddings(
        &mut self,
        token_ids_gpu: &cudarc::driver::CudaSlice<i32>,
        hidden_f32: &cudarc::driver::CudaSlice<f32>,
        n_tokens: u32,
    ) -> Result<Option<forward::PerLayerEmbeddings>, EngineError> {
        let epl = match self.config.embd_per_layer {
            Some(e) => e,
            None => return Ok(None),
        };
        let pe_weight = match self.weights.per_layer_token_embd() {
            Some(w) => w,
            None => return Ok(None),
        };

        let n_layers = self.config.n_layers;
        let dim = self.config.dim;
        let row_width = epl * n_layers;  // total elements per token across all layers

        // Compute bytes per row in the quantized format
        let block_size = pe_weight.quant_type.block_size() as u32;
        let block_bytes = pe_weight.quant_type.block_bytes() as u32;
        let blocks_per_row = row_width / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        // Only log at info level for prefill (many tokens), trace for decode (1 token)
        if n_tokens > 1 {
            info!(
                epl, n_layers, row_width, row_bytes,
                quant = ?pe_weight.quant_type,
                n_tokens,
                "computing per-layer embeddings"
            );
        }

        // Step 1: Gather rows — copy the quantized rows for our token IDs into a contiguous buffer
        let gathered_bytes = n_tokens * row_bytes;
        let mut gathered_quant = self.stream
            .alloc_zeros::<u8>(gathered_bytes as usize)
            .map_err(EngineError::Driver)?;

        self.kernels.ops.gather_rows_quant(
            &pe_weight.data,
            token_ids_gpu,
            &mut gathered_quant,
            row_bytes,
            n_tokens,
        )?;

        // Step 1b: Dequantize all gathered rows to f16: [n_tokens, row_width]
        let total_elements = n_tokens * row_width;
        let mut tok_embd = self.stream
            .alloc_zeros::<half::f16>(total_elements as usize)
            .map_err(EngineError::Driver)?;

        self.kernels.dequant.dequant(
            &gathered_quant,
            &mut tok_embd,
            total_elements,
            pe_weight.quant_type,
        )?;

        // Free gathered quant immediately
        drop(gathered_quant);

        // Step 2: Scale tok_embd by sqrt(epl)
        let tok_scale = (epl as f32).sqrt();
        {
            let src_ptr = &tok_embd as *const cudarc::driver::CudaSlice<half::f16>;
            let dst_ptr = &mut tok_embd as *mut cudarc::driver::CudaSlice<half::f16>;
            unsafe {
                self.kernels.ops.scale_f16(&*src_ptr, &mut *dst_ptr, total_elements, tok_scale)?;
            }
        }

        // Step 3: Project hidden state through per_layer_model_proj
        // per_layer_model_proj is [row_width, dim] quantized (BF16)
        // proj = hidden_f16 @ per_layer_model_proj^T → [n_tokens, row_width]
        let has_proj = self.weights.per_layer_model_proj().is_some()
            && self.weights.per_layer_proj_norm().is_some();

        if has_proj {
            // 3a: Convert hidden f32 → f16 for matmul
            let hidden_elems = n_tokens * dim;
            let mut hidden_f16 = self.stream
                .alloc_zeros::<half::f16>(hidden_elems as usize)
                .map_err(EngineError::Driver)?;
            {
                let mut dst_view = hidden_f16.slice_mut(..);
                self.kernels.ops.copy_f32_to_f16(hidden_f32, &mut dst_view, hidden_elems)?;
            }

            // 3b: Matmul — proj = hidden_f16 @ per_layer_model_proj^T
            //   A = hidden_f16 [n_tokens, dim]  (M=n_tokens, K=dim)
            //   B = per_layer_model_proj [row_width, dim]  (N=row_width, K=dim)
            //   C = proj [n_tokens, row_width]
            let mut proj = self.stream
                .alloc_zeros::<half::f16>(total_elements as usize)
                .map_err(EngineError::Driver)?;

            let proj_w = self.weights.per_layer_model_proj().unwrap();
            self.kernels.gemm.matmul_dequant(
                &hidden_f16,
                &proj_w.data,
                proj_w.quant_type,
                proj_w.n_elements,
                &mut proj,
                n_tokens,    // M
                row_width,   // N
                dim,         // K
                &self.kernels.dequant,
            )?;

            // Free hidden_f16 — no longer needed
            drop(hidden_f16);

            // Step 4: Scale proj by 1/sqrt(dim)
            let proj_scale = 1.0 / (dim as f32).sqrt();
            {
                let src_ptr = &proj as *const cudarc::driver::CudaSlice<half::f16>;
                let dst_ptr = &mut proj as *mut cudarc::driver::CudaSlice<half::f16>;
                unsafe {
                    self.kernels.ops.scale_f16(&*src_ptr, &mut *dst_ptr, total_elements, proj_scale)?;
                }
            }

            // Step 5: RMS-norm proj with per_layer_proj_norm [epl]
            // Conceptually reshape proj as [n_tokens * n_layers, epl] and norm each row
            let norm_rows = n_tokens * n_layers;
            let mut proj_normed = self.stream
                .alloc_zeros::<half::f16>(total_elements as usize)
                .map_err(EngineError::Driver)?;

            let norm_weight = self.weights.per_layer_proj_norm().unwrap();
            self.kernels.ops.rms_norm(
                &proj,
                norm_weight,
                &mut proj_normed,
                norm_rows,
                epl,
                self.config.rms_norm_eps,
            )?;

            // Free un-normed proj
            drop(proj);

            // Step 6: result = (tok_embd_scaled + proj_normed) / sqrt(2)
            // First add: tok_embd + proj_normed → tok_embd (reuse buffer)
            let mut result = self.stream
                .alloc_zeros::<half::f16>(total_elements as usize)
                .map_err(EngineError::Driver)?;
            self.kernels.ops.add_f16(&tok_embd, &proj_normed, &mut result, total_elements)?;

            // Free intermediates
            drop(tok_embd);
            drop(proj_normed);

            // Scale by 1/sqrt(2)
            let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
            {
                let src_ptr = &result as *const cudarc::driver::CudaSlice<half::f16>;
                let dst_ptr = &mut result as *mut cudarc::driver::CudaSlice<half::f16>;
                unsafe {
                    self.kernels.ops.scale_f16(&*src_ptr, &mut *dst_ptr, total_elements, inv_sqrt2)?;
                }
            }

            Ok(Some(forward::PerLayerEmbeddings {
                data: result,
                epl,
                row_width,
                seq_len: n_tokens,
            }))
        } else {
            // No projection weights available — fall back to token embeddings only
            // Scale by 1/sqrt(2) since we already scaled by sqrt(epl)
            let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
            {
                let src_ptr = &tok_embd as *const cudarc::driver::CudaSlice<half::f16>;
                let dst_ptr = &mut tok_embd as *mut cudarc::driver::CudaSlice<half::f16>;
                unsafe {
                    self.kernels.ops.scale_f16(&*src_ptr, &mut *dst_ptr, total_elements, inv_sqrt2)?;
                }
            }

            Ok(Some(forward::PerLayerEmbeddings {
                data: tok_embd,
                epl,
                row_width,
                seq_len: n_tokens,
            }))
        }
    }

    /// Reset the KV cache (start a new conversation).
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}
