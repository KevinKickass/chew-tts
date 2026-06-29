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
use vram_plan::{StreamingPlan, VramPlan};
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
    /// Encoder-only BERT/MiniLM path.
    Bert(arch::bert::BertModelWeights),
    /// Mamba-family SSM path.
    Mamba(arch::mamba::MambaModelWeights),
    /// Some layers streamed from host RAM — double-buffered DMA.
    Streaming(StreamingWeights),
    /// Gemma 4 MoE — own layer structs + streaming.
    Moe(arch::gemma4_moe::MoeModelWeights),
}

impl WeightStorage {
    pub fn token_embd(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        match self {
            WeightStorage::Normal(w) => &w.token_embd,
            WeightStorage::Bert(w) => &w.embeddings.word_embeddings,
            WeightStorage::Mamba(w) => &w.token_embd,
            WeightStorage::Streaming(w) => &w.token_embd,
            WeightStorage::Moe(w) => &w.token_embd,
        }
    }

    pub fn output_norm(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        match self {
            WeightStorage::Normal(w) => &w.output_norm,
            WeightStorage::Bert(_) => panic!("encoder-only BERT weights have no output norm"),
            WeightStorage::Mamba(w) => &w.output_norm,
            WeightStorage::Streaming(w) => &w.output_norm,
            WeightStorage::Moe(w) => &w.output_norm,
        }
    }

    pub fn output(&self) -> &weights::QuantWeight {
        match self {
            WeightStorage::Normal(w) => &w.output,
            WeightStorage::Bert(_) => panic!("encoder-only BERT weights have no LM head"),
            WeightStorage::Mamba(w) => &w.output,
            WeightStorage::Streaming(w) => &w.output,
            WeightStorage::Moe(w) => &w.output,
        }
    }

    pub fn per_layer_token_embd(&self) -> Option<&weights::QuantWeight> {
        match self {
            WeightStorage::Normal(w) => w.per_layer_token_embd.as_ref(),
            WeightStorage::Bert(_) => None,
            WeightStorage::Mamba(_) => None,
            WeightStorage::Streaming(w) => w.per_layer_token_embd.as_ref(),
            WeightStorage::Moe(w) => w.per_layer_token_embd.as_ref(),
        }
    }

    pub fn per_layer_model_proj(&self) -> Option<&weights::QuantWeight> {
        match self {
            WeightStorage::Normal(w) => w.per_layer_model_proj.as_ref(),
            WeightStorage::Bert(_) => None,
            WeightStorage::Mamba(_) => None,
            WeightStorage::Streaming(w) => w.per_layer_model_proj.as_ref(),
            WeightStorage::Moe(w) => w.per_layer_model_proj.as_ref(),
        }
    }

    pub fn per_layer_proj_norm(&self) -> Option<&cudarc::driver::CudaSlice<half::f16>> {
        match self {
            WeightStorage::Normal(w) => w.per_layer_proj_norm.as_ref(),
            WeightStorage::Bert(_) => None,
            WeightStorage::Mamba(_) => None,
            WeightStorage::Streaming(w) => w.per_layer_proj_norm.as_ref(),
            WeightStorage::Moe(w) => w.per_layer_proj_norm.as_ref(),
        }
    }

    pub fn rope_freq_factors(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        match self {
            WeightStorage::Normal(w) => w.rope_freq_factors.as_ref(),
            WeightStorage::Bert(_) => None,
            WeightStorage::Mamba(_) => None,
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
    mamba_state: Option<arch::mamba::MambaRuntimeState>,
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
    #[error("{0}")]
    Unsupported(String),
}

fn all_resident_moe_plan(
    config: &ModelConfig,
    gguf: &GgufFile,
    normal: &VramPlan,
) -> StreamingPlan {
    let mut total_layer_bytes = 0u64;
    let mut max_layer_bytes = 0u64;

    for layer in 0..config.n_layers {
        let prefix = format!("blk.{layer}.");
        let mut layer_bytes = 0u64;
        for t in &gguf.tensors {
            if t.name.starts_with(&prefix) {
                layer_bytes += t.data_size();
            }
        }
        total_layer_bytes += layer_bytes;
        if layer_bytes > max_layer_bytes {
            max_layer_bytes = layer_bytes;
        }
    }

    let per_layer_bytes = if config.n_layers > 0 {
        total_layer_bytes / config.n_layers as u64
    } else {
        0
    };

    StreamingPlan {
        n_resident: config.n_layers,
        n_streamed: 0,
        per_layer_bytes,
        max_layer_bytes,
        fixed_bytes: normal.total_bytes.saturating_sub(total_layer_bytes),
        dma_slot_bytes: 2 * max_layer_bytes,
        kv_cache_bytes: normal.kv_cache_bytes,
        context_length: normal.context_length,
        total_layers: config.n_layers,
    }
}

impl ChewEngine {
    fn validate_prefill_len(&self, seq_len: u32) -> Result<(), EngineError> {
        let max_seq = self.kv_cache.max_seq();
        if seq_len > max_seq {
            return Err(EngineError::Unsupported(format!(
                "prompt too long: {seq_len} tokens exceeds allocated context {max_seq}"
            )));
        }
        let max_batch = self.max_prefill_batch();
        if seq_len > max_batch {
            return Err(EngineError::Unsupported(format!(
                "prompt too long for prefill: {seq_len} tokens exceeds allocated max_batch {max_batch} (context {max_seq})"
            )));
        }
        Ok(())
    }

    fn max_prefill_batch(&self) -> u32 {
        (self.scratch.norm_out.len() / self.config.dim as usize) as u32
    }

    fn clamp_generation_tokens(&self, input_len: u32, requested: u32) -> u32 {
        let max_seq = self.kv_cache.max_seq();
        if input_len >= max_seq {
            return requested.min(1);
        }
        let remaining_decode_steps = max_seq - input_len;
        requested.min(remaining_decode_steps.saturating_add(1))
    }

    fn upload_token_ids(
        &self,
        input_tokens: &[u32],
    ) -> Result<cudarc::driver::CudaSlice<i32>, EngineError> {
        let token_ids_i32: Vec<i32> = input_tokens.iter().map(|&t| t as i32).collect();
        let mut token_ids_gpu = self
            .stream
            .alloc_zeros::<i32>(input_tokens.len())
            .map_err(EngineError::Driver)?;
        self.stream
            .memcpy_htod(&token_ids_i32, &mut token_ids_gpu)
            .map_err(EngineError::Driver)?;
        Ok(token_ids_gpu)
    }

    fn embed_prefill_tokens(
        &mut self,
        token_ids_gpu: &cudarc::driver::CudaSlice<i32>,
        seq_len: u32,
    ) -> Result<cudarc::driver::CudaSlice<f32>, EngineError> {
        let mut hidden = self
            .stream
            .alloc_zeros::<f32>((seq_len * self.config.dim) as usize)
            .map_err(EngineError::Driver)?;
        self.kernels.ops.embed_tokens_f32(
            self.weights.token_embd(),
            token_ids_gpu,
            &mut hidden,
            seq_len,
            self.config.dim,
        )?;

        if self.config.is_gemma4() {
            let scale = (self.config.dim as f32).sqrt();
            self.kernels
                .ops
                .scale_f32_inplace(&mut hidden, seq_len * self.config.dim, scale)?;
        }

        Ok(hidden)
    }

    fn run_prefill_chunk(
        &mut self,
        token_ids_gpu: &cudarc::driver::CudaSlice<i32>,
        hidden: &mut cudarc::driver::CudaSlice<f32>,
        seq_len: u32,
    ) -> Result<(), EngineError> {
        let pe = if self.config.is_gemma4() {
            self.compute_per_layer_embeddings(token_ids_gpu, hidden, seq_len)?
        } else {
            None
        };

        self.run_forward(hidden, pe.as_ref(), seq_len)
    }

    /// Block-diffusion generation (DiffusionGemma). Denoises a fixed-length
    /// canvas via the entropy-bound decoder, returning the argmax tokens
    /// (trimmed at EOS). Single block, `kv_cache=false` (full re-decode/step).
    pub fn generate_diffusion(
        &mut self,
        prompt_tokens: &[u32],
        eb: arch::diffusion_gemma::EbParams,
        seed: u64,
        eos: u32,
    ) -> Result<Vec<u32>, EngineError> {
        use arch::diffusion_gemma as dg;
        let canvas_len = self
            .config
            .canvas_length
            .ok_or_else(|| EngineError::Unsupported("not a diffusion model".into()))?
            as usize;
        if !matches!(self.weights, WeightStorage::Moe(_)) {
            return Err(EngineError::Unsupported("diffusion requires MoE weights".into()));
        }
        let p = prompt_tokens.len();
        let n_tokens = (p + canvas_len) as u32;
        let dim = self.config.dim;
        let vocab = self.config.vocab_size;
        let n_ff = self.config.ff_dim;
        let eps = self.config.rms_norm_eps;
        let n_swa = self.config.sliding_window.unwrap_or(0);
        let softcap = self.config.logit_softcap;
        let dimu = dim as usize;
        let vocabu = vocab as usize;
        let cl = canvas_len;

        let _ = n_tokens;
        let stream = Arc::clone(&self.stream);
        // Prefix-KV: prompt is prefilled once into the KV cache; each step only
        // re-decodes the canvas (C positions) against cached prompt K/V.
        let attn = dg::DiffusionAttn::build_decode(p as u32, cl as u32, n_swa, &stream)?;
        let mut sc_bufs = dg::ScBuffers::alloc(canvas_len as u32, dim, n_ff, vocab, &stream)?;

        let mut canvas_tokens_gpu = stream.alloc_zeros::<i32>(cl)?;
        let mut hidden = stream.alloc_zeros::<f32>(cl * dimu)?;
        let mut canvas_emb = stream.alloc_zeros::<f32>(cl * dimu)?;
        let mut canvas_normed = stream.alloc_zeros::<half::f16>(cl * dimu)?;
        let mut canvas_norm_f32 = stream.alloc_zeros::<f32>(cl * dimu)?;
        let mut canvas_norm_out = stream.alloc_zeros::<half::f16>(cl * dimu)?;
        // Single logit buffer reused as both "previous step's logits" (read by
        // SC at the start of a step) and this step's fresh logits (written by
        // the projection after SC has run). They never overlap in time.
        let mut canvas_logits = stream.alloc_zeros::<half::f16>(cl * vocabu)?;

        // Diagnostics: CHEW_NO_SC disables self-conditioning; CHEW_DIFF_STEPS
        // overrides the step count for fast iteration.
        let no_sc = std::env::var("CHEW_NO_SC").is_ok();

        let mut rng = dg::Rng::new(seed);
        let fixed_canvas: Option<i32> = std::env::var("CHEW_DIFF_FIXED")
            .ok()
            .and_then(|v| v.parse().ok());
        let mut canvas_tokens: Vec<i32> = (0..cl)
            .map(|_| fixed_canvas.unwrap_or_else(|| rng.next_token(vocab) as i32))
            .collect();

        // ── Prefill the prompt once: writes prompt K/V to [0..P), advances to P ──
        {
            let moe = match &mut self.weights {
                WeightStorage::Moe(m) => m,
                _ => unreachable!(),
            };
            let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
            let mut prompt_tokens_gpu = stream.alloc_zeros::<i32>(p)?;
            let mut prompt_hidden = stream.alloc_zeros::<f32>(p * dimu)?;
            stream.memcpy_htod(prompt_i32.as_slice(), &mut prompt_tokens_gpu)?;
            self.kernels.ops.embed_tokens_f32(
                &moe.token_embd,
                &prompt_tokens_gpu,
                &mut prompt_hidden,
                p as u32,
                dim,
            )?;
            self.kernels
                .ops
                .scale_f32_inplace(&mut prompt_hidden, p as u32 * dim, (dim as f32).sqrt())?;
            self.kv_cache.reset();
            // diffusion=None -> causal prompt prefill; advances kv_cache to P.
            arch::gemma4_moe::forward_moe_streaming(
                &mut prompt_hidden,
                moe,
                &self.config,
                &mut self.kernels,
                &mut self.kv_cache,
                &mut self.scratch,
                p as u32,
                None,
                &stream,
                None,
            )?;
        }

        // Device-side entropy-bound: logits stay on the GPU, the kernel reduces
        // each position to argmax/entropy/sample; only these C-sized arrays come
        // back (no 134MB readback).
        let mut rnd_gpu = stream.alloc_zeros::<f32>(cl)?;
        let mut argmax_gpu = stream.alloc_zeros::<u32>(cl)?;
        let mut entropy_gpu = stream.alloc_zeros::<f32>(cl)?;
        let mut sampled_gpu = stream.alloc_zeros::<u32>(cl)?;
        let mut rnd_host = vec![0f32; cl];
        let mut argmax_host = vec![0u32; cl];
        let mut entropy_host = vec![0f32; cl];
        let mut sampled_host = vec![0u32; cl];
        let mut argmax_out = vec![0u32; cl];
        let mut held = 0u32;
        let mut prev_argmax: Option<Vec<u32>> = None;

        let s = std::env::var("CHEW_DIFF_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(eb.steps);
        let envf = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        // Tunables: temp schedule length (was stretched over max steps -> temp
        // stayed high -> slow convergence), early-stop confidence + stability.
        let sched = envf("CHEW_DIFF_SCHED", s as f32).max(1.0);
        let conf = envf("CHEW_DIFF_CONF", eb.confidence);
        let stab = envf("CHEW_DIFF_STAB", eb.stability as f32) as u32;
        for step_idx in 0..s {
            let frac = (step_idx as f32 / sched).min(1.0);
            let t = eb.t_max + (eb.t_min - eb.t_max) * frac;
            let temp_inv = 1.0 / t;
            let sc_use = if step_idx == 0 || no_sc { 0.0 } else { 1.0 };

            let t_gpu = std::time::Instant::now();
            stream.memcpy_htod(canvas_tokens.as_slice(), &mut canvas_tokens_gpu)?;
            {
                let moe = match &mut self.weights {
                    WeightStorage::Moe(m) => m,
                    _ => unreachable!(),
                };
                // Embed canvas tokens only (prompt is in the KV cache).
                self.kernels.ops.embed_tokens_f32(
                    &moe.token_embd,
                    &canvas_tokens_gpu,
                    &mut hidden,
                    cl as u32,
                    dim,
                )?;
                self.kernels
                    .ops
                    .scale_f32_inplace(&mut hidden, cl as u32 * dim, (dim as f32).sqrt())?;
                stream.memcpy_dtod(&hidden.slice(0..cl * dimu), &mut canvas_emb)?;
                let sc = moe
                    .sc
                    .as_ref()
                    .ok_or_else(|| EngineError::Unsupported("missing SC weights".into()))?;
                dg::apply_self_conditioning(
                    &mut self.kernels,
                    sc,
                    &moe.token_embd,
                    &canvas_logits,
                    &canvas_emb,
                    &mut canvas_normed,
                    &mut sc_bufs,
                    cl as u32,
                    dim,
                    n_ff,
                    vocab,
                    sc_use,
                    temp_inv,
                    eps,
                )?;
                self.kernels
                    .ops
                    .copy_f16_to_f32(&canvas_normed, &mut canvas_norm_f32, (cl * dimu) as u32)?;
                stream.memcpy_dtod(&canvas_norm_f32, &mut hidden.slice_mut(0..cl * dimu))?;
                // Reset write position to P: prompt K/V stays, canvas K/V is
                // rewritten at [P..P+C] this step.
                self.kv_cache.set_pos(p as u32);
                arch::gemma4_moe::forward_moe_streaming(
                    &mut hidden,
                    moe,
                    &self.config,
                    &mut self.kernels,
                    &mut self.kv_cache,
                    &mut self.scratch,
                    cl as u32,
                    None,
                    &stream,
                    Some(&attn),
                )?;
                // Project the canvas's final-normed hidden to logits.
                stream.memcpy_dtod(
                    &self.scratch.norm_out.slice(0..cl * dimu),
                    &mut canvas_norm_out,
                )?;
                if std::env::var("CHEW_PROFILE").is_ok() {
                    stream.synchronize().map_err(EngineError::Driver)?;
                }
                let t_proj = std::time::Instant::now();
                crate::forward::project_all_logits(
                    &mut self.kernels,
                    &canvas_norm_out,
                    &moe.output,
                    &mut canvas_logits,
                    cl as u32,
                    vocab,
                    dim,
                )?;
                if std::env::var("CHEW_PROFILE").is_ok() {
                    stream.synchronize().map_err(EngineError::Driver)?;
                    eprintln!("  project_all_logits: {}ms", t_proj.elapsed().as_millis());
                }
            }
            if let Some(cap) = softcap {
                self.kernels
                    .ops
                    .logit_softcap_inplace(&mut canvas_logits, cl as u32 * vocab, cap)?;
            }
            stream.synchronize().map_err(EngineError::Driver)?;
            let gpu_ms = t_gpu.elapsed().as_millis();
            let t_eb = std::time::Instant::now();
            // device-side reduce: argmax/entropy/multinomial per position
            for r in rnd_host.iter_mut() {
                *r = rng.next_f32();
            }
            stream.memcpy_htod(rnd_host.as_slice(), &mut rnd_gpu)?;
            self.kernels.ops.eb_reduce(
                &canvas_logits,
                &rnd_gpu,
                &mut argmax_gpu,
                &mut entropy_gpu,
                &mut sampled_gpu,
                cl as u32,
                vocab,
                temp_inv,
            )?;
            stream.memcpy_dtoh(&argmax_gpu, &mut argmax_host)?;
            stream.memcpy_dtoh(&entropy_gpu, &mut entropy_host)?;
            stream.memcpy_dtoh(&sampled_gpu, &mut sampled_host)?;
            stream.synchronize().map_err(EngineError::Driver)?;
            let step = dg::eb_accept(
                &argmax_host,
                &entropy_host,
                &sampled_host,
                eb.entropy_bound,
                vocab,
                &mut rng,
            );
            eprintln!(
                "step {}/{}: gpu={}ms eb={}ms mean_H={:.4} held={} argmax[0..6]={:?}",
                step_idx + 1,
                s,
                gpu_ms,
                t_eb.elapsed().as_millis(),
                step.mean_entropy,
                held,
                &step.argmax[..6.min(cl)]
            );
            argmax_out.copy_from_slice(&step.argmax);
            for i in 0..cl {
                canvas_tokens[i] = step.next_canvas[i] as i32;
            }
            let stable = prev_argmax.as_ref() == Some(&step.argmax);
            held = if stable { held + 1 } else { 0 };
            let confident = step.mean_entropy < conf;
            prev_argmax = Some(step.argmax);
            if held >= stab && confident {
                break;
            }
        }

        let _ = eos;
        // Debug: return the full raw canvas argmax (no EOS trim) so we can judge
        // coherence end-to-end.
        Ok(argmax_out)
    }

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
        Self::load_with_batch(model_path, alloc, gpu_idx, max_context, None)
    }

    fn load_with_batch(
        model_path: impl AsRef<Path>,
        alloc: &VramAllocator,
        gpu_idx: usize,
        max_context: Option<u32>,
        max_batch_override: Option<u32>,
    ) -> Result<Self, EngineError> {
        let path = model_path.as_ref();
        info!(path = %path.display(), "loading model");

        // 1. Parse GGUF
        let gguf = GgufFile::open(path)?;
        let mut config = ModelConfig::from_gguf(&gguf.header)?;
        if config.is_mamba() {
            let layout = arch::mamba::MambaLayout::inspect(&gguf, &config)?;
            if config.vocab_size != layout.vocab_size {
                info!(
                    header_vocab = config.vocab_size,
                    gguf_vocab = layout.vocab_size,
                    "overriding mamba vocab size from tensor layout"
                );
                config.vocab_size = layout.vocab_size;
            }
            if config.ff_dim == 0 {
                // Mamba GGUFs often omit feed_forward_length; use inner SSM width as fallback.
                config.ff_dim = layout.inner_dim;
            }
        }

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
            let (free, total) = stream
                .context()
                .mem_get_info()
                .map_err(EngineError::Driver)?;
            info!(
                free_mb = free / (1024 * 1024),
                total_mb = total / (1024 * 1024),
                "GPU VRAM"
            );
            free as u64
        };

        let desired_ctx = max_context.unwrap_or(config.context_length.min(32768));

        // Architecture-specific eager paths: do not go through transformer VRAM planning.
        if config.is_bert() || config.is_mamba() {
            let d = config.dim as usize;
            let nh = config.n_heads as usize;
            let hd = config.max_head_dim as usize;
            let ff = config.ff_dim as usize;
            let v = config.vocab_size as usize;
            let max_weight_elems = [nh * hd * d, ff * d, v * d].into_iter().max().unwrap();
            let max_k = config.ff_dim.max(config.dim) as usize;
            let kernels = GpuKernels::load(&stream, max_weight_elems, max_k)?;
            let (weights, mamba_state) = if config.is_bert() {
                (
                    WeightStorage::Bert(arch::bert::BertModelWeights::load(
                        &gguf,
                        &config,
                        alloc,
                        &kernels.dequant,
                        gpu_idx,
                    )?),
                    None,
                )
            } else {
                let w = arch::mamba::MambaModelWeights::load(
                    &gguf,
                    &config,
                    alloc,
                    &kernels.dequant,
                    gpu_idx,
                )?;
                let s = arch::mamba::MambaRuntimeState::new(&w)?;
                (WeightStorage::Mamba(w), Some(s))
            };
            let max_seq = desired_ctx.min(config.context_length).max(1);
            let max_batch = max_batch_override.unwrap_or(max_seq).min(max_seq).max(1);
            let kv_cache = KvCache::alloc(&config, max_seq, &stream)?;
            let scratch = ScratchBuffers::alloc(&config, max_batch, max_seq, &stream)?;

            info!(
                context = max_seq,
                max_batch,
                free_mb = free_bytes / (1024 * 1024),
                arch = %config.arch,
                "engine ready (arch-specific)"
            );

            return Ok(Self {
                config,
                weights,
                mamba_state,
                kernels,
                kv_cache,
                scratch,
                stream,
                gpu_idx,
            });
        }

        // Try normal loading first (min 2k context), then streaming fallback.
        let min_useful_ctx = 2048u32;
        let normal_plan =
            VramPlan::fit_with_batch(&config, &gguf, desired_ctx, max_batch_override, free_bytes);
        let needs_streaming_plan = normal_plan
            .as_ref()
            .filter(|p| p.context_length >= min_useful_ctx)
            .is_none();

        let streaming_plan = if needs_streaming_plan {
            let sp = VramPlan::fit_streaming_with_batch(
                &config,
                &gguf,
                desired_ctx,
                max_batch_override,
                free_bytes,
            );
            if let Some(ref plan) = sp {
                plan.print_report(free_bytes / (1024 * 1024));
                info!(
                    resident = plan.n_resident,
                    streamed = plan.n_streamed,
                    layer_mb = plan.per_layer_bytes / (1024 * 1024),
                    "streaming mode: {} of {} layers in VRAM",
                    plan.n_resident,
                    plan.total_layers,
                );
            }
            sp
        } else {
            None
        };

        let plan = normal_plan
            .clone()
            .or_else(|| {
                // In streaming mode, create a minimal plan for just the resident parts
                streaming_plan.as_ref().map(|sp| {
                    VramPlan::compute(
                        &config,
                        &gguf,
                        sp.context_length,
                        max_batch_override
                            .unwrap_or(sp.context_length.min(512))
                            .min(sp.context_length)
                            .max(1),
                    )
                })
            })
            .ok_or_else(|| {
                EngineError::Load(LoadError::MissingTensor(format!(
                    "not enough VRAM even for streaming: need fixed overhead, have {} MB free",
                    free_bytes / (1024 * 1024)
                )))
            })?;

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

        // 4. Load + dequantize weights (arch-specific dispatch, no implicit fallback).
        let weights = match config.arch.as_str() {
            "gemma4" | "diffusion-gemma" => {
                if config.is_moe() {
                    // MoE path supports both:
                    // - true streaming (some layers in host RAM), or
                    // - fully resident mode (all layers on GPU, n_streamed=0).
                    let full_resident_sp;
                    let sp = if let Some(sp) = streaming_plan.as_ref() {
                        sp
                    } else if let Some(np) = normal_plan.as_ref() {
                        full_resident_sp = all_resident_moe_plan(&config, &gguf, np);
                        &full_resident_sp
                    } else {
                        return Err(EngineError::Load(LoadError::MissingTensor(
                            "MoE model requires either a normal VRAM fit or a streaming plan"
                                .into(),
                        )));
                    };
                    if sp.n_streamed == 0 {
                        info!(
                            "loading MoE in fully resident mode: {} resident, {} streamed",
                            sp.n_resident, sp.n_streamed
                        );
                    } else {
                        info!(
                            "loading MoE in STREAMING mode: {} resident, {} streamed",
                            sp.n_resident, sp.n_streamed
                        );
                    }
                    WeightStorage::Moe(arch::gemma4_moe::MoeModelWeights::load(
                        &gguf,
                        &config,
                        sp,
                        alloc,
                        &kernels.dequant,
                        gpu_idx,
                    )?)
                } else if let Some(ref sp) = streaming_plan {
                    info!(
                        "loading Gemma4 dense in STREAMING mode: {} resident, {} streamed",
                        sp.n_resident, sp.n_streamed
                    );
                    WeightStorage::Streaming(StreamingWeights::load(
                        &gguf,
                        &config,
                        sp,
                        alloc,
                        &kernels.dequant,
                        gpu_idx,
                    )?)
                } else {
                    WeightStorage::Normal(ModelWeights::load(
                        &gguf,
                        &config,
                        alloc,
                        &kernels.dequant,
                        gpu_idx,
                    )?)
                }
            }
            "llama" => {
                if let Some(ref sp) = streaming_plan {
                    info!(
                        "loading Llama in STREAMING mode: {} resident, {} streamed",
                        sp.n_resident, sp.n_streamed
                    );
                    WeightStorage::Streaming(StreamingWeights::load(
                        &gguf,
                        &config,
                        sp,
                        alloc,
                        &kernels.dequant,
                        gpu_idx,
                    )?)
                } else {
                    WeightStorage::Normal(ModelWeights::load(
                        &gguf,
                        &config,
                        alloc,
                        &kernels.dequant,
                        gpu_idx,
                    )?)
                }
            }
            other => {
                return Err(EngineError::Unsupported(format!(
                    "internal loader dispatch error: no transformer VRAM-plan path for arch '{other}'"
                )));
            }
        };

        // 5. Allocate KV cache
        let kv_cache = KvCache::alloc(&config, max_seq, &stream)?;

        // 6. Allocate scratch buffers
        let scratch = ScratchBuffers::alloc(&config, max_batch, max_seq, &stream)?;

        // 7. Expand expert cache with remaining VRAM headroom.
        // Skipped for diffusion: each step re-decodes the full canvas fresh, so
        // there is no cross-token expert reuse to cache — and the freed headroom
        // is needed for the canvas logit/probs buffers at run time.
        let mut weights = weights;
        if !config.is_diffusion() {
            if let WeightStorage::Moe(moe) = &mut weights {
                if let Err(e) = moe.expand_expert_cache(&stream, &config) {
                    tracing::warn!(%e, "failed to expand expert cache (non-fatal)");
                }
            }
        }

        info!(context = max_seq, max_batch, "engine ready");

        Ok(Self {
            config,
            weights,
            mamba_state: None,
            kernels,
            kv_cache,
            scratch,
            stream,
            gpu_idx,
        })
    }

    /// Encode tokens into final hidden states without sampling.
    /// Returns the final hidden state tensor on CPU as row-major f32: [seq_len, dim].
    pub fn encode_hidden(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        self.reset();

        let seq_len = input_tokens.len() as u32;
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        self.validate_prefill_len(seq_len)?;

        if let WeightStorage::Bert(w) = &self.weights {
            let token_ids_gpu = self.upload_token_ids(input_tokens)?;
            let mut hidden = self
                .stream
                .alloc_zeros::<f32>((seq_len * self.config.dim) as usize)
                .map_err(EngineError::Driver)?;
            self.kernels.ops.embed_tokens_f32(
                &w.embeddings.word_embeddings,
                &token_ids_gpu,
                &mut hidden,
                seq_len,
                self.config.dim,
            )?;

            let position_ids: Vec<i32> = (0..seq_len as i32).collect();
            let token_type_ids = vec![0i32; seq_len as usize];
            let mut pos_gpu = self
                .stream
                .alloc_zeros::<i32>(seq_len as usize)
                .map_err(EngineError::Driver)?;
            let mut tok_type_gpu = self
                .stream
                .alloc_zeros::<i32>(seq_len as usize)
                .map_err(EngineError::Driver)?;
            self.stream
                .memcpy_htod(&position_ids, &mut pos_gpu)
                .map_err(EngineError::Driver)?;
            self.stream
                .memcpy_htod(&token_type_ids, &mut tok_type_gpu)
                .map_err(EngineError::Driver)?;

            let mut tmp = self
                .stream
                .alloc_zeros::<f32>((seq_len * self.config.dim) as usize)
                .map_err(EngineError::Driver)?;
            self.kernels.ops.embed_tokens_f32(
                &w.embeddings.position_embeddings,
                &pos_gpu,
                &mut tmp,
                seq_len,
                self.config.dim,
            )?;
            self.kernels
                .ops
                .add_inplace_f32(&mut hidden, &tmp, seq_len * self.config.dim)?;
            self.kernels.ops.embed_tokens_f32(
                &w.embeddings.token_type_embeddings,
                &tok_type_gpu,
                &mut tmp,
                seq_len,
                self.config.dim,
            )?;
            self.kernels
                .ops
                .add_inplace_f32(&mut hidden, &tmp, seq_len * self.config.dim)?;
            self.run_forward(&mut hidden, None, seq_len)?;

            let mut out = vec![0.0f32; (seq_len * self.config.dim) as usize];
            self.stream
                .memcpy_dtoh(&hidden, &mut out)
                .map_err(EngineError::Driver)?;
            return Ok(out);
        }

        let token_ids_gpu = self.upload_token_ids(input_tokens)?;
        let mut hidden = self.embed_prefill_tokens(&token_ids_gpu, seq_len)?;
        self.run_prefill_chunk(&token_ids_gpu, &mut hidden, seq_len)?;

        let mut out = vec![0.0f32; (seq_len * self.config.dim) as usize];
        self.stream
            .memcpy_dtoh(&hidden, &mut out)
            .map_err(EngineError::Driver)?;
        Ok(out)
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
        if self.config.is_bert() {
            return Err(EngineError::Unsupported(
                "generation is unsupported for encoder-only BERT/MiniLM models; use /v1/embeddings or encode_hidden()".into()
            ));
        }
        self.reset();
        let prefill_len = input_tokens.len() as u32;
        self.validate_prefill_len(prefill_len)?;
        let requested_max_new_tokens = max_new_tokens;
        let max_new_tokens = self.clamp_generation_tokens(prefill_len, requested_max_new_tokens);
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        if max_new_tokens < requested_max_new_tokens {
            info!(
                prompt_tokens = prefill_len,
                max_context = self.kv_cache.max_seq(),
                requested_max_new_tokens,
                capped_max_new_tokens = max_new_tokens,
                "generation constrained by context window"
            );
        }
        let generate_t0 = std::time::Instant::now();

        let mut all_tokens: Vec<u32> = input_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();

        let debug_logits = std::env::var("CHEW_DEBUG_LOGITS").is_ok();
        let debug_decode = std::env::var("CHEW_DEBUG_DECODE").is_ok();
        let token_ids_gpu = self.upload_token_ids(input_tokens)?;
        let mut hidden = self.embed_prefill_tokens(&token_ids_gpu, prefill_len)?;

        if self.config.is_gemma4() && debug_decode {
            let last_off = ((prefill_len - 1) * self.config.dim) as usize;
            let mut dbg = vec![0.0f32; 8];
            self.stream
                .memcpy_dtoh(&hidden.slice(last_off..last_off + 8), &mut dbg)
                .map_err(EngineError::Driver)?;
            info!(?dbg, pos = prefill_len - 1, "EMBED last pos first 8 values");
        }

        self.run_prefill_chunk(&token_ids_gpu, &mut hidden, prefill_len)?;

        // Sample first token from logits of last position
        // Logits are f16 — download and convert to f32 for sampling
        let vocab = self.config.vocab_size as usize;
        let mut logits_f16 = vec![half::f16::ZERO; vocab];
        let mut logits_f32 = vec![0.0f32; vocab];
        let logit_view = self.scratch.logits.slice(0..vocab);
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
            let max_logit = logits_f32
                .iter()
                .cloned()
                .filter(|x| x.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            let min_logit = logits_f32
                .iter()
                .cloned()
                .filter(|x| x.is_finite())
                .fold(f32::INFINITY, f32::min);
            info!(
                nan_count,
                inf_count,
                zero_count,
                max_logit,
                min_logit,
                vocab = logits_f32.len(),
                "prefill logits"
            );

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
        let prefill_elapsed = generate_t0.elapsed();

        if next_token == eos_token {
            let e2e_s = prefill_elapsed.as_secs_f64().max(1e-9);
            let prefill_s = prefill_elapsed.as_secs_f64().max(1e-9);
            info!(
                completion_tokens = 1u32,
                prefill_ms = format!("{:.2}", prefill_elapsed.as_secs_f64() * 1000.0),
                prefill_tok_s = format!("{:.2}", prefill_len as f64 / prefill_s),
                decode_tokens = 0u32,
                decode_ms = "0.00",
                decode_tok_s = "0.00",
                e2e_tok_s = format!("{:.2}", 1.0 / e2e_s),
                "generation perf (engine)"
            );
            return Ok(generated);
        }

        // Decode loop: one token at a time
        // Pre-allocate decode buffers ONCE (no per-token GPU allocs!)
        let mut tok_gpu = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(EngineError::Driver)?;
        let mut decode_hidden = self
            .stream
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
            && !self.config.is_bert()
            && !self.config.is_mamba()
            && !self.weights.is_streaming();
        let mut decode_graph: Option<forward::DecodeGraph> = None;

        let greedy = params.temperature == 0.0;
        let gpu_topk_sampling =
            !greedy && params.repeat_penalty == 1.0 && params.top_k > 0 && params.top_k <= 40;
        let mut sample_rng_state =
            0x9E37_79B9_7F4A_7C15u64 ^ (prefill_len as u64) ^ (input_tokens.len() as u64) << 32;
        let mut next_sample_seed = || -> u32 {
            // xorshift64*
            sample_rng_state ^= sample_rng_state << 13;
            sample_rng_state ^= sample_rng_state >> 7;
            sample_rng_state ^= sample_rng_state << 17;
            sample_rng_state as u32
        };
        let profile = std::env::var("CHEW_PROFILE").is_ok();
        let decode_elapsed: std::time::Duration;
        let eos_check_interval = std::env::var("CHEW_EOS_CHECK_INTERVAL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(16);

        if greedy {
            // ZERO-SYNC GREEDY PATH:
            // All tokens stay on GPU. argmax→tok_gpu→embed_tokens chain has no host roundtrip.
            // GPU token buffer stores all generated token IDs. Batch download at end.
            let mut token_buf_gpu = self
                .stream
                .alloc_zeros::<i32>(max_new_tokens as usize)
                .map_err(EngineError::Driver)?;
            let mut n_generated = 0u32;
            let decode_wall_t0 = std::time::Instant::now();

            let decode_t0 = if profile {
                self.stream.synchronize().map_err(EngineError::Driver)?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            for _step in 1..max_new_tokens {
                if _step == 1 {
                    // First decode: upload token from host
                    let last = *all_tokens.last().unwrap();
                    let tok_i32 = [last as i32];
                    self.stream
                        .memcpy_htod(&tok_i32, &mut tok_gpu)
                        .map_err(EngineError::Driver)?;
                }
                // else: tok_gpu already has argmax from previous step

                self.kernels.ops.embed_tokens_f32(
                    self.weights.token_embd(),
                    &tok_gpu,
                    &mut decode_hidden,
                    1,
                    self.config.dim,
                )?;

                // Scale embeddings by sqrt(dim) for Gemma 4
                if self.config.is_gemma4() {
                    let scale = (self.config.dim as f32).sqrt();
                    self.kernels.ops.scale_f32_inplace(
                        &mut decode_hidden,
                        self.config.dim,
                        scale,
                    )?;
                }

                if _step == 1 && debug_decode {
                    let mut tok_host = [0i32];
                    self.stream.memcpy_dtoh(&tok_gpu, &mut tok_host).ok();
                    let mut emb = vec![0.0f32; 4];
                    self.stream
                        .memcpy_dtoh(&decode_hidden.slice(0..4), &mut emb)
                        .ok();
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
                                    &mut decode_hidden,
                                    w,
                                    &self.config,
                                    &mut self.kernels,
                                    &mut self.kv_cache,
                                    &mut self.scratch,
                                    pos,
                                    &self.stream,
                                )?);
                            }
                            WeightStorage::Bert(_)
                            | WeightStorage::Mamba(_)
                            | WeightStorage::Streaming(_)
                            | WeightStorage::Moe(_) => {
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
                        info!(
                            compute_pe_us = t0.elapsed().as_micros(),
                            "PROFILE decode compute_pe"
                        );
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
                    let mut indexed: Vec<(usize, f32)> =
                        logits_f32.iter().cloned().enumerate().collect();
                    indexed
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top10: Vec<(usize, f32)> = indexed.into_iter().take(10).collect();
                    info!(?top10, "decode step1 top 10 logit tokens");
                }

                // GPU argmax → tok_gpu (stays on GPU)
                self.kernels.ops.argmax_f16(
                    &self.scratch.logits,
                    &mut tok_gpu,
                    self.config.vocab_size,
                )?;

                // Copy token to GPU buffer (1 int, async, no sync)
                // Use a tiny device-to-device copy
                {
                    let mut dst =
                        token_buf_gpu.slice_mut(n_generated as usize..(n_generated + 1) as usize);
                    self.stream
                        .memcpy_dtod(&tok_gpu, &mut dst)
                        .map_err(EngineError::Driver)?;
                }
                n_generated += 1;

                // Check EOS in chunks to avoid over-generating far past EOS while
                // keeping host sync overhead low.
                let should_check =
                    n_generated <= eos_check_interval || (n_generated % eos_check_interval == 0);
                if should_check {
                    // Early decode: check last token each step so short generations stop immediately.
                    // Later decode: chunked checks to keep sync overhead low.
                    let (chunk_start, chunk_len) = if n_generated <= eos_check_interval {
                        (n_generated - 1, 1)
                    } else {
                        (n_generated - eos_check_interval, eos_check_interval)
                    };
                    let mut host_chunk = vec![0i32; chunk_len as usize];
                    let src = token_buf_gpu.slice(chunk_start as usize..n_generated as usize);
                    self.stream
                        .memcpy_dtoh(&src, &mut host_chunk)
                        .map_err(EngineError::Driver)?;
                    if let Some(pos) = host_chunk.iter().position(|&t| t as u32 == eos_token) {
                        n_generated = chunk_start + pos as u32 + 1;
                        break;
                    }
                }

                // NO HOST SYNC — entire loop runs without host-GPU synchronization
            }

            if let Some(t0) = decode_t0 {
                self.stream.synchronize().map_err(EngineError::Driver)?;
                let decode_elapsed = t0.elapsed();
                if n_generated > 0 {
                    let ms_per_tok = decode_elapsed.as_secs_f64() * 1000.0 / n_generated as f64;
                    let tps = n_generated as f64 / decode_elapsed.as_secs_f64();
                    info!(
                        n_generated,
                        ms_per_tok = format!("{:.2}", ms_per_tok),
                        decode_tps = format!("{:.1}", tps),
                        graph = use_graph,
                        "decode timing"
                    );
                }
            }

            // Batch download ALL tokens at end (one sync for all)
            if n_generated > 0 {
                let mut host_tokens = vec![0i32; n_generated as usize];
                let src = token_buf_gpu.slice(0..n_generated as usize);
                self.stream
                    .memcpy_dtoh(&src, &mut host_tokens)
                    .map_err(EngineError::Driver)?;
                for &t in &host_tokens {
                    let tok = t as u32;
                    generated.push(tok);
                    if tok == eos_token {
                        break;
                    } // truncate at EOS
                }
            }
            decode_elapsed = decode_wall_t0.elapsed();
        } else if gpu_topk_sampling {
            // ZERO-SYNC TOP-K SAMPLING PATH:
            // Keep sampled token IDs on device and avoid per-token dtoh synchronization.
            let tail_capacity = max_new_tokens.saturating_sub(1).max(1);
            let mut token_buf_gpu = self
                .stream
                .alloc_zeros::<i32>(tail_capacity as usize)
                .map_err(EngineError::Driver)?;
            let mut n_generated = 0u32;
            let decode_wall_t0 = std::time::Instant::now();

            let decode_t0 = if profile {
                self.stream.synchronize().map_err(EngineError::Driver)?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            for step in 1..max_new_tokens {
                if step == 1 {
                    let last = *all_tokens.last().unwrap();
                    let tok_i32 = [last as i32];
                    self.stream
                        .memcpy_htod(&tok_i32, &mut tok_gpu)
                        .map_err(EngineError::Driver)?;
                }

                self.kernels.ops.embed_tokens_f32(
                    self.weights.token_embd(),
                    &tok_gpu,
                    &mut decode_hidden,
                    1,
                    self.config.dim,
                )?;

                if self.config.is_gemma4() {
                    let scale = (self.config.dim as f32).sqrt();
                    self.kernels.ops.scale_f32_inplace(
                        &mut decode_hidden,
                        self.config.dim,
                        scale,
                    )?;
                }

                if debug_decode && step == 1 {
                    let mut tok_host = [0i32];
                    self.stream.memcpy_dtoh(&tok_gpu, &mut tok_host).ok();
                    let mut emb = vec![0.0f32; 4];
                    self.stream
                        .memcpy_dtoh(&decode_hidden.slice(0..4), &mut emb)
                        .ok();
                    info!(?tok_host, ?emb, "decode embed check (sampling-gpu)");
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
                                    &mut decode_hidden,
                                    w,
                                    &self.config,
                                    &mut self.kernels,
                                    &mut self.kv_cache,
                                    &mut self.scratch,
                                    pos,
                                    &self.stream,
                                )?);
                            }
                            WeightStorage::Bert(_)
                            | WeightStorage::Mamba(_)
                            | WeightStorage::Streaming(_)
                            | WeightStorage::Moe(_) => {
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
                        info!(
                            compute_pe_us = t0.elapsed().as_micros(),
                            "PROFILE decode compute_pe"
                        );
                    }
                    self.run_forward(&mut decode_hidden, decode_pe.as_ref(), 1)?;
                }

                let seed = next_sample_seed();
                self.kernels.ops.sample_top_k(
                    &self.scratch.logits,
                    &mut tok_gpu,
                    self.config.vocab_size,
                    params.temperature,
                    params.top_k,
                    params.top_p,
                    seed,
                )?;

                {
                    let mut dst =
                        token_buf_gpu.slice_mut(n_generated as usize..(n_generated + 1) as usize);
                    self.stream
                        .memcpy_dtod(&tok_gpu, &mut dst)
                        .map_err(EngineError::Driver)?;
                }
                n_generated += 1;

                let should_check =
                    n_generated <= eos_check_interval || (n_generated % eos_check_interval == 0);
                if should_check {
                    let (chunk_start, chunk_len) = if n_generated <= eos_check_interval {
                        (n_generated - 1, 1)
                    } else {
                        (n_generated - eos_check_interval, eos_check_interval)
                    };
                    let mut host_chunk = vec![0i32; chunk_len as usize];
                    let src = token_buf_gpu.slice(chunk_start as usize..n_generated as usize);
                    self.stream
                        .memcpy_dtoh(&src, &mut host_chunk)
                        .map_err(EngineError::Driver)?;
                    if let Some(pos) = host_chunk.iter().position(|&t| t as u32 == eos_token) {
                        n_generated = chunk_start + pos as u32 + 1;
                        break;
                    }
                }
            }

            if let Some(t0) = decode_t0 {
                self.stream.synchronize().map_err(EngineError::Driver)?;
                let decode_elapsed = t0.elapsed();
                if n_generated > 0 {
                    let ms_per_tok = decode_elapsed.as_secs_f64() * 1000.0 / n_generated as f64;
                    let tps = n_generated as f64 / decode_elapsed.as_secs_f64();
                    info!(
                        n_generated,
                        ms_per_tok = format!("{:.2}", ms_per_tok),
                        decode_tps = format!("{:.1}", tps),
                        graph = use_graph,
                        "decode timing (sampling-gpu)"
                    );
                }
            }

            if n_generated > 0 {
                let mut host_tokens = vec![0i32; n_generated as usize];
                let src = token_buf_gpu.slice(0..n_generated as usize);
                self.stream
                    .memcpy_dtoh(&src, &mut host_tokens)
                    .map_err(EngineError::Driver)?;
                for &t in &host_tokens {
                    let tok = t as u32;
                    generated.push(tok);
                    if tok == eos_token {
                        break;
                    }
                }
            }
            decode_elapsed = decode_wall_t0.elapsed();
        } else {
            // CPU sampling fallback (needed for repeat penalty / wider sampling configs).
            let decode_wall_t0 = std::time::Instant::now();
            for step in 1..max_new_tokens {
                let last = *all_tokens.last().unwrap();
                let tok_i32 = [last as i32];
                self.stream
                    .memcpy_htod(&tok_i32, &mut tok_gpu)
                    .map_err(EngineError::Driver)?;

                self.kernels.ops.embed_tokens_f32(
                    self.weights.token_embd(),
                    &tok_gpu,
                    &mut decode_hidden,
                    1,
                    self.config.dim,
                )?;

                if self.config.is_gemma4() {
                    let scale = (self.config.dim as f32).sqrt();
                    self.kernels.ops.scale_f32_inplace(
                        &mut decode_hidden,
                        self.config.dim,
                        scale,
                    )?;
                }

                if debug_decode && step == 1 {
                    let mut tok_host = [0i32];
                    self.stream.memcpy_dtoh(&tok_gpu, &mut tok_host).ok();
                    let mut emb = vec![0.0f32; 4];
                    self.stream
                        .memcpy_dtoh(&decode_hidden.slice(0..4), &mut emb)
                        .ok();
                    info!(?tok_host, ?emb, "decode embed check (sampling-cpu)");
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
                                    &mut decode_hidden,
                                    w,
                                    &self.config,
                                    &mut self.kernels,
                                    &mut self.kv_cache,
                                    &mut self.scratch,
                                    pos,
                                    &self.stream,
                                )?);
                            }
                            WeightStorage::Bert(_)
                            | WeightStorage::Mamba(_)
                            | WeightStorage::Streaming(_)
                            | WeightStorage::Moe(_) => {
                                unreachable!("CUDA graph disabled for streaming/MoE mode");
                            }
                        }
                    }
                } else {
                    let decode_pe = if self.config.is_gemma4() {
                        self.compute_per_layer_embeddings(&tok_gpu, &decode_hidden, 1)?
                    } else {
                        None
                    };
                    self.run_forward(&mut decode_hidden, decode_pe.as_ref(), 1)?;
                }

                self.stream
                    .memcpy_dtoh(&self.scratch.logits.slice(0..vocab), &mut logits_f16)
                    .map_err(EngineError::Driver)?;
                for (i, &v) in logits_f16.iter().enumerate() {
                    logits_f32[i] = v.to_f32();
                }
                next_token = sample::sample_token(&mut logits_f32, params, &all_tokens);
                generated.push(next_token);
                all_tokens.push(next_token);
                if next_token == eos_token {
                    break;
                }
            }
            decode_elapsed = decode_wall_t0.elapsed();
        }

        let completion_tokens = generated.len() as u32;
        let decode_tokens = completion_tokens.saturating_sub(1);
        let prefill_s = prefill_elapsed.as_secs_f64().max(1e-9);
        let decode_s = decode_elapsed.as_secs_f64().max(1e-9);
        let e2e_elapsed = generate_t0.elapsed();
        let e2e_s = e2e_elapsed.as_secs_f64().max(1e-9);
        info!(
            completion_tokens,
            prefill_ms = format!("{:.2}", prefill_elapsed.as_secs_f64() * 1000.0),
            prefill_tok_s = format!("{:.2}", prefill_len as f64 / prefill_s),
            decode_tokens,
            decode_ms = format!("{:.2}", decode_elapsed.as_secs_f64() * 1000.0),
            decode_tok_s = format!("{:.2}", decode_tokens as f64 / decode_s),
            e2e_tok_s = format!("{:.2}", completion_tokens as f64 / e2e_s),
            "generation perf (engine)"
        );

        Ok(generated)
    }

    /// Run the forward pass, dispatching based on weight storage type.
    fn run_forward(
        &mut self,
        hidden: &mut cudarc::driver::CudaSlice<f32>,
        pe: Option<&arch::gemma4_common::PerLayerEmbeddings>,
        seq_len: u32,
    ) -> Result<(), EngineError> {
        // Dispatch by weight storage type
        // Streaming and MoE need mutable access to weights (for shell uploads)
        if matches!(self.weights, WeightStorage::Streaming(_)) {
            forward::forward_streaming(
                hidden,
                &mut self.weights,
                &self.config,
                &mut self.kernels,
                &mut self.kv_cache,
                &mut self.scratch,
                seq_len,
                pe,
                &self.stream,
            )?;
        } else if let WeightStorage::Moe(ref mut moe_w) = self.weights {
            arch::gemma4_moe::forward_moe_streaming(
                hidden,
                moe_w,
                &self.config,
                &mut self.kernels,
                &mut self.kv_cache,
                &mut self.scratch,
                seq_len,
                pe,
                &self.stream,
                None,
            )?;
        } else if let WeightStorage::Bert(ref w) = self.weights {
            arch::bert::forward(
                hidden,
                w,
                &self.config,
                &mut self.kernels,
                &mut self.scratch,
                seq_len,
            )?;
        } else if let WeightStorage::Mamba(ref w) = self.weights {
            let state = self.mamba_state.as_mut().ok_or_else(|| {
                EngineError::Unsupported("missing mamba runtime state for mamba weights".into())
            })?;
            arch::mamba::forward(
                hidden,
                w,
                &self.config,
                &mut self.kernels,
                &mut self.scratch,
                seq_len,
                state,
            )?;
        } else if let WeightStorage::Normal(ref w) = self.weights {
            match self.config.arch.as_str() {
                "gemma4" => {
                    arch::gemma4_dense::forward(
                        hidden,
                        w,
                        &self.config,
                        &mut self.kernels,
                        &mut self.kv_cache,
                        &mut self.scratch,
                        seq_len,
                        pe,
                    )?;
                }
                "llama" => {
                    arch::llama::forward(
                        hidden,
                        w,
                        &self.config,
                        &mut self.kernels,
                        &mut self.kv_cache,
                        &mut self.scratch,
                        seq_len,
                    )?;
                }
                other => {
                    return Err(EngineError::Unsupported(format!(
                        "normal-weight forward has no backend for arch '{other}'"
                    )));
                }
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
    ) -> Result<Option<arch::gemma4_common::PerLayerEmbeddings>, EngineError> {
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
        let row_width = epl * n_layers; // total elements per token across all layers

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
        let mut gathered_quant = self
            .stream
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
        let mut tok_embd = self
            .stream
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
                self.kernels
                    .ops
                    .scale_f16(&*src_ptr, &mut *dst_ptr, total_elements, tok_scale)?;
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
            let mut hidden_f16 = self
                .stream
                .alloc_zeros::<half::f16>(hidden_elems as usize)
                .map_err(EngineError::Driver)?;
            {
                let mut dst_view = hidden_f16.slice_mut(..);
                self.kernels
                    .ops
                    .copy_f32_to_f16(hidden_f32, &mut dst_view, hidden_elems)?;
            }

            // 3b: Matmul — proj = hidden_f16 @ per_layer_model_proj^T
            //   A = hidden_f16 [n_tokens, dim]  (M=n_tokens, K=dim)
            //   B = per_layer_model_proj [row_width, dim]  (N=row_width, K=dim)
            //   C = proj [n_tokens, row_width]
            let mut proj = self
                .stream
                .alloc_zeros::<half::f16>(total_elements as usize)
                .map_err(EngineError::Driver)?;

            let proj_w = self.weights.per_layer_model_proj().unwrap();
            self.kernels.gemm.matmul_dequant(
                &hidden_f16,
                &proj_w.data,
                proj_w.quant_type,
                proj_w.n_elements,
                &mut proj,
                n_tokens,  // M
                row_width, // N
                dim,       // K
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
                    self.kernels.ops.scale_f16(
                        &*src_ptr,
                        &mut *dst_ptr,
                        total_elements,
                        proj_scale,
                    )?;
                }
            }

            // Step 5: RMS-norm proj with per_layer_proj_norm [epl]
            // Conceptually reshape proj as [n_tokens * n_layers, epl] and norm each row
            let norm_rows = n_tokens * n_layers;
            let mut proj_normed = self
                .stream
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
            let mut result = self
                .stream
                .alloc_zeros::<half::f16>(total_elements as usize)
                .map_err(EngineError::Driver)?;
            self.kernels
                .ops
                .add_f16(&tok_embd, &proj_normed, &mut result, total_elements)?;

            // Free intermediates
            drop(tok_embd);
            drop(proj_normed);

            // Scale by 1/sqrt(2)
            let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
            {
                let src_ptr = &result as *const cudarc::driver::CudaSlice<half::f16>;
                let dst_ptr = &mut result as *mut cudarc::driver::CudaSlice<half::f16>;
                unsafe {
                    self.kernels.ops.scale_f16(
                        &*src_ptr,
                        &mut *dst_ptr,
                        total_elements,
                        inv_sqrt2,
                    )?;
                }
            }

            Ok(Some(arch::gemma4_common::PerLayerEmbeddings {
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
                    self.kernels.ops.scale_f16(
                        &*src_ptr,
                        &mut *dst_ptr,
                        total_elements,
                        inv_sqrt2,
                    )?;
                }
            }

            Ok(Some(arch::gemma4_common::PerLayerEmbeddings {
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
        if let Some(state) = &mut self.mamba_state {
            state.reset();
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}
