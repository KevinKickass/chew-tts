//! DiffusionGemma block-diffusion specifics.
//!
//! The transformer core is identical to gemma4-MoE (see `gemma4_moe`). This
//! module adds the pieces a block-diffusion decoder needs on top of that core.
//!
//! ## Self-conditioning (SC)
//!
//! DiffusionGemma feeds the *previous* denoising step's logits back into the
//! canvas embedding. Reference graph (llama.cpp diffusion-gemma.cpp:384-426):
//!
//! ```text
//! probs   = softmax(prev_logits * temp_inv)          # [C, vocab]
//! soft    = (probs @ token_embd) * sqrt(n_embd)       # [C, n_embd]
//! normed  = rms_norm(soft) * sc_pre_norm
//! g       = gelu(sc_gate @ normed)                    # [C, n_ff]
//! u       = sc_up @ normed
//! sc_sig  = (sc_down @ (g ⊙ u)) * sc_use              # [C, n_embd], gate {0,1}
//! canvas  = rms_norm(canvas_emb + sc_sig)             # final norm, NO weight
//! ```
//!
//! When `sc_use == 0` (first step) the signal is zeroed and the canvas reduces
//! to `rms_norm(canvas_emb)` — the "zero-SC -> exact forward" property.

use crate::weights::ScWeights;
use chew_kernel::{GpuKernels, KernelError};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

use crate::forward::gemm_q;

/// Additive-mask sentinel for a blocked (query, key) pair. f16-representable and
/// large enough that exp(score + BLOCK) underflows to 0 after the softmax max
/// subtraction, while staying finite (no NaN if a whole row were blocked).
pub const MASK_BLOCK: f32 = -1.0e4;

/// Build the DiffusionGemma UNIFIED attention mask `[n_tokens, n_tokens]` (f16,
/// 0 = allowed, MASK_BLOCK = blocked) for one layer type.
///
/// Layout is `[prompt (P) | canvas (n_tokens - P)]`. Rule (diffusion-gemma.cpp
/// llm_graph_input_attn_diffusion::set_input):
/// - canvas query, global layer: attends everything (bidirectional)
/// - canvas query, SWA layer: all canvas + last (n_swa-1) prompt positions
/// - prompt query: causal over earlier prompt only (never canvas), SWA-clipped
///
/// `swa = false` yields the global-layer mask, `swa = true` the SWA mask.
pub fn build_attention_mask(p: u32, n_tokens: u32, n_swa: u32, swa: bool) -> Vec<half::f16> {
    let allow = half::f16::from_f32(0.0);
    let block = half::f16::from_f32(MASK_BLOCK);
    let (p, n, n_swa) = (p as i64, n_tokens as i64, n_swa as i64);
    let canvas_prompt_lo = p - n_swa + 1;
    let mut m = vec![block; (n * n) as usize];
    for q in 0..n {
        let q_is_canvas = q >= p;
        for k in 0..n {
            let k_is_canvas = k >= p;
            let mut a = if q_is_canvas {
                if swa {
                    k_is_canvas || k >= canvas_prompt_lo
                } else {
                    true
                }
            } else {
                !k_is_canvas && k <= q
            };
            // prompt-query sliding-window clip (standard SWA: drop keys older than n_swa)
            if swa && a && !q_is_canvas && k <= q - n_swa {
                a = false;
            }
            if a {
                m[(q * n + k) as usize] = allow;
            }
        }
    }
    m
}

/// Device-resident attention masks for one diffusion forward pass, built once
/// (the layout `[prompt | canvas]` and lengths are constant across steps).
///
/// UNIFIED mode (`build`): square `[n_tokens, n_tokens]`, recomputes the prompt
/// every step. DECODE mode (`build_decode`): rectangular `[C, P+C]`, used with
/// prefix-KV — the prompt is prefilled once and only the canvas is re-decoded.
pub struct DiffusionAttn {
    /// global-layer mask f16
    pub global_mask: CudaSlice<half::f16>,
    /// SWA-layer mask f16
    pub swa_mask: CudaSlice<half::f16>,
    /// total tokens = prompt_len + canvas_len
    pub n_tokens: u32,
}

/// Build the DECODE-phase mask `[C, P+C]` (f16, 0 = allowed, MASK_BLOCK =
/// blocked) for one layer type, with prefix-KV: canvas query q (absolute
/// position P+q) attends cached prompt keys [0,P) + fresh canvas keys [P,P+C).
/// global: all prompt; SWA: last (n_swa-1) prompt. Canvas always bidirectional.
pub fn build_decode_mask(p: u32, c: u32, n_swa: u32, swa: bool) -> Vec<half::f16> {
    let allow = half::f16::from_f32(0.0);
    let block = half::f16::from_f32(MASK_BLOCK);
    let (p, c, n_swa) = (p as i64, c as i64, n_swa as i64);
    let n_kv = p + c;
    let canvas_prompt_lo = p - n_swa + 1;
    let mut m = vec![block; (c * n_kv) as usize];
    for q in 0..c {
        for k in 0..n_kv {
            let a = if k < p {
                !swa || k >= canvas_prompt_lo
            } else {
                true
            };
            if a {
                m[(q * n_kv + k) as usize] = allow;
            }
        }
    }
    m
}

impl DiffusionAttn {
    /// Build and upload both layer masks for a `[prompt(P) | canvas]` sequence.
    pub fn build(
        p: u32,
        n_tokens: u32,
        n_swa: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let g = build_attention_mask(p, n_tokens, n_swa, false);
        let s = build_attention_mask(p, n_tokens, n_swa, true);
        let mut global_mask = stream.alloc_zeros::<half::f16>(g.len())?;
        let mut swa_mask = stream.alloc_zeros::<half::f16>(s.len())?;
        stream.memcpy_htod(&g, &mut global_mask)?;
        stream.memcpy_htod(&s, &mut swa_mask)?;
        Ok(Self {
            global_mask,
            swa_mask,
            n_tokens,
        })
    }

    /// Build the DECODE-phase masks `[C, P+C]` for prefix-KV diffusion.
    pub fn build_decode(
        p: u32,
        c: u32,
        n_swa: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let g = build_decode_mask(p, c, n_swa, false);
        let s = build_decode_mask(p, c, n_swa, true);
        let mut global_mask = stream.alloc_zeros::<half::f16>(g.len())?;
        let mut swa_mask = stream.alloc_zeros::<half::f16>(s.len())?;
        stream.memcpy_htod(&g, &mut global_mask)?;
        stream.memcpy_htod(&s, &mut swa_mask)?;
        Ok(Self {
            global_mask,
            swa_mask,
            n_tokens: p + c,
        })
    }
}

/// Scratch buffers for the self-conditioning subgraph, sized for the canvas.
pub struct ScBuffers {
    /// softmax(prev_logits * temp_inv): [C, vocab] f16
    pub probs: CudaSlice<half::f16>,
    /// re-embedded soft tokens: [C, n_embd] f16
    pub soft: CudaSlice<half::f16>,
    /// rms_norm(soft) * pre_norm: [C, n_embd] f16
    pub normed: CudaSlice<half::f16>,
    /// sc_gate @ normed: [C, n_ff] f16
    pub gate_out: CudaSlice<half::f16>,
    /// sc_up @ normed: [C, n_ff] f16
    pub up_out: CudaSlice<half::f16>,
    /// gelu(gate) * up: [C, n_ff] f16
    pub glu: CudaSlice<half::f16>,
    /// sc_down @ glu: [C, n_embd] f16
    pub sig: CudaSlice<half::f16>,
    /// canvas_emb + sc_sig: [C, n_embd] f32
    pub summed: CudaSlice<f32>,
}

impl ScBuffers {
    pub fn alloc(
        canvas_len: u32,
        n_embd: u32,
        n_ff: u32,
        vocab: u32,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let c = canvas_len as usize;
        Ok(Self {
            probs: stream.alloc_zeros::<half::f16>(c * vocab as usize)?,
            soft: stream.alloc_zeros::<half::f16>(c * n_embd as usize)?,
            normed: stream.alloc_zeros::<half::f16>(c * n_embd as usize)?,
            gate_out: stream.alloc_zeros::<half::f16>(c * n_ff as usize)?,
            up_out: stream.alloc_zeros::<half::f16>(c * n_ff as usize)?,
            glu: stream.alloc_zeros::<half::f16>(c * n_ff as usize)?,
            sig: stream.alloc_zeros::<half::f16>(c * n_embd as usize)?,
            summed: stream.alloc_zeros::<f32>(c * n_embd as usize)?,
        })
    }
}

/// Entropy-bound decoder parameters (DiffusionGemma defaults from
/// diffusion-cli.cpp / model metadata).
#[derive(Clone, Copy)]
pub struct EbParams {
    pub steps: u32,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub stability: u32,
    pub confidence: f32,
}

impl Default for EbParams {
    fn default() -> Self {
        Self {
            steps: 48,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            stability: 1,
            confidence: 0.005,
        }
    }
}

/// Tiny deterministic RNG (xorshift64*) — avoids a rand dependency and keeps
/// diffusion runs reproducible per seed.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// uniform in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn next_token(&mut self, vocab: u32) -> u32 {
        (self.next_u64() % vocab as u64) as u32
    }
}

/// Result of one entropy-bound denoising step over the host-side canvas logits.
pub struct EbStep {
    /// argmax token per canvas position (the output if we stop now)
    pub argmax: Vec<u32>,
    /// next-step canvas tokens (accepted -> sampled, else fresh random)
    pub next_canvas: Vec<u32>,
    /// mean entropy across the canvas (for the confidence stop)
    pub mean_entropy: f32,
}

/// Process one step's canvas logits `[C, vocab]` (host, f32) per the entropy-
/// bound decoder: per-position argmax + entropy + multinomial sample, then
/// accept the lowest-entropy positions up to `entropy_bound` (cumulative),
/// renoise the rest. Mirrors diffusion.cpp:583-665.
/// Host post-processing for the device-side entropy-bound reduce: accept the
/// lowest-entropy positions up to the cumulative bound, renoise the rest.
pub fn eb_accept(
    argmax: &[u32],
    entropy: &[f32],
    sampled: &[u32],
    entropy_bound: f32,
    vocab: u32,
    rng: &mut Rng,
) -> EbStep {
    let c_len = argmax.len();
    let mut order: Vec<usize> = (0..c_len).collect();
    order.sort_by(|&a, &b| {
        entropy[a].partial_cmp(&entropy[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut accepted = vec![false; c_len];
    let mut cum_e = 0.0f64;
    for &pos in &order {
        if cum_e <= entropy_bound as f64 {
            accepted[pos] = true;
        }
        cum_e += entropy[pos] as f64;
    }
    let next_canvas: Vec<u32> = (0..c_len)
        .map(|pos| if accepted[pos] { sampled[pos] } else { rng.next_token(vocab) })
        .collect();
    let mean_entropy = entropy.iter().sum::<f32>() / c_len as f32;
    EbStep {
        argmax: argmax.to_vec(),
        next_canvas,
        mean_entropy,
    }
}

pub fn eb_step(
    logits: &[f32],
    c_len: usize,
    vocab: usize,
    temp_inv: f32,
    entropy_bound: f32,
    rng: &mut Rng,
) -> EbStep {
    let mut argmax = vec![0u32; c_len];
    let mut sampled = vec![0u32; c_len];
    let mut entropy = vec![0.0f32; c_len];

    for pos in 0..c_len {
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        // argmax + max (scaled)
        let mut amax = 0usize;
        let mut mx = f32::NEG_INFINITY;
        for (v, &z) in row.iter().enumerate() {
            let s = z * temp_inv;
            if s > mx {
                mx = s;
                amax = v;
            }
        }
        argmax[pos] = amax as u32;
        // partition function + entropy
        let mut zsum = 0.0f32;
        for &z in row {
            zsum += (z * temp_inv - mx).exp();
        }
        let mut h = 0.0f32;
        for &z in row {
            let p = (z * temp_inv - mx).exp() / zsum;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        entropy[pos] = h;
        // inverse-CDF multinomial sample
        let target = rng.next_f32() * zsum;
        let mut cum = 0.0f32;
        let mut tok = vocab - 1;
        for (v, &z) in row.iter().enumerate() {
            cum += (z * temp_inv - mx).exp();
            if cum >= target {
                tok = v;
                break;
            }
        }
        sampled[pos] = tok as u32;
    }

    // accept lowest-entropy positions up to the cumulative bound
    let mut order: Vec<usize> = (0..c_len).collect();
    order.sort_by(|&a, &b| entropy[a].partial_cmp(&entropy[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut accepted = vec![false; c_len];
    let mut cum_e = 0.0f64;
    for &pos in &order {
        if cum_e <= entropy_bound as f64 {
            accepted[pos] = true;
        }
        cum_e += entropy[pos] as f64;
    }

    let next_canvas: Vec<u32> = (0..c_len)
        .map(|pos| if accepted[pos] { sampled[pos] } else { rng.next_token(vocab as u32) })
        .collect();
    let mean_entropy = entropy.iter().sum::<f32>() / c_len as f32;

    EbStep {
        argmax,
        next_canvas,
        mean_entropy,
    }
}

/// Compute the diffusion canvas embedding input for the transformer layers:
/// `out = rms_norm(canvas_emb + self_conditioning(prev_logits))`.
///
/// - `canvas_emb`: scaled token embedding of the current canvas, [C, n_embd] f32
///   (already multiplied by sqrt(n_embd) by the embedding step).
/// - `prev_logits`: previous step's raw logits, [C, vocab] f16.
/// - `out`: normalised layer input, [C, n_embd] f32.
/// - `sc_use`: runtime gate, 0.0 on the first step (no prior logits), else 1.0.
#[allow(clippy::too_many_arguments)]
pub fn apply_self_conditioning(
    kernels: &mut GpuKernels,
    sc: &ScWeights,
    token_embd: &CudaSlice<half::f16>,
    prev_logits: &CudaSlice<half::f16>,
    canvas_emb: &CudaSlice<f32>,
    out: &mut CudaSlice<half::f16>,
    bufs: &mut ScBuffers,
    c_len: u32,
    n_embd: u32,
    n_ff: u32,
    vocab: u32,
    sc_use: f32,
    temp_inv: f32,
    eps: f32,
) -> Result<(), KernelError> {
    // First step: no usable prior logits -> canvas = rms_norm(canvas_emb).
    if sc_use == 0.0 {
        return kernels
            .ops
            .rms_norm_f32in_no_weight(canvas_emb, out, c_len, n_embd, eps);
    }

    // probs = softmax(prev_logits * temp_inv) over the vocab dim.
    kernels
        .ops
        .scale_f16(prev_logits, &mut bufs.probs, c_len * vocab, temp_inv)?;
    kernels.ops.softmax(&mut bufs.probs, c_len, vocab)?;

    // soft = probs @ token_embd. token_embd is [vocab, n_embd] row-major and
    // already resident, so use the no-transpose GEMM directly.
    //
    // The reference multiplies soft by sqrt(n_embd) before the norm, but RMSNorm
    // is invariant to positive scalar scaling (eps is negligible at this
    // magnitude), so the scale is folded away here — same result, one op fewer.
    let soft_n = c_len * n_embd;
    kernels
        .gemm
        .matmul_f16_nt(&bufs.probs, token_embd, &mut bufs.soft, c_len, n_embd, vocab)?;

    // normed = rms_norm(soft) * sc_pre_norm
    kernels.ops.rms_norm(
        &bufs.soft,
        &sc.pre_norm,
        &mut bufs.normed,
        c_len,
        n_embd,
        eps,
    )?;

    // SC gated MLP: g = gelu(sc_gate @ normed), u = sc_up @ normed, glu = g ⊙ u
    gemm_q(kernels, &bufs.normed, &sc.gate, &mut bufs.gate_out, c_len, n_ff, n_embd)?;
    gemm_q(kernels, &bufs.normed, &sc.up, &mut bufs.up_out, c_len, n_ff, n_embd)?;
    kernels
        .ops
        .gelu(&bufs.gate_out, &bufs.up_out, &mut bufs.glu, c_len * n_ff)?;

    // sc_sig = sc_down @ glu. The reference scales by sc_use {0,1}; the 0 case
    // is handled by the early return above, so here sc_use == 1.0 (no-op).
    gemm_q(kernels, &bufs.glu, &sc.down, &mut bufs.sig, c_len, n_embd, n_ff)?;

    // canvas = rms_norm(canvas_emb + sc_sig), final norm has no learned weight.
    kernels
        .ops
        .add_f32_f16(canvas_emb, &bufs.sig, &mut bufs.summed, soft_n)?;
    kernels
        .ops
        .rms_norm_f32in_no_weight(&bufs.summed, out, c_len, n_embd, eps)
}
