//! Validate mha_naive_masked + the DiffusionGemma region-aware mask against a
//! CPU reference. Covers GQA and both layer masks (global + SWA).
use chew_engine::arch::diffusion_gemma::{build_attention_mask, MASK_BLOCK};
use chew_kernel::OpsKernels;
use cudarc::driver::CudaContext;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
fn mha_masked_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32], // [seq, kv] additive
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; seq_len * n_heads * head_dim];
    for head in 0..n_heads {
        let kv_head = head / heads_per_kv;
        for q_pos in 0..seq_len {
            let q_off = q_pos * n_heads * head_dim + head * head_dim;
            let mut scores = vec![0.0f32; kv_len];
            for kp in 0..kv_len {
                let k_off = kp * n_kv_heads * head_dim + kv_head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[kp] = dot * scale + mask[q_pos * kv_len + kp];
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let out_off = q_pos * n_heads * head_dim + head * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for kp in 0..kv_len {
                    let v_off = kp * n_kv_heads * head_dim + kv_head * head_dim;
                    acc += scores[kp] * inv * v[v_off + d];
                }
                out[out_off + d] = acc;
            }
        }
    }
    out
}

fn run_case(ops: &OpsKernels, stream: &Arc<CudaStream>, swa: bool) -> f32 {
    let (p, canvas) = (2usize, 3usize);
    let n = p + canvas; // 5 tokens
    let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, 8usize);
    let n_swa = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mask_f16 = build_attention_mask(p as u32, n as u32, n_swa, swa);
    let mask_f32: Vec<f32> = mask_f16.iter().map(|v| v.to_f32()).collect();
    // CPU uses the exact same additive values (incl. MASK_BLOCK) for parity.
    let _ = MASK_BLOCK;

    let tq = n * n_heads * head_dim;
    let tkv = n * n_kv_heads * head_dim;
    let q: Vec<f32> = (0..tq).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
    let k: Vec<f32> = (0..tkv).map(|i| (i as f32 * 0.11).cos() * 0.3).collect();
    let v: Vec<f32> = (0..tkv).map(|i| (i as f32 * 0.05 + 0.2).sin() * 0.3).collect();

    let cpu = mha_masked_cpu(
        &q, &k, &v, &mask_f32, n, n_heads, n_kv_heads, head_dim, n, scale,
    );

    let q16: Vec<half::f16> = q.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k16: Vec<half::f16> = k.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v16: Vec<half::f16> = v.iter().map(|&x| half::f16::from_f32(x)).collect();

    let mut qg = stream.alloc_zeros::<half::f16>(tq).unwrap();
    let mut kg = stream.alloc_zeros::<half::f16>(tkv).unwrap();
    let mut vg = stream.alloc_zeros::<half::f16>(tkv).unwrap();
    let mut mg = stream.alloc_zeros::<half::f16>(n * n).unwrap();
    let mut og = stream.alloc_zeros::<half::f16>(tq).unwrap();
    stream.memcpy_htod(&q16, &mut qg).unwrap();
    stream.memcpy_htod(&k16, &mut kg).unwrap();
    stream.memcpy_htod(&v16, &mut vg).unwrap();
    stream.memcpy_htod(&mask_f16, &mut mg).unwrap();

    let kv = kg.slice(0..tkv);
    let vv = vg.slice(0..tkv);
    let mv = mg.slice(0..n * n);
    ops.mha_naive_masked(
        &qg, &kv, &vv, &mv, &mut og, head_dim as u32, n_heads as u32, n_kv_heads as u32,
        n as u32, n as u32, scale, 0.0,
    )
    .unwrap();

    let mut o16 = vec![half::f16::ZERO; tq];
    stream.memcpy_dtoh(&og, &mut o16).unwrap();
    let gpu: Vec<f32> = o16.iter().map(|v| v.to_f32()).collect();
    let err = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let _ = stream;
    err
}

use cudarc::driver::CudaStream;

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());
    let ops = OpsKernels::load(&stream).unwrap();

    let e_global = run_case(&ops, &stream, false);
    let e_swa = run_case(&ops, &stream, true);
    println!("global-layer mask  max err: {e_global:.5}");
    println!("SWA-layer mask     max err: {e_swa:.5}");

    // Sanity: print a small mask so the region structure is visible.
    let m = build_attention_mask(2, 5, 2, false);
    println!("\nglobal mask [P=2, n=5] (0=allow, X=block):");
    for q in 0..5 {
        let row: String = (0..5)
            .map(|k| if m[q * 5 + k].to_f32() == 0.0 { '.' } else { 'X' })
            .collect();
        println!("  q{q}: {row}");
    }

    assert!(e_global < 0.02, "global mask mismatch");
    assert!(e_swa < 0.02, "swa mask mismatch");
    println!("\nPASS");
}
