//! Validate MHA (fused multi-head attention) kernel against CPU reference.
use chew_kernel::OpsKernels;
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// CPU MHA reference: Q@K^T/scale, causal mask, softmax, @V
/// Q: [seq_len, n_heads, head_dim]     f32
/// K: [kv_len, n_kv_heads, head_dim]   f32 (will be f16 on GPU)
/// V: [kv_len, n_kv_heads, head_dim]   f32 (will be f16 on GPU)
/// out: [seq_len, n_heads, head_dim]    f32
fn mha_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    pos_offset: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let heads_per_kv = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; seq_len * n_heads * head_dim];

    for head in 0..n_heads {
        let kv_head = head / heads_per_kv;

        for q_pos_local in 0..seq_len {
            let q_pos_global = pos_offset + q_pos_local;
            let q_off = q_pos_local * n_heads * head_dim + head * head_dim;

            // Compute scores
            let mut scores = vec![0.0f32; kv_len];
            for kp in 0..kv_len {
                if kp > q_pos_global {
                    scores[kp] = f32::NEG_INFINITY;
                } else {
                    let k_off = kp * n_kv_heads * head_dim + kv_head * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    scores[kp] = dot * scale;
                }
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }

            // Weighted sum of V
            let out_off = q_pos_local * n_heads * head_dim + head * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for kp in 0..kv_len {
                    let v_off = kp * n_kv_heads * head_dim + kv_head * head_dim;
                    acc += scores[kp] * v[v_off + d];
                }
                out[out_off + d] = acc;
            }
        }
    }
    out
}

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());
    let ops = OpsKernels::load(&stream).unwrap();

    // Test 1: simple case (seq=1, n_heads=2, n_kv_heads=2, head_dim=4, kv_len=1, pos=0)
    let seq_len = 1usize;
    let n_heads = 2usize;
    let n_kv_heads = 2usize;
    let head_dim = 4usize;
    let kv_len = 1usize;
    let pos = 0u32;

    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 heads
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25];

    let cpu_out = mha_cpu(
        &q_data,
        &k_data,
        &v_data,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        pos as usize,
    );

    // Q and out are now f16, K and V are f16 (KV cache format)
    let q_f16: Vec<half::f16> = q_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let k_f16: Vec<half::f16> = k_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_f16: Vec<half::f16> = v_data.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut q_gpu = stream.alloc_zeros::<half::f16>(q_data.len()).unwrap();
    let mut k_gpu = stream.alloc_zeros::<half::f16>(k_data.len()).unwrap();
    let mut v_gpu = stream.alloc_zeros::<half::f16>(v_data.len()).unwrap();
    let mut out_gpu = stream.alloc_zeros::<half::f16>(q_data.len()).unwrap();

    stream.memcpy_htod(&q_f16, &mut q_gpu).unwrap();
    stream.memcpy_htod(&k_f16, &mut k_gpu).unwrap();
    stream.memcpy_htod(&v_f16, &mut v_gpu).unwrap();

    let k_view = k_gpu.slice(0..k_data.len());
    let v_view = v_gpu.slice(0..v_data.len());
    ops.mha_fused(
        &q_gpu,
        &k_view,
        &v_view,
        &mut out_gpu,
        head_dim as u32,
        n_heads as u32,
        n_kv_heads as u32,
        seq_len as u32,
        kv_len as u32,
        pos,
    )
    .unwrap();

    let mut gpu_out_f16 = vec![half::f16::ZERO; q_data.len()];
    stream.memcpy_dtoh(&out_gpu, &mut gpu_out_f16).unwrap();
    let gpu_out: Vec<f32> = gpu_out_f16.iter().map(|v| v.to_f32()).collect();

    println!("Test 1: simple (seq=1, heads=2, kv_heads=2, hd=4, kv=1, pos=0)");
    println!("  CPU: {:?}", cpu_out);
    println!("  GPU: {:?}", gpu_out);
    let max_err = cpu_out
        .iter()
        .zip(&gpu_out)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    println!("  Max error: {:.6}", max_err);
    println!("  {}", if max_err < 0.01 { "PASS" } else { "FAIL" });

    // Test 2: causal mask (seq=3, kv=3, pos=0 → tokens at pos 0,1,2)
    let seq_len = 3usize;
    let n_heads = 2usize;
    let n_kv_heads = 2usize;
    let head_dim = 4usize;
    let kv_len = 3usize;
    let pos = 0u32;
    let total_q = seq_len * n_heads * head_dim; // 24
    let total_kv = kv_len * n_kv_heads * head_dim; // 24

    let q_data2: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.1).sin()).collect();
    let k_data2: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.15).cos()).collect();
    let v_data2: Vec<f32> = (0..total_kv)
        .map(|i| (i as f32 * 0.2 + 0.5).sin())
        .collect();

    let cpu_out2 = mha_cpu(
        &q_data2,
        &k_data2,
        &v_data2,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        pos as usize,
    );

    let q_f16: Vec<half::f16> = q_data2.iter().map(|&v| half::f16::from_f32(v)).collect();
    let k_f16: Vec<half::f16> = k_data2.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_f16: Vec<half::f16> = v_data2.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut q_gpu2 = stream.alloc_zeros::<half::f16>(total_q).unwrap();
    let mut k_gpu2 = stream.alloc_zeros::<half::f16>(total_kv).unwrap();
    let mut v_gpu2 = stream.alloc_zeros::<half::f16>(total_kv).unwrap();
    let mut out_gpu2 = stream.alloc_zeros::<half::f16>(total_q).unwrap();

    stream.memcpy_htod(&q_f16, &mut q_gpu2).unwrap();
    stream.memcpy_htod(&k_f16, &mut k_gpu2).unwrap();
    stream.memcpy_htod(&v_f16, &mut v_gpu2).unwrap();

    let k_view2 = k_gpu2.slice(0..total_kv);
    let v_view2 = v_gpu2.slice(0..total_kv);
    ops.mha_fused(
        &q_gpu2,
        &k_view2,
        &v_view2,
        &mut out_gpu2,
        head_dim as u32,
        n_heads as u32,
        n_kv_heads as u32,
        seq_len as u32,
        kv_len as u32,
        pos,
    )
    .unwrap();

    let mut gpu_out2_f16 = vec![half::f16::ZERO; total_q];
    stream.memcpy_dtoh(&out_gpu2, &mut gpu_out2_f16).unwrap();
    let gpu_out2: Vec<f32> = gpu_out2_f16.iter().map(|v| v.to_f32()).collect();

    let max_err2 = cpu_out2
        .iter()
        .zip(&gpu_out2)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    println!("\nTest 2: causal mask (seq=3, kv=3, heads=2, hd=4, pos=0)");
    println!("  CPU first8: {:?}", &cpu_out2[..8]);
    println!("  GPU first8: {:?}", &gpu_out2[..8]);
    println!("  Max error: {:.6}", max_err2);
    println!("  {}", if max_err2 < 0.01 { "PASS" } else { "FAIL" });

    // Test 3: GQA (n_heads=4, n_kv_heads=2 → heads_per_kv=2)
    let seq_len = 2usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = 8usize;
    let kv_len = 2usize;
    let pos = 0u32;
    let total_q = seq_len * n_heads * head_dim;
    let total_kv = kv_len * n_kv_heads * head_dim;
    let total_out = seq_len * n_heads * head_dim;

    let q_data3: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.1).sin()).collect();
    let k_data3: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.15).cos()).collect();
    let v_data3: Vec<f32> = (0..total_kv)
        .map(|i| (i as f32 * 0.2 + 0.5).sin())
        .collect();

    let cpu_out3 = mha_cpu(
        &q_data3,
        &k_data3,
        &v_data3,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        pos as usize,
    );

    let q_f16: Vec<half::f16> = q_data3.iter().map(|&v| half::f16::from_f32(v)).collect();
    let k_f16: Vec<half::f16> = k_data3.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_f16: Vec<half::f16> = v_data3.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut q_gpu3 = stream.alloc_zeros::<half::f16>(total_q).unwrap();
    let mut k_gpu3 = stream.alloc_zeros::<half::f16>(total_kv).unwrap();
    let mut v_gpu3 = stream.alloc_zeros::<half::f16>(total_kv).unwrap();
    let mut out_gpu3 = stream.alloc_zeros::<half::f16>(total_out).unwrap();

    stream.memcpy_htod(&q_f16, &mut q_gpu3).unwrap();
    stream.memcpy_htod(&k_f16, &mut k_gpu3).unwrap();
    stream.memcpy_htod(&v_f16, &mut v_gpu3).unwrap();

    let k_view3 = k_gpu3.slice(0..total_kv);
    let v_view3 = v_gpu3.slice(0..total_kv);
    ops.mha_fused(
        &q_gpu3,
        &k_view3,
        &v_view3,
        &mut out_gpu3,
        head_dim as u32,
        n_heads as u32,
        n_kv_heads as u32,
        seq_len as u32,
        kv_len as u32,
        pos,
    )
    .unwrap();

    let mut gpu_out3_f16 = vec![half::f16::ZERO; total_out];
    stream.memcpy_dtoh(&out_gpu3, &mut gpu_out3_f16).unwrap();
    let gpu_out3: Vec<f32> = gpu_out3_f16.iter().map(|v| v.to_f32()).collect();

    let max_err3 = cpu_out3
        .iter()
        .zip(&gpu_out3)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    println!("\nTest 3: GQA (seq=2, heads=4, kv_heads=2, hd=8, kv=2, pos=0)");
    println!("  CPU first8: {:?}", &cpu_out3[..8]);
    println!("  GPU first8: {:?}", &gpu_out3[..8]);
    println!("  Max error: {:.6}", max_err3);
    println!("  {}", if max_err3 < 0.01 { "PASS" } else { "FAIL" });

    // Test 4: Realistic (seq=17, n_heads=32, n_kv_heads=8, head_dim=128)
    let seq_len = 17usize;
    let n_heads = 32usize;
    let n_kv_heads = 8usize;
    let head_dim = 128usize;
    let kv_len = 17usize;
    let pos = 0u32;
    let total_q = seq_len * n_heads * head_dim;
    let total_kv = kv_len * n_kv_heads * head_dim;
    let total_out = seq_len * n_heads * head_dim;

    let q_data4: Vec<f32> = (0..total_q)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let k_data4: Vec<f32> = (0..total_kv)
        .map(|i| (i as f32 * 0.0015).cos() * 0.1)
        .collect();
    let v_data4: Vec<f32> = (0..total_kv)
        .map(|i| (i as f32 * 0.002 + 0.5).sin() * 0.1)
        .collect();

    let cpu_out4 = mha_cpu(
        &q_data4,
        &k_data4,
        &v_data4,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        pos as usize,
    );

    let q_f16: Vec<half::f16> = q_data4.iter().map(|&v| half::f16::from_f32(v)).collect();
    let k_f16: Vec<half::f16> = k_data4.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_f16: Vec<half::f16> = v_data4.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut q_gpu4 = stream.alloc_zeros::<half::f16>(total_q).unwrap();
    let mut k_gpu4 = stream.alloc_zeros::<half::f16>(total_kv).unwrap();
    let mut v_gpu4 = stream.alloc_zeros::<half::f16>(total_kv).unwrap();
    let mut out_gpu4 = stream.alloc_zeros::<half::f16>(total_out).unwrap();

    stream.memcpy_htod(&q_f16, &mut q_gpu4).unwrap();
    stream.memcpy_htod(&k_f16, &mut k_gpu4).unwrap();
    stream.memcpy_htod(&v_f16, &mut v_gpu4).unwrap();

    let k_view4 = k_gpu4.slice(0..total_kv);
    let v_view4 = v_gpu4.slice(0..total_kv);
    ops.mha_fused(
        &q_gpu4,
        &k_view4,
        &v_view4,
        &mut out_gpu4,
        head_dim as u32,
        n_heads as u32,
        n_kv_heads as u32,
        seq_len as u32,
        kv_len as u32,
        pos,
    )
    .unwrap();

    let mut gpu_out4_f16 = vec![half::f16::ZERO; total_out];
    stream.memcpy_dtoh(&out_gpu4, &mut gpu_out4_f16).unwrap();
    let gpu_out4: Vec<f32> = gpu_out4_f16.iter().map(|v| v.to_f32()).collect();

    let max_err4 = cpu_out4
        .iter()
        .zip(&gpu_out4)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    // Also check if there's a systematic pattern to errors
    let mut errs_by_pos: Vec<(usize, f32)> = Vec::new();
    for s in 0..seq_len {
        let start = s * n_heads * head_dim;
        let end = start + n_heads * head_dim;
        let pos_err = cpu_out4[start..end]
            .iter()
            .zip(&gpu_out4[start..end])
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        errs_by_pos.push((s, pos_err));
    }
    println!("\nTest 4: realistic (seq=17, heads=32, kv_heads=8, hd=128)");
    println!("  Max error: {:.6}", max_err4);
    println!("  Errors by seq_pos: {:?}", errs_by_pos);
    println!("  {}", if max_err4 < 0.05 { "PASS" } else { "FAIL" });
}
