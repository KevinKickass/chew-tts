//! Validate RoPE kernel against CPU reference.
use chew_kernel::OpsKernels;
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// CPU RoPE: x has shape [seq_len, n_heads, head_dim]
fn rope_cpu(x: &mut [f32], seq_len: usize, n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    for s in 0..seq_len {
        for h in 0..n_heads {
            for p in 0..head_dim / 2 {
                let offset = s * n_heads * head_dim + h * head_dim + p * 2;
                let freq = 1.0 / (theta as f64).powf(2.0 * p as f64 / head_dim as f64);
                let angle = (pos + s) as f64 * freq;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;
                let x0 = x[offset];
                let x1 = x[offset + 1];
                x[offset]     = x0 * cos_a - x1 * sin_a;
                x[offset + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());
    let ops = OpsKernels::load(&stream).unwrap();

    // Test 1: simple case (seq_len=1, n_heads=2, head_dim=4, pos=0, theta=10000)
    let seq_len = 1usize;
    let n_heads = 2usize;
    let head_dim = 4usize;
    let pos = 0u32;
    let theta = 10000.0f32;

    let x_orig: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x_cpu = x_orig.clone();
    rope_cpu(&mut x_cpu, seq_len, n_heads, head_dim, pos as usize, theta);

    // RoPE now operates on f16
    let x_f16: Vec<half::f16> = x_orig.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut x_gpu = stream.alloc_zeros::<half::f16>(x_orig.len()).unwrap();
    stream.memcpy_htod(&x_f16, &mut x_gpu).unwrap();

    ops.rope(&mut x_gpu, seq_len as u32, n_heads as u32, head_dim as u32, pos, theta).unwrap();

    let mut gpu_result_f16 = vec![half::f16::ZERO; x_orig.len()];
    stream.memcpy_dtoh(&x_gpu, &mut gpu_result_f16).unwrap();
    let gpu_result: Vec<f32> = gpu_result_f16.iter().map(|v| v.to_f32()).collect();

    println!("Test 1: seq_len=1, n_heads=2, head_dim=4, pos=0, theta=10000");
    println!("  CPU: {:?}", x_cpu);
    println!("  GPU: {:?}", gpu_result);
    let max_err = x_cpu.iter().zip(&gpu_result).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("  Max error: {:.6}", max_err);
    println!("  {}", if max_err < 0.01 { "PASS" } else { "FAIL" });

    // Test 2: pos=5 (non-zero position)
    let pos = 5u32;
    let mut x_cpu2 = x_orig.clone();
    rope_cpu(&mut x_cpu2, seq_len, n_heads, head_dim, pos as usize, theta);

    stream.memcpy_htod(&x_f16, &mut x_gpu).unwrap();
    ops.rope(&mut x_gpu, seq_len as u32, n_heads as u32, head_dim as u32, pos, theta).unwrap();
    let mut gpu2_f16 = vec![half::f16::ZERO; x_orig.len()];
    stream.memcpy_dtoh(&x_gpu, &mut gpu2_f16).unwrap();
    let gpu2: Vec<f32> = gpu2_f16.iter().map(|v| v.to_f32()).collect();

    let max_err2 = x_cpu2.iter().zip(&gpu2).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("\nTest 2: pos=5");
    println!("  CPU: {:?}", x_cpu2);
    println!("  GPU: {:?}", gpu2);
    println!("  Max error: {:.6}", max_err2);
    println!("  {}", if max_err2 < 0.01 { "PASS" } else { "FAIL" });

    // Test 3: realistic (seq_len=3, n_heads=32, head_dim=128, pos=0, theta=500000)
    let seq_len = 3usize;
    let n_heads = 32usize;
    let head_dim = 128usize;
    let pos = 0u32;
    let theta = 500000.0f32;
    let total = seq_len * n_heads * head_dim;
    let x_orig3: Vec<f32> = (0..total).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut x_cpu3 = x_orig3.clone();
    rope_cpu(&mut x_cpu3, seq_len, n_heads, head_dim, pos as usize, theta);

    let x_f16_3: Vec<half::f16> = x_orig3.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut x_gpu3 = stream.alloc_zeros::<half::f16>(total).unwrap();
    stream.memcpy_htod(&x_f16_3, &mut x_gpu3).unwrap();
    ops.rope(&mut x_gpu3, seq_len as u32, n_heads as u32, head_dim as u32, pos, theta).unwrap();
    let mut gpu3_f16 = vec![half::f16::ZERO; total];
    stream.memcpy_dtoh(&x_gpu3, &mut gpu3_f16).unwrap();
    let gpu3: Vec<f32> = gpu3_f16.iter().map(|v| v.to_f32()).collect();

    let max_err3 = x_cpu3.iter().zip(&gpu3).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("\nTest 3: realistic (seq=3, heads=32, head_dim=128, theta=500000)");
    println!("  CPU first8: {:?}", &x_cpu3[..8]);
    println!("  GPU first8: {:?}", &gpu3[..8]);
    println!("  Max error: {:.6}", max_err3);
    println!("  {}", if max_err3 < 0.01 { "PASS" } else { "FAIL" });

    // Test 4: seq_len=3 with pos=10 — each seq position should get pos+seq_idx
    let pos = 10u32;
    let mut x_cpu4 = x_orig3.clone();
    rope_cpu(&mut x_cpu4, seq_len, n_heads, head_dim, pos as usize, theta);
    stream.memcpy_htod(&x_f16_3, &mut x_gpu3).unwrap();
    ops.rope(&mut x_gpu3, seq_len as u32, n_heads as u32, head_dim as u32, pos, theta).unwrap();
    let mut gpu4_f16 = vec![half::f16::ZERO; total];
    stream.memcpy_dtoh(&x_gpu3, &mut gpu4_f16).unwrap();
    let gpu4: Vec<f32> = gpu4_f16.iter().map(|v| v.to_f32()).collect();

    let max_err4 = x_cpu4.iter().zip(&gpu4).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("\nTest 4: pos=10 (multiple seq positions get pos+0, pos+1, pos+2)");
    println!("  Max error: {:.6}", max_err4);
    println!("  {}", if max_err4 < 0.01 { "PASS" } else { "FAIL" });
}
