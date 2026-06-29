//! Validate RMSNorm kernel against CPU reference.
use chew_kernel::OpsKernels;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn rms_norm_cpu(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ss / dim as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v / rms * w)
        .collect()
}

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());
    let ops = OpsKernels::load(&stream).unwrap();

    // Test 1: simple values — f32 input → f16 output via rms_norm_f32in
    let dim = 8;
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight: Vec<f32> = vec![1.0; dim];
    let eps = 1e-5f32;

    let cpu_result = rms_norm_cpu(&x, &weight, eps);

    let w_f16: Vec<half::f16> = weight.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut x_gpu = stream.alloc_zeros::<f32>(dim).unwrap();
    let mut w_gpu = stream.alloc_zeros::<half::f16>(dim).unwrap();
    let mut out_gpu = stream.alloc_zeros::<half::f16>(dim).unwrap();
    stream.memcpy_htod(&x, &mut x_gpu).unwrap();
    stream.memcpy_htod(&w_f16, &mut w_gpu).unwrap();

    ops.rms_norm_f32in(&x_gpu, &w_gpu, &mut out_gpu, 1, dim as u32, eps)
        .unwrap();

    let mut gpu_result_f16 = vec![half::f16::ZERO; dim];
    stream.memcpy_dtoh(&out_gpu, &mut gpu_result_f16).unwrap();
    let gpu_result: Vec<f32> = gpu_result_f16.iter().map(|v| v.to_f32()).collect();

    println!("Test 1: simple values (dim=8) — rms_norm_f32in (f32→f16)");
    println!("  CPU: {:?}", cpu_result);
    println!("  GPU: {:?}", gpu_result);
    let max_err = cpu_result
        .iter()
        .zip(&gpu_result)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    println!("  Max error: {:.6}", max_err);
    println!("  {}", if max_err < 0.01 { "PASS" } else { "FAIL" });

    // Test 2: realistic dim (4096) with random-ish values
    let dim = 4096;
    let x: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.1).sin() * 0.01)).collect();
    let weight: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 * 0.07).cos() * 0.5 + 1.0))
        .collect();
    let eps = 1e-5f32;

    let cpu_result = rms_norm_cpu(&x, &weight, eps);

    let w_f16: Vec<half::f16> = weight.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut x_gpu = stream.alloc_zeros::<f32>(dim).unwrap();
    let mut w_gpu = stream.alloc_zeros::<half::f16>(dim).unwrap();
    let mut out_gpu = stream.alloc_zeros::<half::f16>(dim).unwrap();
    stream.memcpy_htod(&x, &mut x_gpu).unwrap();
    stream.memcpy_htod(&w_f16, &mut w_gpu).unwrap();

    ops.rms_norm_f32in(&x_gpu, &w_gpu, &mut out_gpu, 1, dim as u32, eps)
        .unwrap();

    let mut gpu_result_f16 = vec![half::f16::ZERO; dim];
    stream.memcpy_dtoh(&out_gpu, &mut gpu_result_f16).unwrap();
    let gpu_result: Vec<f32> = gpu_result_f16.iter().map(|v| v.to_f32()).collect();

    let max_err = cpu_result
        .iter()
        .zip(&gpu_result)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let first5_cpu: Vec<f32> = cpu_result[..5].to_vec();
    let first5_gpu: Vec<f32> = gpu_result[..5].to_vec();
    println!("\nTest 2: dim=4096 (realistic)");
    println!("  CPU first5: {:?}", first5_cpu);
    println!("  GPU first5: {:?}", first5_gpu);
    println!("  Max error: {:.6}", max_err);
    println!("  {}", if max_err < 0.01 { "PASS" } else { "FAIL" });

    // Test 3: multi-row (seq_len=2, dim=4096) — tests that rows are independent
    let dim = 4096;
    let rows = 2;
    let x_row0: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.1).sin() * 0.01)).collect();
    let x_row1: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.2).cos() * 0.02)).collect();
    let weight: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 * 0.07).cos() * 0.5 + 1.0))
        .collect();

    let cpu0 = rms_norm_cpu(&x_row0, &weight, eps);
    let cpu1 = rms_norm_cpu(&x_row1, &weight, eps);

    let mut x_all: Vec<f32> = Vec::new();
    x_all.extend(&x_row0);
    x_all.extend(&x_row1);
    let w_f16: Vec<half::f16> = weight.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut x_gpu = stream.alloc_zeros::<f32>(dim * rows).unwrap();
    let mut w_gpu = stream.alloc_zeros::<half::f16>(dim).unwrap();
    let mut out_gpu = stream.alloc_zeros::<half::f16>(dim * rows).unwrap();
    stream.memcpy_htod(&x_all, &mut x_gpu).unwrap();
    stream.memcpy_htod(&w_f16, &mut w_gpu).unwrap();

    ops.rms_norm_f32in(&x_gpu, &w_gpu, &mut out_gpu, rows as u32, dim as u32, eps)
        .unwrap();

    let mut gpu_result_f16 = vec![half::f16::ZERO; dim * rows];
    stream.memcpy_dtoh(&out_gpu, &mut gpu_result_f16).unwrap();
    let gpu_result: Vec<f32> = gpu_result_f16.iter().map(|v| v.to_f32()).collect();
    let gpu0 = &gpu_result[..dim];
    let gpu1 = &gpu_result[dim..];

    let max_err0 = cpu0
        .iter()
        .zip(gpu0)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let max_err1 = cpu1
        .iter()
        .zip(gpu1)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    println!("\nTest 3: multi-row (2x4096)");
    println!(
        "  Row 0 max error: {:.6} {}",
        max_err0,
        if max_err0 < 0.01 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Row 1 max error: {:.6} {}",
        max_err1,
        if max_err1 < 0.01 { "PASS" } else { "FAIL" }
    );
}
