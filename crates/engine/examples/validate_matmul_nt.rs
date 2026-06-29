//! Validate Gemm::matmul_f16_nt: C[m,n] = A[m,k] @ B[k,n] (B not transposed).
//! Compares against a CPU reference for non-symmetric data to catch any
//! transpose/orientation bug (the classic "nur Müll" failure mode).
use chew_kernel::Gemm;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());

    // Deterministic non-symmetric data. m=3, k=4, n=5.
    let (m, k, n) = (3usize, 4usize, 5usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.5) - 2.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.25) - 1.0).collect();
    let expected = cpu_matmul(&a, &b, m, n, k);

    let a_f16: Vec<half::f16> = a.iter().map(|&v| half::f16::from_f32(v)).collect();
    let b_f16: Vec<half::f16> = b.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut a_gpu = stream.alloc_zeros::<half::f16>(m * k).unwrap();
    let mut b_gpu = stream.alloc_zeros::<half::f16>(k * n).unwrap();
    let mut c_gpu = stream.alloc_zeros::<half::f16>(m * n).unwrap();
    stream.memcpy_htod(&a_f16, &mut a_gpu).unwrap();
    stream.memcpy_htod(&b_f16, &mut b_gpu).unwrap();

    let gemm = Gemm::new(&stream, (k * n).max(m * n)).unwrap();
    gemm.matmul_f16_nt(&a_gpu, &b_gpu, &mut c_gpu, m as u32, n as u32, k as u32)
        .unwrap();

    let mut c_f16 = vec![half::f16::ZERO; m * n];
    stream.memcpy_dtoh(&c_gpu, &mut c_f16).unwrap();
    let got: Vec<f32> = c_f16.iter().map(|v| v.to_f32()).collect();

    println!("matmul_f16_nt  C[{m},{n}] = A[{m},{k}] @ B[{k},{n}]");
    let mut max_err = 0.0f32;
    for i in 0..m * n {
        max_err = max_err.max((got[i] - expected[i]).abs());
    }
    println!("  expected: {expected:?}");
    println!("  got:      {got:?}");
    println!("  max abs err: {max_err:.4}");
    assert!(max_err < 0.1, "matmul_f16_nt orientation/precision mismatch");
    println!("  PASS");
}
