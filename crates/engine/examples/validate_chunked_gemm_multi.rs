//! Validate chunked GEMM with M>1 (multiple rows) against non-chunked reference.
//! This tests whether the strided writes with ldc=full_N work for multi-row inputs.
use chew_kernel::Gemm;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());

    // Test: M=4, N=12, K=8. Scratch only fits 6*8=48 elements, forcing 2 chunks of 6.
    let m = 4u32;
    let n = 12u32;
    let k = 8u32;

    // A: [4, 8]
    let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.1).sin()).collect();
    // B: [12, 8] — weight matrix
    let b_data: Vec<f32> = (0..n*k).map(|i| (i as f32 * 0.07 + 0.3).cos()).collect();

    let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let b_f16: Vec<half::f16> = b_data.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut a_gpu = stream.alloc_zeros::<half::f16>((m*k) as usize).unwrap();
    let mut b_gpu = stream.alloc_zeros::<half::f16>((n*k) as usize).unwrap();
    stream.memcpy_htod(&a_f16, &mut a_gpu).unwrap();
    stream.memcpy_htod(&b_f16, &mut b_gpu).unwrap();

    // Reference: non-chunked GEMM (scratch big enough for everything)
    let mut c_ref_gpu = stream.alloc_zeros::<half::f16>((m*n) as usize).unwrap();
    let gemm_full = Gemm::new(&stream, (n*k) as usize).unwrap();
    gemm_full.matmul_f16(&a_gpu, &b_gpu, &mut c_ref_gpu, m, n, k).unwrap();

    let mut c_ref_f16 = vec![half::f16::ZERO; (m*n) as usize];
    stream.memcpy_dtoh(&c_ref_gpu, &mut c_ref_f16).unwrap();
    let c_ref: Vec<f32> = c_ref_f16.iter().map(|v| v.to_f32()).collect();

    // Chunked: set scratch to force chunking (only 48 elements = 6 rows of K=8)
    let mut _gemm_small = Gemm::new(&stream, 48).unwrap();  // 48 elements = 6 rows max per chunk

    // Better test: realistic scenario with huge N
    // Let's use N=20, K=8, scratch=64 (8 rows per chunk) → 3 chunks: 8+8+4
    let m = 3u32;
    let n = 20u32;
    let k = 8u32;

    let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.1).sin()).collect();
    let b_data: Vec<f32> = (0..n*k).map(|i| (i as f32 * 0.07 + 0.3).cos()).collect();

    let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let b_f16: Vec<half::f16> = b_data.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut a_gpu = stream.alloc_zeros::<half::f16>((m*k) as usize).unwrap();
    let mut b_gpu = stream.alloc_zeros::<half::f16>((n*k) as usize).unwrap();
    stream.memcpy_htod(&a_f16, &mut a_gpu).unwrap();
    stream.memcpy_htod(&b_f16, &mut b_gpu).unwrap();

    // Reference: full GEMM
    let mut c_ref_gpu = stream.alloc_zeros::<half::f16>((m*n) as usize).unwrap();
    let gemm_full = Gemm::new(&stream, (n*k) as usize).unwrap();
    gemm_full.matmul_f16(&a_gpu, &b_gpu, &mut c_ref_gpu, m, n, k).unwrap();
    let mut c_ref_f16 = vec![half::f16::ZERO; (m*n) as usize];
    stream.memcpy_dtoh(&c_ref_gpu, &mut c_ref_f16).unwrap();
    let c_ref: Vec<f32> = c_ref_f16.iter().map(|v| v.to_f32()).collect();

    // CPU reference
    let mut c_cpu = vec![0.0f32; (m*n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for l in 0..k as usize {
                sum += a_data[i * k as usize + l] * b_data[j * k as usize + l];
            }
            c_cpu[i * n as usize + j] = sum;
        }
    }

    // Check full GEMM against CPU
    let max_err_full = c_cpu.iter().zip(&c_ref).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);

    println!("Test: M={}, N={}, K={}", m, n, k);
    println!("  Full GEMM vs CPU max_err: {:.6} {}", max_err_full,
        if max_err_full < 0.05 { "PASS" } else { "FAIL" });

    // Now test chunked GEMM manually by calling gemm_chunked
    // Simulate chunked GEMM: chunk_n=8 (so 3 chunks: 8,8,4)
    let chunk_n = 8u32;
    let mut c_chunked_gpu = stream.alloc_zeros::<half::f16>((m*n) as usize).unwrap();

    let alpha = half::f16::from_f32(1.0);
    let beta = half::f16::from_f32(0.0);

    let blas = cudarc::cublas::CudaBlas::new(Arc::clone(&stream)).unwrap();

    let mut n_done = 0u32;
    while n_done < n {
        let chunk = (n - n_done).min(chunk_n);
        let b_chunk = b_gpu.slice((n_done * k) as usize..((n_done + chunk) * k) as usize);

        let cfg = cudarc::cublas::GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: chunk as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,  // full N, not chunk
        };

        let mut c_view = c_chunked_gpu.slice_mut(n_done as usize..);

        unsafe {
            cudarc::cublas::Gemm::gemm(&blas, cfg, &b_chunk, &a_gpu, &mut c_view).unwrap();
        }

        n_done += chunk;
    }

    let mut c_chunked_f16 = vec![half::f16::ZERO; (m*n) as usize];
    stream.memcpy_dtoh(&c_chunked_gpu, &mut c_chunked_f16).unwrap();
    let c_chunked: Vec<f32> = c_chunked_f16.iter().map(|v| v.to_f32()).collect();

    let max_err_chunked = c_cpu.iter().zip(&c_chunked).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("  Chunked GEMM vs CPU max_err: {:.6} {}", max_err_chunked,
        if max_err_chunked < 0.05 { "PASS" } else { "FAIL" });

    // Print specific values for first row
    println!("\n  CPU row 0: {:?}", &c_cpu[..n as usize]);
    println!("  Full row 0: {:?}", &c_ref[..n as usize]);
    println!("  Chunked row 0: {:?}", &c_chunked[..n as usize]);

    // Print last row (row m-1)
    let last_start = ((m-1)*n) as usize;
    let last_end = (m*n) as usize;
    println!("\n  CPU row {}: {:?}", m-1, &c_cpu[last_start..last_end]);
    println!("  Full row {}: {:?}", m-1, &c_ref[last_start..last_end]);
    println!("  Chunked row {}: {:?}", m-1, &c_chunked[last_start..last_end]);
}
