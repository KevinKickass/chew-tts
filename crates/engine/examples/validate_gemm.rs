//! Validate cuBLAS GEMM by computing a known matrix multiplication.
//! Tests: C = A @ B^T where A=[m,k], B=[n,k], C=[m,n]
use chew_kernel::Gemm;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn main() {
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());

    // Small test: m=1, n=3, k=4
    // A = [1, 4]: [1, 2, 3, 4]
    // B = [3, 4]: [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    // C = A @ B^T = [1*1+2*0+3*0+4*0, 1*0+2*1+3*0+4*0, 1*0+2*0+3*1+4*0] = [1, 2, 3]

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ];

    let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let b_f16: Vec<half::f16> = b_data.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut a_gpu = stream.alloc_zeros::<half::f16>(4).unwrap();
    let mut b_gpu = stream.alloc_zeros::<half::f16>(12).unwrap();
    let mut c_gpu = stream.alloc_zeros::<half::f16>(3).unwrap();

    stream.memcpy_htod(&a_f16, &mut a_gpu).unwrap();
    stream.memcpy_htod(&b_f16, &mut b_gpu).unwrap();

    let gemm = Gemm::new(&stream, 12).unwrap();
    gemm.matmul_f16(&a_gpu, &b_gpu, &mut c_gpu, 1, 3, 4).unwrap();

    let mut c_host_f16 = vec![half::f16::ZERO; 3];
    stream.memcpy_dtoh(&c_gpu, &mut c_host_f16).unwrap();
    let c_host: Vec<f32> = c_host_f16.iter().map(|v| v.to_f32()).collect();

    println!("Test 1: C = A @ B^T (identity-like B)");
    println!("  Expected: [1.0, 2.0, 3.0]");
    println!("  Got:      [{}, {}, {}]", c_host[0], c_host[1], c_host[2]);

    let ok = (c_host[0] - 1.0).abs() < 0.01
        && (c_host[1] - 2.0).abs() < 0.01
        && (c_host[2] - 3.0).abs() < 0.01;
    println!("  {}", if ok { "PASS" } else { "FAIL" });

    // Test 2: m=2, n=3, k=4
    // A = [[1,2,3,4],[5,6,7,8]]
    // B = [[1,1,0,0],[0,0,1,1],[1,0,1,0]]
    // C = A @ B^T
    // C[0,0] = 1+2 = 3, C[0,1] = 3+4 = 7, C[0,2] = 1+3 = 4
    // C[1,0] = 5+6 = 11, C[1,1] = 7+8 = 15, C[1,2] = 5+7 = 12
    let a2_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b2_data: Vec<f32> = vec![
        1.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 0.0,
    ];

    let a2_f16: Vec<half::f16> = a2_data.iter().map(|&v| half::f16::from_f32(v)).collect();
    let b2_f16: Vec<half::f16> = b2_data.iter().map(|&v| half::f16::from_f32(v)).collect();

    let mut a2_gpu = stream.alloc_zeros::<half::f16>(8).unwrap();
    let mut b2_gpu = stream.alloc_zeros::<half::f16>(12).unwrap();
    let mut c2_gpu = stream.alloc_zeros::<half::f16>(6).unwrap();

    stream.memcpy_htod(&a2_f16, &mut a2_gpu).unwrap();
    stream.memcpy_htod(&b2_f16, &mut b2_gpu).unwrap();

    gemm.matmul_f16(&a2_gpu, &b2_gpu, &mut c2_gpu, 2, 3, 4).unwrap();

    let mut c2_host_f16 = vec![half::f16::ZERO; 6];
    stream.memcpy_dtoh(&c2_gpu, &mut c2_host_f16).unwrap();
    let c2_host: Vec<f32> = c2_host_f16.iter().map(|v| v.to_f32()).collect();

    println!("\nTest 2: C = A @ B^T (m=2, n=3, k=4)");
    println!("  Expected: [3, 7, 4, 11, 15, 12]");
    println!("  Got:      [{}, {}, {}, {}, {}, {}]",
        c2_host[0], c2_host[1], c2_host[2],
        c2_host[3], c2_host[4], c2_host[5]);

    let ok2 = (c2_host[0] - 3.0).abs() < 0.01
        && (c2_host[1] - 7.0).abs() < 0.01
        && (c2_host[2] - 4.0).abs() < 0.01
        && (c2_host[3] - 11.0).abs() < 0.01
        && (c2_host[4] - 15.0).abs() < 0.01
        && (c2_host[5] - 12.0).abs() < 0.01;
    println!("  {}", if ok2 { "PASS" } else { "FAIL" });
}
