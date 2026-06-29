//! Validate the output projection matmul (which uses chunked GEMM).
//! Creates a simple input vector, runs the matmul on GPU, and compares
//! specific output elements against CPU-computed dot products.
use chew_gguf::{GgmlType, GgufFile};
use chew_kernel::{DequantKernels, Gemm};
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// Dequant one row of Q6_K weight on CPU (256 elements at a time)
fn dequant_q6k_row_cpu(raw: &[u8], row_idx: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 210;
    let row_start = row_idx * row_bytes;

    let mut result = vec![0.0f32; k];
    for b in 0..blocks_per_row {
        let block = &raw[row_start + b * 210..row_start + (b + 1) * 210];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

        for idx in 0..256 {
            let sub = idx / 16;
            let scale = sc[sub] as i8;
            let ql_val = if idx % 2 == 0 {
                ql[idx / 2] & 0x0F
            } else {
                (ql[idx / 2] >> 4) & 0x0F
            };
            let qh_val = (qh[idx / 4] >> (2 * (idx % 4))) & 0x03;
            let q = (ql_val | (qh_val << 4)) as i32 - 32;
            result[b * 256 + idx] = d * scale as f32 * q as f32;
        }
    }
    result
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_output_matmul <model.gguf>");
    let gguf = GgufFile::open(&path).expect("failed to open GGUF");

    // Get output.weight tensor
    let (tensor, raw) = gguf
        .tensor_data_by_name("output.weight")
        .expect("no output.weight");
    let k = 4096u32;
    let n = (tensor.n_elements() as u32) / k; // vocab_size
    println!(
        "output.weight: {:?}, n={}, k={}, n_elements={}",
        tensor.ggml_type,
        n,
        k,
        tensor.n_elements()
    );

    // Create simple input: all ones
    let input_f32: Vec<f32> = vec![1.0f32; k as usize];
    let input_f16: Vec<half::f16> = input_f32.iter().map(|&x| half::f16::from_f32(x)).collect();

    // CPU reference: dot product of input with specific rows
    let test_rows = [0usize, 1, 100, 60704, 122342, n as usize - 1];
    let mut cpu_dots = Vec::new();
    for &row in &test_rows {
        let weight_row = dequant_q6k_row_cpu(raw, row, k as usize);
        let dot: f32 = weight_row.iter().sum(); // dot with all-ones = sum
        cpu_dots.push(dot);
        println!("CPU dot product (row {}): {:.6}", row, dot);
    }

    // GPU matmul
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());

    let dequant = DequantKernels::load(&stream).unwrap();
    let mut gemm = Gemm::new(&stream, (n * k) as usize).unwrap();

    // Upload input
    let mut a_gpu = stream.alloc_zeros::<half::f16>(k as usize).unwrap();
    stream.memcpy_htod(&input_f16, &mut a_gpu).unwrap();

    // Upload quantized weights
    let mut w_gpu = stream.alloc_zeros::<u8>(raw.len()).unwrap();
    stream.memcpy_htod(raw, &mut w_gpu).unwrap();

    // Output
    let mut c_gpu = stream.alloc_zeros::<half::f16>(n as usize).unwrap();

    // Run matmul_dequant (this will use chunked GEMM internally)
    gemm.matmul_dequant(
        &a_gpu,
        &w_gpu,
        tensor.ggml_type,
        tensor.n_elements() as u32,
        &mut c_gpu,
        1,
        n,
        k,
        &dequant,
    )
    .unwrap();

    // Download result
    let mut c_host = vec![half::f16::ZERO; n as usize];
    stream.memcpy_dtoh(&c_gpu, &mut c_host).unwrap();

    println!("\nGPU results:");
    for (i, &row) in test_rows.iter().enumerate() {
        let gpu_val = c_host[row].to_f32();
        let cpu_val = cpu_dots[i];
        let err = (gpu_val - cpu_val).abs();
        let pct = if cpu_val.abs() > 1e-6 {
            err / cpu_val.abs() * 100.0
        } else {
            0.0
        };
        let status = if pct < 1.0 { "OK" } else { "MISMATCH" };
        println!(
            "  row {:6}: gpu={:10.4} cpu={:10.4} err={:.4} ({:.1}%) {}",
            row, gpu_val, cpu_val, err, pct, status
        );
    }

    // Check top-5 by GPU value
    let mut indexed: Vec<(usize, f32)> = c_host
        .iter()
        .enumerate()
        .map(|(i, h)| (i, h.to_f32()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nGPU top 5 outputs:");
    for (idx, val) in indexed.iter().take(5) {
        println!("  row {:6}: {:.4}", idx, val);
    }
}
