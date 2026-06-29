//! Validate token_embd dequantization at specific rows (including BOS=128000).
//! CPU-dequants Q4_K blocks for reference, then compares against GPU dequant.
use chew_gguf::{GgmlType, GgufFile};
use chew_kernel::DequantKernels;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn get_scale_min_q4k(sub: usize, scales: &[u8]) -> (u8, u8) {
    if sub < 4 {
        (scales[sub] & 0x3F, scales[sub + 4] & 0x3F)
    } else {
        let off = sub - 4;
        let sc = (scales[off + 8] & 0x0F) | ((scales[off] >> 6) << 4);
        let m = (scales[off + 8] >> 4) | ((scales[off + 4] >> 6) << 4);
        (sc, m)
    }
}

fn dequant_q4k_block_cpu(block: &[u8]) -> Vec<f32> {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut result = vec![0.0f32; 256];
    let mut q_ptr = 0usize;
    for j in (0..256).step_by(64) {
        let sub0 = (j / 64) * 2;
        let sub1 = sub0 + 1;
        let (sc0, m0) = get_scale_min_q4k(sub0, scales);
        let (sc1, m1) = get_scale_min_q4k(sub1, scales);
        for l in 0..32 {
            let qb = qs[q_ptr + l];
            result[j + l] = d * sc0 as f32 * (qb & 0x0F) as f32 - dmin * m0 as f32;
            result[j + l + 32] = d * sc1 as f32 * ((qb >> 4) & 0x0F) as f32 - dmin * m1 as f32;
        }
        q_ptr += 32;
    }
    result
}

/// CPU-dequant a full row (dim elements) of a Q4_K tensor
fn dequant_q4k_row_cpu(raw: &[u8], row: usize, dim: usize) -> Vec<f32> {
    let blocks_per_row = dim / 256;
    let row_bytes = blocks_per_row * 144;
    let row_start = row * row_bytes;
    let mut result = Vec::with_capacity(dim);
    for b in 0..blocks_per_row {
        let offset = row_start + b * 144;
        let block = &raw[offset..offset + 144];
        result.extend(dequant_q4k_block_cpu(block));
    }
    result
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_embedding <model.gguf>");
    let gguf = GgufFile::open(&path).expect("failed to open GGUF");

    let (tensor, raw) = gguf
        .tensor_data_by_name("token_embd.weight")
        .expect("no token_embd.weight");
    let n_elements = tensor.n_elements() as usize;
    let dim = 4096usize;
    let vocab = n_elements / dim;
    println!(
        "token_embd: {:?}, vocab={}, dim={}, n_elements={}",
        tensor.ggml_type, vocab, dim, n_elements
    );
    println!("raw bytes: {}", raw.len());

    assert_eq!(tensor.ggml_type, GgmlType::Q4_K, "expected Q4_K embedding");

    // Test rows: row 0, row 1, BOS=128000, some others
    let test_rows = [0usize, 1, 100, 128000, 128001, vocab - 1];

    // CPU reference for each row
    println!("\n=== CPU dequant reference ===");
    for &row in &test_rows {
        let cpu_row = dequant_q4k_row_cpu(raw, row, dim);
        let first8: Vec<f32> = cpu_row[..8].to_vec();
        let nonzero = cpu_row.iter().filter(|&&v| v.abs() > 1e-10).count();
        let sum: f32 = cpu_row.iter().sum();
        println!(
            "  row {:6}: first8={:?}  nonzero={}/{}  sum={:.6}",
            row, first8, nonzero, dim, sum
        );
    }

    // GPU dequant of full tensor
    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());
    let dequant = DequantKernels::load(&stream).expect("failed to load dequant kernels");

    // Upload raw quantized bytes
    let mut src_gpu = stream.alloc_zeros::<u8>(raw.len()).unwrap();
    stream.memcpy_htod(raw, &mut src_gpu).unwrap();

    // Allocate f16 output
    let mut dst_gpu = stream.alloc_zeros::<half::f16>(n_elements).unwrap();

    // Dequantize
    dequant
        .dequant(&src_gpu, &mut dst_gpu, n_elements as u32, GgmlType::Q4_K)
        .unwrap();

    // Download specific rows and compare
    println!("\n=== GPU vs CPU comparison ===");
    for &row in &test_rows {
        let offset = row * dim;
        let slice = dst_gpu.slice(offset..offset + dim);
        let mut gpu_f16 = vec![half::f16::ZERO; dim];
        stream.memcpy_dtoh(&slice, &mut gpu_f16).unwrap();
        let gpu_f32: Vec<f32> = gpu_f16.iter().map(|h| h.to_f32()).collect();

        let cpu_row = dequant_q4k_row_cpu(raw, row, dim);

        // Compare
        let mut max_err = 0.0f32;
        let mut mismatches = 0;
        for i in 0..dim {
            let err = (gpu_f32[i] - cpu_row[i]).abs();
            if err > max_err {
                max_err = err;
            }
            if err > 0.01 {
                mismatches += 1;
            }
        }

        let gpu_first8: Vec<f32> = gpu_f32[..8].to_vec();
        let cpu_first8: Vec<f32> = cpu_row[..8].to_vec();
        let gpu_nonzero = gpu_f32.iter().filter(|&&v| v.abs() > 1e-10).count();
        let status = if mismatches == 0 { "PASS" } else { "FAIL" };

        println!(
            "  row {:6}: max_err={:.6}  mismatches={}  {}",
            row, max_err, mismatches, status
        );
        println!("    CPU first8: {:?}", cpu_first8);
        println!("    GPU first8: {:?}", gpu_first8);
        println!("    GPU nonzero: {}/{}", gpu_nonzero, dim);
    }
}
