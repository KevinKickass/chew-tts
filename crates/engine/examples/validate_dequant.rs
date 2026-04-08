//! Validate Q4_K and Q6_K dequantization against CPU reference.
use chew_gguf::{GgufFile, GgmlType};
use chew_kernel::DequantKernels;
use cudarc::driver::CudaContext;
use std::sync::Arc;

fn dequant_q4k_cpu(block: &[u8]) -> Vec<f32> {
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

/// Q6_K CPU dequant (reference: llama.cpp ggml-quants.c)
/// Block layout: ql[128] | qh[64] | scales[16] | d[2] = 210 bytes, 256 elements
fn dequant_q6k_cpu(block: &[u8]) -> Vec<f32> {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

    let mut result = vec![0.0f32; 256];
    for idx in 0..256 {
        let sub = idx / 16;
        let scale = sc[sub] as i8;

        // Low 4 bits from ql
        let ql_val = if idx % 2 == 0 {
            ql[idx / 2] & 0x0F
        } else {
            (ql[idx / 2] >> 4) & 0x0F
        };

        // High 2 bits from qh
        let qh_val = (qh[idx / 4] >> (2 * (idx % 4))) & 0x03;

        let q = (ql_val | (qh_val << 4)) as i32 - 32;
        result[idx] = d * scale as f32 * q as f32;
    }
    result
}

fn validate(name: &str, cpu: &[f32], gpu: &[f32], n: usize) {
    let mut max_err = 0.0f32;
    let mut max_err_idx = 0;
    let mut mismatches = 0;
    for i in 0..n {
        let err = (cpu[i] - gpu[i]).abs();
        if err > 0.002 {
            if mismatches < 10 {
                println!("  MISMATCH [{:3}]: cpu={:.8} gpu={:.8} err={:.8}", i, cpu[i], gpu[i], err);
            }
            mismatches += 1;
        }
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }
    println!("  Max error: {:.8} at index {}", max_err, max_err_idx);
    println!("  Mismatches (>0.002): {}/{}", mismatches, n);
    if mismatches == 0 {
        println!("  {} PASS", name);
    } else {
        println!("  {} FAIL", name);
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: validate_dequant <model.gguf>");
    let gguf = GgufFile::open(&path).expect("failed to open GGUF");

    cudarc::driver::result::init().unwrap();
    let ctx = CudaContext::new(0).expect("no GPU");
    let stream = Arc::new(ctx.default_stream());
    let dequant = DequantKernels::load(&stream).expect("failed to load dequant kernels");

    // === Test Q4_K ===
    let tensor = gguf.tensors.iter()
        .find(|t| t.ggml_type == GgmlType::Q4_K)
        .expect("no Q4_K tensor");
    let raw = gguf.tensor_data(tensor).unwrap();
    let block = &raw[..144];
    let cpu = dequant_q4k_cpu(block);

    let mut qg = stream.alloc_zeros::<u8>(144).unwrap();
    stream.memcpy_htod(block, &mut qg).unwrap();
    let mut og = stream.alloc_zeros::<half::f16>(256).unwrap();
    dequant.dequant(&qg, &mut og, 256, GgmlType::Q4_K).unwrap();
    let mut gh = vec![half::f16::ZERO; 256];
    stream.memcpy_dtoh(&og, &mut gh).unwrap();
    let gpu: Vec<f32> = gh.iter().map(|h| h.to_f32()).collect();

    println!("=== Q4_K ({}) ===", tensor.name);
    println!("  CPU[0..4]: {:?}", &cpu[0..4]);
    println!("  GPU[0..4]: {:?}", &gpu[0..4]);
    validate("Q4_K", &cpu, &gpu, 256);

    // === Test Q6_K ===
    let tensor6 = gguf.tensors.iter()
        .find(|t| t.ggml_type == GgmlType::Q6_K)
        .expect("no Q6_K tensor");
    let raw6 = gguf.tensor_data(tensor6).unwrap();
    let block6 = &raw6[..210];
    let cpu6 = dequant_q6k_cpu(block6);

    let mut qg6 = stream.alloc_zeros::<u8>(210).unwrap();
    stream.memcpy_htod(block6, &mut qg6).unwrap();
    let mut og6 = stream.alloc_zeros::<half::f16>(256).unwrap();
    dequant.dequant(&qg6, &mut og6, 256, GgmlType::Q6_K).unwrap();
    let mut gh6 = vec![half::f16::ZERO; 256];
    stream.memcpy_dtoh(&og6, &mut gh6).unwrap();
    let gpu6: Vec<f32> = gh6.iter().map(|h| h.to_f32()).collect();

    println!("\n=== Q6_K ({}) ===", tensor6.name);
    println!("  CPU[0..4]: {:?}", &cpu6[0..4]);
    println!("  GPU[0..4]: {:?}", &gpu6[0..4]);
    validate("Q6_K", &cpu6, &gpu6, 256);

    // Also test multi-block Q6_K (2 blocks = 512 elements)
    if raw6.len() >= 420 {
        let blocks2 = &raw6[..420];
        let mut cpu_2b = dequant_q6k_cpu(&blocks2[..210]);
        cpu_2b.extend(dequant_q6k_cpu(&blocks2[210..420]));

        let mut qg2 = stream.alloc_zeros::<u8>(420).unwrap();
        stream.memcpy_htod(blocks2, &mut qg2).unwrap();
        let mut og2 = stream.alloc_zeros::<half::f16>(512).unwrap();
        dequant.dequant(&qg2, &mut og2, 512, GgmlType::Q6_K).unwrap();
        let mut gh2 = vec![half::f16::ZERO; 512];
        stream.memcpy_dtoh(&og2, &mut gh2).unwrap();
        let gpu_2b: Vec<f32> = gh2.iter().map(|h| h.to_f32()).collect();

        println!("\n=== Q6_K 2-block ({}) ===", tensor6.name);
        validate("Q6_K 2-block", &cpu_2b, &gpu_2b, 512);
    }
}
