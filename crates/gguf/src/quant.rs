/// Dequantization routines for GGUF quantization types.
///
/// These run on CPU to extract f32 weights from quantized blocks.
/// For GPU inference, the dequant happens in CUDA kernels (chew-kernel crate).
/// This module is used for validation, debugging, and CPU fallback.
use crate::types::GgmlType;

/// Dequantize a block of quantized data to f32.
///
/// `data` is the raw quantized bytes for one or more blocks.
/// Returns a Vec of f32 weights.
pub fn dequantize(data: &[u8], ggml_type: GgmlType, n_elements: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n_elements];
    match ggml_type {
        GgmlType::F32 => dequant_f32(data, &mut output),
        GgmlType::F16 => dequant_f16(data, &mut output),
        GgmlType::BF16 => dequant_bf16(data, &mut output),
        GgmlType::Q8_0 => dequant_q8_0(data, &mut output),
        GgmlType::Q4_0 => dequant_q4_0(data, &mut output),
        GgmlType::Q4_K => dequant_q4_k(data, &mut output),
        GgmlType::Q6_K => dequant_q6_k(data, &mut output),
        GgmlType::Q2_K => dequant_q2_k(data, &mut output),
        _ => {
            // Unsupported type — fill with zeros, log warning
            tracing::warn!(ggml_type = %ggml_type, "CPU dequant not implemented, returning zeros");
        }
    }
    output
}

// --- F32 / F16 / BF16 (trivial) ---

fn dequant_f32(data: &[u8], output: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(4).enumerate().take(output.len()) {
        output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

fn dequant_f16(data: &[u8], output: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(2).enumerate().take(output.len()) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        output[i] = half::f16::from_bits(bits).to_f32();
    }
}

fn dequant_bf16(data: &[u8], output: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(2).enumerate().take(output.len()) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        output[i] = half::bf16::from_bits(bits).to_f32();
    }
}

// --- Q8_0: 32 weights per block, f16 scale + 32 * i8 ---

fn dequant_q8_0(data: &[u8], output: &mut [f32]) {
    let block_size = 34; // 2 (f16 scale) + 32 (i8 quants)
    for (bi, block) in data.chunks_exact(block_size).enumerate() {
        let d = f16_to_f32(block[0], block[1]);
        let base = bi * 32;
        for j in 0..32 {
            if base + j < output.len() {
                let q = block[2 + j] as i8;
                output[base + j] = d * q as f32;
            }
        }
    }
}

// --- Q4_0: 32 weights per block, f16 scale + 16 bytes (4-bit quants) ---

fn dequant_q4_0(data: &[u8], output: &mut [f32]) {
    let block_size = 18; // 2 + 16
    for (bi, block) in data.chunks_exact(block_size).enumerate() {
        let d = f16_to_f32(block[0], block[1]);
        let base = bi * 32;
        for j in 0..32 {
            if base + j >= output.len() {
                break;
            }
            let byte = block[2 + j / 2];
            let q = if j % 2 == 0 {
                (byte & 0x0F) as i32 - 8
            } else {
                ((byte >> 4) & 0x0F) as i32 - 8
            };
            output[base + j] = d * q as f32;
        }
    }
}

// --- Q4_K: 256 weights per block ---
// f16 d + f16 dmin + 12 bytes scales + 128 bytes qs

fn dequant_q4_k(data: &[u8], output: &mut [f32]) {
    let block_bytes = 144; // 2 + 2 + 12 + 128
    for (bi, block) in data.chunks_exact(block_bytes).enumerate() {
        let d = f16_to_f32(block[0], block[1]);
        let dmin = f16_to_f32(block[2], block[3]);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let base = bi * 256;

        for sub in 0..8 {
            // 6-bit scales packed into 12 bytes
            let (sc, m) = decode_q4k_scale(scales, sub);
            let scale = d * sc as f32;
            let min = dmin * m as f32;

            for j in 0..32 {
                let idx = sub * 32 + j;
                if base + idx >= output.len() {
                    return;
                }
                let byte = qs[idx / 2 + (sub / 2) * 32]; // interleaved layout
                let q = if idx % 2 == 0 {
                    (byte & 0x0F) as f32
                } else {
                    ((byte >> 4) & 0x0F) as f32
                };
                output[base + idx] = q * scale - min;
            }
        }
    }
}

fn decode_q4k_scale(scales: &[u8], sub: usize) -> (u8, u8) {
    // Q4_K packs 6-bit scale + 6-bit min for 8 sub-blocks into 12 bytes
    // Lower 4 bits in first 8 bytes, upper 2 bits packed in bytes 8-11
    if sub < 4 {
        let sc = (scales[sub] & 0x3F) as u8;
        let m = (scales[sub + 4] & 0x3F) as u8;
        (sc, m)
    } else {
        let off = sub - 4;
        let sc_lo = (scales[off] >> 6) as u8;
        let sc_hi = ((scales[off + 8 - 4] >> ((off % 2) * 2)) & 0x03) as u8;
        let m_lo = (scales[off + 4] >> 6) as u8;
        let m_hi = ((scales[off + 8 - 4] >> ((off % 2) * 2 + 4)) & 0x03) as u8;
        (sc_lo | (sc_hi << 2), m_lo | (m_hi << 2))
    }
}

// --- Q6_K: 256 weights per block ---
// f16 d + 128 ql + 64 qh + 16 scales

fn dequant_q6_k(data: &[u8], output: &mut [f32]) {
    let block_bytes = 210; // 2 + 128 + 64 + 16
    for (bi, block) in data.chunks_exact(block_bytes).enumerate() {
        let d = f16_to_f32(block[0], block[1]);
        let ql = &block[2..130];
        let qh = &block[130..194];
        let scales = &block[194..210];

        let base = bi * 256;

        for k in 0..256 {
            if base + k >= output.len() {
                return;
            }
            let sub = k / 16;
            let sc = scales[sub] as i8;

            // Lower 4 bits from ql, upper 2 bits from qh
            let ql_val = if k % 2 == 0 {
                ql[k / 2] & 0x0F
            } else {
                (ql[k / 2] >> 4) & 0x0F
            };

            let qh_val = (qh[k / 4] >> ((k % 4) * 2)) & 0x03;
            let q = (ql_val | (qh_val << 4)) as i32 - 32;

            output[base + k] = d * sc as f32 * q as f32;
        }
    }
}

// --- Q2_K: 256 weights per block ---
// f16 d + f16 dmin + 16 scales + 64 qs

fn dequant_q2_k(data: &[u8], output: &mut [f32]) {
    let block_bytes = 84; // 2 + 2 + 16 + 64
    for (bi, block) in data.chunks_exact(block_bytes).enumerate() {
        let d = f16_to_f32(block[0], block[1]);
        let dmin = f16_to_f32(block[2], block[3]);
        let scales = &block[4..20];
        let qs = &block[20..84];

        let base = bi * 256;

        for sub in 0..16 {
            let sc = (scales[sub] & 0x0F) as f32;
            let m = (scales[sub] >> 4) as f32;

            for j in 0..16 {
                let idx = sub * 16 + j;
                if base + idx >= output.len() {
                    return;
                }
                let byte_idx = idx / 4;
                let bit_off = (idx % 4) * 2;
                let q = ((qs[byte_idx] >> bit_off) & 0x03) as f32;
                output[base + idx] = d * sc * q - dmin * m;
            }
        }
    }
}

// --- Helpers ---

fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    half::f16::from_bits(bits).to_f32()
}
