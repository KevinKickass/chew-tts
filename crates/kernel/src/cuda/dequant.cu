// Dequantization kernels: quantized blocks → f16
//
// All dequant kernels output __half* (f16) for VRAM efficiency.
// Intermediate computations within each kernel use f32 for accuracy,
// then convert to f16 on output.

#include <cuda_fp16.h>

// NVRTC has no system headers — define integer types directly
typedef unsigned char      uint8_t;
typedef signed char        int8_t;
typedef unsigned short     uint16_t;
typedef short              int16_t;
typedef unsigned int       uint32_t;
typedef int                int32_t;

extern "C" {

// --- Q8_0: 32 weights per block, f16 scale + 32 * i8 ---
__global__ void dequant_q8_0(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 32;
    int in_block  = idx % 32;

    const uint8_t* block = (const uint8_t*)src + block_idx * 34;
    float d = __half2float(*(const __half*)block);
    int8_t q = ((const int8_t*)(block + 2))[in_block];

    dst[idx] = __float2half(d * (float)q);
}

// --- Q4_0: 32 weights per block, f16 scale + 16 bytes (4-bit quants) ---
__global__ void dequant_q4_0(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 32;
    int in_block  = idx % 32;

    const uint8_t* block = (const uint8_t*)src + block_idx * 18;
    float d = __half2float(*(const __half*)block);

    uint8_t byte = block[2 + in_block / 2];
    int q;
    if (in_block % 2 == 0) {
        q = (byte & 0x0F) - 8;
    } else {
        q = ((byte >> 4) & 0x0F) - 8;
    }

    dst[idx] = __float2half(d * (float)q);
}

// --- Q4_K: 256 weights per super-block ---
__global__ void dequant_q4_k(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;
    int sub       = in_block / 32;

    const uint8_t* block = (const uint8_t*)src + block_idx * 144;

    float d    = __half2float(*(const __half*)block);
    float dmin = __half2float(*(const __half*)(block + 2));
    const uint8_t* scales = block + 4;
    const uint8_t* qs     = block + 16;

    uint8_t sc, m;
    if (sub < 4) {
        sc = scales[sub] & 0x3F;
        m  = scales[sub + 4] & 0x3F;
    } else {
        int off = sub - 4;
        sc = (scales[off + 8] & 0x0F) | ((scales[off] >> 6) << 4);
        m  = (scales[off + 8] >> 4)    | ((scales[off + 4] >> 6) << 4);
    }

    // Q4_K nibble layout: pairs of sub-blocks share 32 bytes.
    // Even sub-block → low nibble, odd sub-block → high nibble.
    int in_sub = in_block % 32;
    int pair = sub / 2;           // 0,1,2,3
    int byte_idx = pair * 32 + in_sub;
    uint8_t q_nibble;
    if (sub % 2 == 0) {
        q_nibble = qs[byte_idx] & 0x0F;
    } else {
        q_nibble = (qs[byte_idx] >> 4) & 0x0F;
    }

    float val = d * (float)sc * (float)q_nibble - dmin * (float)m;
    dst[idx] = __float2half(val);
}

// --- Q5_K: 256 weights per super-block ---
// Layout: d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128] = 176 bytes
// Same scale/min encoding as Q4_K, but 5 bits per weight:
// Low 4 bits from qs (same nibble layout as Q4_K), high bit from qh.
__global__ void dequant_q5_k(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;
    int sub       = in_block / 32;

    const uint8_t* block = (const uint8_t*)src + block_idx * 176;

    float d    = __half2float(*(const __half*)block);
    float dmin = __half2float(*(const __half*)(block + 2));
    const uint8_t* scales = block + 4;    // 12 bytes
    const uint8_t* qh     = block + 16;   // 32 bytes (high bits)
    const uint8_t* qs     = block + 48;   // 128 bytes (low nibbles)

    // Scale/min decoding: same as Q4_K
    uint8_t sc, m;
    if (sub < 4) {
        sc = scales[sub] & 0x3F;
        m  = scales[sub + 4] & 0x3F;
    } else {
        int off = sub - 4;
        sc = (scales[off + 8] & 0x0F) | ((scales[off] >> 6) << 4);
        m  = (scales[off + 8] >> 4)    | ((scales[off + 4] >> 6) << 4);
    }

    // Low 4 bits: same nibble layout as Q4_K
    int in_sub = in_block % 32;
    int pair = sub / 2;
    int byte_idx = pair * 32 + in_sub;
    uint8_t q_lo;
    if (sub % 2 == 0) {
        q_lo = qs[byte_idx] & 0x0F;
    } else {
        q_lo = (qs[byte_idx] >> 4) & 0x0F;
    }

    // High bit: qh[in_sub] bit (sub)
    // GGML layout: for 64*j+offset+l, high bit = qh[l] >> (2*j + offset/32)
    // Simplified: qh[in_sub] >> sub
    uint8_t q_hi = (qh[in_sub] >> sub) & 1;

    int q = q_lo | (q_hi << 4);  // 5-bit quantized value

    float val = d * (float)sc * (float)q - dmin * (float)m;
    dst[idx] = __float2half(val);
}

// --- Q6_K: 256 weights per super-block ---
// Layout: ql[128] | qh[64] | scales[16] | d[2] = 210 bytes
__global__ void dequant_q6_k(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    const uint8_t* block = (const uint8_t*)src + block_idx * 210;

    const uint8_t* ql = block;             // offset 0, 128 bytes
    const uint8_t* qh = block + 128;       // offset 128, 64 bytes
    const int8_t* sc = (const int8_t*)(block + 192);  // offset 192, 16 bytes
    float d = __half2float(*(const __half*)(block + 208)); // offset 208, 2 bytes

    // Q6_K layout (matching GGML dequantize_row_q6_K):
    // Two 128-element halves. Each half processes 64 ql bytes + 32 qh bytes.
    // Within each half (128 elements at positions base..base+127):
    //   positions [base+ 0..base+31]: ql[0..31]  low nibble + qh[0..31] bits 0-1
    //   positions [base+32..base+63]: ql[32..63] low nibble + qh[0..31] bits 2-3
    //   positions [base+64..base+95]: ql[0..31]  high nibble + qh[0..31] bits 4-5
    //   positions [base+96..base+127]:ql[32..63] high nibble + qh[0..31] bits 6-7

    int half    = in_block / 128;          // 0 or 1
    int in_half = in_block % 128;          // 0..127
    int group   = in_half / 32;            // 0,1,2,3 within the half
    int l       = in_half % 32;            // position within the group

    int ql_base = half * 64;               // 0 for first half, 64 for second
    int qh_base = half * 32;               // 0 for first half, 32 for second

    uint8_t ql_val;
    uint8_t qh_val;
    switch (group) {
        case 0: // low nibble of ql[0..31]
            ql_val = ql[ql_base + l] & 0x0F;
            qh_val = (qh[qh_base + l] >> 0) & 0x03;
            break;
        case 1: // low nibble of ql[32..63]
            ql_val = ql[ql_base + 32 + l] & 0x0F;
            qh_val = (qh[qh_base + l] >> 2) & 0x03;
            break;
        case 2: // high nibble of ql[0..31]
            ql_val = (ql[ql_base + l] >> 4) & 0x0F;
            qh_val = (qh[qh_base + l] >> 4) & 0x03;
            break;
        case 3: // high nibble of ql[32..63]
            ql_val = (ql[ql_base + 32 + l] >> 4) & 0x0F;
            qh_val = (qh[qh_base + l] >> 6) & 0x03;
            break;
    }

    int q = (int)(ql_val | (qh_val << 4)) - 32;
    int8_t scale = sc[in_block / 16];

    dst[idx] = __float2half(d * (float)scale * (float)q);
}

// --- Q2_K: 256 weights per super-block ---
// Layout: scales[16] | qs[64] | d[2] | dmin[2] = 84 bytes
__global__ void dequant_q2_k(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;
    int sub       = in_block / 16;

    const uint8_t* block = (const uint8_t*)src + block_idx * 84;

    // Layout: scales at 0, qs at 16, d at 80, dmin at 82
    const uint8_t* scales = block;                         // offset 0, 16 bytes
    const uint8_t* qs     = block + 16;                    // offset 16, 64 bytes
    float d    = __half2float(*(const __half*)(block + 80));  // offset 80, 2 bytes
    float dmin = __half2float(*(const __half*)(block + 82));  // offset 82, 2 bytes

    float sc = (float)(scales[sub] & 0x0F);
    float m  = (float)(scales[sub] >> 4);

    int byte_idx = in_block / 4;
    int bit_off  = (in_block % 4) * 2;
    float q = (float)((qs[byte_idx] >> bit_off) & 0x03);

    dst[idx] = __float2half(d * sc * q - dmin * m);
}

// --- Q3_K: 256 weights per super-block, 3-bit + scales ---
// Layout: f16 d, 32 hmask, 64 qs (256*3bit packed), 12 scales
__global__ void dequant_q3_k(const void* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    const uint8_t* block = (const uint8_t*)src + block_idx * 110;
    // Layout: hmask[32], qs[64], scales[12], d[2]
    const uint8_t* hmask  = block;
    const uint8_t* qs     = block + 32;
    const uint8_t* scales_raw = block + 96;
    float d = __half2float(*(const __half*)(block + 108));

    int sub = in_block / 16;

    // Reconstruct 6-bit scale
    int8_t scale;
    if (sub < 8) {
        uint8_t raw = scales_raw[sub];
        scale = (int8_t)(raw & 0x0F) - 8;
        if (sub >= 4) {
            raw = scales_raw[sub - 4];
            scale = (int8_t)((raw >> 4) & 0x0F) - 8;
        }
    } else {
        // sub 8..15 use the 4 extra scale bytes
        int si = sub - 8;
        uint8_t raw = scales_raw[8 + si / 2];
        if (si % 2 == 0) {
            scale = (int8_t)(raw & 0x0F) - 8;
        } else {
            scale = (int8_t)((raw >> 4) & 0x0F) - 8;
        }
    }

    // Get 2-bit value from qs
    int byte_pos = in_block / 4;
    int bit_shift = (in_block % 4) * 2;
    int q2 = (qs[byte_pos] >> bit_shift) & 0x03;

    // Get high bit from hmask
    int hmask_byte = in_block / 8;
    int hmask_bit  = in_block % 8;
    int hb = (hmask[hmask_byte] >> hmask_bit) & 1;

    int q = q2 | (hb << 2);  // 3-bit value [0..7]
    q -= 4;                   // center: [-4..3]

    dst[idx] = __float2half(d * (float)scale * (float)q);
}

// --- BF16 → f16 ---
__global__ void dequant_bf16(const uint16_t* __restrict__ src,
                              __half* __restrict__ dst,
                              int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    // BF16: sign(1) + exp(8) + mantissa(7)
    // F32:  sign(1) + exp(8) + mantissa(23) — just shift left by 16
    uint16_t bf = src[idx];
    uint32_t f32_bits = (uint32_t)bf << 16;
    float val;
    memcpy(&val, &f32_bits, sizeof(float));
    dst[idx] = __float2half(val);
}

// --- IQ2_S: 256 weights per super-block ---
// Approximate dequant for IQ2_S (2.5 bpw with signs)
// Layout: f16 d, qs[64], qh[8], scales[32] — total ~106 bytes
__global__ void dequant_iq2_s(const void* __restrict__ src,
                               __half* __restrict__ dst,
                               int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    // IQ2_S block: 2 (d) + 64 (qs) + 8 (qh) + 32 (scales) = 106 bytes
    const uint8_t* block = (const uint8_t*)src + block_idx * 106;
    float d = __half2float(*(const __half*)block);
    const uint8_t* qs     = block + 2;
    const uint8_t* qh     = block + 66;
    const uint8_t* scales = block + 74;

    int sub = in_block / 8;  // 32 sub-groups of 8

    // Scale from scales table
    float sc = (float)(scales[sub] & 0x0F) + 1.0f;
    int sign_bits = scales[sub] >> 4;

    // Get 2-bit quantized value
    int byte_idx = in_block / 4;
    int bit_off = (in_block % 4) * 2;
    int q = (qs[byte_idx] >> bit_off) & 0x03;

    // High bit from qh
    int qh_byte = in_block / 8;
    int qh_bit = in_block % 8;
    int h = (qh[qh_byte] >> qh_bit) & 1;

    // Apply sign
    int in_sub = in_block % 8;
    int sign = (sign_bits >> (in_sub % 4)) & 1;

    float val = d * sc * ((float)(q + h * 4) - 3.0f);
    if (sign) val = -val;

    dst[idx] = __float2half(val);
}

// --- IQ3_XXS: 256 weights per block ---
// Layout: f16 d, qs[96] (3bit*256/8=96), signs[16] (256/16=16)
__global__ void dequant_iq3_xxs(const void* __restrict__ src,
                                 __half* __restrict__ dst,
                                 int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    // IQ3_XXS: 2 (d) + 96 (qs) + 16 (extra) = 114 bytes
    const uint8_t* block = (const uint8_t*)src + block_idx * 114;
    float d = __half2float(*(const __half*)block);
    const uint8_t* qs = block + 2;
    const uint8_t* extra = block + 98;

    // 3 bits per weight: bit position = in_block * 3
    int bit_pos = in_block * 3;
    int byte_idx = bit_pos / 8;
    int bit_off = bit_pos % 8;

    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 96) raw |= (uint16_t)qs[byte_idx + 1] << 8;

    int q = (raw >> bit_off) & 0x07;  // 3-bit [0..7]

    // Sign from extra bytes
    int sign_byte = in_block / 8;
    int sign_bit = in_block % 8;
    int sign = (extra[sign_byte] >> sign_bit) & 1;

    float val = d * ((float)q - 3.5f);
    if (sign) val = -val;

    dst[idx] = __float2half(val);
}

// --- IQ3_S: 256 weights per block ---
// Layout: f16 d, qs[96], qh[32], scales[8] — total ~138 bytes
__global__ void dequant_iq3_s(const void* __restrict__ src,
                               __half* __restrict__ dst,
                               int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    // IQ3_S: 2 (d) + 96 (qs) + 32 (qh) + 8 (scales) = 138 bytes
    const uint8_t* block = (const uint8_t*)src + block_idx * 138;
    float d = __half2float(*(const __half*)block);
    const uint8_t* qs = block + 2;
    const uint8_t* qh = block + 98;
    const uint8_t* signs = block + 130;

    int sub = in_block / 32;

    // 3-bit value
    int bit_pos = in_block * 3;
    int byte_idx = bit_pos / 8;
    int bit_off = bit_pos % 8;

    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 96) raw |= (uint16_t)qs[byte_idx + 1] << 8;

    int q = (raw >> bit_off) & 0x07;

    // High bit from qh
    int qh_byte = in_block / 8;
    int qh_bit = in_block % 8;
    int h = (qh[qh_byte] >> qh_bit) & 1;
    if (h) q += 8;

    // Sign
    int sign_byte = in_block / 8;
    int sign_bit = in_block % 8;
    int sign = (signs[sign_byte] >> sign_bit) & 1;

    // Scale from sub-block
    float sc = (float)((signs[sub] >> 4) & 0x0F) + 1.0f;

    float val = d * sc * ((float)q - 7.5f);
    if (sign) val = -val;

    dst[idx] = __float2half(val);
}

// --- IQ4_XS: 256 weights per block ---
// Layout: f16 d, u16 scales_h, qs[128], scales_l[16] — total ~148 bytes
__global__ void dequant_iq4_xs(const void* __restrict__ src,
                                __half* __restrict__ dst,
                                int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    // IQ4_XS: 2 (d) + 2 (scales_h) + 128 (qs) + 16 (scales_l) = 148 bytes
    const uint8_t* block = (const uint8_t*)src + block_idx * 148;
    float d = __half2float(*(const __half*)block);
    uint16_t scales_h = *(const uint16_t*)(block + 2);
    const uint8_t* qs = block + 4;
    const uint8_t* scales_l = block + 132;

    int sub = in_block / 32;

    // Reconstruct 6-bit scale
    int sc_lo = (scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0F;
    int sc_hi = (scales_h >> sub) & 1;
    float sc = (float)((sc_hi << 4) | sc_lo) + 1.0f;

    // 4-bit quantized value
    int in_sub = in_block % 32;
    int byte_idx = sub * 16 + in_sub / 2;
    int q;
    if (in_sub % 2 == 0) {
        q = qs[byte_idx] & 0x0F;
    } else {
        q = (qs[byte_idx] >> 4) & 0x0F;
    }

    float val = d * sc * ((float)q - 8.0f);
    dst[idx] = __float2half(val);
}

// --- F16 passthrough ---
// --- TQ2_0: 256 weights per block, ternary quantization ---
// Layout: d(f16) + qs[64] = 66 bytes per 256 elements
// Each byte holds 4 x 2-bit values. Value = (q - 1) * d → {-d, 0, d, 2d}
// Element ordering: 2 groups of 128, each group has 4 passes of 32 bytes
__global__ void dequant_tq2_0(const void* __restrict__ src,
                               __half* __restrict__ dst,
                               int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int block_idx = idx / 256;
    int in_block  = idx % 256;

    const uint8_t* block = (const uint8_t*)src + block_idx * 66;
    float d = __half2float(*(const __half*)block);
    const uint8_t* qs = block + 2;

    // Element mapping within the block:
    // The dequant iterates: j in [0, 32, 64...] stepping by 32, l in [0..3], m in [0..31]
    // Output index = (j/32) * 128 + l * 32 + m
    // So for in_block: group = in_block / 128, subgroup = (in_block % 128) / 32, m = in_block % 32
    // byte index = group * 32 + m, shift = subgroup * 2
    int group = in_block / 128;      // 0 or 1
    int subgroup = (in_block % 128) / 32;  // 0..3
    int m = in_block % 32;

    int byte_idx = group * 32 + m;
    int shift = subgroup * 2;
    int q = (qs[byte_idx] >> shift) & 3;

    float val = (float)(q - 1) * d;
    dst[idx] = __float2half(val);
}

__global__ void dequant_f16(const __half* __restrict__ src,
                             __half* __restrict__ dst,
                             int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    dst[idx] = src[idx];
}

// --- F32 → f16 ---
__global__ void dequant_f32(const float* __restrict__ src,
                             __half* __restrict__ dst,
                             int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    dst[idx] = __float2half(src[idx]);
}

} // extern "C"
