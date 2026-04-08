// Optimized GEMV kernels for quantized weights × Q8_1 input.
//
// Based on llama.cpp's approach:
// 1. Quantize input f16 vector to Q8_1 (half2 ds + int8 qs[32] = 36 bytes/block)
// 2. Use __dp4a for int8 dot products (4 MADs per instruction)
//
// Q8_1 block layout (36 bytes per 32 elements):
//   half2 ds: ds.x = d (scale), ds.y = sum (sum of original values)
//   int8_t qs[32]: quantized values
//
// Grid: (N,), Block: (256,)

#include <cuda_fp16.h>

typedef unsigned char  uint8_t;
typedef signed char    int8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;

// Q8_1 block size in bytes: half2(4) + qs[32] = 36
#define Q8_1_BYTES 36

// Load int32 from 2-byte-aligned data (Q6_K blocks are 210 bytes, not 4-byte aligned).
// Matches llama.cpp's get_int_b2.
static __device__ __forceinline__ int get_int_b2(const void* x, const int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = x16[2 * i32 + 0] << 0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
}

// Load int32 from 4-byte-aligned data. Matches llama.cpp's get_int_b4.
static __device__ __forceinline__ int get_int_b4(const void* x, const int i32) {
    return ((const int*)x)[i32];
}

extern "C" {

// ============================================================================
// Quantize f16 vector to Q8_1
// ============================================================================
// Grid: ((K+255)/256,), Block: (256,)
// Each warp (32 threads) produces one Q8_1 block (36 bytes)
__global__ void quantize_x_q8_1(const __half* __restrict__ x,
                                 void* __restrict__ x_q8,
                                 int K) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= K) return;

    const int block_idx = idx / 32;
    const int in_block  = idx % 32;

    // Read f16 value and convert to float
    const float val = __half2float(x[idx]);

    // Warp-level reduce: max absolute value for scale
    float amax = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
    }

    // Warp-level reduce: sum for dmin correction
    float warp_sum = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_xor_sync(0xFFFFFFFF, warp_sum, offset);
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const int8_t q = (int8_t)__float2int_rn(val * id);

    // Write Q8_1 block: half2 ds (4 bytes) + int8_t qs[32]
    uint8_t* block_ptr = (uint8_t*)x_q8 + block_idx * Q8_1_BYTES;
    if (in_block == 0) {
        // Pack d and sum as half2
        *(half2*)block_ptr = make_half2(__float2half(d), __float2half(warp_sum));
    }
    ((int8_t*)(block_ptr + 4))[in_block] = q;
}

// ============================================================================
// Q4_K × Q8_1 GEMV using dp4a
// ============================================================================
// Q4_K super-block: 256 elements, 144 bytes
//   [0..1]   half d
//   [2..3]   half dmin
//   [4..15]  uint8_t scales[12] (packed 6-bit scales and mins)
//   [16..143] uint8_t qs[128]   (4-bit quants, 2 nibbles per byte)
//
// Each Q4_K super-block maps to 8 Q8_1 blocks (256/32 = 8).
// We process in groups of 32 elements (sub-blocks). Each sub-block has its
// own 6-bit scale and min.
//
// Thread mapping: 256 threads, 8 warps.
// Each thread processes VDR=2 sub-block pairs per super-block iteration.
// The llama.cpp approach: each thread loads 2 int32 from qs (= 8 bytes = 16 nibbles),
// masks nibbles, and uses dp4a against Q8_1 values.
//
// Grid: (N,), Block: (256,)
__global__ void gemv_q4_k(const void* __restrict__ W,
                           const __half* __restrict__ x_unused,
                           __half* __restrict__ out,
                           int N, int K,
                           const void* __restrict__ x_q8) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;  // [0, 255]

    const int blocks_per_row = K / 256;
    const uint8_t* row_data = (const uint8_t*)W + (long long)row * blocks_per_row * 144;

    // Each thread handles a slice of the super-block.
    // Following llama.cpp: qi = 32 (ints per Q4_K block when cast to int*), vdr = 2
    // iqs = vdr * (tid % (qi/vdr)) = 2 * (tid % 16) -> 0,2,4,...,30
    // bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2)) = 2 * ((iqs/2) / 4) = 2 * (tid%16 / 4)
    // Each thread iterates: kbx = tid/(qi/vdr) = tid/16, kbx += blocks_per_iter
    // blocks_per_iter = vdr * nwarps * 32 / qi = 2 * 8 * 32 / 32 = 16
    // But we only have blocks_per_row super-blocks, so we iterate in steps of 16.

    // Simplified: each of the 256 threads picks a position within each super-block.
    // With qi=32, vdr=2: thread tid processes iqs = 2*(tid%16), starting at super-block tid/16.
    // That gives 16 starting super-blocks, stepping by 16.

    const int iqs = 2 * (tid % 16);     // position within super-block (0,2,4,...,30)
    const int bq8_offset = 2 * (iqs / 8);  // which pair of Q8_1 blocks (0,2,4,6)

    float sumf = 0.0f;

    for (int sb = tid / 16; sb < blocks_per_row; sb += 16) {
        const uint8_t* block = row_data + sb * 144;

        // Load d, dmin as half2
        const half2 dm = *(const half2*)block;
        const float dm_x = __half2float(__low2half(dm));   // d
        const float dm_y = __half2float(__high2half(dm));  // dmin

        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        // Load Q4_K quant data: 2 int32 values from qs
        // q4 pointer: qs + 16*bq8_offset + 4*((iqs/2)%4)
        const int* q4 = (const int*)(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        int v0 = q4[0];   // 4 bytes = 8 nibbles (low half)
        int v1 = q4[4];   // next 4 bytes (high half, 16 bytes apart)

        // Decode scales and mins (6-bit packed)
        const uint16_t* sc16 = (const uint16_t*)scales;
        uint16_t aux[2];
        const int j = bq8_offset / 2;
        if (j < 2) {
            aux[0] = sc16[j + 0] & 0x3f3f;
            aux[1] = sc16[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((sc16[j + 2] >> 0) & 0x0f0f) | ((sc16[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((sc16[j + 2] >> 4) & 0x0f0f) | ((sc16[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t* sc = (const uint8_t*)aux;
        const uint8_t* m  = sc + 2;

        // Load Q8_1 data for the two iterations (QR4_K = 2)
        int u[4];     // Q8_1 quant values as int32 (4 int8 per int32)
        float d8[2];  // Q8_1 scales

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint8_t* bq8 = (const uint8_t*)x_q8 + (sb * 8 + bq8_offset + i) * Q8_1_BYTES;
            d8[i] = __half2float(__low2half(*(const half2*)bq8));

            const int* q8 = (const int*)(bq8 + 4) + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }

        // vec_dot_q4_K_q8_1_impl_vmmq
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {  // QR4_K = 2
            const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;

            // dp4a: 4 int8 multiplies + accumulate
            const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
            const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));

            sumf_d += d8[i] * (dot1 * sc[i]);
            sumf_m += d8[i] * (dot2 * m[i]);
        }

        sumf += dm_x * sumf_d - dm_y * sumf_m;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[8];
    if (tid % 32 == 0) warp_sums[tid / 32] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) total += warp_sums[i];
        out[row] = __float2half(total);
    }
}

// ============================================================================
// Q6_K × Q8_1 GEMV using dp4a
// ============================================================================
// Q6_K super-block: 256 elements, 210 bytes
//   [0..127]   uint8_t ql[128]   (low 4 bits)
//   [128..191] uint8_t qh[64]    (high 2 bits)
//   [192..207] int8_t scales[16]
//   [208..209] half d
//
// IMPORTANT: Q6_K blocks are 210 bytes = NOT 4-byte aligned!
// Must use get_int_b2 (2-byte loads) for weight data, not int* casts.
//
// Following llama.cpp's vec_dot_q6_K_q8_1 + vec_dot_q6_K_q8_1_impl_mmvq:
// qi = 32, vdr = 1, QR6_K = 2
// Grid: (N,), Block: (256,)
__global__ void gemv_q6_k(const void* __restrict__ W,
                           const __half* __restrict__ x_unused,
                           __half* __restrict__ out,
                           int N, int K,
                           const void* __restrict__ x_q8) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const uint8_t* row_data = (const uint8_t*)W + (long long)row * blocks_per_row * 210;

    // qi = 32, vdr = 1
    // blocks_per_iter = 1 * 8 * 32 / 32 = 8
    const int iqs = tid % 32;

    float sumf = 0.0f;

    for (int sb = tid / 32; sb < blocks_per_row; sb += 8) {
        const uint8_t* block = row_data + sb * 210;
        const uint8_t* ql = block;         // 128 bytes
        const uint8_t* qh = block + 128;   // 64 bytes
        const int8_t*  scales = (const int8_t*)(block + 192);  // 16 bytes
        // d is at block + 208, 2-byte aligned (half)
        const float d = __half2float(*(const __half*)(block + 208));

        // Compute offsets matching llama.cpp
        // QI6_K = 32 (256 / (4*2))
        // bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4)
        //            = 2 * 2 * (iqs / 16) + (iqs % 16) / 8
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        // scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8)
        //              = 8 * (iqs / 16) + (iqs % 16) / 4
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        // vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4))
        //          = 2 * ((iqs % 16) / 8)
        const int vh_shift = 2 * ((iqs % 16) / 8);

        // Use get_int_b2 for 2-byte-aligned weight loads (Q6_K blocks are 210 bytes!)
        const int vl = get_int_b2(ql, iqs);
        // QI6_K/4 = 8
        const int vh = get_int_b2(qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;

        const int8_t* sc = scales + scale_offset;

        // Load Q8_1 data for QR6_K=2 iterations
        int u[2];
        float d8[2];

        #pragma unroll
        for (int i = 0; i < 2; ++i) {  // QR6_K = 2
            const uint8_t* bq8 = (const uint8_t*)x_q8 + (sb * 8 + bq8_offset + 2 * i) * Q8_1_BYTES;
            d8[i] = __half2float(__low2half(*(const half2*)bq8));
            // Q8_1 blocks are 36 bytes (4-byte aligned), safe to use get_int_b4
            u[i] = get_int_b4(bq8 + 4, iqs % 8);
        }

        // vec_dot_q6_K_q8_1_impl_mmvq
        float local_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {  // QR6_K = 2
            const int sc_val = sc[4 * i];
            const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;

            // Combine low 4 bits and high 2 bits, subtract 32
            const int vi = __vsubss4((vil | vih), 0x20202020);

            local_sum += d8[i] * (__dp4a(vi, u[i], 0) * sc_val);
        }

        sumf += d * local_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);
    }

    __shared__ float warp_sums[8];
    if (tid % 32 == 0) warp_sums[tid / 32] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) total += warp_sums[i];
        out[row] = __float2half(total);
    }
}

// ============================================================================
// Q8_0 × Q8_1 GEMV using dp4a
// ============================================================================
// Q8_0 block: 34 bytes per 32 elements
//   half d (2 bytes) + int8_t qs[32]
//
// Both sides are int8, so dp4a is perfect: process 4 elements per dp4a call.
// Each thread loads 1 int32 from Q8_0 qs (= 4 int8 values) and 1 int32 from Q8_1 qs.
// VDR_Q8_0 = 2: each thread processes 2 int32 pairs per Q8_0 block.
//
// Grid: (N,), Block: (256,)
__global__ void gemv_q8_0(const void* __restrict__ W,
                           const __half* __restrict__ x_unused,
                           __half* __restrict__ out,
                           int N, int K,
                           const void* __restrict__ x_q8) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 32;
    const uint8_t* row_data = (const uint8_t*)W + (long long)row * blocks_per_row * 34;

    // qi = QI8_0 = 8, vdr = 2
    // iqs = 2 * (tid % 4) -> 0, 2, 4, 6
    // blocks_per_iter = 2 * 8 * 32 / 8 = 64
    // Each thread starts at block tid/4, steps by 64.
    const int iqs = 2 * (tid % 4);

    float sumf = 0.0f;

    for (int blk = tid / 4; blk < blocks_per_row; blk += 64) {
        const uint8_t* w_block = row_data + blk * 34;
        const __half w_d = *(const __half*)w_block;

        // Load 2 int32 from Q8_0 qs
        const int* w_qs = (const int*)(w_block + 2);
        int v0 = w_qs[iqs + 0];
        int v1 = w_qs[iqs + 1];

        // Load corresponding Q8_1 data
        const uint8_t* bq8 = (const uint8_t*)x_q8 + blk * Q8_1_BYTES;
        const __half x_d = __low2half(*(const half2*)bq8);
        const int* x_qs = (const int*)(bq8 + 4);
        int u0 = x_qs[iqs + 0];
        int u1 = x_qs[iqs + 1];

        int sumi = __dp4a(v0, u0, 0);
        sumi = __dp4a(v1, u1, sumi);

        sumf += __half2float(w_d) * __half2float(x_d) * (float)sumi;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);
    }

    __shared__ float warp_sums[8];
    if (tid % 32 == 0) warp_sums[tid / 32] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) total += warp_sums[i];
        out[row] = __float2half(total);
    }
}

} // extern "C"
