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
// Q4_K × Q8_1 GEMV using dp4a — 4 warps (128 threads), 1 row per block
// ============================================================================
// Reduced from 8 warps to 4: lower register pressure → better occupancy.
// Grid: (N,), Block: (128,)
#define NWARPS_Q4K 4
__launch_bounds__(NWARPS_Q4K * 32, 1)
__global__ void gemv_q4_k(const void* __restrict__ W,
                           const __half* __restrict__ x_unused,
                           __half* __restrict__ out,
                           int N, int K,
                           const void* __restrict__ x_q8) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;  // [0, 127]

    const int blocks_per_row = K / 256;
    const uint8_t* row_data = (const uint8_t*)W + (long long)row * blocks_per_row * 144;

    const int iqs = 2 * (tid % 16);
    const int bq8_offset = 2 * (iqs / 8);

    // 4 warps, 128 threads: blocks_per_iter = 2 * 4 * 32 / 32 = 8
    float sumf = 0.0f;

    for (int sb = tid / 16; sb < blocks_per_row; sb += 8) {
        const uint8_t* block = row_data + sb * 144;

        const half2 dm = *(const half2*)block;
        const float dm_x = __half2float(__low2half(dm));
        const float dm_y = __half2float(__high2half(dm));

        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        const int* q4 = (const int*)(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        int v0 = q4[0];
        int v1 = q4[4];

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

        int u[4];
        float d8[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint8_t* bq8 = (const uint8_t*)x_q8 + (sb * 8 + bq8_offset + i) * Q8_1_BYTES;
            d8[i] = __half2float(__low2half(*(const half2*)bq8));
            const int* q8 = (const int*)(bq8 + 4) + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }

        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
            const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
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

    // Cross-warp reduction (only 4 warps)
    __shared__ float warp_sums[NWARPS_Q4K];
    if (tid % 32 == 0) warp_sums[tid / 32] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        out[row] = __float2half(total);
    }
}

// ============================================================================
// Q6_K × Q8_1 GEMV using dp4a — 4 warps (128 threads), 1 row per block
// ============================================================================
// Grid: (N,), Block: (128,)
#define NWARPS_Q6K 4
__launch_bounds__(NWARPS_Q6K * 32, 2)
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
    const int iqs = tid % 32;

    float sumf = 0.0f;

    for (int sb = tid / 32; sb < blocks_per_row; sb += NWARPS_Q6K) {
        const uint8_t* block = row_data + sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t*  scales = (const int8_t*)(block + 192);
        const float d = __half2float(*(const __half*)(block + 208));

        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);

        const int vl = get_int_b2(ql, iqs);
        const int vh = get_int_b2(qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        const int8_t* sc = scales + scale_offset;

        int u[2];
        float d8[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint8_t* bq8 = (const uint8_t*)x_q8 + (sb * 8 + bq8_offset + 2 * i) * Q8_1_BYTES;
            d8[i] = __half2float(__low2half(*(const half2*)bq8));
            u[i] = get_int_b4(bq8 + 4, iqs % 8);
        }

        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int sc_val = sc[4 * i];
            const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = __vsubss4((vil | vih), 0x20202020);
            local_sum += d8[i] * (__dp4a(vi, u[i], 0) * sc_val);
        }
        sumf += d * local_sum;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);
    }

    __shared__ float warp_sums[NWARPS_Q6K];
    if (tid % 32 == 0) warp_sums[tid / 32] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        out[row] = __float2half(total);
    }
}

// ============================================================================
// Q8_0 × Q8_1 GEMV using dp4a — 4 warps (128 threads), 1 row per block
// ============================================================================
// Grid: (N,), Block: (128,)
#define NWARPS_Q80 4
__launch_bounds__(NWARPS_Q80 * 32, 2)
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

    const int iqs = 2 * (tid % 4);
    float sumf = 0.0f;

    for (int blk = tid / 4; blk < blocks_per_row; blk += 32) {
        const uint8_t* w_block = row_data + blk * 34;
        const __half w_d = *(const __half*)w_block;
        const int* w_qs = (const int*)(w_block + 2);
        int v0 = w_qs[iqs + 0];
        int v1 = w_qs[iqs + 1];

        const uint8_t* bq8 = (const uint8_t*)x_q8 + blk * Q8_1_BYTES;
        const __half x_d = __low2half(*(const half2*)bq8);
        const int* x_qs = (const int*)(bq8 + 4);
        int u0 = x_qs[iqs + 0];
        int u1 = x_qs[iqs + 1];

        int sumi = __dp4a(v0, u0, 0);
        sumi = __dp4a(v1, u1, sumi);
        sumf += __half2float(w_d) * __half2float(x_d) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);
    }

    __shared__ float warp_sums[NWARPS_Q80];
    if (tid % 32 == 0) warp_sums[tid / 32] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        out[row] = __float2half(total);
    }
}

// ============================================================================
// Fused Gate+Up GEMV: computes BOTH gate[N,K] and up[N,K] in one kernel
// ============================================================================
// Saves 1 kernel launch and reads Q8_1 input only once for both matrices.
// Grid: (N,), Block: (128,)
// Writes: gate_out[N] and up_out[N]
#define NWARPS_DUAL 4
__launch_bounds__(NWARPS_DUAL * 32, 1)
__global__ void gemv_dual_q4_k(const void* __restrict__ W_gate,
                                const void* __restrict__ W_up,
                                __half* __restrict__ out_gate,
                                __half* __restrict__ out_up,
                                int N, int K,
                                const void* __restrict__ x_q8) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const uint8_t* gate_data = (const uint8_t*)W_gate + (long long)row * blocks_per_row * 144;
    const uint8_t* up_data   = (const uint8_t*)W_up   + (long long)row * blocks_per_row * 144;

    const int iqs = 2 * (tid % 16);
    const int bq8_offset = 2 * (iqs / 8);

    float sumf_gate = 0.0f;
    float sumf_up   = 0.0f;

    for (int sb = tid / 16; sb < blocks_per_row; sb += 8) {
        // Load Q8_1 data ONCE (shared between gate and up)
        int u[4];
        float d8[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint8_t* bq8 = (const uint8_t*)x_q8 + (sb * 8 + bq8_offset + i) * Q8_1_BYTES;
            d8[i] = __half2float(__low2half(*(const half2*)bq8));
            const int* q8 = (const int*)(bq8 + 4) + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }

        // Process GATE weight
        {
            const uint8_t* block = gate_data + sb * 144;
            const half2 dm = *(const half2*)block;
            const float dm_x = __half2float(__low2half(dm));
            const float dm_y = __half2float(__high2half(dm));
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;
            const int* q4 = (const int*)(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
            int v0 = q4[0]; int v1 = q4[4];
            const uint16_t* sc16 = (const uint16_t*)scales;
            uint16_t aux[2]; const int j = bq8_offset / 2;
            if (j < 2) { aux[0] = sc16[j+0]&0x3f3f; aux[1] = sc16[j+2]&0x3f3f; }
            else { aux[0]=((sc16[j+2]>>0)&0x0f0f)|((sc16[j-2]&0xc0c0)>>2); aux[1]=((sc16[j+2]>>4)&0x0f0f)|((sc16[j-0]&0xc0c0)>>2); }
            const uint8_t* sc = (const uint8_t*)aux; const uint8_t* m = sc + 2;
            float sd=0,sm=0;
            #pragma unroll
            for (int i=0;i<2;++i) {
                const int v0i=(v0>>(4*i))&0x0F0F0F0F; const int v1i=(v1>>(4*i))&0x0F0F0F0F;
                sd += d8[i]*(__dp4a(v1i,u[2*i+1],__dp4a(v0i,u[2*i+0],0))*sc[i]);
                sm += d8[i]*(__dp4a(0x01010101,u[2*i+1],__dp4a(0x01010101,u[2*i+0],0))*m[i]);
            }
            sumf_gate += dm_x*sd - dm_y*sm;
        }

        // Process UP weight (same Q8_1 data reused)
        {
            const uint8_t* block = up_data + sb * 144;
            const half2 dm = *(const half2*)block;
            const float dm_x = __half2float(__low2half(dm));
            const float dm_y = __half2float(__high2half(dm));
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;
            const int* q4 = (const int*)(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
            int v0 = q4[0]; int v1 = q4[4];
            const uint16_t* sc16 = (const uint16_t*)scales;
            uint16_t aux[2]; const int j = bq8_offset / 2;
            if (j < 2) { aux[0] = sc16[j+0]&0x3f3f; aux[1] = sc16[j+2]&0x3f3f; }
            else { aux[0]=((sc16[j+2]>>0)&0x0f0f)|((sc16[j-2]&0xc0c0)>>2); aux[1]=((sc16[j+2]>>4)&0x0f0f)|((sc16[j-0]&0xc0c0)>>2); }
            const uint8_t* sc = (const uint8_t*)aux; const uint8_t* m = sc + 2;
            float sd=0,sm=0;
            #pragma unroll
            for (int i=0;i<2;++i) {
                const int v0i=(v0>>(4*i))&0x0F0F0F0F; const int v1i=(v1>>(4*i))&0x0F0F0F0F;
                sd += d8[i]*(__dp4a(v1i,u[2*i+1],__dp4a(v0i,u[2*i+0],0))*sc[i]);
                sm += d8[i]*(__dp4a(0x01010101,u[2*i+1],__dp4a(0x01010101,u[2*i+0],0))*m[i]);
            }
            sumf_up += dm_x*sd - dm_y*sm;
        }
    }

    // Warp reduction for both
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sumf_gate += __shfl_down_sync(0xFFFFFFFF, sumf_gate, off);
        sumf_up   += __shfl_down_sync(0xFFFFFFFF, sumf_up, off);
    }

    __shared__ float ws_gate[NWARPS_DUAL];
    __shared__ float ws_up[NWARPS_DUAL];
    if (tid % 32 == 0) { ws_gate[tid/32] = sumf_gate; ws_up[tid/32] = sumf_up; }
    __syncthreads();

    if (tid == 0) {
        float g = ws_gate[0]+ws_gate[1]+ws_gate[2]+ws_gate[3];
        float u = ws_up[0]+ws_up[1]+ws_up[2]+ws_up[3];
        out_gate[row] = __float2half(g);
        out_up[row]   = __float2half(u);
    }
}

// gemv_qkv_q4_k removed — replaced by separate Q GEMV + K+V dual GEMV.
// The 3-matrix fused kernel had register pressure issues causing corrupted output.

} // extern "C"
