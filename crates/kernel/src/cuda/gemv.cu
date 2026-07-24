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
#include <cuda_bf16.h>

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
        int v0 = get_int_b2(w_block + 2, iqs + 0);
        int v1 = get_int_b2(w_block + 2, iqs + 1);

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

// ============================================================================
// Dense f16 GEMV for native Safetensors weights
// ============================================================================
// One 256-thread block computes one output row with f32 accumulation.
__global__ void gemv_f16(const __half* __restrict__ x,
                         const __half* __restrict__ weight,
                         __half* __restrict__ out,
                         int N,
                         int K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const __half* w = weight + (long long)row * K;
    float sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        sum += __half2float(x[col]) * __half2float(w[col]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) out[row] = __float2half(sum);
    }
}

__global__ void gemv_bf16(const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ weight,
                          __nv_bfloat16* __restrict__ out,
                          int N,
                          int K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const __nv_bfloat16* w = weight + (long long)row * K;
    float sum = 0.0f;
    const int pairs = K >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    if ((K & 1) == 0) {
        for (int col = tid; col < pairs; col += blockDim.x) {
            const float2 xv = __bfloat1622float2(x2[col]);
            const float2 wv = __bfloat1622float2(w2[col]);
            sum = fmaf(xv.x, wv.x, sum);
            sum = fmaf(xv.y, wv.y, sum);
        }
    } else {
        for (int col = tid; col < K; col += blockDim.x) {
            sum += __bfloat162float(x[col]) * __bfloat162float(w[col]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) out[row] = __float2bfloat16(sum);
    }
}

// Q/K/V share the same normalized input. The Q projection may have more rows
// than K/V under grouped-query attention, so the grid follows Q and only the
// common prefix computes all three projections.
__global__ void gemv_qkv_bf16(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ q_weight,
                              const __nv_bfloat16* __restrict__ k_weight,
                              const __nv_bfloat16* __restrict__ v_weight,
                              __nv_bfloat16* __restrict__ q_out,
                              __nv_bfloat16* __restrict__ k_out,
                              __nv_bfloat16* __restrict__ v_out,
                              int QN,
                              int KVN,
                              int K) {
    const int row = blockIdx.x;
    if (row >= QN) return;
    const int tid = threadIdx.x;
    const int pairs = K >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* qw2 = reinterpret_cast<const __nv_bfloat162*>(
        q_weight + (long long)row * K);
    const bool has_kv = row < KVN;
    const __nv_bfloat162* kw2 = has_kv
        ? reinterpret_cast<const __nv_bfloat162*>(k_weight + (long long)row * K)
        : nullptr;
    const __nv_bfloat162* vw2 = has_kv
        ? reinterpret_cast<const __nv_bfloat162*>(v_weight + (long long)row * K)
        : nullptr;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    if ((K & 1) == 0) {
        for (int col = tid; col < pairs; col += blockDim.x) {
            const float2 value = __bfloat1622float2(x2[col]);
            const float2 qv = __bfloat1622float2(qw2[col]);
            q_sum = fmaf(value.x, qv.x, q_sum);
            q_sum = fmaf(value.y, qv.y, q_sum);
            if (has_kv) {
                const float2 kv = __bfloat1622float2(kw2[col]);
                const float2 vv = __bfloat1622float2(vw2[col]);
                k_sum = fmaf(value.x, kv.x, k_sum);
                k_sum = fmaf(value.y, kv.y, k_sum);
                v_sum = fmaf(value.x, vv.x, v_sum);
                v_sum = fmaf(value.y, vv.y, v_sum);
            }
        }
    } else {
        for (int col = tid; col < K; col += blockDim.x) {
            const float value = __bfloat162float(x[col]);
            q_sum += value * __bfloat162float(q_weight[(long long)row * K + col]);
            if (has_kv) {
                k_sum += value * __bfloat162float(k_weight[(long long)row * K + col]);
                v_sum += value * __bfloat162float(v_weight[(long long)row * K + col]);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xFFFFFFFF, q_sum, offset);
        k_sum += __shfl_down_sync(0xFFFFFFFF, k_sum, offset);
        v_sum += __shfl_down_sync(0xFFFFFFFF, v_sum, offset);
    }
    __shared__ float q_warp_sums[8];
    __shared__ float k_warp_sums[8];
    __shared__ float v_warp_sums[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        q_warp_sums[warp] = q_sum;
        k_warp_sums[warp] = k_sum;
        v_warp_sums[warp] = v_sum;
    }
    __syncthreads();
    if (warp == 0) {
        q_sum = lane < 8 ? q_warp_sums[lane] : 0.0f;
        k_sum = lane < 8 ? k_warp_sums[lane] : 0.0f;
        v_sum = lane < 8 ? v_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum += __shfl_down_sync(0xFFFFFFFF, q_sum, offset);
            k_sum += __shfl_down_sync(0xFFFFFFFF, k_sum, offset);
            v_sum += __shfl_down_sync(0xFFFFFFFF, v_sum, offset);
        }
        if (lane == 0) {
            q_out[row] = __float2bfloat16(q_sum);
            if (has_kv) {
                k_out[row] = __float2bfloat16(k_sum);
                v_out[row] = __float2bfloat16(v_sum);
            }
        }
    }
}

// Gate and up projections share the input read and launch overhead.
__global__ void gemv_dual_f16(const __half* __restrict__ x,
                              const __half* __restrict__ gate_weight,
                              const __half* __restrict__ up_weight,
                              __half* __restrict__ gate_out,
                              __half* __restrict__ up_out,
                              int N,
                              int K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const __half* gate = gate_weight + (long long)row * K;
    const __half* up = up_weight + (long long)row * K;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        const float value = __half2float(x[col]);
        gate_sum += value * __half2float(gate[col]);
        up_sum += value * __half2float(up[col]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
        up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
    }
    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        gate_warp_sums[warp] = gate_sum;
        up_warp_sums[warp] = up_sum;
    }
    __syncthreads();
    if (warp == 0) {
        gate_sum = lane < 8 ? gate_warp_sums[lane] : 0.0f;
        up_sum = lane < 8 ? up_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
            up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
        }
        if (lane == 0) {
            gate_out[row] = __float2half(gate_sum);
            up_out[row] = __float2half(up_sum);
        }
    }
}

__global__ void gemv_dual_bf16(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ gate_weight,
                               const __nv_bfloat16* __restrict__ up_weight,
                               __nv_bfloat16* __restrict__ gate_out,
                               __nv_bfloat16* __restrict__ up_out,
                               int N,
                               int K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const __nv_bfloat16* gate = gate_weight + (long long)row * K;
    const __nv_bfloat16* up = up_weight + (long long)row * K;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const int pairs = K >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const __nv_bfloat162* up2 = reinterpret_cast<const __nv_bfloat162*>(up);
    if ((K & 1) == 0) {
        for (int col = tid; col < pairs; col += blockDim.x) {
            const float2 value = __bfloat1622float2(x2[col]);
            const float2 gate_value = __bfloat1622float2(gate2[col]);
            const float2 up_value = __bfloat1622float2(up2[col]);
            gate_sum = fmaf(value.x, gate_value.x, gate_sum);
            gate_sum = fmaf(value.y, gate_value.y, gate_sum);
            up_sum = fmaf(value.x, up_value.x, up_sum);
            up_sum = fmaf(value.y, up_value.y, up_sum);
        }
    } else {
        for (int col = tid; col < K; col += blockDim.x) {
            const float value = __bfloat162float(x[col]);
            gate_sum += value * __bfloat162float(gate[col]);
            up_sum += value * __bfloat162float(up[col]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
        up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
    }
    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        gate_warp_sums[warp] = gate_sum;
        up_warp_sums[warp] = up_sum;
    }
    __syncthreads();
    if (warp == 0) {
        gate_sum = lane < 8 ? gate_warp_sums[lane] : 0.0f;
        up_sum = lane < 8 ? up_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_down_sync(0xFFFFFFFF, gate_sum, offset);
            up_sum += __shfl_down_sync(0xFFFFFFFF, up_sum, offset);
        }
        if (lane == 0) {
            gate_out[row] = __float2bfloat16(gate_sum);
            up_out[row] = __float2bfloat16(up_sum);
        }
    }
}

} // extern "C"
