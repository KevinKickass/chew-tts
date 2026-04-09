// Transformer operation kernels
//
// Mixed precision: hidden state stays f32 (residual stream), intermediate
// buffers are f16 for VRAM efficiency. Bridge ops convert between them.
// KV cache is f16. Weight/embedding tables are f16.

#include <cuda_fp16.h>

typedef unsigned char  uint8_t;
typedef signed char    int8_t;

extern "C" {

// --- RMSNorm (f16 input, f16 output) ---
// For f16 → f16 normalization (general purpose).
// Weight is f16. Internal computation uses f32.
__global__ void rms_norm(const __half* __restrict__ x,
                          const __half* __restrict__ weight,
                          __half* __restrict__ out,
                          int dim,
                          float eps) {
    int row = blockIdx.x;
    const __half* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]) / rms;
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * w);
    }
}

// --- RMSNorm (f32 input, f16 output) ---
// Bridge: f32 hidden state → f16 for GEMM input.
__global__ void rms_norm_f32in(const float* __restrict__ x,
                                const __half* __restrict__ weight,
                                __half* __restrict__ out,
                                int dim,
                                float eps) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i] / rms;
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * w);
    }
}

// --- Element-wise Add: f16 + f16 -> f16 ---
__global__ void add_f16(const __half* __restrict__ a,
                         const __half* __restrict__ b,
                         __half* __restrict__ out,
                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

// --- Element-wise Add: f32 + f16 -> f32 (Residual Connection) ---
__global__ void add_f32_f16(const float* __restrict__ a,
                             const __half* __restrict__ b,
                             float* __restrict__ out,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + __half2float(b[idx]);
}

// --- Fused Add + RMSNorm ---
// hidden[i] = hidden[i] + delta[i]  (f32 + f16 → f32, in-place)
// norm_out[i] = (hidden[i] / rms) * weight[i]  (f32 → f16)
// Saves one kernel launch + one full pass over hidden.
__global__ void fused_add_rmsnorm(float* __restrict__ hidden,
                                   const __half* __restrict__ delta,
                                   const __half* __restrict__ weight,
                                   __half* __restrict__ norm_out,
                                   int dim,
                                   float eps) {
    int row = blockIdx.x;
    float* h_row = hidden + row * dim;
    __half* n_row = norm_out + row * dim;
    const __half* d_row = delta + row * dim;

    extern __shared__ float sdata[];

    // Pass 1: add + sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = h_row[i] + __half2float(d_row[i]);
        h_row[i] = v;
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    // Pass 2: normalize + scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        n_row[i] = __float2half(h_row[i] / rms * __half2float(weight[i]));
    }
}

// --- RMSNorm (f32→f16) + Q8_1 Quantize ---
// Combines rms_norm_f32in + quantize_input into one kernel.
__global__ void rms_norm_f32in_q8(const float* __restrict__ x,
                                   const __half* __restrict__ weight,
                                   __half* __restrict__ out,
                                   void* __restrict__ x_q8,
                                   int dim,
                                   float eps) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm_val = x_row[i] / rms * __half2float(weight[i]);
        out_row[i] = __float2half(norm_val);

        // Q8_1 quantize
        float amax = fabsf(norm_val);
        float wsum = norm_val;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
            wsum += __shfl_xor_sync(0xFFFFFFFF, wsum, offset);
        }
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        int8_t q = (int8_t)__float2int_rn(norm_val * id);

        int q8_block = i / 32;
        int q8_in = i % 32;
        uint8_t* block_ptr = (uint8_t*)x_q8 + q8_block * 36;
        if (q8_in == 0) {
            *(half2*)(block_ptr) = make_half2(__float2half(d), __float2half(wsum));
        }
        ((int8_t*)(block_ptr + 4))[q8_in] = q;
    }
}

// --- Fused Add + RMSNorm + Q8_1 Quantize ---
// Same as fused_add_rmsnorm but ALSO writes Q8_1 quantized norm_out.
// Eliminates a separate quantize_input kernel launch.
// x_q8 format: half2 ds + int8_t qs[32] = 36 bytes per 32-element block.
__global__ void fused_add_rmsnorm_q8(float* __restrict__ hidden,
                                      const __half* __restrict__ delta,
                                      const __half* __restrict__ weight,
                                      __half* __restrict__ norm_out,
                                      void* __restrict__ x_q8,
                                      int dim,
                                      float eps) {
    int row = blockIdx.x;
    float* h_row = hidden + row * dim;
    __half* n_row = norm_out + row * dim;
    const __half* d_row = delta + row * dim;

    extern __shared__ float sdata[];

    // Pass 1: add + sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = h_row[i] + __half2float(d_row[i]);
        h_row[i] = v;
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    // Pass 2: normalize + scale + Q8_1 quantize
    // Each warp handles consecutive 32 elements. With stride access and 256 threads:
    // Iteration 0: threads 0-255 handle elements 0-255
    // Iteration 1: threads 0-255 handle elements 256-511
    // Within each iteration: warp 0 handles elements [iter*256 + 0..31]
    //                        warp 1 handles elements [iter*256 + 32..63]
    //                        etc.
    // So each warp has consecutive 32 elements → can do Q8_1 block quantize.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm_val = h_row[i] / rms * __half2float(weight[i]);
        n_row[i] = __float2half(norm_val);

        // Q8_1 quantize: warp-level reduce for this group of 32 elements
        float amax = fabsf(norm_val);
        float wsum = norm_val;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
            wsum += __shfl_xor_sync(0xFFFFFFFF, wsum, offset);
        }

        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        int8_t q = (int8_t)__float2int_rn(norm_val * id);

        // Write Q8_1 block: half2 ds (4 bytes) + int8_t qs[32]
        int q8_block = i / 32;
        int q8_in = i % 32;
        uint8_t* block_ptr = (uint8_t*)x_q8 + q8_block * 36;
        if (q8_in == 0) {
            *(half2*)(block_ptr) = make_half2(__float2half(d), __float2half(wsum));
        }
        ((int8_t*)(block_ptr + 4))[q8_in] = q;
    }
}

// --- Fused Add (no norm, just residual update in-place) ---
// hidden[i] += delta[i]  (f32 + f16 → f32)
__global__ void add_inplace_f32_f16(float* __restrict__ hidden,
                                     const __half* __restrict__ delta,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    hidden[idx] += __half2float(delta[idx]);
}

// --- Embedding Lookup (f32 output) ---
// Reads f16 embedding table, writes f32 hidden state.
__global__ void embed_tokens_f32(const __half* __restrict__ embd,
                                  const int* __restrict__ token_ids,
                                  float* __restrict__ out,
                                  int dim) {
    int tok_idx = blockIdx.x;
    int token_id = token_ids[tok_idx];
    const __half* src = embd + token_id * dim;
    float* dst = out + tok_idx * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = __half2float(src[i]);
    }
}

// --- RoPE (Rotary Position Embeddings) --- f16 in-place
// Applies rotary embeddings to Q and K tensors in f16.
// x has shape [seq_len, n_heads, head_dim]
// pos is the starting position index.
__global__ void rope(__half* __restrict__ x,
                      int head_dim,
                      int n_heads,
                      int pos,
                      float theta_base) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    int offset = seq_idx * n_heads * head_dim + head_idx * head_dim + pair_idx * 2;

    float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
    float angle = (float)(pos + seq_idx) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(x[offset]);
    float x1 = __half2float(x[offset + 1]);

    x[offset]     = __float2half(x0 * cos_a - x1 * sin_a);
    x[offset + 1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// --- Fused RoPE(Q) + RoPE(K) + KV cache write ---
// One kernel launch instead of 4 (rope_q, rope_k, copy_k, copy_v)
// Grid: (seq_len, n_heads + n_kv_heads, 1), Block: (head_dim/2, 1, 1)
// For head_idx < n_heads: apply RoPE to Q
// For head_idx >= n_heads: apply RoPE to K, and copy K+V to cache
__global__ void fused_rope_kv(
    __half* __restrict__ q,          // [seq_len, n_heads, head_dim]
    __half* __restrict__ k,          // [seq_len, n_kv_heads, head_dim]
    const __half* __restrict__ v,    // [seq_len, n_kv_heads, head_dim]
    __half* __restrict__ k_cache,    // KV cache K at write position
    __half* __restrict__ v_cache,    // KV cache V at write position
    int head_dim, int n_heads, int n_kv_heads,
    int pos, float theta_base) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
    float angle = (float)(pos + seq_idx) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    if (head_idx < n_heads) {
        // RoPE on Q
        int off = seq_idx * n_heads * head_dim + head_idx * head_dim + pair_idx * 2;
        float x0 = __half2float(q[off]);
        float x1 = __half2float(q[off + 1]);
        q[off]     = __float2half(x0 * cos_a - x1 * sin_a);
        q[off + 1] = __float2half(x0 * sin_a + x1 * cos_a);
    } else {
        // RoPE on K + copy K to cache + copy V to cache
        int kv_head = head_idx - n_heads;
        int off = seq_idx * n_kv_heads * head_dim + kv_head * head_dim + pair_idx * 2;
        float x0 = __half2float(k[off]);
        float x1 = __half2float(k[off + 1]);
        float k0 = x0 * cos_a - x1 * sin_a;
        float k1 = x0 * sin_a + x1 * cos_a;
        k[off]     = __float2half(k0);
        k[off + 1] = __float2half(k1);
        // Write to KV cache
        k_cache[off] = __float2half(k0);
        k_cache[off + 1] = __float2half(k1);
        // Copy V (no RoPE on V)
        v_cache[off]     = v[off];
        v_cache[off + 1] = v[off + 1];
    }
}

// --- SiLU + Q8_1 quantize fused ---
// Computes SiLU(gate) * up, writes f16 output, AND quantizes to Q8_1.
// Quantizes from the f16 round-tripped value (matching separate silu+quantize pipeline).
// Grid: ((n+255)/256,), Block: (256,)
__global__ void silu_q8(const __half* __restrict__ gate,
                         const __half* __restrict__ up,
                         __half* __restrict__ out,
                         void* __restrict__ x_q8,
                         int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // SiLU(gate) * up
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    __half val_f16 = __float2half(g / (1.0f + expf(-g)) * u);
    out[idx] = val_f16;

    // Q8_1 quantize from f16 (matches separate quantize_input precision)
    float val = __half2float(val_f16);
    const int lane = threadIdx.x % 32;

    float amax = fabsf(val);
    float wsum = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
        wsum += __shfl_xor_sync(0xFFFFFFFF, wsum, offset);
    }

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)__float2int_rn(val * id);

    int q8_block = idx / 32;
    uint8_t* block_ptr = (uint8_t*)x_q8 + q8_block * 36;
    if (lane == 0) {
        *(half2*)(block_ptr) = make_half2(__float2half(d), __float2half(wsum));
    }
    ((int8_t*)(block_ptr + 4))[lane] = q;
}

// --- MHA output quantize fused ---
// After MHA writes to attn_mha_out, quantize it to Q8_1 for the output projection GEMV.
// This fuses the separate quantize_input call.
// Grid: ((n+255)/256,), Block: (256,)
__global__ void quantize_f16_q8_1(const __half* __restrict__ x,
                                    void* __restrict__ x_q8,
                                    int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float val = __half2float(x[idx]);
    const int lane = threadIdx.x % 32;

    float amax = fabsf(val);
    float wsum = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
        wsum += __shfl_xor_sync(0xFFFFFFFF, wsum, offset);
    }

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)__float2int_rn(val * id);

    int q8_block = idx / 32;
    uint8_t* block_ptr = (uint8_t*)x_q8 + q8_block * 36;
    if (lane == 0) {
        *(half2*)(block_ptr) = make_half2(__float2half(d), __float2half(wsum));
    }
    ((int8_t*)(block_ptr + 4))[lane] = q;
}

// --- RoPE + KV cache write fused ---
// One block per (seq_pos, head). Threads iterate over pairs.
// Grid: (seq_len, n_heads + n_kv_heads,), Block: (min(head_dim/2, 256),)
__global__ void rope_kv_write(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ k_cache_base,
    __half* __restrict__ v_cache_base,
    int head_dim, int n_heads, int n_kv_heads,
    int pos, float theta_base,
    int kv_stride, int kv_offset) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int half_dim = head_dim / 2;

    for (int pair_idx = threadIdx.x; pair_idx < half_dim; pair_idx += blockDim.x) {
        float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
        float angle = (float)(pos + seq_idx) * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        if (head_idx < n_heads) {
            int off = seq_idx * n_heads * head_dim + head_idx * head_dim + pair_idx * 2;
            float x0 = __half2float(q[off]);
            float x1 = __half2float(q[off + 1]);
            q[off]     = __float2half(x0 * cos_a - x1 * sin_a);
            q[off + 1] = __float2half(x0 * sin_a + x1 * cos_a);
        } else {
            int kv_head = head_idx - n_heads;
            int scratch_off = seq_idx * n_kv_heads * head_dim + kv_head * head_dim + pair_idx * 2;
            float x0 = __half2float(k[scratch_off]);
            float x1 = __half2float(k[scratch_off + 1]);
            float k0 = x0 * cos_a - x1 * sin_a;
            float k1 = x0 * sin_a + x1 * cos_a;
            k[scratch_off]     = __float2half(k0);
            k[scratch_off + 1] = __float2half(k1);
            int cache_off = kv_offset + seq_idx * kv_stride + kv_head * head_dim + pair_idx * 2;
            k_cache_base[cache_off]     = __float2half(k0);
            k_cache_base[cache_off + 1] = __float2half(k1);
            v_cache_base[cache_off]     = v[scratch_off];
            v_cache_base[cache_off + 1] = v[scratch_off + 1];
        }
    }
}

// --- SiLU (Sigmoid Linear Unit) --- f16 version
// out = SiLU(gate) * up
__global__ void silu(const __half* __restrict__ gate,
                      const __half* __restrict__ up,
                      __half* __restrict__ out,
                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float s = g / (1.0f + expf(-g));  // SiLU(gate)
    out[idx] = __float2half(s * u);   // SiLU(gate) * up
}

// --- Softmax --- f16 version
// In-place softmax over dim elements per row.
// One block per row.
__global__ void softmax(__half* __restrict__ x,
                         int dim) {
    int row = blockIdx.x;
    __half* x_row = x + row * dim;

    extern __shared__ float sdata[];

    // Find max
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        if (v > local_max) local_max = v;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = expf(__half2float(x_row[i]) - max_val);
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum = sdata[0];

    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        x_row[i] = __float2half(expf(__half2float(x_row[i]) - max_val) / sum);
    }
}

// --- Copy f32 → f16 ---
// For writing f32 hidden state into f16 buffers.
__global__ void copy_f32_to_f16(const float* __restrict__ src,
                                 __half* __restrict__ dst,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(src[idx]);
}

// --- Copy f16 → f16 ---
// For writing f16 K/V projections into f16 KV cache.
__global__ void copy_f16(const __half* __restrict__ src,
                          __half* __restrict__ dst,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx];
}

// --- Multi-Head Attention (with GQA) --- f16 Q, f16 KV cache, f16 output
//
// Q layout: [seq_len, n_heads, head_dim]        f16
// K layout: [kv_len, n_kv_heads, head_dim]      f16 (from KV cache)
// V layout: [kv_len, n_kv_heads, head_dim]      f16 (from KV cache)
// out:      [seq_len, n_heads, head_dim]         f16
//
// Grid: (n_heads, seq_len, 1)
// Block: (threads, 1, 1)
//
// --- Fused MHA with parallel softmax ---
// Score materialization in shared memory, parallel softmax, weighted V sum.
// --- Tiled MHA (used for both regular and graph paths) ---
// Processes KV in tiles of MHA_TILE_KV with online softmax across tiles.
// Fixed shared memory: (MHA_TILE_KV + blockDim.x) floats.
// This replaces the old score-materializing MHA that needed kv_len * 4 bytes smem.
#ifndef MHA_TILE_KV
#define MHA_TILE_KV 128
#endif
__global__ void mha_fused(const __half* __restrict__ q,
                           const __half* __restrict__ k,
                           const __half* __restrict__ v,
                           __half* __restrict__ out,
                           int head_dim,
                           int n_heads,
                           int n_kv_heads,
                           int seq_len,
                           int kv_len,
                           int pos_offset,
                           float scale) {
    int head = blockIdx.x;
    int q_pos_local = blockIdx.y;
    int q_pos_global = pos_offset + q_pos_local;
    int kv_head = head / (n_heads / n_kv_heads);

    const __half* q_vec = q + q_pos_local * n_heads * head_dim + head * head_dim;

    extern __shared__ float smem[];
    float* tile_scores = smem;
    float* scratch = smem + MHA_TILE_KV;

    // Online softmax state
    float global_max = -1e30f;
    float global_sum = 0.0f;

    // Per-thread V accumulator
    float v_acc[4] = {0,0,0,0};
    int elems = (head_dim + (int)blockDim.x - 1) / (int)blockDim.x;

    for (int tile_start = 0; tile_start < kv_len; tile_start += MHA_TILE_KV) {
        int tile_end = tile_start + MHA_TILE_KV;
        if (tile_end > kv_len) tile_end = kv_len;
        int tile_size = tile_end - tile_start;

        // Compute scores for this tile
        for (int t = 0; t < tile_size; t++) {
            int kp = tile_start + t;
            if (kp > q_pos_global) {
                if (threadIdx.x == 0) tile_scores[t] = -1e30f;
                __syncthreads();
                continue;
            }
            const __half* k_vec = k + kp * n_kv_heads * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
            scratch[threadIdx.x] = dot;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) tile_scores[t] = scratch[0] * scale;
            __syncthreads();
        }

        // Tile max
        float tile_max = -1e30f;
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x)
            if (tile_scores[t] > tile_max) tile_max = tile_scores[t];
        scratch[threadIdx.x] = tile_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
            __syncthreads();
        }
        tile_max = scratch[0];

        // Online softmax merge
        float new_max = fmaxf(global_max, tile_max);
        float old_scale_factor = expf(global_max - new_max);
        float tile_scale_factor = expf(tile_max - new_max);

        for (int e = 0; e < elems; e++) v_acc[e] *= old_scale_factor;
        global_sum *= old_scale_factor;

        for (int t = threadIdx.x; t < tile_size; t += blockDim.x)
            tile_scores[t] = expf(tile_scores[t] - tile_max) * tile_scale_factor;
        __syncthreads();

        float tile_sum = 0.0f;
        for (int t = 0; t < tile_size; t++) tile_sum += tile_scores[t];
        global_sum += tile_sum;

        // Accumulate weighted V
        for (int e = 0; e < elems; e++) {
            int d = threadIdx.x + e * blockDim.x;
            if (d < head_dim) {
                float acc = 0.0f;
                for (int t = 0; t < tile_size; t++) {
                    int kp = tile_start + t;
                    acc += tile_scores[t] * __half2float(v[kp * n_kv_heads * head_dim + kv_head * head_dim + d]);
                }
                v_acc[e] += acc;
            }
        }
        global_max = new_max;
    }

    // Write output
    __half* out_vec = out + q_pos_local * n_heads * head_dim + head * head_dim;
    float inv = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;
    for (int e = 0; e < elems; e++) {
        int d = threadIdx.x + e * blockDim.x;
        if (d < head_dim) out_vec[d] = __float2half(v_acc[e] * inv);
    }
}

// --- Argmax over f16 vector ---
// Finds the index of the maximum value. One block, 256 threads.
// Grid: (1,), Block: (256,)
__global__ void argmax_f16(const __half* __restrict__ x, int* __restrict__ out, int n) {
    __shared__ float smax[256];
    __shared__ int   sidx[256];

    float local_max = -1e30f;
    int   local_idx = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(x[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    smax[threadIdx.x] = local_max;
    sidx[threadIdx.x] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x]) {
            smax[threadIdx.x] = smax[threadIdx.x + s];
            sidx[threadIdx.x] = sidx[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out[0] = sidx[0];
}


// --- GPU Top-K + Softmax + Sampling (single-pass) ---
// Each thread maintains a local top-K via insertion sort (K small = ~40).
// Single pass through vocab, then serial merge in thread 0.
// Grid: (1,), Block: (256,)
#define SAMPLE_K 40
__global__ void sample_top_k(const __half* __restrict__ logits,
                              int* __restrict__ out,
                              int vocab_size,
                              float temperature,
                              int top_k_param,
                              float top_p,
                              unsigned int random_seed) {
    const int k = min(top_k_param, SAMPLE_K);

    // Thread-local top-K via insertion sort
    float lv[SAMPLE_K];
    int   li[SAMPLE_K];
    for (int i = 0; i < k; i++) { lv[i] = -1e30f; li[i] = 0; }

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > lv[k-1]) {
            lv[k-1] = v; li[k-1] = i;
            for (int j = k-2; j >= 0; j--) {
                if (lv[j+1] > lv[j]) {
                    float tv = lv[j]; lv[j] = lv[j+1]; lv[j+1] = tv;
                    int ti = li[j]; li[j] = li[j+1]; li[j+1] = ti;
                } else break;
            }
        }
    }

    // Write thread-local top-K to shared, merge in thread 0
    __shared__ float sv[256 * 2];   // 256 threads × top-2 values (val, idx pairs)
    __shared__ int   si[256 * 2];

    // Each thread stores its top-2 (enough for merging top-40 from 256 threads)
    int store = min(k, 2);
    for (int i = 0; i < store; i++) {
        sv[threadIdx.x * store + i] = lv[i];
        si[threadIdx.x * store + i] = li[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Merge: collect all 256*store candidates, find global top-K
        // Simple: iterate candidates, maintain sorted top-K
        float gv[SAMPLE_K]; int gi[SAMPLE_K];
        for (int i = 0; i < k; i++) { gv[i] = -1e30f; gi[i] = 0; }

        for (int t = 0; t < 256 * store; t++) {
            float v = sv[t];
            if (v > gv[k-1]) {
                gv[k-1] = v; gi[k-1] = si[t];
                for (int j = k-2; j >= 0; j--) {
                    if (gv[j+1] > gv[j]) {
                        float tv = gv[j]; gv[j] = gv[j+1]; gv[j+1] = tv;
                        int ti = gi[j]; gi[j] = gi[j+1]; gi[j+1] = ti;
                    } else break;
                }
            }
        }

        // Temperature + softmax + top-p + sample
        if (temperature > 0.0f && temperature != 1.0f)
            for (int i = 0; i < k; i++) gv[i] /= temperature;

        float mx = gv[0], sum = 0.0f;
        for (int i = 0; i < k; i++) { gv[i] = expf(gv[i] - mx); sum += gv[i]; }
        for (int i = 0; i < k; i++) gv[i] /= sum;

        int cutoff = k;
        if (top_p < 1.0f) {
            float cs = 0.0f;
            for (int i = 0; i < k; i++) { cs += gv[i]; if (cs >= top_p) { cutoff = i+1; break; } }
            sum = 0.0f;
            for (int i = 0; i < cutoff; i++) sum += gv[i];
            for (int i = 0; i < cutoff; i++) gv[i] /= sum;
        }

        unsigned int x = random_seed;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        float r = (float)(x >> 8) / (float)(1 << 24);
        float cs = 0.0f;
        for (int i = 0; i < cutoff; i++) { cs += gv[i]; if (r < cs) { out[0] = gi[i]; return; } }
        out[0] = gi[0];
    }
}
// =============================================================
// CUDA Graph-compatible kernels
// =============================================================
// These read dynamic per-step parameters from device memory
// instead of kernel arguments, allowing CUDA Graph replay
// without re-capture.
//
// decode_params layout (int array in device memory):
//   [0] = pos          (current sequence position)
//   [1] = kv_len       (total KV cache length = pos + 1 for decode)
//   [2] = kv_offset    (element offset for KV cache write = pos * n_kv_heads * head_dim)

// --- RoPE reading pos from device memory ---
__global__ void rope_graph(__half* __restrict__ x,
                           const int* __restrict__ decode_params,
                           int head_dim,
                           int n_heads,
                           float theta_base) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    int pos = decode_params[0];
    int offset = seq_idx * n_heads * head_dim + head_idx * head_dim + pair_idx * 2;

    float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
    float angle = (float)(pos + seq_idx) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(x[offset]);
    float x1 = __half2float(x[offset + 1]);

    x[offset]     = __float2half(x0 * cos_a - x1 * sin_a);
    x[offset + 1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// --- Copy f16 with offset (for KV cache writes without changing pointers) ---
__global__ void copy_f16_with_offset(const __half* __restrict__ src,
                                      __half* __restrict__ dst_base,
                                      const int* __restrict__ decode_params,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int kv_offset = decode_params[2];
    dst_base[kv_offset + idx] = src[idx];
}

// --- MHA reading kv_len and pos_offset from device memory ---
// Uses base KV cache pointers (not views) — compatible with graph replay.
// Shared memory must be allocated for max_kv_len at capture time.
// --- Tiled MHA for CUDA Graph (fixed shared memory) ---
// Processes KV positions in tiles of TILE_KV. Uses online softmax across tiles.
// Shared memory: TILE_KV + blockDim.x floats = CONSTANT regardless of kv_len.
// This is the key: CUDA Graph can capture this with fixed smem.
#define MHA_TILE_KV 128
__global__ void mha_fused_graph(
    const __half* __restrict__ q,
    const __half* __restrict__ k_base,
    const __half* __restrict__ v_base,
    __half* __restrict__ out,
    const int* __restrict__ decode_params,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    float scale) {
    int head = blockIdx.x;
    int q_pos_local = blockIdx.y;

    int kv_len = decode_params[1];
    int pos_offset = decode_params[0];
    int q_pos_global = pos_offset + q_pos_local;
    int kv_head = head / (n_heads / n_kv_heads);

    const __half* q_vec = q + q_pos_local * n_heads * head_dim + head * head_dim;

    // Fixed shared memory: tile_scores[TILE_KV] + scratch[blockDim.x]
    extern __shared__ float smem[];
    float* tile_scores = smem;
    float* scratch = smem + MHA_TILE_KV;

    // Online softmax state per thread
    float global_max = -1e30f;
    float global_sum = 0.0f;

    // Per-thread V accumulator (head_dim elements spread across threads)
    // For head_dim=128, 128 threads: 1 element per thread
    float v_acc[4] = {0,0,0,0}; // supports head_dim up to 512 with 128 threads
    int elems = (head_dim + blockDim.x - 1) / (int)blockDim.x;

    // Process KV in tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += MHA_TILE_KV) {
        int tile_end = tile_start + MHA_TILE_KV;
        if (tile_end > kv_len) tile_end = kv_len;
        int tile_size = tile_end - tile_start;

        // --- Compute scores for this tile ---
        for (int t = 0; t < tile_size; t++) {
            int kp = tile_start + t;
            if (kp > q_pos_global) {
                if (threadIdx.x == 0) tile_scores[t] = -1e30f;
                __syncthreads();
                continue;
            }

            const __half* k_vec = k_base + kp * n_kv_heads * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
            }
            scratch[threadIdx.x] = dot;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) tile_scores[t] = scratch[0] * scale;
            __syncthreads();
        }

        // --- Tile max ---
        float tile_max = -1e30f;
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {
            if (tile_scores[t] > tile_max) tile_max = tile_scores[t];
        }
        scratch[threadIdx.x] = tile_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
            __syncthreads();
        }
        tile_max = scratch[0];

        // --- Online softmax: merge tile with global state ---
        float new_max = fmaxf(global_max, tile_max);
        float old_scale_factor = expf(global_max - new_max);
        float tile_scale_factor = expf(tile_max - new_max);

        // Rescale V accumulator from old max
        for (int e = 0; e < elems; e++) {
            v_acc[e] *= old_scale_factor;
        }
        global_sum *= old_scale_factor;

        // Exp scores relative to tile_max, then scale to new_max
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {
            tile_scores[t] = expf(tile_scores[t] - tile_max) * tile_scale_factor;
        }
        __syncthreads();

        // Sum this tile's weights
        float tile_sum = 0.0f;
        for (int t = 0; t < tile_size; t++) {
            tile_sum += tile_scores[t];
        }
        global_sum += tile_sum;

        // Accumulate weighted V for this tile
        for (int e = 0; e < elems; e++) {
            int d = threadIdx.x + e * blockDim.x;
            if (d < head_dim) {
                float acc = 0.0f;
                for (int t = 0; t < tile_size; t++) {
                    int kp = tile_start + t;
                    const __half* v_vec = v_base + kp * n_kv_heads * head_dim + kv_head * head_dim;
                    acc += tile_scores[t] * __half2float(v_vec[d]);
                }
                v_acc[e] += acc;
            }
        }

        global_max = new_max;
    }

    // --- Write output: v_acc / global_sum ---
    __half* out_vec = out + q_pos_local * n_heads * head_dim + head * head_dim;
    float inv = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;
    for (int e = 0; e < elems; e++) {
        int d = threadIdx.x + e * blockDim.x;
        if (d < head_dim) {
            out_vec[d] = __float2half(v_acc[e] * inv);
        }
    }
}

// (Old mha_fused_graph with dynamic smem removed — replaced by tiled version above)

// =============================================================
// Gemma 4 kernels
// =============================================================

// --- GELU activation: out = GELU(gate) * up ---
// Gemma 4 uses GELU instead of SiLU.
__global__ void gelu(const __half* __restrict__ gate,
                     const __half* __restrict__ up,
                     __half* __restrict__ out,
                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    // GELU approximation (tanh version, matches PyTorch)
    float gelu_g = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    out[idx] = __float2half(gelu_g * u);
}

// --- RMSNorm without weight multiplication (for V norm) ---
// Just normalizes by RMS, no learned weight scaling.
// f16 input → f16 output.
__global__ void rms_norm_no_weight(const __half* __restrict__ x,
                                    __half* __restrict__ out,
                                    int dim,
                                    float eps) {
    int row = blockIdx.x;
    const __half* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = __float2half(__half2float(x_row[i]) / rms);
    }
}

// --- Scale f16 tensor by scalar ---
// out[i] = x[i] * scale
__global__ void scale_f16(const __half* __restrict__ x,
                          __half* __restrict__ out,
                          int n,
                          float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(__half2float(x[idx]) * scale);
}

// --- Scale f32 tensor by scalar in-place ---
// x[i] *= scale
__global__ void scale_f32_inplace(float* __restrict__ x,
                                  int n,
                                  float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] *= scale;
}

// --- Logit softcapping: out = tanh(x / cap) * cap ---
// Applied after the final logit projection in Gemma 4.
__global__ void logit_softcap(const __half* __restrict__ x,
                              __half* __restrict__ out,
                              int n,
                              float cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(x[idx]);
    out[idx] = __float2half(tanhf(v / cap) * cap);
}

// --- RoPE NeoX (interleaved first/second half) --- f16 in-place
// NeoX-style: pairs are (x[i], x[i + d/2]) instead of (x[2i], x[2i+1])
// x has shape [seq_len, n_heads, head_dim]
__global__ void rope_neox(__half* __restrict__ x,
                          int head_dim,
                          int n_heads,
                          int pos,
                          float theta_base) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;  // 0..head_dim/2-1

    int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    int base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int idx0 = base + pair_idx;           // first half
    int idx1 = base + pair_idx + half_dim; // second half

    float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
    float angle = (float)(pos + seq_idx) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(x[idx0]);
    float x1 = __half2float(x[idx1]);

    x[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
    x[idx1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// --- RoPE NeoX with proportional frequency factors --- f16 in-place
// For Gemma 4 full-attention layers: freq_factors[pair_idx] divides the frequency.
// When freq_factor = 1e30, angle ≈ 0, so cos=1, sin=0 → no rotation (identity).
// n_rope_dims: number of dimensions to actually rotate (rest are identity).
// freq_factors has shape [head_dim/2].
__global__ void rope_neox_freqs(__half* __restrict__ x,
                                const float* __restrict__ freq_factors,
                                int head_dim,
                                int n_heads,
                                int pos,
                                float theta_base) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;

    int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    int base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int idx0 = base + pair_idx;
    int idx1 = base + pair_idx + half_dim;

    float ff = freq_factors[pair_idx];
    float freq = 1.0f / (powf(theta_base, (float)(2 * pair_idx) / (float)head_dim) * ff);
    float angle = (float)(pos + seq_idx) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(x[idx0]);
    float x1 = __half2float(x[idx1]);

    x[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
    x[idx1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// --- RoPE NeoX graph-compatible (reads pos from device memory) ---
__global__ void rope_neox_graph(__half* __restrict__ x,
                                const int* __restrict__ decode_params,
                                int head_dim,
                                int n_heads,
                                float theta_base) {
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;

    int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    int pos = decode_params[0];
    int base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int idx0 = base + pair_idx;
    int idx1 = base + pair_idx + half_dim;

    float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
    float angle = (float)(pos + seq_idx) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(x[idx0]);
    float x1 = __half2float(x[idx1]);

    x[idx0] = __float2half(x0 * cos_a - x1 * sin_a);
    x[idx1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// --- MHA with custom attention scale (no 1/sqrt(d)) ---
// Same as mha_fused but takes explicit scale parameter, for Gemma 4
// which uses attention_scale=1.0 (pre-normalized via QK norms).
// (This is already handled by the existing mha_fused which takes scale param)

// --- Post-norm fused add: out = rmsnorm(delta) * weight, hidden += out ---
// For Gemma 4 post-attention/post-FFN norms.
// delta is f16, hidden is f32, weight is f16, out is f16.
__global__ void post_norm_add(float* __restrict__ hidden,
                              const __half* __restrict__ delta,
                              const __half* __restrict__ weight,
                              __half* __restrict__ norm_out,
                              int dim,
                              float eps) {
    int row = blockIdx.x;
    const __half* d_row = delta + row * dim;
    float* h_row = hidden + row * dim;
    __half* n_row = norm_out + row * dim;

    extern __shared__ float sdata[];

    // Compute RMS of delta
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(d_row[i]);
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)dim + eps);

    // norm_out = rmsnorm(delta) * weight, hidden += norm_out
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normed = __half2float(d_row[i]) / rms * __half2float(weight[i]);
        n_row[i] = __float2half(normed);
        h_row[i] += normed;
    }
}

// --- Logit softcap in-place ---
__global__ void logit_softcap_inplace(__half* __restrict__ x, int n, float cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(x[idx]);
    x[idx] = __float2half(tanhf(v / cap) * cap);
}

// --- Element-wise multiply: out = a * b (f16) ---
__global__ void mul_f16(const __half* __restrict__ a,
                        const __half* __restrict__ b,
                        __half* __restrict__ out,
                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
}

// --- Standalone GELU activation: out = GELU(x) (f16) ---
// Does NOT multiply by up — just applies GELU to x.
__global__ void gelu_act(const __half* __restrict__ x,
                         __half* __restrict__ out,
                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __half2float(x[idx]);
    float gelu_g = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    out[idx] = __float2half(gelu_g);
}

// --- Gather quantized rows by token ID ---
// Copies rows from a quantized tensor into a contiguous output buffer.
// src: quantized tensor data (all rows contiguous)
// token_ids: [n_tokens] int32 token indices
// dst: output buffer [n_tokens * row_bytes] bytes
// row_bytes: bytes per row in the quantized format
// n_tokens: number of tokens to gather
__global__ void gather_rows_quant(const unsigned char* __restrict__ src,
                                  const int* __restrict__ token_ids,
                                  unsigned char* __restrict__ dst,
                                  int row_bytes,
                                  int n_tokens) {
    // Grid: (n_tokens, ceil(row_bytes/blockDim.x), 1)
    int tok_idx = blockIdx.x;
    int byte_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (tok_idx >= n_tokens || byte_idx >= row_bytes) return;

    int token_id = token_ids[tok_idx];
    long long src_off = (long long)token_id * (long long)row_bytes + byte_idx;
    long long dst_off = (long long)tok_idx * (long long)row_bytes + byte_idx;
    dst[dst_off] = src[src_off];
}

// --- Per-layer embedding strided multiply ---
// For each token t and embedding dim j:
//   out[t * epl + j] = a[t * epl + j] * embd[t * row_width + layer_off + j]
// a, out: [n_tokens, epl] contiguous f16
// embd: [n_tokens, row_width] contiguous f16 (full per-layer embeddings for all layers)
// layer_off: column offset for the current layer
__global__ void pe_strided_mul(const __half* __restrict__ a,
                               const __half* __restrict__ embd,
                               __half* __restrict__ out,
                               int epl,
                               int row_width,
                               int layer_off,
                               int n_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * epl;
    if (idx >= total) return;

    int t = idx / epl;
    int j = idx % epl;
    float av = __half2float(a[idx]);
    float ev = __half2float(embd[t * row_width + layer_off + j]);
    out[idx] = __float2half(av * ev);
}

} // extern "C"
