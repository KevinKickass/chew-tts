// Transformer operation kernels
//
// Mixed precision: hidden state stays f32 (residual stream), intermediate
// buffers are f16 for VRAM efficiency. Bridge ops convert between them.
// KV cache is f16. Weight/embedding tables are f16.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

// --- RMS Norm f32 input, NO weight, output f16 ---
// out[row,i] = x[row,i] / rms(x[row,:])
__global__ void rms_norm_f32in_no_weight(const float* __restrict__ x,
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
        out_row[i] = __float2half(x_row[i] / rms);
    }
}

// --- LayerNorm (f32 input, f16 output) ---
// out[row,i] = ((x[row,i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
__global__ void layer_norm_f32in(const float* __restrict__ x,
                                 const __half* __restrict__ weight,
                                 const __half* __restrict__ bias,
                                 __half* __restrict__ out,
                                 int dim,
                                 float eps) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    __half* out_row = out + row * dim;

    extern __shared__ float sdata[];
    float* sum_buf = sdata;
    float* sq_buf = sdata + blockDim.x;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v;
        local_sq += v * v;
    }
    sum_buf[threadIdx.x] = local_sum;
    sq_buf[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_buf[threadIdx.x] += sum_buf[threadIdx.x + s];
            sq_buf[threadIdx.x] += sq_buf[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = sum_buf[0] / (float)dim;
    float var = sq_buf[0] / (float)dim - mean * mean;
    float inv_std = rsqrtf(fmaxf(var, 0.0f) + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm = (x_row[i] - mean) * inv_std;
        float w = __half2float(weight[i]);
        float b = __half2float(bias[i]);
        out_row[i] = __float2half(norm * w + b);
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

// --- Row-wise bias add on f16 matrix ---
// out[row, col] = x[row, col] + bias[col]
__global__ void add_bias_f16(const __half* __restrict__ x,
                             const __half* __restrict__ bias,
                             __half* __restrict__ out,
                             int dim) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    int idx = row * dim + col;
    out[idx] = __hadd(x[idx], bias[col]);
}

// --- Row-wise bias add in-place on f16 matrix ---
__global__ void add_bias_f16_inplace(__half* __restrict__ x,
                                     const __half* __restrict__ bias,
                                     int dim) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    int idx = row * dim + col;
    x[idx] = __hadd(x[idx], bias[col]);
}

// --- Copy f16 -> f32 ---
__global__ void copy_f16_to_f32(const __half* __restrict__ src,
                                float* __restrict__ dst,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __half2float(src[idx]);
}

// --- In-place add: f32 += f32 ---
__global__ void add_inplace_f32(const float* __restrict__ delta,
                                float* __restrict__ x,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += delta[idx];
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
// --- Flash Attention MHA (sub-warp dot products, warp-independent) ---
// Based on llama.cpp fattn-vec design for f16 K/V, D=128.
// 128 threads = 4 warps. Sub-groups of 8 threads compute one Q·K dot product.
// Each warp processes 32 KV positions per outer iteration (4 sub-groups × 8 iterations).
// Warps are INDEPENDENT — only __syncwarp() in the main loop, no __syncthreads().
// V accumulation uses smem within each warp to broadcast attention weights.
// Final cross-warp reduction at the end via smem.
// Grid: (n_heads, seq_len), Block: (32, 4) [lane, warp]
// smem: 4 * 32 floats for KQ weights + 4*D floats for final VKQ combine = fixed size
#define FA_NTH_KQ 8
#define FA_NTH_V  8
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
                           int window,
                           float scale,
                           float softcap) {
    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int causal_limit = pos_offset + q_pos;
    const int causal_start = window > 0 ? ((causal_limit - window + 1) > 0 ? (causal_limit - window + 1) : 0) : 0;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int lane = threadIdx.x;  // 0..31
    const int warp = threadIdx.y;  // 0..3
    const int tid = warp * 32 + lane;
    const int kv_stride = n_kv_heads * head_dim;
    const int D = head_dim;  // alias

    const __half* Q_ptr = q + q_pos * n_heads * D + head * D;

    // Load Q into registers: each sub-group of FA_NTH_KQ=8 threads holds all of Q.
    // Thread lane holds D/(2*8) float2 values: 8 for D=128, 16 for D=256, 32 for D=512.
    const int q_idx = lane % FA_NTH_KQ;
    const int q_elems = D / (2 * FA_NTH_KQ);  // 8 for D=128, 32 for D=512
    float2 Q_reg[32];
    #pragma unroll
    for (int i = 0; i < q_elems; i++) {
        int d = q_idx * q_elems + i;
        Q_reg[i] = make_float2(
            __half2float(Q_ptr[2*d]) * scale,
            __half2float(Q_ptr[2*d + 1]) * scale
        );
    }

    // Per-warp V accumulator: each of 32 lanes owns D/32 dimensions.
    // For D=128: 4 f16 values = 2 float2 per lane.
    // For D=256: 8 f16 = 4 float2. For D=512: 16 f16 = 8 float2.
    const int v_elems_f16 = D / 32;  // 4 for D=128
    const int v_elems = v_elems_f16 / 2;  // 2 float2 for D=128
    float2 VKQ[8] = {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};

    float KQ_max = -1e30f;
    float KQ_sum = 0.0f;

    // smem layout: 128 floats for KQ weights (32 per warp)
    extern __shared__ float smem[];
    float* KQ_smem = smem + warp * 32;

    // Main loop: each warp processes 32 KV positions per iteration
    const __half* K_warp = k + warp * 32 * kv_stride + kv_head * D;
    const __half* V_warp = v + warp * 32 * kv_stride + kv_head * D;

    for (int kv_base = warp * 32; kv_base < kv_len; kv_base += 128,
         K_warp += 128 * kv_stride, V_warp += 128 * kv_stride) {

        // Phase 1: Compute 32 Q·K scores
        // Each lane ultimately needs the score for KV position (kv_base + lane).
        // Sub-groups of 8 threads compute one dot product together.
        // Lane `lane` participates in computing score for position (lane & ~7) + i_KQ.
        // Lane `lane` keeps the score when i_KQ == (lane % 8).
        float my_score = -1e30f;
        float KQ_max_new = KQ_max;

        #pragma unroll
        for (int i_KQ = 0; i_KQ < FA_NTH_KQ; i_KQ++) {
            int kp_local = (lane & ~(FA_NTH_KQ - 1)) + i_KQ;
            int kp = kv_base + kp_local;

            float dot = 0.0f;
            if (kp < kv_len && kp >= causal_start && kp <= causal_limit) {
                const __half* k_ptr = K_warp + kp_local * kv_stride;
                #pragma unroll
                for (int i = 0; i < q_elems; i++) {
                    int d = q_idx * q_elems + i;
                    dot += Q_reg[i].x * __half2float(k_ptr[2*d]);
                    dot += Q_reg[i].y * __half2float(k_ptr[2*d + 1]);
                }
            } else {
                dot = -1e30f;
            }

            #pragma unroll
            for (int off = FA_NTH_KQ / 2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

            if (softcap > 0.0f && dot > -1e20f) {
                dot = tanhf(dot / softcap) * softcap;
            }

            KQ_max_new = fmaxf(KQ_max_new, dot);

            if ((lane % FA_NTH_KQ) == i_KQ)
                my_score = dot;
        }
        KQ_smem[lane] = my_score;

        // Reduce KQ_max_new across warp
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, off));

        // Online softmax rescale
        float rescale = expf(KQ_max - KQ_max_new);
        KQ_sum *= rescale;
        #pragma unroll
        for (int i = 0; i < v_elems; i++) {
            VKQ[i].x *= rescale;
            VKQ[i].y *= rescale;
        }
        KQ_max = KQ_max_new;

        // Compute exp weights and store in smem
        // Each lane computes weight for its KV position
        {
            float score = KQ_smem[lane];
            float w = (score > -1e20f) ? expf(score - KQ_max) : 0.0f;
            KQ_sum += w;
            KQ_smem[lane] = w;
        }
        __syncwarp();

        // Phase 2: V accumulation
        // Each lane owns D/32 dimensions. Iterate over all 32 KV positions in this warp.
        for (int kp_local = 0; kp_local < 32; kp_local++) {
            int kp = kv_base + kp_local;
            float w = KQ_smem[kp_local];
            if (w > 0.0f && kp < kv_len) {
                const __half* v_ptr = V_warp + kp_local * kv_stride;
                #pragma unroll
                for (int i = 0; i < v_elems; i++) {
                    int d = lane * v_elems + i;
                    VKQ[i].x += w * __half2float(v_ptr[2*d]);
                    VKQ[i].y += w * __half2float(v_ptr[2*d + 1]);
                }
            }
        }
    }

    // Final: combine warps via smem
    // Each warp has its own KQ_max, KQ_sum, VKQ. Need to merge.
    __syncthreads();

    // Reduce KQ_sum within each warp (each lane has 1 weight per 32-KV batch)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        KQ_sum += __shfl_xor_sync(0xFFFFFFFF, KQ_sum, off);

    if (lane == 0) {
        smem[warp] = KQ_max;
        smem[4 + warp] = KQ_sum;
    }
    __syncthreads();

    // Find global max across warps
    float global_max = smem[0];
    for (int w = 1; w < 4; w++) global_max = fmaxf(global_max, smem[w]);

    // Rescale this warp's VKQ to global max
    float warp_rescale = expf(KQ_max - global_max);
    #pragma unroll
    for (int i = 0; i < v_elems; i++) {
        VKQ[i].x *= warp_rescale;
        VKQ[i].y *= warp_rescale;
    }
    float warp_sum_rescaled = KQ_sum * warp_rescale;

    // Cross-warp combine: store VKQ to smem, reduce across warps
    float* vkq_smem = smem + 8;
    for (int i = 0; i < v_elems; i++) {
        int d = lane * v_elems + i;
        vkq_smem[warp * D + 2*d]     = VKQ[i].x;
        vkq_smem[warp * D + 2*d + 1] = VKQ[i].y;
    }
    __syncthreads();

    float global_sum = 0;
    for (int w = 0; w < 4; w++) global_sum += smem[4 + w] * expf(smem[w] - global_max);
    float inv = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    if (warp == 0) {
        __half* out_ptr = out + q_pos * n_heads * D + head * D;
        for (int i = 0; i < v_elems; i++) {
            int d = lane * v_elems + i;
            float sx = 0, sy = 0;
            for (int w = 0; w < 4; w++) {
                sx += vkq_smem[w * D + 2*d];
                sy += vkq_smem[w * D + 2*d + 1];
            }
            out_ptr[2*d]     = __float2half(sx * inv);
            out_ptr[2*d + 1] = __float2half(sy * inv);
        }
    }
}

// Correctness-first MHA fallback for larger head sizes / special paths.
// One block per (head, q_pos). Thread 0 computes the full attention.
__global__ void mha_naive(const __half* __restrict__ q,
                          const __half* __restrict__ k,
                          const __half* __restrict__ v,
                          __half* __restrict__ out,
                          int head_dim,
                          int n_heads,
                          int n_kv_heads,
                          int seq_len,
                          int kv_len,
                          int pos_offset,
                          int window,
                          float scale,
                          float softcap) {
    if (threadIdx.x != 0 || threadIdx.y != 0) {
        return;
    }

    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int causal_limit = pos_offset + q_pos;
    const int causal_start = window > 0 ? ((causal_limit - window + 1) > 0 ? (causal_limit - window + 1) : 0) : 0;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const int D = head_dim;

    const __half * Q_ptr = q + q_pos * n_heads * D + head * D;
    __half * out_ptr = out + q_pos * n_heads * D + head * D;

    extern __shared__ float smem[];
    float * scores = smem;      // kv_len
    float * probs  = smem + kv_len;

    float max_score = -1e30f;
    for (int kp = 0; kp < kv_len; ++kp) {
        float dot = -1e30f;
        if (kp >= causal_start && kp <= causal_limit) {
            const __half * K_ptr = k + kp * kv_stride + kv_head * D;
            dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += __half2float(Q_ptr[d]) * __half2float(K_ptr[d]);
            }
            dot *= scale;
            if (softcap > 0.0f) {
                dot = tanhf(dot / softcap) * softcap;
            }
            max_score = fmaxf(max_score, dot);
        }
        scores[kp] = dot;
    }

    float sum = 0.0f;
    for (int kp = 0; kp < kv_len; ++kp) {
        float p = 0.0f;
        if (scores[kp] > -1e20f) {
            p = expf(scores[kp] - max_score);
            sum += p;
        }
        probs[kp] = p;
    }

    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int kp = 0; kp < kv_len; ++kp) {
            float p = probs[kp] * inv_sum;
            if (p == 0.0f) {
                continue;
            }
            const __half * V_ptr = v + kp * kv_stride + kv_head * D;
            acc += p * __half2float(V_ptr[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// Bidirectional/full-attention fallback for encoder models.
// Same implementation style as mha_naive, but without a causal mask.
__global__ void mha_naive_full(const __half* __restrict__ q,
                               const __half* __restrict__ k,
                               const __half* __restrict__ v,
                               __half* __restrict__ out,
                               int head_dim,
                               int n_heads,
                               int n_kv_heads,
                               int seq_len,
                               int kv_len,
                               int pos_offset,
                               float scale,
                               float softcap) {
    if (threadIdx.x != 0 || threadIdx.y != 0) {
        return;
    }

    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    (void)pos_offset;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const int D = head_dim;

    const __half * Q_ptr = q + q_pos * n_heads * D + head * D;
    __half * out_ptr = out + q_pos * n_heads * D + head * D;

    extern __shared__ float smem[];
    float * scores = smem;
    float * probs  = smem + kv_len;

    float max_score = -3.402823466e+38F;
    for (int kp = 0; kp < kv_len; kp++) {
        const __half * K_ptr = k + kp * kv_stride + kv_head * D;
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            score += __half2float(Q_ptr[d]) * __half2float(K_ptr[d]);
        }
        score *= scale;
        if (softcap > 0.0f) score = tanhf(score / softcap) * softcap;
        scores[kp] = score;
        max_score = fmaxf(max_score, score);
    }

    float sum = 0.0f;
    for (int kp = 0; kp < kv_len; kp++) {
        float p = expf(scores[kp] - max_score);
        probs[kp] = p;
        sum += p;
    }

    float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int kp = 0; kp < kv_len; kp++) {
            const __half * V_ptr = v + kp * kv_stride + kv_head * D;
            acc += (probs[kp] * inv_sum) * __half2float(V_ptr[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// Full bidirectional ESPnet relative-position attention. One block handles
// one (head, query-position) pair. S3Gen sequences are short enough that this
// correctness-first implementation is practical.
__global__ void mha_relative_full(const __half* __restrict__ q,
                                  const __half* __restrict__ k,
                                  const __half* __restrict__ v,
                                  const __half* __restrict__ pos,
                                  const __half* __restrict__ bias_u,
                                  const __half* __restrict__ bias_v,
                                  __half* __restrict__ out,
                                  int head_dim,
                                  int n_heads,
                                  int seq_len) {
    if (threadIdx.x != 0 || threadIdx.y != 0) return;
    const int head = blockIdx.x;
    const int query_pos = blockIdx.y;
    const int width = n_heads * head_dim;
    const int q_base = query_pos * width + head * head_dim;

    extern __shared__ float scratch[];
    float* scores = scratch;
    float* probs = scratch + seq_len;
    const float scale = rsqrtf((float)head_dim);
    float maximum = -3.402823466e+38F;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
        const int k_base = key_pos * width + head * head_dim;
        const int p_base =
            (seq_len - 1 - query_pos + key_pos) * width + head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float qv = __half2float(q[q_base + d]);
            score += (qv + __half2float(bias_u[head * head_dim + d]))
                * __half2float(k[k_base + d]);
            score += (qv + __half2float(bias_v[head * head_dim + d]))
                * __half2float(pos[p_base + d]);
        }
        score *= scale;
        scores[key_pos] = score;
        maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
        probs[key_pos] = expf(scores[key_pos] - maximum);
        denominator += probs[key_pos];
    }
    const float inv_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float value = 0.0f;
        for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
            const int v_index = key_pos * width + head * head_dim + d;
            value += probs[key_pos] * inv_denominator * __half2float(v[v_index]);
        }
        out[q_base + d] = __float2half(value);
    }
}

// --- Naive MHA with an explicit additive attention mask (DiffusionGemma) ---
// Same as mha_naive_full, but adds mask[q_pos * kv_len + kp] to each score
// before the softmax. Mask convention: 0.0 = allowed, large-negative = blocked.
// Used for non-causal region-aware attention where causality cannot be
// expressed as a position window. Grid: (n_heads, seq_len), Block: (1,1).
__global__ void mha_naive_masked(const __half* __restrict__ q,
                                 const __half* __restrict__ k,
                                 const __half* __restrict__ v,
                                 const __half* __restrict__ mask,
                                 __half* __restrict__ out,
                                 int head_dim,
                                 int n_heads,
                                 int n_kv_heads,
                                 int seq_len,
                                 int kv_len,
                                 int pos_offset,
                                 float scale,
                                 float softcap) {
    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    (void)pos_offset;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    const int D = head_dim;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    const __half * Q_ptr = q + q_pos * n_heads * D + head * D;
    __half * out_ptr = out + q_pos * n_heads * D + head * D;
    const __half * mask_row = mask + (long)q_pos * kv_len;

    extern __shared__ float smem[];
    float * scores = smem;          // [kv_len], reused as probs in place
    __shared__ float red[256];      // block reduction

    // scores: each thread handles a strided set of key positions
    for (int kp = tid; kp < kv_len; kp += nt) {
        const __half * K_ptr = k + kp * kv_stride + kv_head * D;
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            score += __half2float(Q_ptr[d]) * __half2float(K_ptr[d]);
        }
        score *= scale;
        if (softcap > 0.0f) score = tanhf(score / softcap) * softcap;
        score += __half2float(mask_row[kp]);
        scores[kp] = score;
    }
    __syncthreads();

    // max-reduction over scores
    float lmax = -3.402823466e+38F;
    for (int kp = tid; kp < kv_len; kp += nt) lmax = fmaxf(lmax, scores[kp]);
    red[tid] = lmax;
    __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) {
        if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
        __syncthreads();
    }
    float max_score = red[0];
    __syncthreads();

    // exp + sum-reduction (write probs back into scores)
    float lsum = 0.0f;
    for (int kp = tid; kp < kv_len; kp += nt) {
        float p = expf(scores[kp] - max_score);
        scores[kp] = p;
        lsum += p;
    }
    red[tid] = lsum;
    __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        __syncthreads();
    }
    float inv_sum = (red[0] > 0.0f) ? 1.0f / red[0] : 0.0f;
    __syncthreads();

    // @V: each thread handles a strided set of output dims
    for (int d = tid; d < D; d += nt) {
        float acc = 0.0f;
        for (int kp = 0; kp < kv_len; kp++) {
            const __half * V_ptr = v + kp * kv_stride + kv_head * D;
            acc += scores[kp] * __half2float(V_ptr[d]);
        }
        out_ptr[d] = __float2half(acc * inv_sum);
    }
}

// Full bidirectional MHA over independent, equally-sized sequences packed
// contiguously along the row dimension. Grid: (heads, seq_len, batches).
__global__ void mha_naive_batched_full(const __half* __restrict__ q,
                                       const __half* __restrict__ k,
                                       const __half* __restrict__ v,
                                       __half* __restrict__ out,
                                       int head_dim,
                                       int n_heads,
                                       int n_kv_heads,
                                       int seq_len,
                                       float scale) {
    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int q_stride = n_heads * head_dim;
    const int kv_stride = n_kv_heads * head_dim;
    const int row_offset = batch * seq_len;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    const __half *Q_ptr =
        q + (row_offset + q_pos) * q_stride + head * head_dim;
    __half *out_ptr =
        out + (row_offset + q_pos) * q_stride + head * head_dim;

    extern __shared__ float scores[];
    __shared__ float red[256];

    // One warp cooperates on each Q·K dot product. This keeps the K loads
    // coalesced; assigning a complete key to one thread made every lane read
    // a different, widely-strided row and dominated the flow runtime.
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = nt >> 5;
    for (int kp = warp; kp < seq_len; kp += warps) {
        const __half *K_ptr =
            k + (row_offset + kp) * kv_stride + kv_head * head_dim;
        float score = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            score += __half2float(Q_ptr[d]) * __half2float(K_ptr[d]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }
        if (lane == 0) scores[kp] = score * scale;
    }
    __syncthreads();

    float lmax = -3.402823466e+38F;
    for (int kp = tid; kp < seq_len; kp += nt) {
        lmax = fmaxf(lmax, scores[kp]);
    }
    red[tid] = lmax;
    __syncthreads();
    for (int offset = nt / 2; offset > 0; offset >>= 1) {
        if (tid < offset) red[tid] = fmaxf(red[tid], red[tid + offset]);
        __syncthreads();
    }
    const float maximum = red[0];

    float local_sum = 0.0f;
    for (int kp = tid; kp < seq_len; kp += nt) {
        const float probability = expf(scores[kp] - maximum);
        scores[kp] = probability;
        local_sum += probability;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int offset = nt / 2; offset > 0; offset >>= 1) {
        if (tid < offset) red[tid] += red[tid + offset];
        __syncthreads();
    }
    const float inverse_sum = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    __syncthreads();

    for (int d = tid; d < head_dim; d += nt) {
        float value = 0.0f;
        for (int kp = 0; kp < seq_len; ++kp) {
            const __half *V_ptr =
                v + (row_offset + kp) * kv_stride + kv_head * head_dim;
            value += scores[kp] * __half2float(V_ptr[d]);
        }
        out_ptr[d] = __float2half(value * inverse_sum);
    }
}

__global__ void attention_pack_qkv_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ q_packed,
    __half* __restrict__ k_packed,
    __half* __restrict__ v_transposed,
    int total_rows,
    int sequence_len,
    int heads,
    int head_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = heads * head_dim;
    const int count = total_rows * width;
    if (index >= count) return;
    const int row = index / width;
    const int feature = index - row * width;
    const int head = feature / head_dim;
    const int dim = feature - head * head_dim;
    const int batch = row / sequence_len;
    const int position = row - batch * sequence_len;
    const int head_batch = batch * heads + head;
    const int packed = (head_batch * sequence_len + position) * head_dim + dim;
    q_packed[packed] = q[index];
    k_packed[packed] = k[index];
    v_transposed[(head_batch * head_dim + dim) * sequence_len + position] = v[index];
}

__global__ void attention_unpack_f16(
    const __half* __restrict__ packed,
    __half* __restrict__ output,
    int total_rows,
    int sequence_len,
    int heads,
    int head_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = heads * head_dim;
    const int count = total_rows * width;
    if (index >= count) return;
    const int row = index / width;
    const int feature = index - row * width;
    const int head = feature / head_dim;
    const int dim = feature - head * head_dim;
    const int batch = row / sequence_len;
    const int position = row - batch * sequence_len;
    const int head_batch = batch * heads + head;
    output[index] =
        packed[(head_batch * sequence_len + position) * head_dim + dim];
}

__global__ void softmax_rows_scaled_f16_inplace(
    __half* __restrict__ values,
    int rows,
    int columns,
    float scale) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    __half *row_values = values + (long)row * columns;
    __shared__ float reduction[256];

    float maximum = -3.402823466e+38F;
    for (int column = tid; column < columns; column += blockDim.x) {
        maximum = fmaxf(maximum, __half2float(row_values[column]) * scale);
    }
    reduction[tid] = maximum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]);
        }
        __syncthreads();
    }
    maximum = reduction[0];

    float sum = 0.0f;
    for (int column = tid; column < columns; column += blockDim.x) {
        const float probability =
            expf(__half2float(row_values[column]) * scale - maximum);
        row_values[column] = __float2half(probability);
        sum += probability;
    }
    reduction[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) reduction[tid] += reduction[tid + offset];
        __syncthreads();
    }
    const float inverse = reduction[0] > 0.0f ? 1.0f / reduction[0] : 0.0f;
    for (int column = tid; column < columns; column += blockDim.x) {
        row_values[column] =
            __float2half(__half2float(row_values[column]) * inverse);
    }
}

// --- Entropy-bound reduce: per canvas position (one block), compute argmax,
// entropy, and an inverse-CDF multinomial sample over the vocab. Reads logits
// directly on-device (no 134MB readback). Grid: (c_len,), Block: (256,) ---
__global__ void eb_reduce(const __half* __restrict__ logits,
                          int vocab,
                          float temp_inv,
                          const float* __restrict__ rnd,
                          unsigned int* __restrict__ argmax,
                          float* __restrict__ entropy,
                          unsigned int* __restrict__ sampled) {
    int pos = blockIdx.x;
    const __half* row = logits + (size_t)pos * vocab;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    __shared__ float s_val[256];
    __shared__ int s_idx[256];

    // pass 1: max(scaled) + argmax
    float lmax = -1e30f;
    int larg = 0;
    for (int v = tid; v < vocab; v += nt) {
        float s = __half2float(row[v]) * temp_inv;
        if (s > lmax) { lmax = s; larg = v; }
    }
    s_val[tid] = lmax; s_idx[tid] = larg;
    __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) {
        if (tid < o && s_val[tid + o] > s_val[tid]) {
            s_val[tid] = s_val[tid + o]; s_idx[tid] = s_idx[tid + o];
        }
        __syncthreads();
    }
    float mx = s_val[0];
    int am = s_idx[0];

    // pass 2: partition sum
    float lsum = 0.0f;
    for (int v = tid; v < vocab; v += nt)
        lsum += __expf(__half2float(row[v]) * temp_inv - mx);
    s_val[tid] = lsum;
    __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) {
        if (tid < o) s_val[tid] += s_val[tid + o];
        __syncthreads();
    }
    float zsum = s_val[0];

    // pass 3: entropy
    float lh = 0.0f;
    for (int v = tid; v < vocab; v += nt) {
        float p = __expf(__half2float(row[v]) * temp_inv - mx) / zsum;
        if (p > 0.0f) lh -= p * __logf(p);
    }
    s_val[tid] = lh;
    __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) {
        if (tid < o) s_val[tid] += s_val[tid + o];
        __syncthreads();
    }

    // pass 4: inverse-CDF multinomial (single thread)
    if (tid == 0) {
        argmax[pos] = (unsigned int)am;
        entropy[pos] = s_val[0];
        float target = rnd[pos] * zsum;
        float cum = 0.0f;
        int tok = vocab - 1;
        for (int v = 0; v < vocab; v++) {
            cum += __expf(__half2float(row[v]) * temp_inv - mx);
            if (cum >= target) { tok = v; break; }
        }
        sampled[pos] = (unsigned int)tok;
    }
}

// --- Gather rows by index: dst[i,:] = src[idx[i],:] (f16) ---
// Grid: (n_rows,), Block: (256,)
__global__ void gather_rows_f16(const __half* __restrict__ src,
                                const int* __restrict__ idx,
                                __half* __restrict__ dst,
                                int dim) {
    int row = blockIdx.x;
    int s = idx[row] * dim;
    int d = row * dim;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        dst[d + j] = src[s + j];
    }
}

// --- Scatter-add rows with per-row weight: dst[idx[i],:] += w[i] * src[i,:] ---
// dst is f32 accumulator. Grid: (n_rows,), Block: (256,). idx within one call
// must be disjoint (one expert's tokens are unique) -> no atomics needed.
__global__ void scatter_add_rows_f16(const __half* __restrict__ src,
                                     const int* __restrict__ idx,
                                     const float* __restrict__ w,
                                     float* __restrict__ dst,
                                     int dim) {
    int row = blockIdx.x;
    int dbase = idx[row] * dim;
    int sbase = row * dim;
    float wt = w[row];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        dst[dbase + j] += wt * __half2float(src[sbase + j]);
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

// Exact sampler for vocabularies up to 3,072 entries such as Qwen3-TTS.
// Each of 256 threads owns at most 12 strided logits. Local candidates are
// sorted in registers; repeated block reductions then merge the exact global
// top-k without downloading logits or approximating the candidate set.
#define SMALL_SAMPLE_K 64
#define SMALL_SAMPLE_LOCAL 12
__global__ void sample_top_k_small(const __half* __restrict__ logits,
                                   int* __restrict__ out,
                                   int vocab_size,
                                   float temperature,
                                   int top_k_param,
                                   unsigned int random_seed) {
    if (blockIdx.x != 0) return;
    const int k = min(max(top_k_param, 1), SMALL_SAMPLE_K);
    float local_values[SMALL_SAMPLE_LOCAL];
    int local_ids[SMALL_SAMPLE_LOCAL];
    for (int i = 0; i < SMALL_SAMPLE_LOCAL; i++) {
        local_values[i] = -1e30f;
        local_ids[i] = 0;
    }
    for (int token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        const float value = __half2float(logits[token]);
        if (value <= local_values[SMALL_SAMPLE_LOCAL - 1]) continue;
        local_values[SMALL_SAMPLE_LOCAL - 1] = value;
        local_ids[SMALL_SAMPLE_LOCAL - 1] = token;
        for (int slot = SMALL_SAMPLE_LOCAL - 2;
             slot >= 0 && local_values[slot + 1] > local_values[slot];
             slot--) {
            const float old_value = local_values[slot];
            local_values[slot] = local_values[slot + 1];
            local_values[slot + 1] = old_value;
            const int old_id = local_ids[slot];
            local_ids[slot] = local_ids[slot + 1];
            local_ids[slot + 1] = old_id;
        }
    }

    __shared__ float reduce_values[256];
    __shared__ int reduce_ids[256];
    __shared__ float selected_values[SMALL_SAMPLE_K];
    __shared__ int selected_ids[SMALL_SAMPLE_K];
    __shared__ int selected_owner;
    int cursor = 0;
    for (int rank = 0; rank < k; rank++) {
        reduce_values[threadIdx.x] =
            cursor < SMALL_SAMPLE_LOCAL ? local_values[cursor] : -1e30f;
        reduce_ids[threadIdx.x] =
            cursor < SMALL_SAMPLE_LOCAL ? local_ids[cursor] : 0;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride &&
                reduce_values[threadIdx.x + stride] > reduce_values[threadIdx.x]) {
                reduce_values[threadIdx.x] = reduce_values[threadIdx.x + stride];
                reduce_ids[threadIdx.x] = reduce_ids[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected_values[rank] = reduce_values[0];
            selected_ids[rank] = reduce_ids[0];
            selected_owner = reduce_ids[0] & 255;
        }
        __syncthreads();
        if ((int)threadIdx.x == selected_owner) cursor++;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float divisor = temperature > 0.0f ? temperature : 1.0f;
        const float maximum = selected_values[0] / divisor;
        float total = 0.0f;
        for (int i = 0; i < k; i++) {
            selected_values[i] = expf(selected_values[i] / divisor - maximum);
            total += selected_values[i];
        }
        unsigned int random = random_seed ? random_seed : 0x9e3779b9u;
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        float threshold = ((float)(random >> 8) / (float)(1 << 24)) * total;
        for (int i = 0; i < k; i++) {
            if (threshold <= selected_values[i]) {
                out[0] = selected_ids[i];
                return;
            }
            threshold -= selected_values[i];
        }
        out[0] = selected_ids[k - 1];
    }
}

// Exact filtered sampler for Qwen3-TTS semantic speech IDs. Speech tokens
// [0, speech_vocab_size) and one EOS token are allowed; previous tokens receive
// the same sign-aware repetition penalty as the reference CPU sampler.
__global__ void sample_top_k_small_filtered(
    const __half* __restrict__ logits,
    const int* __restrict__ previous,
    int* __restrict__ out,
    int vocab_size,
    int speech_vocab_size,
    int eos_token,
    int previous_count,
    float temperature,
    float repetition_penalty,
    int top_k_param,
    unsigned int random_bits) {
    if (blockIdx.x != 0) return;
    const int k = min(max(top_k_param, 1), SMALL_SAMPLE_K);
    float local_values[SMALL_SAMPLE_LOCAL];
    int local_ids[SMALL_SAMPLE_LOCAL];
    for (int i = 0; i < SMALL_SAMPLE_LOCAL; i++) {
        local_values[i] = -1e30f;
        local_ids[i] = 0;
    }
    for (int token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        if (token >= speech_vocab_size && token != eos_token) continue;
        float value = __half2float(logits[token]);
        if (repetition_penalty != 1.0f) {
            bool repeated = false;
            for (int i = 0; i < previous_count; i++) {
                if (previous[i] == token) {
                    repeated = true;
                    break;
                }
            }
            if (repeated) {
                value = value >= 0.0f
                    ? value / repetition_penalty
                    : value * repetition_penalty;
            }
        }
        if (value <= local_values[SMALL_SAMPLE_LOCAL - 1]) continue;
        local_values[SMALL_SAMPLE_LOCAL - 1] = value;
        local_ids[SMALL_SAMPLE_LOCAL - 1] = token;
        for (int slot = SMALL_SAMPLE_LOCAL - 2;
             slot >= 0 && local_values[slot + 1] > local_values[slot];
             slot--) {
            const float old_value = local_values[slot];
            local_values[slot] = local_values[slot + 1];
            local_values[slot + 1] = old_value;
            const int old_id = local_ids[slot];
            local_ids[slot] = local_ids[slot + 1];
            local_ids[slot + 1] = old_id;
        }
    }

    __shared__ float reduce_values[256];
    __shared__ int reduce_ids[256];
    __shared__ float selected_values[SMALL_SAMPLE_K];
    __shared__ int selected_ids[SMALL_SAMPLE_K];
    __shared__ int selected_owner;
    int cursor = 0;
    for (int rank = 0; rank < k; rank++) {
        reduce_values[threadIdx.x] =
            cursor < SMALL_SAMPLE_LOCAL ? local_values[cursor] : -1e30f;
        reduce_ids[threadIdx.x] =
            cursor < SMALL_SAMPLE_LOCAL ? local_ids[cursor] : 0;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride &&
                reduce_values[threadIdx.x + stride] > reduce_values[threadIdx.x]) {
                reduce_values[threadIdx.x] = reduce_values[threadIdx.x + stride];
                reduce_ids[threadIdx.x] = reduce_ids[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected_values[rank] = reduce_values[0];
            selected_ids[rank] = reduce_ids[0];
            selected_owner = reduce_ids[0] & 255;
        }
        __syncthreads();
        if ((int)threadIdx.x == selected_owner) cursor++;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float divisor = fmaxf(temperature, 1e-5f);
        const float maximum = selected_values[0] / divisor;
        float total = 0.0f;
        for (int i = 0; i < k; i++) {
            selected_values[i] = expf(selected_values[i] / divisor - maximum);
            total += selected_values[i];
        }
        float threshold =
            ((float)(random_bits >> 8) / (float)(1 << 24)) * total;
        for (int i = 0; i < k; i++) {
            if (threshold <= selected_values[i]) {
                out[0] = selected_ids[i];
                return;
            }
            threshold -= selected_values[i];
        }
        out[0] = selected_ids[k - 1];
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
// Graph-compatible Flash Attention: same algorithm as mha_fused but reads
// kv_len and pos from device memory (decode_params) for CUDA Graph replay.
// Block: (32, 4), smem: (8 + 4*D) floats.
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
    const int head = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int kv_len = decode_params[1];
    const int pos_offset = decode_params[0];
    const int causal_limit = pos_offset + q_pos;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int kv_stride = n_kv_heads * head_dim;
    const int D = head_dim;

    const __half* Q_ptr = q + q_pos * n_heads * D + head * D;

    const int q_idx = lane % FA_NTH_KQ;
    const int q_elems = D / (2 * FA_NTH_KQ);
    float2 Q_reg[32];
    #pragma unroll
    for (int i = 0; i < q_elems; i++) {
        int d = q_idx * q_elems + i;
        Q_reg[i] = make_float2(
            __half2float(Q_ptr[2*d]) * scale,
            __half2float(Q_ptr[2*d + 1]) * scale);
    }

    const int v_elems = D / (2 * 32);
    float2 VKQ[8] = {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};
    float KQ_max = -1e30f;
    float KQ_sum = 0.0f;

    extern __shared__ float smem[];
    float* KQ_smem = smem + warp * 32;

    const __half* K_warp = k_base + warp * 32 * kv_stride + kv_head * D;
    const __half* V_warp = v_base + warp * 32 * kv_stride + kv_head * D;

    for (int kv_base = warp * 32; kv_base < kv_len; kv_base += 128,
         K_warp += 128 * kv_stride, V_warp += 128 * kv_stride) {

        float my_score = -1e30f;
        float KQ_max_new = KQ_max;

        #pragma unroll
        for (int i_KQ = 0; i_KQ < FA_NTH_KQ; i_KQ++) {
            int kp_local = (lane & ~(FA_NTH_KQ - 1)) + i_KQ;
            int kp = kv_base + kp_local;
            float dot = 0.0f;
            if (kp < kv_len && kp <= causal_limit) {
                const __half* k_ptr = K_warp + kp_local * kv_stride;
                #pragma unroll
                for (int i = 0; i < q_elems; i++) {
                    int d = q_idx * q_elems + i;
                    dot += Q_reg[i].x * __half2float(k_ptr[2*d]);
                    dot += Q_reg[i].y * __half2float(k_ptr[2*d + 1]);
                }
            } else { dot = -1e30f; }
            #pragma unroll
            for (int off = FA_NTH_KQ / 2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            KQ_max_new = fmaxf(KQ_max_new, dot);
            if ((lane % FA_NTH_KQ) == i_KQ) my_score = dot;
        }
        KQ_smem[lane] = my_score;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, off));

        float rescale = expf(KQ_max - KQ_max_new);
        KQ_sum *= rescale;
        #pragma unroll
        for (int i = 0; i < v_elems; i++) { VKQ[i].x *= rescale; VKQ[i].y *= rescale; }
        KQ_max = KQ_max_new;

        float w = (my_score > -1e20f) ? expf(my_score - KQ_max) : 0.0f;
        KQ_sum += w;
        KQ_smem[lane] = w;
        __syncwarp();

        for (int kp_local = 0; kp_local < 32; kp_local++) {
            int kp = kv_base + kp_local;
            float sw = KQ_smem[kp_local];
            if (sw > 0.0f && kp < kv_len) {
                const __half* v_ptr = V_warp + kp_local * kv_stride;
                #pragma unroll
                for (int i = 0; i < v_elems; i++) {
                    int d = lane * v_elems + i;
                    VKQ[i].x += sw * __half2float(v_ptr[2*d]);
                    VKQ[i].y += sw * __half2float(v_ptr[2*d + 1]);
                }
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        KQ_sum += __shfl_xor_sync(0xFFFFFFFF, KQ_sum, off);
    __syncthreads();

    if (lane == 0) { smem[warp] = KQ_max; smem[4 + warp] = KQ_sum; }
    __syncthreads();

    float global_max = smem[0];
    for (int w = 1; w < 4; w++) global_max = fmaxf(global_max, smem[w]);
    float warp_rescale = expf(KQ_max - global_max);
    #pragma unroll
    for (int i = 0; i < v_elems; i++) { VKQ[i].x *= warp_rescale; VKQ[i].y *= warp_rescale; }

    float* vkq_smem = smem + 8;
    for (int i = 0; i < v_elems; i++) {
        int d = lane * v_elems + i;
        vkq_smem[warp * D + 2*d] = VKQ[i].x;
        vkq_smem[warp * D + 2*d + 1] = VKQ[i].y;
    }
    __syncthreads();

    float global_sum = 0;
    for (int w = 0; w < 4; w++) global_sum += smem[4 + w] * expf(smem[w] - global_max);
    float inv = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    if (warp == 0) {
        __half* out_ptr = out + q_pos * n_heads * D + head * D;
        for (int i = 0; i < v_elems; i++) {
            int d = lane * v_elems + i;
            float sx = 0, sy = 0;
            for (int w = 0; w < 4; w++) {
                sx += vkq_smem[w * D + 2*d];
                sy += vkq_smem[w * D + 2*d + 1];
            }
            out_ptr[2*d] = __float2half(sx * inv);
            out_ptr[2*d + 1] = __float2half(sy * inv);
        }
    }
}

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

// --- Batched GELU split for fused expert gate+up output ---
// fused shape: [batch, 2 * expert_ff]
// out   shape: [batch, expert_ff]
__global__ void gelu_split_batch(const __half* __restrict__ fused,
                                 __half* __restrict__ out,
                                 int expert_ff,
                                 int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = expert_ff * batch;
    if (idx >= total) return;

    int row = idx / expert_ff;
    int col = idx - row * expert_ff;
    const __half* row_ptr = fused + row * expert_ff * 2;
    float g = __half2float(row_ptr[col]);
    float u = __half2float(row_ptr[expert_ff + col]);
    float gelu_g = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    out[idx] = __float2half(gelu_g * u);
}

// --- Weighted row sum ---
// rows shape: [batch, dim]
// weights shape: [batch] (f32)
// out shape: [dim]
__global__ void weighted_sum_rows_f16(const __half* __restrict__ rows,
                                      const float* __restrict__ weights,
                                      __half* __restrict__ out,
                                      int dim,
                                      int batch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= dim) return;

    float acc = 0.0f;
    for (int row = 0; row < batch; ++row) {
        acc += __half2float(rows[row * dim + col]) * weights[row];
    }
    out[col] = __float2half(acc);
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

// Reflection-pad one frame on the left of channel-major [channels, frames].
// PyTorch ReflectionPad1d((1, 0)) maps the new first frame to input frame 1.
__global__ void reflection_pad_left_f16(const __half* __restrict__ input,
                                        __half* __restrict__ output,
                                        int channels,
                                        int frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_frames = frames + 1;
    int n = channels * output_frames;
    if (idx >= n) return;
    int channel = idx / output_frames;
    int frame = idx - channel * output_frames;
    int input_frame = frame == 0 ? 1 : frame - 1;
    output[idx] = input[channel * frames + input_frame];
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

// --- Broadcast multiply: out[i] = a[i] * b[i % stride] (f16) ---
// Used for row-wise multiply with a vector: a is [rows, stride], b is [stride]
__global__ void mul_f16_broadcast(const __half* __restrict__ a,
                                   const __half* __restrict__ b,
                                   __half* __restrict__ out,
                                   int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx % stride]));
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

// --- Fused MoE Router: RMS-norm + scale + GEMV + softcap + softmax + top-k ---
// Single kernel replaces 6 separate launches. One thread per expert computes its dot product.
//
// hidden: [dim] f32 input (residual stream)
// gate_scale: [dim] f16 per-element scale
// gate_weights: [n_experts, dim] f16 (row-major, pre-dequanted)
// out_ids: [top_k] int32 selected expert indices
// out_weights: [top_k] float renormalized weights
// dim, n_experts, top_k: scalar params
// eps, inv_sqrt_dim, softcap: float params
//
// Launch: <<<1, n_experts>>> with shared_mem = (dim + n_experts + blockDim.x) * sizeof(float)
__global__ void fused_moe_router(const float* __restrict__ hidden,
                                  const __half* __restrict__ gate_scale,
                                  const __half* __restrict__ gate_weights,
                                  int* __restrict__ out_ids,
                                  float* __restrict__ out_weights,
                                  int dim,
                                  int n_experts,
                                  int top_k,
                                  float eps,
                                  float inv_sqrt_dim,
                                  float softcap) {
    extern __shared__ float sdata[];
    // Layout: [dim] normed input | [n_experts] probs | [blockDim.x] reduce
    float* s_input = sdata;
    float* probs = sdata + dim;
    float* reduce = probs + n_experts;

    int tid = threadIdx.x;

    // Step 1: Cooperative load + RMS norm of hidden into shared memory
    // All threads participate to load dim elements (dim >> n_experts typically)
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = hidden[i];
        local_sum += v * v;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float rms = sqrtf(reduce[0] / (float)dim + eps);
    float scale = inv_sqrt_dim / rms;

    // Apply RMS norm + gate_scale + inv_sqrt_dim, store in shared mem
    for (int i = tid; i < dim; i += blockDim.x) {
        s_input[i] = hidden[i] * scale * __half2float(gate_scale[i]);
    }
    __syncthreads();

    // Step 2: Each thread computes dot product for its expert
    float logit = 0.0f;
    if (tid < n_experts) {
        const __half* row = gate_weights + tid * dim;
        for (int i = 0; i < dim; i++) {
            logit += s_input[i] * __half2float(row[i]);
        }
        if (softcap > 0.0f) {
            logit = tanhf(logit / softcap) * softcap;
        }
    }

    // Step 3: Softmax
    float v = (tid < n_experts) ? logit : -1e30f;
    reduce[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float max_val = reduce[0];

    float exp_v = (tid < n_experts) ? expf(v - max_val) : 0.0f;
    reduce[tid] = exp_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float sum_exp = reduce[0];

    if (tid < n_experts) {
        probs[tid] = exp_v / sum_exp;
    }
    __syncthreads();

    // Step 4: Top-k + renormalize (thread 0)
    if (tid == 0) {
        float renorm_sum = 0.0f;
        for (int ki = 0; ki < top_k && ki < n_experts; ki++) {
            float best = -1.0f;
            int best_idx = 0;
            for (int i = 0; i < n_experts; i++) {
                if (probs[i] > best) {
                    best = probs[i];
                    best_idx = i;
                }
            }
            out_ids[ki] = best_idx;
            out_weights[ki] = best;
            renorm_sum += best;
            probs[best_idx] = -1.0f;
        }
        if (renorm_sum > 0.0f) {
            for (int ki = 0; ki < top_k; ki++) {
                out_weights[ki] /= renorm_sum;
            }
        }
    }
}

// Keep the standalone softmax_topk for potential future use
__global__ void softmax_topk(const __half* __restrict__ logits,
                             int* __restrict__ out_ids,
                             float* __restrict__ out_weights,
                             int n_experts,
                             int top_k,
                             float softcap) {
    extern __shared__ float sdata[];
    float* probs = sdata;
    float* reduce = sdata + n_experts;
    int tid = threadIdx.x;
    float v = (tid < n_experts) ? __half2float(logits[tid]) : -1e30f;
    if (tid < n_experts && softcap > 0.0f) v = tanhf(v / softcap) * softcap;
    reduce[tid] = v; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid+s]); __syncthreads(); }
    float max_val = reduce[0];
    float exp_v = (tid < n_experts) ? expf(v - max_val) : 0.0f;
    reduce[tid] = exp_v; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) reduce[tid] += reduce[tid+s]; __syncthreads(); }
    if (tid < n_experts) probs[tid] = exp_v / reduce[0]; __syncthreads();
    if (tid == 0) {
        float rs = 0.0f;
        for (int ki = 0; ki < top_k; ki++) {
            float best = -1.0f; int bi = 0;
            for (int i = 0; i < n_experts; i++) { if (probs[i] > best) { best = probs[i]; bi = i; } }
            out_ids[ki] = bi; out_weights[ki] = best; rs += best; probs[bi] = -1.0f;
        }
        if (rs > 0.0f) for (int ki = 0; ki < top_k; ki++) out_weights[ki] /= rs;
    }
}

// Causal grouped Conv1d over channel-first data.
// x: [in_channels, seq_len], weight: [out_channels, in_channels/groups, kernel].
__global__ void conv1d_causal_f16(const __half* __restrict__ x,
                                  const __half* __restrict__ weight,
                                  const __half* __restrict__ bias,
                                  __half* __restrict__ out,
                                  int in_channels,
                                  int out_channels,
                                  int seq_len,
                                  int kernel_size,
                                  int dilation,
                                  int groups) {
    // Sequence length can exceed CUDA's 65,535-block Y limit for longer
    // audio. Keep positions on the much larger X grid dimension.
    const int position = blockIdx.x;
    const int out_channel = blockIdx.y;
    if (out_channel >= out_channels || position >= seq_len) return;
    const int channels_per_group = in_channels / groups;
    const int outputs_per_group = out_channels / groups;
    const int group = out_channel / outputs_per_group;
    const int input_start = group * channels_per_group;
    const int work = channels_per_group * kernel_size;
    float sum = 0.0f;
    for (int item = threadIdx.x; item < work; item += blockDim.x) {
        const int local_channel = item / kernel_size;
        const int kernel_index = item % kernel_size;
        const int input_position =
            position - (kernel_size - 1 - kernel_index) * dilation;
        if (input_position >= 0) {
            const int input_channel = input_start + local_channel;
            const int input_index = input_channel * seq_len + input_position;
            const int weight_index =
                (out_channel * channels_per_group + local_channel) * kernel_size
                + kernel_index;
            sum += __half2float(x[input_index]) * __half2float(weight[weight_index]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) {
            out[out_channel * seq_len + position] =
                __float2half(sum + __half2float(bias[out_channel]));
        }
    }
}

__global__ void conv1d_padded_f16(const __half* __restrict__ x,
                                  const __half* __restrict__ weight,
                                  const __half* __restrict__ bias,
                                  __half* __restrict__ out,
                                  int in_channels,
                                  int out_channels,
                                  int seq_len,
                                  int kernel_size,
                                  int left_padding) {
    const int position = blockIdx.x;
    const int out_channel = blockIdx.y;
    float sum = 0.0f;
    const int work = in_channels * kernel_size;
    for (int item = threadIdx.x; item < work; item += blockDim.x) {
        const int input_channel = item / kernel_size;
        const int kernel_index = item % kernel_size;
        const int source = position + kernel_index - left_padding;
        if (source >= 0 && source < seq_len) {
            sum += __half2float(x[input_channel * seq_len + source])
                * __half2float(weight[
                    (out_channel * in_channels + input_channel) * kernel_size
                    + kernel_index]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0)
            out[out_channel * seq_len + position] =
                __float2half(sum + __half2float(bias[out_channel]));
    }
}

// General PyTorch-compatible Conv1d over channel-first f16 data.
// weight: [out_channels, in_channels, kernel_size].
__global__ void conv1d_general_f16(const __half* __restrict__ x,
                                   const __half* __restrict__ weight,
                                   const __half* __restrict__ bias,
                                   __half* __restrict__ out,
                                   int in_channels,
                                   int out_channels,
                                   int input_len,
                                   int output_len,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   int dilation) {
    const int position = blockIdx.x;
    const int out_channel = blockIdx.y;
    if (out_channel >= out_channels || position >= output_len) return;
    float sum = 0.0f;
    const int work = in_channels * kernel_size;
    for (int item = threadIdx.x; item < work; item += blockDim.x) {
        const int input_channel = item / kernel_size;
        const int kernel_index = item - input_channel * kernel_size;
        const int source = position * stride - padding + kernel_index * dilation;
        if (source >= 0 && source < input_len) {
            sum += __half2float(x[input_channel * input_len + source])
                * __half2float(weight[
                    (out_channel * in_channels + input_channel) * kernel_size
                    + kernel_index]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0)
            out[out_channel * output_len + position] =
                __float2half(sum + __half2float(bias[out_channel]));
    }
}

// Streaming causal convolution. x contains history followed by new positions;
// out contains only the new positions.
__global__ void conv1d_causal_offset_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int in_channels,
    int out_channels,
    int input_len,
    int output_len,
    int history_len,
    int kernel_size,
    int dilation,
    int groups) {
    const int position = blockIdx.x;
    const int out_channel = blockIdx.y;
    if (out_channel >= out_channels || position >= output_len) return;
    const int absolute_position = history_len + position;
    const int channels_per_group = in_channels / groups;
    const int outputs_per_group = out_channels / groups;
    const int group = out_channel / outputs_per_group;
    const int input_start = group * channels_per_group;
    const int work = channels_per_group * kernel_size;
    float sum = 0.0f;
    for (int item = threadIdx.x; item < work; item += blockDim.x) {
        const int local_channel = item / kernel_size;
        const int kernel_index = item % kernel_size;
        const int input_position =
            absolute_position - (kernel_size - 1 - kernel_index) * dilation;
        if (input_position >= 0 && input_position < input_len) {
            const int input_channel = input_start + local_channel;
            const int input_index = input_channel * input_len + input_position;
            const int weight_index =
                (out_channel * channels_per_group + local_channel) * kernel_size
                + kernel_index;
            sum += __half2float(x[input_index]) * __half2float(weight[weight_index]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) {
            out[out_channel * output_len + position] =
                __float2half(sum + __half2float(bias[out_channel]));
        }
    }
}

// Unfold channel-first causal Conv1d input into row-major GEMM rows:
// [channels, sequence] -> [sequence, channels * kernel].
__global__ void unfold_causal_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int channels,
    int seq_len,
    int kernel_size,
    int dilation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_width = channels * kernel_size;
    const int n = seq_len * row_width;
    if (index >= n) return;
    const int position = index / row_width;
    const int item = index - position * row_width;
    const int channel = item / kernel_size;
    const int kernel_index = item - channel * kernel_size;
    const int input_position =
        position - (kernel_size - 1 - kernel_index) * dilation;
    out[index] = input_position >= 0
        ? x[channel * seq_len + input_position]
        : __float2half(0.0f);
}

// Batch-aware causal unfold. Sequences are concatenated along the time axis
// in channel-first storage; context never crosses a sequence boundary.
__global__ void unfold_causal_batched_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int channels,
    int total_len,
    int sequence_len,
    int kernel_size,
    int dilation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_width = channels * kernel_size;
    const int n = total_len * row_width;
    if (index >= n) return;
    const int position = index / row_width;
    const int item = index - position * row_width;
    const int channel = item / kernel_size;
    const int kernel_index = item - channel * kernel_size;
    const int sequence_start = (position / sequence_len) * sequence_len;
    const int input_position =
        position - (kernel_size - 1 - kernel_index) * dilation;
    out[index] = input_position >= sequence_start
        ? x[channel * total_len + input_position]
        : __float2half(0.0f);
}

// Unfold a general channel-first Conv1d input into row-major GEMM rows.
// [channels, input_len] -> [output_len, channels * kernel_size].
__global__ void unfold_conv1d_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int channels,
    int input_len,
    int output_len,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_width = channels * kernel_size;
    const int n = output_len * row_width;
    if (index >= n) return;
    const int position = index / row_width;
    const int item = index - position * row_width;
    const int channel = item / kernel_size;
    const int kernel_index = item - channel * kernel_size;
    const int source =
        position * stride - padding + kernel_index * dilation;
    out[index] = source >= 0 && source < input_len
        ? x[channel * input_len + source]
        : __float2half(0.0f);
}

// Gather two adjacent input frames per channel for a polyphase
// ConvTranspose1d GEMM. Output is [input_len, channels * 2].
__global__ void unfold_adjacent_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int channels,
    int input_len,
    int first_offset) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_width = channels * 2;
    const int n = input_len * row_width;
    if (index >= n) return;
    const int position = index / row_width;
    const int item = index - position * row_width;
    const int channel = item / 2;
    const int tap = item - channel * 2;
    const int source = position + first_offset + tap;
    out[index] = source >= 0 && source < input_len
        ? x[channel * input_len + source]
        : __float2half(0.0f);
}

// Scatter one row-major ConvTranspose phase into channel-first output.
__global__ void scatter_conv_transpose_phase_f16(
    const __half* __restrict__ phase_input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int input_len,
    int out_channels,
    int stride,
    int phase) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = input_len * out_channels;
    if (index >= n) return;
    const int position = index / out_channels;
    const int channel = index - position * out_channels;
    const int output_len = input_len * stride;
    output[channel * output_len + position * stride + phase] =
        __hadd(phase_input[index], bias[channel]);
}

// Causal ConvTranspose1d over channel-first data. The untrimmed tail is
// omitted, so output length is exactly input_len * stride.
// weight: [in_channels, out_channels, kernel_size].
__global__ void conv_transpose1d_causal_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int in_channels,
    int out_channels,
    int input_len,
    int kernel_size,
    int stride) {
    const int position = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int output_len = input_len * stride;
    if (out_channel >= out_channels || position >= output_len) return;
    float sum = 0.0f;
    const int phase = position % stride;
    const int kernels_for_phase =
        phase < kernel_size ? 1 + (kernel_size - 1 - phase) / stride : 0;
    const int work = in_channels * kernels_for_phase;
    for (int item = threadIdx.x; item < work; item += blockDim.x) {
        const int input_channel = item / kernels_for_phase;
        const int kernel_index = phase + (item % kernels_for_phase) * stride;
        const int source = position - kernel_index;
        if (source >= 0) {
            const int input_position = source / stride;
            if (input_position < input_len) {
                const int input_index = input_channel * input_len + input_position;
                const int weight_index =
                    (input_channel * out_channels + out_channel) * kernel_size
                    + kernel_index;
                sum += __half2float(x[input_index])
                    * __half2float(weight[weight_index]);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) {
            out[out_channel * output_len + position] =
                __float2half(sum + __half2float(bias[out_channel]));
        }
    }
}

// PyTorch-compatible ConvTranspose1d with dilation=1 and output_padding=0.
// weight: [in_channels, out_channels, kernel_size].
__global__ void conv_transpose1d_general_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int in_channels,
    int out_channels,
    int input_len,
    int output_len,
    int kernel_size,
    int stride,
    int padding) {
    const int position = blockIdx.x;
    const int out_channel = blockIdx.y;
    if (out_channel >= out_channels || position >= output_len) return;
    float sum = 0.0f;
    const int work = in_channels * kernel_size;
    for (int item = threadIdx.x; item < work; item += blockDim.x) {
        const int input_channel = item / kernel_size;
        const int kernel_index = item - input_channel * kernel_size;
        const int numerator = position + padding - kernel_index;
        if (numerator >= 0 && numerator % stride == 0) {
            const int source = numerator / stride;
            if (source < input_len) {
                sum += __half2float(x[input_channel * input_len + source])
                    * __half2float(weight[
                        (input_channel * out_channels + out_channel) * kernel_size
                        + kernel_index]);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0)
            out[out_channel * output_len + position] =
                __float2half(sum + __half2float(bias[out_channel]));
    }
}

// Transpose a row-major [rows, cols] f16 matrix.
__global__ void transpose_f16(const __half* __restrict__ x,
                              __half* __restrict__ out,
                              int rows,
                              int cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = rows * cols;
    if (index < n) {
        const int row = index / cols;
        const int col = index % cols;
        out[col * rows + row] = x[index];
    }
}

// Exact erf-based GELU used by the codec's ConvNeXt blocks.
__global__ void gelu_erf_f16(const __half* __restrict__ x,
                             __half* __restrict__ out,
                             int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(x[index]);
        out[index] = __float2half(
            0.5f * value * (1.0f + erff(value * 0.7071067811865475f)));
    }
}

__global__ void silu_act_f16(const __half* __restrict__ x,
                             __half* __restrict__ out,
                             int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(x[index]);
        out[index] = __float2half(value / (1.0f + expf(-value)));
    }
}

__global__ void leaky_relu_f16(const __half* __restrict__ x,
                               __half* __restrict__ out,
                               int n,
                               float negative_slope) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(x[index]);
        out[index] = __float2half(value >= 0.0f ? value : value * negative_slope);
    }
}

__global__ void elu_f16(const __half* __restrict__ x,
                        __half* __restrict__ out,
                        int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(x[index]);
        out[index] = __float2half(value >= 0.0f ? value : expm1f(value));
    }
}

// One PyTorch-compatible LSTM cell. Gate order is input, forget, cell, output.
__global__ void lstm_cell_f16(const __half* __restrict__ input_gates,
                              const __half* __restrict__ hidden_gates,
                              const __half* __restrict__ bias_ih,
                              const __half* __restrict__ bias_hh,
                              __half* __restrict__ hidden,
                              float* __restrict__ cell,
                              __half* __restrict__ sequence_output,
                              int hidden_size,
                              int timestep,
                              int output_timestep) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden_size) return;
    float gates[4];
    #pragma unroll
    for (int gate = 0; gate < 4; ++gate) {
        const int index = gate * hidden_size + channel;
        gates[gate] = __half2float(input_gates[
            timestep * 4 * hidden_size + index])
            + __half2float(hidden_gates[index])
            + __half2float(bias_ih[index])
            + __half2float(bias_hh[index]);
    }
    const float input = 1.0f / (1.0f + expf(-gates[0]));
    const float forget = 1.0f / (1.0f + expf(-gates[1]));
    const float candidate = tanhf(gates[2]);
    const float output = 1.0f / (1.0f + expf(-gates[3]));
    const float new_cell = forget * cell[channel] + input * candidate;
    const float new_hidden = output * tanhf(new_cell);
    cell[channel] = new_cell;
    hidden[channel] = __float2half(new_hidden);
    sequence_output[output_timestep * hidden_size + channel] =
        __float2half(new_hidden);
}

// InstanceNorm1d followed by Kokoro's style affine:
// (1 + gamma[channel]) * normalized + beta[channel].
__global__ void instance_norm_affine_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ out,
    int channels,
    int frames,
    float epsilon) {
    const int channel = blockIdx.x;
    if (channel >= channels) return;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int frame = threadIdx.x; frame < frames; frame += blockDim.x) {
        const float value = __half2float(x[channel * frames + frame]);
        sum += value;
        sum_sq += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    __shared__ float warp_sum[8];
    __shared__ float warp_sq[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sum[warp] = sum;
        warp_sq[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sum[lane] : 0.0f;
        sum_sq = lane < 8 ? warp_sq[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (lane == 0) {
            warp_sum[0] = sum / frames;
            const float variance = fmaxf(0.0f, sum_sq / frames - warp_sum[0] * warp_sum[0]);
            warp_sq[0] = rsqrtf(variance + epsilon);
        }
    }
    __syncthreads();
    const float mean = warp_sum[0];
    const float inverse = warp_sq[0];
    const float scale = 1.0f + __half2float(gamma[channel]);
    const float shift = __half2float(beta[channel]);
    for (int frame = threadIdx.x; frame < frames; frame += blockDim.x) {
        const int index = channel * frames + frame;
        out[index] = __float2half(
            scale * (__half2float(x[index]) - mean) * inverse + shift);
    }
}

// Depthwise ConvTranspose1d used by Kokoro's learned 2x AdaIN upsamplers.
__global__ void conv_transpose1d_depthwise_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out,
    int channels,
    int input_len,
    int output_len,
    int kernel_size,
    int stride,
    int padding) {
    const int position = blockIdx.x;
    const int channel = blockIdx.y;
    if (channel >= channels || position >= output_len) return;
    float sum = 0.0f;
    for (int kernel_index = threadIdx.x;
         kernel_index < kernel_size;
         kernel_index += blockDim.x) {
        const int numerator = position + padding - kernel_index;
        if (numerator >= 0 && numerator % stride == 0) {
            const int source = numerator / stride;
            if (source < input_len) {
                sum += __half2float(x[channel * input_len + source])
                    * __half2float(weight[channel * kernel_size + kernel_index]);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (threadIdx.x == 0)
        out[channel * output_len + position] =
            __float2half(sum + __half2float(bias[channel]));
}

__global__ void mish_f16(const __half* __restrict__ x,
                         __half* __restrict__ out,
                         int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(x[index]);
        // Stable softplus for large positive inputs.
        const float softplus = value > 20.0f ? value : log1pf(expf(value));
        out[index] = __float2half(value * tanhf(softplus));
    }
}

__global__ void repeat_interleave_f16(const __half* __restrict__ x,
                                      __half* __restrict__ out,
                                      int channels,
                                      int seq_len,
                                      int repeats) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_len = seq_len * repeats;
    const int n = channels * output_len;
    if (index < n) {
        const int channel = index / output_len;
        const int position = index % output_len;
        out[index] = x[channel * seq_len + position / repeats];
    }
}

__global__ void concat_f32_f16_rows(const float* __restrict__ left,
                                    const __half* __restrict__ right,
                                    float* __restrict__ out,
                                    int rows,
                                    int left_dim,
                                    int right_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = left_dim + right_dim;
    if (index < rows * width) {
        const int row = index / width;
        const int col = index % width;
        out[index] = col < left_dim
            ? left[row * left_dim + col]
            : __half2float(right[row * right_dim + col - left_dim]);
    }
}

// Channel-wise SnakeBeta activation over channel-first data.
__global__ void snake_beta_f16(const __half* __restrict__ x,
                               const __half* __restrict__ alpha,
                               const __half* __restrict__ beta,
                               __half* __restrict__ out,
                               int channels,
                               int seq_len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = channels * seq_len;
    if (index < n) {
        const int channel = index / seq_len;
        const float value = __half2float(x[index]);
        const float frequency = expf(__half2float(alpha[channel]));
        const float magnitude = expf(__half2float(beta[channel])) + 1e-9f;
        const float periodic = sinf(frequency * value);
        out[index] = __float2half(value + periodic * periodic / magnitude);
    }
}

// Channel-wise Snake activation with linear (not log-scale) alpha.
__global__ void snake_f16(const __half* __restrict__ x,
                          const __half* __restrict__ alpha,
                          __half* __restrict__ out,
                          int channels,
                          int seq_len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = channels * seq_len;
    if (index < n) {
        const int channel = index / seq_len;
        const float value = __half2float(x[index]);
        const float frequency = __half2float(alpha[channel]);
        const float periodic = sinf(frequency * value);
        out[index] = __float2half(
            value + periodic * periodic / (frequency + 1e-9f));
    }
}

__global__ void clamp_f16(const __half* __restrict__ x,
                          __half* __restrict__ out,
                          int n,
                          float minimum,
                          float maximum) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(x[index]);
        out[index] = __float2half(fminf(maximum, fmaxf(minimum, value)));
    }
}

// --- Qwen BF16 primitives -------------------------------------------------

__global__ void rms_norm_f32in_bf16(const float* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ weight,
                                     __nv_bfloat16* __restrict__ out,
                                     int dim,
                                     float eps) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    __nv_bfloat16* out_row = out + row * dim;
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
        out_row[i] = __float2bfloat16(
            x_row[i] / rms * __bfloat162float(weight[i]));
    }
}

__global__ void add_bias_bf16_inplace(__nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ bias,
                                       int dim) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    int idx = row * dim + col;
    x[idx] = __float2bfloat16(
        __bfloat162float(x[idx]) + __bfloat162float(bias[col]));
}

__global__ void copy_bf16_to_f32(const __nv_bfloat16* __restrict__ src,
                                  float* __restrict__ dst,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __bfloat162float(src[idx]);
}

__global__ void copy_bf16_to_f16(const __nv_bfloat16* __restrict__ src,
                                  __half* __restrict__ dst,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(__bfloat162float(src[idx]));
}

__global__ void copy_f16_to_bf16(const __half* __restrict__ src,
                                  __nv_bfloat16* __restrict__ dst,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(__half2float(src[idx]));
}

__global__ void add_inplace_f32_bf16(float* __restrict__ hidden,
                                      const __nv_bfloat16* __restrict__ delta,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) hidden[idx] += __bfloat162float(delta[idx]);
}

__global__ void silu_bf16(const __nv_bfloat16* __restrict__ gate,
                           const __nv_bfloat16* __restrict__ up,
                           __nv_bfloat16* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    out[idx] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

__global__ void silu_act_bf16(const __nv_bfloat16* __restrict__ x,
                              __nv_bfloat16* __restrict__ out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float value = __bfloat162float(x[idx]);
    out[idx] = __float2bfloat16(value / (1.0f + expf(-value)));
}

__global__ void gather_rows_bf16(const __nv_bfloat16* __restrict__ src,
                                  const int* __restrict__ idx,
                                  __nv_bfloat16* __restrict__ dst,
                                  int dim) {
    int row = blockIdx.x;
    int s = idx[row] * dim;
    int d = row * dim;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        dst[d + j] = src[s + j];
    }
}

} // extern "C"
