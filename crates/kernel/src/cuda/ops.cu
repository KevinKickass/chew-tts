// Transformer operation kernels
//
// Mixed precision: hidden state stays f32 (residual stream), intermediate
// buffers are f16 for VRAM efficiency. Bridge ops convert between them.
// KV cache is f16. Weight/embedding tables are f16.

#include <cuda_fp16.h>

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
__global__ void mha_fused(const __half* __restrict__ q,       // f16
                           const __half* __restrict__ k,      // f16 KV cache
                           const __half* __restrict__ v,      // f16 KV cache
                           __half* __restrict__ out,           // f16
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

    int kv_head = head / (n_heads / n_kv_heads);  // GQA mapping

    // Q pointer for this (query_pos, head) — f16
    const __half* q_vec = q + q_pos_local * n_heads * head_dim + head * head_dim;

    // Shared memory: first kv_len floats for scores, then blockDim.x for reduction
    extern __shared__ float smem[];
    float* scores = smem;
    float* scratch = smem + kv_len;

    // --- Step 1: Compute attention scores ---
    for (int kp = 0; kp < kv_len; kp++) {
        if (kp > q_pos_global) {
            // Causal mask
            if (threadIdx.x == 0) scores[kp] = -1e30f;
            continue;
        }

        // K pointer for this (kv_pos, kv_head) — f16 from KV cache
        const __half* k_vec = k + kp * n_kv_heads * head_dim + kv_head * head_dim;

        // Dot product with reduction: f16 Q dot f16 K (accumulate in f32)
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        scratch[threadIdx.x] = dot;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                scratch[threadIdx.x] += scratch[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            scores[kp] = scratch[0] * scale;
        }
        __syncthreads();
    }

    // --- Step 2: Softmax over scores ---
    if (threadIdx.x == 0) {
        float max_val = -1e30f;
        for (int kp = 0; kp < kv_len; kp++) {
            if (scores[kp] > max_val) max_val = scores[kp];
        }
        scratch[0] = max_val;
    }
    __syncthreads();
    float max_val = scratch[0];

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int kp = 0; kp < kv_len; kp++) {
            float e = expf(scores[kp] - max_val);
            scores[kp] = e;
            sum += e;
        }
        for (int kp = 0; kp < kv_len; kp++) {
            scores[kp] /= sum;
        }
    }
    __syncthreads();

    // --- Step 3: Weighted sum of V (f16 from KV cache) → f16 output ---
    __half* out_vec = out + q_pos_local * n_heads * head_dim + head * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kp = 0; kp < kv_len; kp++) {
            const __half* v_vec = v + kp * n_kv_heads * head_dim + kv_head * head_dim;
            acc += scores[kp] * __half2float(v_vec[d]);
        }
        out_vec[d] = __float2half(acc);
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
__global__ void mha_fused_graph(
    const __half* __restrict__ q,
    const __half* __restrict__ k_base,     // Full KV cache K base pointer
    const __half* __restrict__ v_base,     // Full KV cache V base pointer
    __half* __restrict__ out,
    const int* __restrict__ decode_params, // [pos, kv_len, kv_offset]
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    float scale) {
    int head = blockIdx.x;
    int q_pos_local = blockIdx.y;

    int kv_len = decode_params[1];
    int pos_offset = decode_params[0]; // For decode, pos_offset = pos

    int q_pos_global = pos_offset + q_pos_local;
    int kv_head = head / (n_heads / n_kv_heads);

    const __half* q_vec = q + q_pos_local * n_heads * head_dim + head * head_dim;

    extern __shared__ float smem[];
    float* scores = smem;
    float* scratch = smem + kv_len;

    // --- Step 1: Compute attention scores ---
    for (int kp = 0; kp < kv_len; kp++) {
        if (kp > q_pos_global) {
            if (threadIdx.x == 0) scores[kp] = -1e30f;
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
        if (threadIdx.x == 0) scores[kp] = scratch[0] * scale;
        __syncthreads();
    }

    // --- Step 2: Softmax ---
    if (threadIdx.x == 0) {
        float max_val = -1e30f;
        for (int kp = 0; kp < kv_len; kp++)
            if (scores[kp] > max_val) max_val = scores[kp];
        scratch[0] = max_val;
    }
    __syncthreads();
    float max_val = scratch[0];

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int kp = 0; kp < kv_len; kp++) {
            float e = expf(scores[kp] - max_val);
            scores[kp] = e;
            sum += e;
        }
        for (int kp = 0; kp < kv_len; kp++) scores[kp] /= sum;
    }
    __syncthreads();

    // --- Step 3: Weighted sum of V ---
    __half* out_vec = out + q_pos_local * n_heads * head_dim + head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kp = 0; kp < kv_len; kp++) {
            const __half* v_vec = v_base + kp * n_kv_heads * head_dim + kv_head * head_dim;
            acc += scores[kp] * __half2float(v_vec[d]);
        }
        out_vec[d] = __float2half(acc);
    }
}

} // extern "C"
