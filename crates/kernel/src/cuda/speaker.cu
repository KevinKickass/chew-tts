#include <cuda_fp16.h>

extern "C" {

__global__ void unfold_reflect_f16(const __half* __restrict__ input,
                                   __half* __restrict__ output,
                                   int channels,
                                   int seq_len,
                                   int kernel_size,
                                   int dilation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = channels * kernel_size;
    const int total = seq_len * width;
    if (index >= total) return;
    const int position = index / width;
    const int item = index % width;
    const int channel = item / kernel_size;
    const int tap = item % kernel_size;
    const int padding = dilation * (kernel_size - 1) / 2;
    int source = position + tap * dilation - padding;
    if (seq_len > 1) {
        while (source < 0 || source >= seq_len) {
            source = source < 0 ? -source : 2 * seq_len - 2 - source;
        }
    } else {
        source = 0;
    }
    output[index] = input[channel * seq_len + source];
}

__global__ void relu_f16(__half* values, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) values[index] = __float2half(fmaxf(0.0f, __half2float(values[index])));
}

__global__ void tanh_f16(__half* values, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) values[index] = __float2half(tanhf(__half2float(values[index])));
}

__global__ void sigmoid_f16(__half* values, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float value = __half2float(values[index]);
        values[index] = __float2half(1.0f / (1.0f + expf(-value)));
    }
}

__global__ void channel_mean_f16(const __half* __restrict__ input,
                                 __half* __restrict__ mean,
                                 int channels,
                                 int seq_len) {
    const int channel = blockIdx.x;
    if (channel >= channels) return;
    float sum = 0.0f;
    for (int position = threadIdx.x; position < seq_len; position += blockDim.x) {
        sum += __half2float(input[channel * seq_len + position]);
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
        if (lane == 0) mean[channel] = __float2half(sum / seq_len);
    }
}

__global__ void channel_scale_f16(const __half* __restrict__ input,
                                  const __half* __restrict__ scale,
                                  __half* __restrict__ output,
                                  int channels,
                                  int seq_len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = channels * seq_len;
    if (index < total) {
        output[index] = __hmul(input[index], scale[index / seq_len]);
    }
}

__global__ void append_channel_block_f16(const __half* __restrict__ input,
                                         __half* __restrict__ output,
                                         int input_channels,
                                         int output_channels,
                                         int channel_offset,
                                         int seq_len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = input_channels * seq_len;
    if (index < total) {
        const int channel = index / seq_len;
        const int position = index % seq_len;
        output[(channel_offset + channel) * seq_len + position] = input[index];
    }
}

__global__ void append_context_f16(const __half* __restrict__ input,
                                   const __half* __restrict__ mean,
                                   const __half* __restrict__ stddev,
                                   __half* __restrict__ output,
                                   int channels,
                                   int seq_len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = channels * seq_len;
    if (index < total) {
        const int channel = index / seq_len;
        const int position = index % seq_len;
        output[channel * seq_len + position] = input[index];
        output[(channels + channel) * seq_len + position] = mean[channel];
        output[(2 * channels + channel) * seq_len + position] = stddev[channel];
    }
}

__global__ void channel_stats_f16(const __half* __restrict__ input,
                                  const __half* __restrict__ weights,
                                  __half* __restrict__ mean,
                                  __half* __restrict__ stddev,
                                  int channels,
                                  int seq_len) {
    const int channel = blockIdx.x;
    if (channel >= channels) return;
    float local_mean = 0.0f;
    for (int position = threadIdx.x; position < seq_len; position += blockDim.x) {
        const float value = __half2float(input[channel * seq_len + position]);
        const float weight = weights
            ? __half2float(weights[channel * seq_len + position])
            : 1.0f / seq_len;
        local_mean += weight * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_mean += __shfl_down_sync(0xFFFFFFFF, local_mean, offset);
    }
    __shared__ float mean_sums[8];
    __shared__ float shared_mean;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) mean_sums[warp] = local_mean;
    __syncthreads();
    if (warp == 0) {
        local_mean = lane < 8 ? mean_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_mean += __shfl_down_sync(0xFFFFFFFF, local_mean, offset);
        }
        if (lane == 0) {
            const __half rounded_mean = __float2half(local_mean);
            mean[channel] = rounded_mean;
            shared_mean = __half2float(rounded_mean);
        }
    }
    __syncthreads();
    float variance = 0.0f;
    for (int position = threadIdx.x; position < seq_len; position += blockDim.x) {
        const float value = __half2float(input[channel * seq_len + position]);
        const float weight = weights
            ? __half2float(weights[channel * seq_len + position])
            : 1.0f / seq_len;
        const float delta = value - shared_mean;
        variance += weight * delta * delta;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        variance += __shfl_down_sync(0xFFFFFFFF, variance, offset);
    }
    __shared__ float variance_sums[8];
    if (lane == 0) variance_sums[warp] = variance;
    __syncthreads();
    if (warp == 0) {
        variance = lane < 8 ? variance_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            variance += __shfl_down_sync(0xFFFFFFFF, variance, offset);
        }
        if (lane == 0) stddev[channel] = __float2half(sqrtf(fmaxf(1e-12f, variance)));
    }
}

__global__ void softmax_channels_f16(__half* values, int channels, int seq_len) {
    const int channel = blockIdx.x;
    if (channel >= channels) return;
    __half* row = values + channel * seq_len;
    float maximum = -3.402823466e+38F;
    for (int position = threadIdx.x; position < seq_len; position += blockDim.x) {
        maximum = fmaxf(maximum, __half2float(row[position]));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xFFFFFFFF, maximum, offset));
    }
    __shared__ float warp_values[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 8 ? warp_values[lane] : -3.402823466e+38F;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xFFFFFFFF, maximum, offset));
        }
        if (lane == 0) warp_values[0] = maximum;
    }
    __syncthreads();
    maximum = warp_values[0];
    float sum = 0.0f;
    for (int position = threadIdx.x; position < seq_len; position += blockDim.x) {
        sum += expf(__half2float(row[position]) - maximum);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane == 0) warp_values[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_values[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) warp_values[0] = sum;
    }
    __syncthreads();
    sum = warp_values[0];
    for (int position = threadIdx.x; position < seq_len; position += blockDim.x) {
        row[position] = __float2half(expf(__half2float(row[position]) - maximum) / sum);
    }
}

} // extern "C"
