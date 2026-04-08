// chew-kernel: CUDA kernels for inference
//
// Planned:
// - Dequantization (Q4_K, Q6_K, IQ1_S, IQ2_XXS, etc.) → f16 on GPU
// - RMSNorm
// - RoPE (rotary position embeddings)
// - SiLU activation
// - Softmax
// - GEMM via cuBLAS (not a kernel, but wrapped here)
// - Fused attention (eventually)

pub struct KernelContext;
