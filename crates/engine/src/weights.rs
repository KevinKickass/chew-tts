use crate::config::ModelConfig;
use chew_gguf::{GgmlType, GgufFile, GgufError};
use chew_vram::VramAllocator;
use cudarc::driver::CudaSlice;
use chew_kernel::{DequantKernels, KernelError};
use tracing::info;

/// A weight tensor stored quantized on GPU.
/// Only dequantized on-the-fly during GEMM — saves massive VRAM.
pub struct QuantWeight {
    /// Raw quantized bytes on GPU
    pub data: CudaSlice<u8>,
    /// Quantization format
    pub quant_type: GgmlType,
    /// Number of logical elements (e.g. rows * cols)
    pub n_elements: u32,
}

/// All model weights living on GPU.
/// Large matrices (GEMM operands) stay quantized.
/// Small vectors (norms, embeddings) are dequantized to f16.
pub struct ModelWeights {
    /// Token embeddings: [vocab_size, dim] — f16 (needed for embed_tokens kernel)
    pub token_embd: CudaSlice<half::f16>,
    /// Per-layer weights
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight: [dim] — f16
    pub output_norm: CudaSlice<half::f16>,
    /// Output projection (lm_head): [vocab_size, dim] — quantized
    pub output: QuantWeight,
}

/// Weights for one transformer layer.
pub struct LayerWeights {
    /// Attention input norm: [dim] — f16 (small)
    pub attn_norm: CudaSlice<half::f16>,
    /// Q projection: [n_heads * head_dim, dim] — quantized
    pub attn_q: QuantWeight,
    /// K projection: [n_kv_heads * head_dim, dim] — quantized
    pub attn_k: QuantWeight,
    /// V projection: [n_kv_heads * head_dim, dim] — quantized
    pub attn_v: QuantWeight,
    /// Output projection: [dim, n_heads * head_dim] — quantized
    pub attn_output: QuantWeight,
    /// FFN input norm: [dim] — f16 (small)
    pub ffn_norm: CudaSlice<half::f16>,
    /// FFN gate: [ff_dim, dim] — quantized
    pub ffn_gate: QuantWeight,
    /// FFN up: [ff_dim, dim] — quantized
    pub ffn_up: QuantWeight,
    /// FFN down: [dim, ff_dim] — quantized
    pub ffn_down: QuantWeight,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("GGUF: {0}")]
    Gguf(#[from] GgufError),
    #[error("VRAM: {0}")]
    Vram(#[from] chew_vram::VramError),
    #[error("kernel: {0}")]
    Kernel(#[from] KernelError),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
}

impl ModelWeights {
    /// Load a model from GGUF into GPU memory.
    ///
    /// Large weight matrices stay quantized on GPU (only ~2.3 GB for IQ2 model).
    /// Small vectors (norms) are dequantized to f16.
    pub fn load(
        gguf: &GgufFile,
        config: &ModelConfig,
        alloc: &VramAllocator,
        dequant: &DequantKernels,
        gpu_idx: usize,
    ) -> Result<Self, LoadError> {
        info!(
            arch = %config.arch,
            layers = config.n_layers,
            dim = config.dim,
            "loading model weights to GPU"
        );

        // Embeddings must be f16 for the embed_tokens kernel
        let token_embd = upload_and_dequant(gguf, "token_embd.weight", alloc, dequant, gpu_idx)?;
        let output_norm =
            upload_and_dequant(gguf, "output_norm.weight", alloc, dequant, gpu_idx)?;

        // Output projection: quantized (large matrix)
        let output = if gguf.find_tensor("output.weight").is_some() {
            upload_quantized(gguf, "output.weight", alloc, gpu_idx)?
        } else {
            info!("output.weight not found, using tied embeddings as f16 fallback");
            // Tied embeddings — keep as quantized from token_embd source
            upload_quantized(gguf, "token_embd.weight", alloc, gpu_idx)?
        };

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers {
            let pfx = format!("blk.{i}");
            info!(layer = i, "loading layer weights");

            let layer = LayerWeights {
                // Norms are tiny — dequant to f16
                attn_norm: upload_and_dequant(
                    gguf, &format!("{pfx}.attn_norm.weight"), alloc, dequant, gpu_idx,
                )?,
                ffn_norm: upload_and_dequant(
                    gguf, &format!("{pfx}.ffn_norm.weight"), alloc, dequant, gpu_idx,
                )?,
                // Large matrices — keep quantized
                attn_q: upload_quantized(gguf, &format!("{pfx}.attn_q.weight"), alloc, gpu_idx)?,
                attn_k: upload_quantized(gguf, &format!("{pfx}.attn_k.weight"), alloc, gpu_idx)?,
                attn_v: upload_quantized(gguf, &format!("{pfx}.attn_v.weight"), alloc, gpu_idx)?,
                attn_output: upload_quantized(gguf, &format!("{pfx}.attn_output.weight"), alloc, gpu_idx)?,
                ffn_gate: upload_quantized(gguf, &format!("{pfx}.ffn_gate.weight"), alloc, gpu_idx)?,
                ffn_up: upload_quantized(gguf, &format!("{pfx}.ffn_up.weight"), alloc, gpu_idx)?,
                ffn_down: upload_quantized(gguf, &format!("{pfx}.ffn_down.weight"), alloc, gpu_idx)?,
            };
            layers.push(layer);
        }

        info!("all weights loaded to GPU");

        Ok(Self {
            token_embd,
            layers,
            output_norm,
            output,
        })
    }
}

/// Upload raw quantized tensor bytes to GPU — no dequantization.
fn upload_quantized(
    gguf: &GgufFile,
    name: &str,
    alloc: &VramAllocator,
    gpu_idx: usize,
) -> Result<QuantWeight, LoadError> {
    let (tensor_info, host_data) = gguf
        .tensor_data_by_name(name)
        .map_err(|_| LoadError::MissingTensor(name.to_string()))?;

    let stream = alloc.stream(gpu_idx);

    let mut gpu_buf = stream
        .alloc_zeros::<u8>(host_data.len())
        .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
    stream
        .memcpy_htod(host_data, &mut gpu_buf)
        .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;

    Ok(QuantWeight {
        data: gpu_buf,
        quant_type: tensor_info.ggml_type,
        n_elements: tensor_info.n_elements() as u32,
    })
}

/// Upload raw quantized tensor data to GPU and dequantize directly to f16.
/// Used for small tensors (norms, embeddings) that need f16 for kernels.
/// Dequant kernels output f16 directly, so no extra conversion needed.
fn upload_and_dequant(
    gguf: &GgufFile,
    name: &str,
    alloc: &VramAllocator,
    dequant: &DequantKernels,
    gpu_idx: usize,
) -> Result<CudaSlice<half::f16>, LoadError> {
    let (tensor_info, host_data) = gguf
        .tensor_data_by_name(name)
        .map_err(|_| LoadError::MissingTensor(name.to_string()))?;

    let n_elements = tensor_info.n_elements() as usize;
    let stream = alloc.stream(gpu_idx);

    // 1. Upload raw quantized bytes to GPU
    let mut src_gpu = stream
        .alloc_zeros::<u8>(host_data.len())
        .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
    stream
        .memcpy_htod(host_data, &mut src_gpu)
        .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;

    // 2. Dequantize directly to f16 output
    let mut dst_gpu = stream
        .alloc_zeros::<half::f16>(n_elements)
        .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
    dequant.dequant(&src_gpu, &mut dst_gpu, n_elements as u32, tensor_info.ggml_type)?;
    // src_gpu dropped here, freeing quantized bytes

    Ok(dst_gpu)
}
