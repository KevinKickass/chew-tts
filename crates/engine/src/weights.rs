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
    // Gemma 4 per-layer embedding weights (shared across layers)
    /// per_layer_token_embd: [n_embd_per_layer * n_layers, vocab_size] — quantized
    pub per_layer_token_embd: Option<QuantWeight>,
    /// per_layer_model_proj: [n_embd_per_layer * n_layers, dim] — quantized
    pub per_layer_model_proj: Option<QuantWeight>,
    /// per_layer_proj_norm: [n_embd_per_layer] — f16
    pub per_layer_proj_norm: Option<CudaSlice<half::f16>>,
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

    // === Gemma 4 specific (all Optional) ===
    /// QK norm weights: [head_dim] — f16
    pub attn_q_norm: Option<CudaSlice<half::f16>>,
    pub attn_k_norm: Option<CudaSlice<half::f16>>,
    /// Post-attention norm: [dim] — f16
    pub post_attention_norm: Option<CudaSlice<half::f16>>,
    /// Post-FFN norm: [dim] — f16
    pub post_ffw_norm: Option<CudaSlice<half::f16>>,
    /// Post-norm (after everything): [dim] — f16
    pub post_norm: Option<CudaSlice<half::f16>>,
    /// Per-layer input gate: [dim, embd_per_layer] — quantized
    pub inp_gate: Option<QuantWeight>,
    /// Per-layer projection: [embd_per_layer, dim] — quantized
    pub proj: Option<QuantWeight>,
    /// Layer output scale: scalar f32
    pub layer_output_scale: Option<f32>,
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

        let is_gemma4 = config.is_gemma4();

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

        // Gemma 4 per-layer embedding weights
        let per_layer_token_embd = if is_gemma4 && gguf.find_tensor("per_layer_token_embd.weight").is_some() {
            Some(upload_quantized(gguf, "per_layer_token_embd.weight", alloc, gpu_idx)?)
        } else {
            None
        };
        let per_layer_model_proj = if is_gemma4 && gguf.find_tensor("per_layer_model_proj.weight").is_some() {
            Some(upload_quantized(gguf, "per_layer_model_proj.weight", alloc, gpu_idx)?)
        } else {
            None
        };
        let per_layer_proj_norm = if is_gemma4 && gguf.find_tensor("per_layer_proj_norm.weight").is_some() {
            Some(upload_and_dequant(gguf, "per_layer_proj_norm.weight", alloc, dequant, gpu_idx)?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers {
            let pfx = format!("blk.{i}");
            info!(layer = i, "loading layer weights");

            // Load Gemma 4 optional tensors
            let attn_q_norm = try_upload_and_dequant(gguf, &format!("{pfx}.attn_q_norm.weight"), alloc, dequant, gpu_idx)?;
            let attn_k_norm = try_upload_and_dequant(gguf, &format!("{pfx}.attn_k_norm.weight"), alloc, dequant, gpu_idx)?;
            let post_attention_norm = try_upload_and_dequant(gguf, &format!("{pfx}.post_attention_norm.weight"), alloc, dequant, gpu_idx)?;
            let post_ffw_norm = try_upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm.weight"), alloc, dequant, gpu_idx)?;
            let post_norm = try_upload_and_dequant(gguf, &format!("{pfx}.post_norm.weight"), alloc, dequant, gpu_idx)?;
            let inp_gate = try_upload_quantized(gguf, &format!("{pfx}.inp_gate.weight"), alloc, gpu_idx)?;
            let proj = try_upload_quantized(gguf, &format!("{pfx}.proj.weight"), alloc, gpu_idx)?;

            // Layer output scale: read F32 scalar from GGUF
            let layer_output_scale = if let Some((_ti, data)) = gguf.tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight")).ok() {
                if data.len() >= 4 {
                    let val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    Some(val)
                } else {
                    None
                }
            } else {
                None
            };

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
                // Gemma 4 specific
                attn_q_norm,
                attn_k_norm,
                post_attention_norm,
                post_ffw_norm,
                post_norm,
                inp_gate,
                proj,
                layer_output_scale,
            };
            layers.push(layer);
        }

        info!("all weights loaded to GPU");

        Ok(Self {
            token_embd,
            layers,
            output_norm,
            output,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
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

/// Try to upload and dequant — returns None if tensor doesn't exist.
fn try_upload_and_dequant(
    gguf: &GgufFile,
    name: &str,
    alloc: &VramAllocator,
    dequant: &DequantKernels,
    gpu_idx: usize,
) -> Result<Option<CudaSlice<half::f16>>, LoadError> {
    if gguf.find_tensor(name).is_some() {
        Ok(Some(upload_and_dequant(gguf, name, alloc, dequant, gpu_idx)?))
    } else {
        Ok(None)
    }
}

/// Try to upload quantized — returns None if tensor doesn't exist.
fn try_upload_quantized(
    gguf: &GgufFile,
    name: &str,
    alloc: &VramAllocator,
    gpu_idx: usize,
) -> Result<Option<QuantWeight>, LoadError> {
    if gguf.find_tensor(name).is_some() {
        Ok(Some(upload_quantized(gguf, name, alloc, gpu_idx)?))
    } else {
        Ok(None)
    }
}
