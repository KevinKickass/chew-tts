use crate::config::ModelConfig;
use crate::vram_plan::StreamingPlan;
use chew_gguf::{GgmlType, GgufFile, GgufError};
use chew_vram::VramAllocator;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream};
use chew_kernel::{DequantKernels, KernelError};
use std::sync::Arc;
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
    /// RoPE proportional frequency factors: [max_head_dim/2] — f32
    /// For Gemma 4: 1.0 for rotated dims, 1e30 for identity dims
    pub rope_freq_factors: Option<CudaSlice<f32>>,
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

/// Layout of one layer's tensors within the host-side buffer.
/// Records byte offsets and metadata for each weight tensor.
pub struct HostLayerLayout {
    /// Byte offset of this layer's data in the host buffer
    pub offset: usize,
    /// Total bytes for this layer
    pub size: usize,
    // Per-tensor: (offset_within_layer, byte_size, quant_type, n_elements)
    pub attn_q: TensorSlot,
    pub attn_k: TensorSlot,
    pub attn_v: TensorSlot,
    pub attn_output: TensorSlot,
    pub ffn_gate: TensorSlot,
    pub ffn_up: TensorSlot,
    pub ffn_down: TensorSlot,
    // Gemma4 optional tensors
    pub inp_gate: Option<TensorSlot>,
    pub proj: Option<TensorSlot>,
    pub layer_output_scale: Option<f32>,
}

/// Describes where a single tensor lives within a layer blob.
pub struct TensorSlot {
    /// Byte offset within the layer blob
    pub off: usize,
    /// Byte size
    pub size: usize,
    /// Quantization type
    pub quant_type: GgmlType,
    /// Number of logical elements
    pub n_elements: u32,
}

/// Per-layer norm weights that are always resident on GPU (tiny, ~0.8MB total).
pub struct StreamingLayerNorms {
    pub attn_norm: CudaSlice<half::f16>,
    pub ffn_norm: CudaSlice<half::f16>,
    // Gemma4 optional norms
    pub attn_q_norm: Option<CudaSlice<half::f16>>,
    pub attn_k_norm: Option<CudaSlice<half::f16>>,
    pub post_attention_norm: Option<CudaSlice<half::f16>>,
    pub post_ffw_norm: Option<CudaSlice<half::f16>>,
    pub post_norm: Option<CudaSlice<half::f16>>,
}

/// Streaming weights: some layers on GPU, rest streamed from host RAM.
///
/// The key idea: resident layers live on GPU permanently as normal `LayerWeights`.
/// Streamed layers are stored in a host-side buffer. Two GPU "shell" `LayerWeights`
/// act as double-buffers — before processing a streamed layer, we DMA the host
/// tensor data into the active shell's CudaSlice fields.
///
/// All layer norms are always on GPU (tiny, ~0.8MB total) so the forward pass
/// can fuse residual-add + RMSNorm for the next layer without waiting for DMA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingShellSlot {
    pub loaded: Option<usize>,
    pub in_flight: Option<usize>,
    pub locked: bool,
    pub last_used_tick: u64,
}

pub struct StreamingWeights {
    // Always on GPU (same as ModelWeights)
    pub token_embd: CudaSlice<half::f16>,
    pub output_norm: CudaSlice<half::f16>,
    pub output: QuantWeight,

    // Per-layer norm weights — ALL on GPU (tiny)
    pub layer_norms: Vec<StreamingLayerNorms>,

    // Resident layer weights — first N layers fully on GPU
    pub resident_layers: Vec<LayerWeights>,
    pub n_resident: usize,

    // Host-side layer data for streamed layers (layers n_resident..n_layers)
    pub host_layer_data: Vec<u8>,
    pub host_layer_offsets: Vec<HostLayerLayout>,

    // Double-buffer GPU shells for streaming layers
    pub shell_a: LayerWeights,
    pub shell_b: LayerWeights,
    pub dma_stream: Arc<CudaStream>,
    pub shell_ready: [Option<CudaEvent>; 2],
    pub shell_slots: [StreamingShellSlot; 2],

    // Gemma4 specific (same as ModelWeights)
    pub per_layer_token_embd: Option<QuantWeight>,
    pub per_layer_model_proj: Option<QuantWeight>,
    pub per_layer_proj_norm: Option<CudaSlice<half::f16>>,
    pub rope_freq_factors: Option<CudaSlice<f32>>,
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

        // RoPE proportional frequency factors (Gemma 4 full-attention layers)
        let rope_freq_factors = if gguf.find_tensor("rope_freqs.weight").is_some() {
            let (ti, host_data) = gguf.tensor_data_by_name("rope_freqs.weight")
                .map_err(|_| LoadError::MissingTensor("rope_freqs.weight".into()))?;
            let n = ti.n_elements() as usize;
            let stream = alloc.stream(gpu_idx);
            // This is F32 data — upload directly
            assert!(host_data.len() == n * 4, "rope_freqs.weight expected F32");
            let f32_data: Vec<f32> = host_data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut gpu_buf = stream.alloc_zeros::<f32>(n)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            stream.memcpy_htod(&f32_data, &mut gpu_buf)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            info!(n_factors = n, "loaded rope_freqs.weight");
            Some(gpu_buf)
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
            rope_freq_factors,
        })
    }
}

impl StreamingWeights {
    /// Load a model with streaming: first N layers on GPU, rest in host RAM.
    pub fn load(
        gguf: &GgufFile,
        config: &ModelConfig,
        plan: &StreamingPlan,
        alloc: &VramAllocator,
        dequant: &DequantKernels,
        gpu_idx: usize,
    ) -> Result<Self, LoadError> {
        let n_resident = plan.n_resident as usize;
        let n_layers = config.n_layers as usize;
        let is_gemma4 = config.is_gemma4();

        info!(
            arch = %config.arch,
            layers = n_layers,
            resident = n_resident,
            streamed = n_layers - n_resident,
            "loading streaming weights"
        );

        // 1. Global tensors (same as ModelWeights)
        let token_embd = upload_and_dequant(gguf, "token_embd.weight", alloc, dequant, gpu_idx)?;
        let output_norm = upload_and_dequant(gguf, "output_norm.weight", alloc, dequant, gpu_idx)?;

        let output = if gguf.find_tensor("output.weight").is_some() {
            upload_quantized(gguf, "output.weight", alloc, gpu_idx)?
        } else {
            info!("output.weight not found, using tied embeddings as f16 fallback");
            upload_quantized(gguf, "token_embd.weight", alloc, gpu_idx)?
        };

        // Gemma4 extras
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
        let rope_freq_factors = if gguf.find_tensor("rope_freqs.weight").is_some() {
            let (ti, host_data) = gguf.tensor_data_by_name("rope_freqs.weight")
                .map_err(|_| LoadError::MissingTensor("rope_freqs.weight".into()))?;
            let n = ti.n_elements() as usize;
            let stream = alloc.stream(gpu_idx);
            assert!(host_data.len() == n * 4, "rope_freqs.weight expected F32");
            let f32_data: Vec<f32> = host_data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut gpu_buf = stream.alloc_zeros::<f32>(n)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            stream.memcpy_htod(&f32_data, &mut gpu_buf)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            Some(gpu_buf)
        } else {
            None
        };

        // 2. Load ALL layer norms to GPU (tiny, always resident)
        let mut layer_norms = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let pfx = format!("blk.{i}");
            let norms = StreamingLayerNorms {
                attn_norm: upload_and_dequant(gguf, &format!("{pfx}.attn_norm.weight"), alloc, dequant, gpu_idx)?,
                ffn_norm: upload_and_dequant(gguf, &format!("{pfx}.ffn_norm.weight"), alloc, dequant, gpu_idx)?,
                attn_q_norm: try_upload_and_dequant(gguf, &format!("{pfx}.attn_q_norm.weight"), alloc, dequant, gpu_idx)?,
                attn_k_norm: try_upload_and_dequant(gguf, &format!("{pfx}.attn_k_norm.weight"), alloc, dequant, gpu_idx)?,
                post_attention_norm: try_upload_and_dequant(gguf, &format!("{pfx}.post_attention_norm.weight"), alloc, dequant, gpu_idx)?,
                post_ffw_norm: try_upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm.weight"), alloc, dequant, gpu_idx)?,
                post_norm: try_upload_and_dequant(gguf, &format!("{pfx}.post_norm.weight"), alloc, dequant, gpu_idx)?,
            };
            layer_norms.push(norms);
        }

        // 3. Load resident layers (0..n_resident) to GPU
        let mut resident_layers = Vec::with_capacity(n_resident);
        for i in 0..n_resident {
            let pfx = format!("blk.{i}");
            info!(layer = i, "loading resident layer weights");

            let layer = load_layer_weights(gguf, &pfx, alloc, dequant, gpu_idx, is_gemma4,
                &layer_norms[i])?;
            resident_layers.push(layer);
        }

        // 4. Load streamed layers (n_resident..n_layers) into host buffer
        let mut host_layer_data = Vec::new();
        let mut host_layer_offsets = Vec::with_capacity(n_layers - n_resident);

        for i in n_resident..n_layers {
            let pfx = format!("blk.{i}");
            info!(layer = i, "loading streamed layer to host RAM");

            let layer_start = host_layer_data.len();
            let layout = load_layer_to_host(gguf, &pfx, &mut host_layer_data, is_gemma4)?;
            let layout = HostLayerLayout {
                offset: layer_start,
                size: host_layer_data.len() - layer_start,
                ..layout
            };
            host_layer_offsets.push(layout);
        }

        info!(
            host_mb = host_layer_data.len() / (1024 * 1024),
            "streamed layer data loaded to host RAM"
        );

        // 5. Allocate two GPU shells (double-buffer) sized for the largest layer
        let stream = alloc.stream(gpu_idx);
        let shell_a = allocate_layer_shell(config, plan.max_layer_bytes as usize, stream, is_gemma4)?;
        let shell_b = allocate_layer_shell(config, plan.max_layer_bytes as usize, stream, is_gemma4)?;
        let dma_stream = stream.context().new_stream()
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;

        info!("streaming weights loaded: {} resident, {} streamed",
            n_resident, n_layers - n_resident);

        Ok(Self {
            token_embd,
            output_norm,
            output,
            layer_norms,
            resident_layers,
            n_resident,
            host_layer_data,
            host_layer_offsets,
            shell_a,
            shell_b,
            dma_stream,
            shell_ready: [None, None],
            shell_slots: [
                StreamingShellSlot { loaded: None, in_flight: None, locked: false, last_used_tick: 0 },
                StreamingShellSlot { loaded: None, in_flight: None, locked: false, last_used_tick: 0 },
            ],
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            rope_freq_factors,
        })
    }

}

/// Load a full LayerWeights to GPU (for resident layers).
/// Norms come from the already-loaded StreamingLayerNorms (device-to-device copy).
fn load_layer_weights(
    gguf: &GgufFile,
    pfx: &str,
    alloc: &VramAllocator,
    dequant: &DequantKernels,
    gpu_idx: usize,
    _is_gemma4: bool,
    _norms: &StreamingLayerNorms,
) -> Result<LayerWeights, LoadError> {
    let attn_q_norm = try_upload_and_dequant(gguf, &format!("{pfx}.attn_q_norm.weight"), alloc, dequant, gpu_idx)?;
    let attn_k_norm = try_upload_and_dequant(gguf, &format!("{pfx}.attn_k_norm.weight"), alloc, dequant, gpu_idx)?;
    let post_attention_norm = try_upload_and_dequant(gguf, &format!("{pfx}.post_attention_norm.weight"), alloc, dequant, gpu_idx)?;
    let post_ffw_norm = try_upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm.weight"), alloc, dequant, gpu_idx)?;
    let post_norm = try_upload_and_dequant(gguf, &format!("{pfx}.post_norm.weight"), alloc, dequant, gpu_idx)?;
    let inp_gate = try_upload_quantized(gguf, &format!("{pfx}.inp_gate.weight"), alloc, gpu_idx)?;
    let proj = try_upload_quantized(gguf, &format!("{pfx}.proj.weight"), alloc, gpu_idx)?;

    let layer_output_scale = if let Ok((_ti, data)) = gguf.tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight")) {
        if data.len() >= 4 {
            Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
        } else {
            None
        }
    } else {
        None
    };

    Ok(LayerWeights {
        attn_norm: upload_and_dequant(gguf, &format!("{pfx}.attn_norm.weight"), alloc, dequant, gpu_idx)?,
        ffn_norm: upload_and_dequant(gguf, &format!("{pfx}.ffn_norm.weight"), alloc, dequant, gpu_idx)?,
        attn_q: upload_quantized(gguf, &format!("{pfx}.attn_q.weight"), alloc, gpu_idx)?,
        attn_k: upload_quantized(gguf, &format!("{pfx}.attn_k.weight"), alloc, gpu_idx)?,
        attn_v: upload_quantized(gguf, &format!("{pfx}.attn_v.weight"), alloc, gpu_idx)?,
        attn_output: upload_quantized(gguf, &format!("{pfx}.attn_output.weight"), alloc, gpu_idx)?,
        ffn_gate: upload_quantized(gguf, &format!("{pfx}.ffn_gate.weight"), alloc, gpu_idx)?,
        ffn_up: upload_quantized(gguf, &format!("{pfx}.ffn_up.weight"), alloc, gpu_idx)?,
        ffn_down: upload_quantized(gguf, &format!("{pfx}.ffn_down.weight"), alloc, gpu_idx)?,
        attn_q_norm,
        attn_k_norm,
        post_attention_norm,
        post_ffw_norm,
        post_norm,
        inp_gate,
        proj,
        layer_output_scale,
    })
}

/// Read a layer's quantized weight tensors into a host Vec<u8>, recording offsets.
fn load_layer_to_host(
    gguf: &GgufFile,
    pfx: &str,
    buf: &mut Vec<u8>,
    is_gemma4: bool,
) -> Result<HostLayerLayout, LoadError> {
    let layer_start = buf.len();

    fn read_tensor(gguf: &GgufFile, name: &str, buf: &mut Vec<u8>, base: usize) -> Result<TensorSlot, LoadError> {
        let (ti, data) = gguf.tensor_data_by_name(name)
            .map_err(|_| LoadError::MissingTensor(name.to_string()))?;
        let off = buf.len() - base;
        buf.extend_from_slice(data);
        Ok(TensorSlot {
            off,
            size: data.len(),
            quant_type: ti.ggml_type,
            n_elements: ti.n_elements() as u32,
        })
    }

    fn try_read_tensor(gguf: &GgufFile, name: &str, buf: &mut Vec<u8>, base: usize) -> Result<Option<TensorSlot>, LoadError> {
        if gguf.find_tensor(name).is_some() {
            Ok(Some(read_tensor(gguf, name, buf, base)?))
        } else {
            Ok(None)
        }
    }

    let attn_q = read_tensor(gguf, &format!("{pfx}.attn_q.weight"), buf, layer_start)?;
    let attn_k = read_tensor(gguf, &format!("{pfx}.attn_k.weight"), buf, layer_start)?;
    let attn_v = read_tensor(gguf, &format!("{pfx}.attn_v.weight"), buf, layer_start)?;
    let attn_output = read_tensor(gguf, &format!("{pfx}.attn_output.weight"), buf, layer_start)?;
    let ffn_gate = read_tensor(gguf, &format!("{pfx}.ffn_gate.weight"), buf, layer_start)?;
    let ffn_up = read_tensor(gguf, &format!("{pfx}.ffn_up.weight"), buf, layer_start)?;
    let ffn_down = read_tensor(gguf, &format!("{pfx}.ffn_down.weight"), buf, layer_start)?;

    let inp_gate = if is_gemma4 {
        try_read_tensor(gguf, &format!("{pfx}.inp_gate.weight"), buf, layer_start)?
    } else {
        None
    };
    let proj = if is_gemma4 {
        try_read_tensor(gguf, &format!("{pfx}.proj.weight"), buf, layer_start)?
    } else {
        None
    };

    let layer_output_scale = if let Ok((_ti, data)) = gguf.tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight")) {
        if data.len() >= 4 {
            Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
        } else {
            None
        }
    } else {
        None
    };

    Ok(HostLayerLayout {
        offset: layer_start,
        size: buf.len() - layer_start,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        ffn_gate,
        ffn_up,
        ffn_down,
        inp_gate,
        proj,
        layer_output_scale,
    })
}

/// Allocate a LayerWeights "shell" with pre-sized GPU buffers for streaming.
/// All CudaSlice fields are allocated to the max size any streamed layer needs.
fn allocate_layer_shell(
    config: &ModelConfig,
    _max_layer_bytes: usize,
    stream: &Arc<CudaStream>,
    is_gemma4: bool,
) -> Result<LayerWeights, LoadError> {
    let d = config.dim as usize;
    let ff = config.ff_dim as usize;
    let nh = config.n_heads as usize;
    let nkv = config.n_kv_heads as usize;
    let hd = config.max_head_dim as usize;

    // Estimate max sizes for each tensor type.
    // Actual bytes depend on quant type, but we allocate for the worst case.
    // The largest possible size for any tensor is max_layer_bytes (the whole layer).
    // We use max_layer_bytes as a safe upper bound for each tensor allocation.
    // This wastes some VRAM but keeps things simple (2x max_layer_bytes total for shells).

    // Actually, let's be smarter: compute max tensor byte size from dimension estimates.
    // For IQ2/Q4 etc., bytes = n_elements * block_bytes / block_size.
    // But we don't know the quant type at allocation time. Use max_layer_bytes per tensor.
    // The total shell VRAM is already budgeted in the StreamingPlan as dma_slot_bytes.

    // For the quantized weight buffers, allocate max_layer_bytes each (overkill but safe).
    // The norm buffers (f16) are small — dim elements.
    let alloc_u8 = |size: usize| -> Result<CudaSlice<u8>, LoadError> {
        stream.alloc_zeros::<u8>(size)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))
    };
    let alloc_f16 = |size: usize| -> Result<CudaSlice<half::f16>, LoadError> {
        stream.alloc_zeros::<half::f16>(size)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))
    };

    let mk_qw = |size: usize| -> Result<QuantWeight, LoadError> {
        Ok(QuantWeight {
            data: alloc_u8(size)?,
            quant_type: GgmlType::F16, // placeholder, updated on upload
            n_elements: 0,
        })
    };

    // Each weight matrix in a layer: at most max_layer_bytes (conservative).
    // A tighter bound: largest single tensor is ffn_gate/ffn_up at ff_dim * dim * bpw/8,
    // but we don't know bpw. max_layer_bytes / 7 is a reasonable per-tensor bound.
    // Actually just use max_layer_bytes for each — we already budgeted 2 * max_layer_bytes
    // in the VRAM plan. The shell re-uses the same VRAM, so overlapping is fine.
    // NO WAIT: we need 7 separate CudaSlice allocations, each of which takes real VRAM.
    // We need to be smarter.

    // Use a single CudaSlice<u8> per shell of size max_layer_bytes, then create
    // QuantWeight views... but QuantWeight owns CudaSlice, so that doesn't work easily.
    //
    // Alternative: allocate per-tensor with realistic sizes.
    // For Q4_K: bytes = n_elements * 144/256 = ~0.5625 * n_elements
    // For IQ2: bytes = n_elements * ... varies
    // Just use n_elements * 2 bytes (f16 size) as upper bound — quantized is always smaller.
    let q_size = nh * hd * d * 2;
    let k_size = nkv * hd * d * 2;
    let v_size = nkv * hd * d * 2;
    let o_size = d * nh * hd * 2;
    let gate_size = ff * d * 2;
    let up_size = ff * d * 2;
    let down_size = d * ff * 2;

    let inp_gate_qw = if is_gemma4 {
        if let Some(epl) = config.embd_per_layer {
            Some(mk_qw(d * epl as usize * 2)?)
        } else {
            None
        }
    } else {
        None
    };
    let proj_qw = if is_gemma4 {
        if let Some(epl) = config.embd_per_layer {
            Some(mk_qw(epl as usize * d * 2)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(LayerWeights {
        attn_norm: alloc_f16(d)?,
        ffn_norm: alloc_f16(d)?,
        attn_q: mk_qw(q_size)?,
        attn_k: mk_qw(k_size)?,
        attn_v: mk_qw(v_size)?,
        attn_output: mk_qw(o_size)?,
        ffn_gate: mk_qw(gate_size)?,
        ffn_up: mk_qw(up_size)?,
        ffn_down: mk_qw(down_size)?,
        attn_q_norm: if is_gemma4 { Some(alloc_f16(hd)?) } else { None },
        attn_k_norm: if is_gemma4 { Some(alloc_f16(hd)?) } else { None },
        post_attention_norm: if is_gemma4 { Some(alloc_f16(d)?) } else { None },
        post_ffw_norm: if is_gemma4 { Some(alloc_f16(d)?) } else { None },
        post_norm: if is_gemma4 { Some(alloc_f16(d)?) } else { None },
        inp_gate: inp_gate_qw,
        proj: proj_qw,
        layer_output_scale: None,
    })
}

/// Upload raw quantized tensor bytes to GPU — no dequantization.
pub(crate) fn upload_quantized(
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
pub(crate) fn upload_and_dequant(
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
