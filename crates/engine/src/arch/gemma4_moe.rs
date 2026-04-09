//! Gemma 4 MoE (Mixture of Experts) — e.g. Gemma 4 26B-A4B
//!
//! Key differences from dense Gemma 4:
//! - Per-layer n_kv_heads (SWA=8, Full=2)
//! - Shared KV: full-attention layers have no attn_v, V = K
//! - MoE FFN: 128 experts, top-8 routing, plus shared dense FFN
//! - Expert weights as 3D tensors: [dim, expert_dim, n_experts]
//! - Additional norms: post_ffw_norm_1, post_ffw_norm_2, pre_ffw_norm_2
//! - No per-layer embeddings (embd_per_layer = 0)

use crate::config::ModelConfig;
use crate::weights::{QuantWeight, LoadError, upload_quantized, upload_and_dequant};
use chew_gguf::{GgmlType, GgufFile};
use chew_vram::VramAllocator;
use chew_kernel::DequantKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;
use tracing::info;

// ─── Layer weights ──────────────────────────────────────────────────────────

/// Weights for one Gemma 4 MoE transformer layer.
pub struct MoeLayerWeights {
    // Attention
    pub attn_norm: CudaSlice<half::f16>,
    pub attn_q: QuantWeight,
    pub attn_k: QuantWeight,
    /// None for full-attention layers (shared KV: V = K)
    pub attn_v: Option<QuantWeight>,
    pub attn_output: QuantWeight,
    pub attn_q_norm: CudaSlice<half::f16>,
    pub attn_k_norm: CudaSlice<half::f16>,
    pub post_attention_norm: CudaSlice<half::f16>,

    // Shared dense FFN (always applied)
    pub ffn_norm: CudaSlice<half::f16>,
    pub ffn_gate: QuantWeight,
    pub ffn_up: QuantWeight,
    pub ffn_down: QuantWeight,
    pub post_ffw_norm: CudaSlice<half::f16>,

    // MoE expert FFN
    /// Router: [dim, n_experts]
    pub moe_gate: QuantWeight,
    /// Fused gate+up for all experts: [dim, expert_ff_dim*2, n_experts]
    pub moe_gate_up_exps: QuantWeight,
    /// Down projection for all experts: [expert_ff_dim, dim, n_experts]
    pub moe_down_exps: QuantWeight,
    /// MoE norms
    pub pre_ffw_norm_2: CudaSlice<half::f16>,
    pub post_ffw_norm_2: CudaSlice<half::f16>,

    pub layer_output_scale: Option<f32>,
}

/// Norms that stay GPU-resident during streaming (tiny, always loaded).
pub struct MoeStreamingNorms {
    pub attn_norm: CudaSlice<half::f16>,
    pub attn_q_norm: CudaSlice<half::f16>,
    pub attn_k_norm: CudaSlice<half::f16>,
    pub post_attention_norm: CudaSlice<half::f16>,
    pub ffn_norm: CudaSlice<half::f16>,
    pub post_ffw_norm: CudaSlice<half::f16>,
    pub pre_ffw_norm_2: CudaSlice<half::f16>,
    pub post_ffw_norm_2: CudaSlice<half::f16>,
}

/// Host-side layout for one streamed MoE layer.
pub struct MoeHostLayout {
    pub offset: usize,
    pub size: usize,
    pub attn_q: TensorSlot,
    pub attn_k: TensorSlot,
    pub attn_v: Option<TensorSlot>,
    pub attn_output: TensorSlot,
    pub ffn_gate: TensorSlot,
    pub ffn_up: TensorSlot,
    pub ffn_down: TensorSlot,
    pub moe_gate: TensorSlot,
    pub moe_gate_up_exps: TensorSlot,
    pub moe_down_exps: TensorSlot,
    pub layer_output_scale: Option<f32>,
}

/// Where a tensor lives within the host buffer.
pub struct TensorSlot {
    pub off: usize,
    pub size: usize,
    pub quant_type: GgmlType,
    pub n_elements: u32,
}

// ─── Weight loading ─────────────────────────────────────────────────────────

/// Load one MoE layer directly to GPU (for resident layers).
pub fn load_layer_to_gpu(
    gguf: &GgufFile,
    layer: usize,
    config: &ModelConfig,
    alloc: &VramAllocator,
    dequant: &DequantKernels,
    gpu_idx: usize,
) -> Result<MoeLayerWeights, LoadError> {
    let pfx = format!("blk.{layer}");
    info!(layer, "loading MoE layer to GPU");

    let attn_norm = upload_and_dequant(gguf, &format!("{pfx}.attn_norm.weight"), alloc, dequant, gpu_idx)?;
    let attn_q = upload_quantized(gguf, &format!("{pfx}.attn_q.weight"), alloc, gpu_idx)?;
    let attn_k = upload_quantized(gguf, &format!("{pfx}.attn_k.weight"), alloc, gpu_idx)?;

    // Shared KV: full-attention layers have no attn_v
    let attn_v = if gguf.find_tensor(&format!("{pfx}.attn_v.weight")).is_some() {
        Some(upload_quantized(gguf, &format!("{pfx}.attn_v.weight"), alloc, gpu_idx)?)
    } else {
        None
    };

    let attn_output = upload_quantized(gguf, &format!("{pfx}.attn_output.weight"), alloc, gpu_idx)?;
    let attn_q_norm = upload_and_dequant(gguf, &format!("{pfx}.attn_q_norm.weight"), alloc, dequant, gpu_idx)?;
    let attn_k_norm = upload_and_dequant(gguf, &format!("{pfx}.attn_k_norm.weight"), alloc, dequant, gpu_idx)?;
    let post_attention_norm = upload_and_dequant(gguf, &format!("{pfx}.post_attention_norm.weight"), alloc, dequant, gpu_idx)?;

    // Shared dense FFN
    let ffn_norm = upload_and_dequant(gguf, &format!("{pfx}.ffn_norm.weight"), alloc, dequant, gpu_idx)?;
    let ffn_gate = upload_quantized(gguf, &format!("{pfx}.ffn_gate.weight"), alloc, gpu_idx)?;
    let ffn_up = upload_quantized(gguf, &format!("{pfx}.ffn_up.weight"), alloc, gpu_idx)?;
    let ffn_down = upload_quantized(gguf, &format!("{pfx}.ffn_down.weight"), alloc, gpu_idx)?;
    let post_ffw_norm = upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm.weight"), alloc, dequant, gpu_idx)?;

    // MoE expert weights
    let moe_gate = upload_quantized(gguf, &format!("{pfx}.ffn_gate_inp.weight"), alloc, gpu_idx)?;
    let moe_gate_up_exps = upload_quantized(gguf, &format!("{pfx}.ffn_gate_up_exps.weight"), alloc, gpu_idx)?;
    let moe_down_exps = upload_quantized(gguf, &format!("{pfx}.ffn_down_exps.weight"), alloc, gpu_idx)?;

    // MoE norms
    let pre_ffw_norm_2 = upload_and_dequant(gguf, &format!("{pfx}.pre_ffw_norm_2.weight"), alloc, dequant, gpu_idx)?;
    let post_ffw_norm_2 = upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm_2.weight"), alloc, dequant, gpu_idx)?;

    // Layer output scale
    let layer_output_scale = gguf.tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight"))
        .ok()
        .and_then(|(_ti, data)| {
            if data.len() >= 4 {
                Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            } else {
                None
            }
        });

    Ok(MoeLayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        attn_q_norm,
        attn_k_norm,
        post_attention_norm,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm,
        moe_gate,
        moe_gate_up_exps,
        moe_down_exps,
        pre_ffw_norm_2,
        post_ffw_norm_2,
        layer_output_scale,
    })
}

/// Load norms for a MoE layer (always GPU-resident during streaming).
pub fn load_layer_norms(
    gguf: &GgufFile,
    layer: usize,
    alloc: &VramAllocator,
    dequant: &DequantKernels,
    gpu_idx: usize,
) -> Result<MoeStreamingNorms, LoadError> {
    let pfx = format!("blk.{layer}");
    Ok(MoeStreamingNorms {
        attn_norm: upload_and_dequant(gguf, &format!("{pfx}.attn_norm.weight"), alloc, dequant, gpu_idx)?,
        attn_q_norm: upload_and_dequant(gguf, &format!("{pfx}.attn_q_norm.weight"), alloc, dequant, gpu_idx)?,
        attn_k_norm: upload_and_dequant(gguf, &format!("{pfx}.attn_k_norm.weight"), alloc, dequant, gpu_idx)?,
        post_attention_norm: upload_and_dequant(gguf, &format!("{pfx}.post_attention_norm.weight"), alloc, dequant, gpu_idx)?,
        ffn_norm: upload_and_dequant(gguf, &format!("{pfx}.ffn_norm.weight"), alloc, dequant, gpu_idx)?,
        post_ffw_norm: upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm.weight"), alloc, dequant, gpu_idx)?,
        pre_ffw_norm_2: upload_and_dequant(gguf, &format!("{pfx}.pre_ffw_norm_2.weight"), alloc, dequant, gpu_idx)?,
        post_ffw_norm_2: upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm_2.weight"), alloc, dequant, gpu_idx)?,
    })
}

/// Load one MoE layer's quantized weights into host RAM for streaming.
pub fn load_layer_to_host(
    gguf: &GgufFile,
    layer: usize,
    buf: &mut Vec<u8>,
) -> Result<MoeHostLayout, LoadError> {
    let pfx = format!("blk.{layer}");
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
    let attn_v = try_read_tensor(gguf, &format!("{pfx}.attn_v.weight"), buf, layer_start)?;
    let attn_output = read_tensor(gguf, &format!("{pfx}.attn_output.weight"), buf, layer_start)?;
    let ffn_gate = read_tensor(gguf, &format!("{pfx}.ffn_gate.weight"), buf, layer_start)?;
    let ffn_up = read_tensor(gguf, &format!("{pfx}.ffn_up.weight"), buf, layer_start)?;
    let ffn_down = read_tensor(gguf, &format!("{pfx}.ffn_down.weight"), buf, layer_start)?;
    let moe_gate = read_tensor(gguf, &format!("{pfx}.ffn_gate_inp.weight"), buf, layer_start)?;
    let moe_gate_up_exps = read_tensor(gguf, &format!("{pfx}.ffn_gate_up_exps.weight"), buf, layer_start)?;
    let moe_down_exps = read_tensor(gguf, &format!("{pfx}.ffn_down_exps.weight"), buf, layer_start)?;

    let layer_output_scale = gguf.tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight"))
        .ok()
        .and_then(|(_ti, data)| {
            if data.len() >= 4 {
                Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            } else {
                None
            }
        });

    let size = buf.len() - layer_start;

    Ok(MoeHostLayout {
        offset: layer_start,
        size,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        ffn_gate,
        ffn_up,
        ffn_down,
        moe_gate,
        moe_gate_up_exps,
        moe_down_exps,
        layer_output_scale,
    })
}

// ─── Streaming shell ────────────────────────────────────────────────────────

/// Allocate a MoE shell with pre-sized GPU buffers for streaming.
pub fn allocate_shell(
    config: &ModelConfig,
    stream: &Arc<CudaStream>,
) -> Result<MoeLayerWeights, LoadError> {
    let d = config.dim as usize;
    let ff = config.ff_dim as usize;
    let nh = config.n_heads as usize;
    let nkv = config.n_kv_heads as usize; // max kv heads
    let hd = config.max_head_dim as usize;
    let n_exp = config.n_experts as usize;
    let exp_ff = config.expert_ff_dim as usize;

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
            quant_type: GgmlType::F16,
            n_elements: 0,
        })
    };

    // Use f16 size (2 bytes/elem) as upper bound for quantized tensor sizes
    Ok(MoeLayerWeights {
        attn_norm: alloc_f16(d)?,
        attn_q: mk_qw(nh * hd * d * 2)?,
        attn_k: mk_qw(nkv * hd * d * 2)?,
        attn_v: Some(mk_qw(nkv * hd * d * 2)?), // allocate max; Some layers won't use it
        attn_output: mk_qw(d * nh * hd * 2)?,
        attn_q_norm: alloc_f16(hd)?,
        attn_k_norm: alloc_f16(hd)?,
        post_attention_norm: alloc_f16(d)?,
        ffn_norm: alloc_f16(d)?,
        ffn_gate: mk_qw(ff * d * 2)?,
        ffn_up: mk_qw(ff * d * 2)?,
        ffn_down: mk_qw(d * ff * 2)?,
        post_ffw_norm: alloc_f16(d)?,
        moe_gate: mk_qw(d * n_exp * 2)?,
        moe_gate_up_exps: mk_qw(d * exp_ff * 2 * n_exp * 2)?,
        moe_down_exps: mk_qw(exp_ff * d * n_exp * 2)?,
        pre_ffw_norm_2: alloc_f16(d)?,
        post_ffw_norm_2: alloc_f16(d)?,
        layer_output_scale: None,
    })
}

/// Upload host layer data into a GPU shell.
pub fn upload_to_shell(
    host_data: &[u8],
    layout: &MoeHostLayout,
    shell: &mut MoeLayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    let base = layout.offset;

    fn upload_slot(host_data: &[u8], base: usize, slot: &TensorSlot, qw: &mut QuantWeight, stream: &Arc<CudaStream>) -> Result<(), String> {
        let src = &host_data[base + slot.off..base + slot.off + slot.size];
        stream.memcpy_htod(src, &mut qw.data).map_err(|e| e.to_string())?;
        qw.quant_type = slot.quant_type;
        qw.n_elements = slot.n_elements;
        Ok(())
    }

    upload_slot(host_data, base, &layout.attn_q, &mut shell.attn_q, stream)?;
    upload_slot(host_data, base, &layout.attn_k, &mut shell.attn_k, stream)?;
    if let (Some(slot), Some(qw)) = (&layout.attn_v, &mut shell.attn_v) {
        upload_slot(host_data, base, slot, qw, stream)?;
    }
    upload_slot(host_data, base, &layout.attn_output, &mut shell.attn_output, stream)?;
    upload_slot(host_data, base, &layout.ffn_gate, &mut shell.ffn_gate, stream)?;
    upload_slot(host_data, base, &layout.ffn_up, &mut shell.ffn_up, stream)?;
    upload_slot(host_data, base, &layout.ffn_down, &mut shell.ffn_down, stream)?;
    upload_slot(host_data, base, &layout.moe_gate, &mut shell.moe_gate, stream)?;
    upload_slot(host_data, base, &layout.moe_gate_up_exps, &mut shell.moe_gate_up_exps, stream)?;
    upload_slot(host_data, base, &layout.moe_down_exps, &mut shell.moe_down_exps, stream)?;
    shell.layer_output_scale = layout.layer_output_scale;

    Ok(())
}

/// Copy norms from GPU-resident norms into a shell.
pub fn copy_norms_to_shell(
    norms: &MoeStreamingNorms,
    shell: &mut MoeLayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    stream.memcpy_dtod(&norms.attn_norm, &mut shell.attn_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.attn_q_norm, &mut shell.attn_q_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.attn_k_norm, &mut shell.attn_k_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.post_attention_norm, &mut shell.post_attention_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.ffn_norm, &mut shell.ffn_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.post_ffw_norm, &mut shell.post_ffw_norm).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.pre_ffw_norm_2, &mut shell.pre_ffw_norm_2).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.post_ffw_norm_2, &mut shell.post_ffw_norm_2).map_err(|e| e.to_string())?;
    Ok(())
}

// ─── Expert slicing ─────────────────────────────────────────────────────────

/// Info needed to address one expert's weights within a 3D tensor.
pub struct ExpertSlice {
    /// Byte offset into the parent tensor's GPU buffer
    pub byte_offset: usize,
    /// Byte size of this expert's weight data
    pub byte_size: usize,
    /// Number of logical elements for this expert
    pub n_elements: u32,
    /// Quantization type (inherited from parent)
    pub quant_type: GgmlType,
}

/// Compute slice info for expert `k` within a 3D expert tensor.
/// The tensor has shape [..., n_experts] stored contiguously.
pub fn expert_slice_info(tensor: &QuantWeight, expert_idx: u32, n_experts: u32) -> ExpertSlice {
    let total_bytes = tensor.data.len();
    let expert_bytes = total_bytes / n_experts as usize;
    let expert_elements = tensor.n_elements / n_experts;

    ExpertSlice {
        byte_offset: expert_idx as usize * expert_bytes,
        byte_size: expert_bytes,
        n_elements: expert_elements,
        quant_type: tensor.quant_type,
    }
}
