//! Gemma 4 MoE (Mixture of Experts) — e.g. Gemma 4 26B-A4B
//!
//! Key differences from dense Gemma 4:
//! - Per-layer n_kv_heads (SWA=8, Full=2)
//! - Shared KV: full-attention layers have no attn_v, V = K
//! - MoE FFN: 128 experts, top-8 routing, plus shared dense FFN
//! - Expert weights as 3D tensors: [dim, expert_dim, n_experts]
//! - Additional norms: post_ffw_norm_1, post_ffw_norm_2, pre_ffw_norm_2
//! - Gemma 4 per-layer embeddings use the same inp_gate/proj/post_norm path as dense Gemma 4

use crate::config::ModelConfig;
use crate::weights::{LoadError, QuantWeight, upload_and_dequant, upload_quantized};
use chew_gguf::{GgmlType, GgufFile};
use chew_kernel::DequantKernels;
use chew_vram::VramAllocator;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use std::sync::Arc;
use tracing::info;

// ─── Layer weights ──────────────────────────────────────────────────────────

/// Weights for one Gemma 4 MoE transformer layer.
pub struct MoeLayerWeights {
    // Attention
    pub attn_norm: CudaSlice<half::f16>,
    pub attn_q: QuantWeight,
    pub attn_k: QuantWeight,
    /// Quantized V projection storage for layers that have a dedicated V tensor.
    pub attn_v: Option<QuantWeight>,
    /// Whether this layer has a dedicated V projection tensor. Full-attention shared-KV
    /// layers leave this false and reuse K as V, matching llama.cpp's Gemma4 MoE path.
    pub has_attn_v: bool,
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
    /// Router weights: [dim, n_experts]
    pub moe_gate: QuantWeight,
    /// Router scale: [dim] — applied to rms_norm(attn_out) / sqrt(dim) before router
    pub moe_gate_scale: CudaSlice<half::f16>,
    /// Host copy of router weights in row-major [expert, dim] F32.
    pub moe_gate_host: Arc<[f32]>,
    /// Host copy of router scale [dim] F32.
    pub moe_gate_scale_host: Arc<[f32]>,
    /// Fused gate+up for all experts: [dim, expert_ff_dim*2, n_experts]
    pub moe_gate_up_exps: QuantWeight,
    /// Down projection for all experts: [expert_ff_dim, dim, n_experts]
    pub moe_down_exps: QuantWeight,
    /// MoE norms
    pub pre_ffw_norm_2: CudaSlice<half::f16>,
    pub post_ffw_norm_1: CudaSlice<half::f16>,
    pub post_ffw_norm_2: CudaSlice<half::f16>,
    pub post_norm: Option<CudaSlice<half::f16>>,
    pub inp_gate: Option<QuantWeight>,
    pub proj: Option<QuantWeight>,

    pub layer_output_scale: Option<f32>,
    /// Per-expert scale for down projection: [n_experts] on CPU
    pub moe_down_scale: Arc<[f32]>,
}

/// Norms that stay GPU-resident during streaming (tiny, always loaded).
pub struct MoeStreamingNorms {
    pub attn_norm: CudaSlice<half::f16>,
    pub attn_q_norm: CudaSlice<half::f16>,
    pub attn_k_norm: CudaSlice<half::f16>,
    pub post_attention_norm: CudaSlice<half::f16>,
    pub ffn_norm: CudaSlice<half::f16>,
    pub post_ffw_norm: CudaSlice<half::f16>,
    pub moe_gate_scale: CudaSlice<half::f16>,
    pub pre_ffw_norm_2: CudaSlice<half::f16>,
    pub post_ffw_norm_1: CudaSlice<half::f16>,
    pub post_ffw_norm_2: CudaSlice<half::f16>,
    pub post_norm: Option<CudaSlice<half::f16>>,
}

/// Host-side layout for one streamed MoE layer.
#[derive(Clone)]
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
    pub inp_gate: Option<TensorSlot>,
    pub proj: Option<TensorSlot>,
    pub layer_output_scale: Option<f32>,
    pub moe_down_scale: Arc<[f32]>,
    pub moe_gate_host: Arc<[f32]>,
    pub moe_gate_scale_host: Arc<[f32]>,
}

/// Where a tensor lives within the host buffer.
#[derive(Clone, Copy)]
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
    _config: &ModelConfig,
    alloc: &VramAllocator,
    dequant: &DequantKernels,
    gpu_idx: usize,
) -> Result<MoeLayerWeights, LoadError> {
    let pfx = format!("blk.{layer}");
    info!(layer, "loading MoE layer to GPU");

    let attn_norm = upload_and_dequant(
        gguf,
        &format!("{pfx}.attn_norm.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let attn_q = upload_quantized(gguf, &format!("{pfx}.attn_q.weight"), alloc, gpu_idx)?;
    let attn_k = upload_quantized(gguf, &format!("{pfx}.attn_k.weight"), alloc, gpu_idx)?;

    // Shared KV: full-attention layers have no attn_v
    let attn_v = if gguf.find_tensor(&format!("{pfx}.attn_v.weight")).is_some() {
        Some(upload_quantized(
            gguf,
            &format!("{pfx}.attn_v.weight"),
            alloc,
            gpu_idx,
        )?)
    } else {
        None
    };
    let has_attn_v = attn_v.is_some();

    let attn_output = upload_quantized(gguf, &format!("{pfx}.attn_output.weight"), alloc, gpu_idx)?;
    let attn_q_norm = upload_and_dequant(
        gguf,
        &format!("{pfx}.attn_q_norm.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let attn_k_norm = upload_and_dequant(
        gguf,
        &format!("{pfx}.attn_k_norm.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let post_attention_norm = upload_and_dequant(
        gguf,
        &format!("{pfx}.post_attention_norm.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;

    // Shared dense FFN
    let ffn_norm = upload_and_dequant(
        gguf,
        &format!("{pfx}.ffn_norm.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let ffn_gate = upload_quantized(gguf, &format!("{pfx}.ffn_gate.weight"), alloc, gpu_idx)?;
    let ffn_up = upload_quantized(gguf, &format!("{pfx}.ffn_up.weight"), alloc, gpu_idx)?;
    let ffn_down = upload_quantized(gguf, &format!("{pfx}.ffn_down.weight"), alloc, gpu_idx)?;
    let post_ffw_norm = upload_and_dequant(
        gguf,
        &format!("{pfx}.post_ffw_norm.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;

    // MoE expert weights
    let moe_gate = upload_quantized(gguf, &format!("{pfx}.ffn_gate_inp.weight"), alloc, gpu_idx)?;
    let moe_gate_scale = upload_and_dequant(
        gguf,
        &format!("{pfx}.ffn_gate_inp.scale"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let moe_gate_host = gguf
        .tensor_data_by_name(&format!("{pfx}.ffn_gate_inp.weight"))
        .map(|(_ti, data)| {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_default();
    let moe_gate_scale_host = gguf
        .tensor_data_by_name(&format!("{pfx}.ffn_gate_inp.scale"))
        .map(|(_ti, data)| {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_default();
    let moe_gate_up_exps = upload_quantized(
        gguf,
        &format!("{pfx}.ffn_gate_up_exps.weight"),
        alloc,
        gpu_idx,
    )?;
    let moe_down_exps =
        upload_quantized(gguf, &format!("{pfx}.ffn_down_exps.weight"), alloc, gpu_idx)?;

    // MoE norms
    let pre_ffw_norm_2 = upload_and_dequant(
        gguf,
        &format!("{pfx}.pre_ffw_norm_2.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let post_ffw_norm_1 = upload_and_dequant(
        gguf,
        &format!("{pfx}.post_ffw_norm_1.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let post_ffw_norm_2 = upload_and_dequant(
        gguf,
        &format!("{pfx}.post_ffw_norm_2.weight"),
        alloc,
        dequant,
        gpu_idx,
    )?;
    let post_norm = if gguf
        .find_tensor(&format!("{pfx}.post_norm.weight"))
        .is_some()
    {
        Some(upload_and_dequant(
            gguf,
            &format!("{pfx}.post_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?)
    } else {
        None
    };
    let inp_gate = if gguf
        .find_tensor(&format!("{pfx}.inp_gate.weight"))
        .is_some()
    {
        Some(upload_quantized(
            gguf,
            &format!("{pfx}.inp_gate.weight"),
            alloc,
            gpu_idx,
        )?)
    } else {
        None
    };
    let proj = if gguf.find_tensor(&format!("{pfx}.proj.weight")).is_some() {
        Some(upload_quantized(
            gguf,
            &format!("{pfx}.proj.weight"),
            alloc,
            gpu_idx,
        )?)
    } else {
        None
    };

    // Layer output scale
    let layer_output_scale = gguf
        .tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight"))
        .ok()
        .and_then(|(_ti, data)| {
            if data.len() >= 4 {
                Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            } else {
                None
            }
        });

    // Per-expert down scale: [n_experts] F32
    let moe_down_scale = gguf
        .tensor_data_by_name(&format!("{pfx}.ffn_down_exps.scale"))
        .map(|(_ti, data)| {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_default();

    Ok(MoeLayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        has_attn_v,
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
        moe_gate_scale,
        moe_gate_host: Arc::from(moe_gate_host),
        moe_gate_scale_host: Arc::from(moe_gate_scale_host),
        moe_gate_up_exps,
        moe_down_exps,
        pre_ffw_norm_2,
        post_ffw_norm_1,
        post_ffw_norm_2,
        post_norm,
        inp_gate,
        proj,
        layer_output_scale,
        moe_down_scale: Arc::from(moe_down_scale),
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
        attn_norm: upload_and_dequant(
            gguf,
            &format!("{pfx}.attn_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        attn_q_norm: upload_and_dequant(
            gguf,
            &format!("{pfx}.attn_q_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        attn_k_norm: upload_and_dequant(
            gguf,
            &format!("{pfx}.attn_k_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        post_attention_norm: upload_and_dequant(
            gguf,
            &format!("{pfx}.post_attention_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        ffn_norm: upload_and_dequant(
            gguf,
            &format!("{pfx}.ffn_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        post_ffw_norm: upload_and_dequant(
            gguf,
            &format!("{pfx}.post_ffw_norm.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        moe_gate_scale: upload_and_dequant(
            gguf,
            &format!("{pfx}.ffn_gate_inp.scale"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        pre_ffw_norm_2: upload_and_dequant(
            gguf,
            &format!("{pfx}.pre_ffw_norm_2.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        post_ffw_norm_1: upload_and_dequant(
            gguf,
            &format!("{pfx}.post_ffw_norm_1.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        post_ffw_norm_2: upload_and_dequant(
            gguf,
            &format!("{pfx}.post_ffw_norm_2.weight"),
            alloc,
            dequant,
            gpu_idx,
        )?,
        post_norm: if gguf
            .find_tensor(&format!("{pfx}.post_norm.weight"))
            .is_some()
        {
            Some(upload_and_dequant(
                gguf,
                &format!("{pfx}.post_norm.weight"),
                alloc,
                dequant,
                gpu_idx,
            )?)
        } else {
            None
        },
    })
}

/// Plan one MoE layer's streamed host layout.
pub fn plan_layer_host_layout(
    gguf: &GgufFile,
    layer: usize,
    base_offset: usize,
) -> Result<MoeHostLayout, LoadError> {
    let pfx = format!("blk.{layer}");

    fn tensor_slot(
        gguf: &GgufFile,
        name: &str,
        cursor: &mut usize,
        base: usize,
    ) -> Result<TensorSlot, LoadError> {
        let ti = gguf
            .find_tensor(name)
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;
        let off = *cursor - base;
        let size = ti.data_size() as usize;
        *cursor += size;
        Ok(TensorSlot {
            off,
            size,
            quant_type: ti.ggml_type,
            n_elements: ti.n_elements() as u32,
        })
    }

    fn try_tensor_slot(
        gguf: &GgufFile,
        name: &str,
        cursor: &mut usize,
        base: usize,
    ) -> Result<Option<TensorSlot>, LoadError> {
        if gguf.find_tensor(name).is_some() {
            Ok(Some(tensor_slot(gguf, name, cursor, base)?))
        } else {
            Ok(None)
        }
    }

    let mut cursor = base_offset;
    let attn_q = tensor_slot(
        gguf,
        &format!("{pfx}.attn_q.weight"),
        &mut cursor,
        base_offset,
    )?;
    let attn_k = tensor_slot(
        gguf,
        &format!("{pfx}.attn_k.weight"),
        &mut cursor,
        base_offset,
    )?;
    let attn_v = try_tensor_slot(
        gguf,
        &format!("{pfx}.attn_v.weight"),
        &mut cursor,
        base_offset,
    )?;
    let attn_output = tensor_slot(
        gguf,
        &format!("{pfx}.attn_output.weight"),
        &mut cursor,
        base_offset,
    )?;
    let ffn_gate = tensor_slot(
        gguf,
        &format!("{pfx}.ffn_gate.weight"),
        &mut cursor,
        base_offset,
    )?;
    let ffn_up = tensor_slot(
        gguf,
        &format!("{pfx}.ffn_up.weight"),
        &mut cursor,
        base_offset,
    )?;
    let ffn_down = tensor_slot(
        gguf,
        &format!("{pfx}.ffn_down.weight"),
        &mut cursor,
        base_offset,
    )?;
    let moe_gate = tensor_slot(
        gguf,
        &format!("{pfx}.ffn_gate_inp.weight"),
        &mut cursor,
        base_offset,
    )?;
    let moe_gate_up_exps = tensor_slot(
        gguf,
        &format!("{pfx}.ffn_gate_up_exps.weight"),
        &mut cursor,
        base_offset,
    )?;
    let moe_down_exps = tensor_slot(
        gguf,
        &format!("{pfx}.ffn_down_exps.weight"),
        &mut cursor,
        base_offset,
    )?;
    let inp_gate = try_tensor_slot(
        gguf,
        &format!("{pfx}.inp_gate.weight"),
        &mut cursor,
        base_offset,
    )?;
    let proj = try_tensor_slot(
        gguf,
        &format!("{pfx}.proj.weight"),
        &mut cursor,
        base_offset,
    )?;

    let layer_output_scale = gguf
        .tensor_data_by_name(&format!("{pfx}.layer_output_scale.weight"))
        .ok()
        .and_then(|(_ti, data)| {
            if data.len() >= 4 {
                Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            } else {
                None
            }
        });

    let moe_down_scale = gguf
        .tensor_data_by_name(&format!("{pfx}.ffn_down_exps.scale"))
        .map(|(_ti, data)| {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_default();
    let moe_gate_host = gguf
        .tensor_data_by_name(&format!("{pfx}.ffn_gate_inp.weight"))
        .map(|(_ti, data)| {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_default();
    let moe_gate_scale_host = gguf
        .tensor_data_by_name(&format!("{pfx}.ffn_gate_inp.scale"))
        .map(|(_ti, data)| {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_default();

    let size = cursor - base_offset;

    Ok(MoeHostLayout {
        offset: base_offset,
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
        inp_gate,
        proj,
        layer_output_scale,
        moe_down_scale: Arc::from(moe_down_scale),
        moe_gate_host: Arc::from(moe_gate_host),
        moe_gate_scale_host: Arc::from(moe_gate_scale_host),
    })
}

/// Copy one planned streamed layer directly from the GGUF mmap into pinned host RAM.
pub fn fill_layer_host_data(
    gguf: &GgufFile,
    layer: usize,
    dst: &mut [u8],
    layout: &MoeHostLayout,
) -> Result<(), LoadError> {
    let pfx = format!("blk.{layer}");

    fn copy_tensor(
        gguf: &GgufFile,
        name: &str,
        dst: &mut [u8],
        slot: &TensorSlot,
    ) -> Result<(), LoadError> {
        let (_ti, data) = gguf
            .tensor_data_by_name(name)
            .map_err(|_| LoadError::MissingTensor(name.to_string()))?;
        let start = slot.off;
        let end = start + slot.size;
        dst[start..end].copy_from_slice(data);
        Ok(())
    }

    copy_tensor(gguf, &format!("{pfx}.attn_q.weight"), dst, &layout.attn_q)?;
    copy_tensor(gguf, &format!("{pfx}.attn_k.weight"), dst, &layout.attn_k)?;
    if let Some(slot) = &layout.attn_v {
        copy_tensor(gguf, &format!("{pfx}.attn_v.weight"), dst, slot)?;
    }
    copy_tensor(
        gguf,
        &format!("{pfx}.attn_output.weight"),
        dst,
        &layout.attn_output,
    )?;
    copy_tensor(
        gguf,
        &format!("{pfx}.ffn_gate.weight"),
        dst,
        &layout.ffn_gate,
    )?;
    copy_tensor(gguf, &format!("{pfx}.ffn_up.weight"), dst, &layout.ffn_up)?;
    copy_tensor(
        gguf,
        &format!("{pfx}.ffn_down.weight"),
        dst,
        &layout.ffn_down,
    )?;
    copy_tensor(
        gguf,
        &format!("{pfx}.ffn_gate_inp.weight"),
        dst,
        &layout.moe_gate,
    )?;
    copy_tensor(
        gguf,
        &format!("{pfx}.ffn_gate_up_exps.weight"),
        dst,
        &layout.moe_gate_up_exps,
    )?;
    copy_tensor(
        gguf,
        &format!("{pfx}.ffn_down_exps.weight"),
        dst,
        &layout.moe_down_exps,
    )?;
    if let Some(slot) = &layout.inp_gate {
        copy_tensor(gguf, &format!("{pfx}.inp_gate.weight"), dst, slot)?;
    }
    if let Some(slot) = &layout.proj {
        copy_tensor(gguf, &format!("{pfx}.proj.weight"), dst, slot)?;
    }

    Ok(())
}

// ─── Streaming shell ────────────────────────────────────────────────────────

/// Allocate a MoE shell with pre-sized GPU buffers for streaming.
/// Sizes are computed from actual GGUF tensor byte sizes (worst case across layers).
pub fn allocate_shell(
    config: &ModelConfig,
    gguf: &GgufFile,
    stream: &Arc<CudaStream>,
) -> Result<MoeLayerWeights, LoadError> {
    let d = config.dim as usize;
    let hd = config.max_head_dim as usize;

    let alloc_u8 = |size: usize| -> Result<CudaSlice<u8>, LoadError> {
        stream
            .alloc_zeros::<u8>(size.max(64)) // minimum 64 bytes
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))
    };
    let alloc_f16 = |size: usize| -> Result<CudaSlice<half::f16>, LoadError> {
        stream
            .alloc_zeros::<half::f16>(size)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))
    };
    let mk_qw = |size: usize| -> Result<QuantWeight, LoadError> {
        Ok(QuantWeight {
            data: alloc_u8(size)?,
            quant_type: GgmlType::F16,
            n_elements: 0,
        })
    };

    // Compute max byte size for each tensor role across all layers
    let tensor_names = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
        "inp_gate.weight",
        "proj.weight",
    ];
    let mut max_sizes = [0usize; 12];
    for layer in 0..config.n_layers {
        for (i, name) in tensor_names.iter().enumerate() {
            let full_name = format!("blk.{layer}.{name}");
            if let Some(ti) = gguf.find_tensor(&full_name) {
                max_sizes[i] = max_sizes[i].max(ti.data_size() as usize);
            }
        }
    }

    info!(
        attn_q_mb = max_sizes[0] / (1024 * 1024),
        gate_up_exps_mb = max_sizes[8] / (1024 * 1024),
        down_exps_mb = max_sizes[9] / (1024 * 1024),
        "MoE shell sizes from GGUF"
    );
    let tiny_quant_bytes = 64usize;

    Ok(MoeLayerWeights {
        attn_norm: alloc_f16(d)?,
        attn_q: mk_qw(max_sizes[0])?,               // attn_q
        attn_k: mk_qw(max_sizes[1])?,               // attn_k
        attn_v: Some(mk_qw(max_sizes[2].max(64))?), // attn_v (0 for full-attn layers)
        attn_output: mk_qw(max_sizes[3])?,          // attn_output
        attn_q_norm: alloc_f16(hd)?,
        attn_k_norm: alloc_f16(hd)?,
        post_attention_norm: alloc_f16(d)?,
        ffn_norm: alloc_f16(d)?,
        ffn_gate: mk_qw(max_sizes[4])?, // ffn_gate (shared)
        ffn_up: mk_qw(max_sizes[5])?,   // ffn_up (shared)
        ffn_down: mk_qw(max_sizes[6])?, // ffn_down (shared)
        post_ffw_norm: alloc_f16(d)?,
        moe_gate: mk_qw(tiny_quant_bytes)?, // router uses persistent router_gates instead
        moe_gate_scale: alloc_f16(d)?,
        moe_gate_host: Arc::from([]),
        moe_gate_scale_host: Arc::from([]),
        moe_gate_up_exps: mk_qw(tiny_quant_bytes)?, // streamed on-demand
        moe_down_exps: mk_qw(tiny_quant_bytes)?,    // streamed on-demand
        pre_ffw_norm_2: alloc_f16(d)?,
        post_ffw_norm_1: alloc_f16(d)?,
        post_ffw_norm_2: alloc_f16(d)?,
        post_norm: Some(alloc_f16(d)?),
        inp_gate: Some(mk_qw(max_sizes[10].max(64))?),
        proj: Some(mk_qw(max_sizes[11].max(64))?),
        layer_output_scale: None,
        moe_down_scale: Arc::from([]),
        has_attn_v: false,
    })
}

/// Upload host layer data into a GPU shell.
pub fn upload_to_shell(
    host_data: &PinnedHostSlice<u8>,
    layout: &MoeHostLayout,
    shell: &mut MoeLayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<(u128, u128), String> {
    let base = layout.offset;
    let host_slice = host_data.as_slice().map_err(|e| e.to_string())?;
    let t_htod = std::time::Instant::now();
    let t_dtod = std::time::Instant::now();

    fn upload_slot(
        host_slice: &[u8],
        base: usize,
        slot: &TensorSlot,
        qw: &mut QuantWeight,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let src = &host_slice[base + slot.off..base + slot.off + slot.size];
        let mut dst = qw.data.slice_mut(0..slot.size);
        stream
            .memcpy_htod(src, &mut dst)
            .map_err(|e| e.to_string())?;
        qw.quant_type = slot.quant_type;
        qw.n_elements = slot.n_elements;
        Ok(())
    }

    upload_slot(host_slice, base, &layout.attn_q, &mut shell.attn_q, stream)?;
    upload_slot(host_slice, base, &layout.attn_k, &mut shell.attn_k, stream)?;
    if let (Some(slot), Some(qw)) = (&layout.attn_v, &mut shell.attn_v) {
        upload_slot(host_slice, base, slot, qw, stream)?;
    }
    shell.has_attn_v = layout.attn_v.is_some();
    upload_slot(
        host_slice,
        base,
        &layout.attn_output,
        &mut shell.attn_output,
        stream,
    )?;
    upload_slot(
        host_slice,
        base,
        &layout.ffn_gate,
        &mut shell.ffn_gate,
        stream,
    )?;
    upload_slot(host_slice, base, &layout.ffn_up, &mut shell.ffn_up, stream)?;
    upload_slot(
        host_slice,
        base,
        &layout.ffn_down,
        &mut shell.ffn_down,
        stream,
    )?;
    // Router gate weights are permanently GPU-resident (router_gates), no shell DMA needed.
    // Streamed expert tensors (gate_up_exps, down_exps) are fetched on-demand per selected expert.
    if let (Some(slot), Some(qw)) = (&layout.inp_gate, &mut shell.inp_gate) {
        upload_slot(host_slice, base, slot, qw, stream)?;
    }
    if let (Some(slot), Some(qw)) = (&layout.proj, &mut shell.proj) {
        upload_slot(host_slice, base, slot, qw, stream)?;
    }
    let htod_us = t_htod.elapsed().as_micros();
    shell.layer_output_scale = layout.layer_output_scale;
    shell.moe_gate.quant_type = layout.moe_gate.quant_type;
    shell.moe_gate.n_elements = layout.moe_gate.n_elements;
    shell.moe_gate_up_exps.quant_type = layout.moe_gate_up_exps.quant_type;
    shell.moe_gate_up_exps.n_elements = layout.moe_gate_up_exps.n_elements;
    shell.moe_down_exps.quant_type = layout.moe_down_exps.quant_type;
    shell.moe_down_exps.n_elements = layout.moe_down_exps.n_elements;
    shell.moe_down_scale = Arc::clone(&layout.moe_down_scale);
    shell.moe_gate_host = Arc::clone(&layout.moe_gate_host);
    shell.moe_gate_scale_host = Arc::clone(&layout.moe_gate_scale_host);

    Ok((htod_us, t_dtod.elapsed().as_micros()))
}

fn enqueue_shell_prefetch(
    moe_weights: &mut MoeModelWeights,
    streamed_idx: usize,
    slot: usize,
    wait_on: Option<&CudaEvent>,
) -> Result<(CudaEvent, u128, u128, u128, u128), KernelError> {
    if let Some(done) = wait_on {
        moe_weights
            .dma_stream
            .wait(done)
            .map_err(|e| KernelError::Launch(e.to_string()))?;
    }

    let layer_idx = moe_weights.n_resident + streamed_idx;
    let layout = &moe_weights.host_layer_offsets[streamed_idx];
    let norms = &moe_weights.layer_norms[layer_idx];
    let shell = if slot == 0 {
        &mut moe_weights.shell_a
    } else {
        &mut moe_weights.shell_b
    };

    let (htod_us, dtod_us) = upload_to_shell(
        &moe_weights.host_layer_data,
        layout,
        shell,
        &moe_weights.dma_stream,
    )
    .map_err(KernelError::Launch)?;
    let t_norms = std::time::Instant::now();
    copy_norms_to_shell(norms, shell, &moe_weights.dma_stream).map_err(KernelError::Launch)?;
    let norms_us = t_norms.elapsed().as_micros();
    let t_event = std::time::Instant::now();
    let ev = moe_weights
        .dma_stream
        .record_event(None)
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    Ok((
        ev,
        htod_us,
        dtod_us,
        norms_us,
        t_event.elapsed().as_micros(),
    ))
}

/// Copy norms from GPU-resident norms into a shell.
pub fn copy_norms_to_shell(
    norms: &MoeStreamingNorms,
    shell: &mut MoeLayerWeights,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    stream
        .memcpy_dtod(&norms.attn_norm, &mut shell.attn_norm)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.attn_q_norm, &mut shell.attn_q_norm)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.attn_k_norm, &mut shell.attn_k_norm)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.post_attention_norm, &mut shell.post_attention_norm)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.ffn_norm, &mut shell.ffn_norm)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.post_ffw_norm, &mut shell.post_ffw_norm)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.moe_gate_scale, &mut shell.moe_gate_scale)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.pre_ffw_norm_2, &mut shell.pre_ffw_norm_2)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.post_ffw_norm_1, &mut shell.post_ffw_norm_1)
        .map_err(|e| e.to_string())?;
    stream
        .memcpy_dtod(&norms.post_ffw_norm_2, &mut shell.post_ffw_norm_2)
        .map_err(|e| e.to_string())?;
    if let (Some(src), Some(dst)) = (&norms.post_norm, &mut shell.post_norm) {
        stream.memcpy_dtod(src, dst).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ─── Full model container ────────────────────────────────────────────────────

pub struct ExpertDmaSlot {
    pub gate_up: QuantWeight,
    pub down: QuantWeight,
    pub cached_streamed_idx: Option<usize>,
    pub cached_expert_id: Option<usize>,
    pub ready: Option<CudaEvent>,
    pub age: u64,
    /// Generation counter: slots loaded in the current token's generation are
    /// preferred eviction targets (they won't be reused until next token).
    /// Cross-token ("stale") entries are preserved for cache hits in later layers.
    pub load_generation: u64,
}

/// All weights for a Gemma 4 MoE model in streaming mode.
pub struct MoeModelWeights {
    pub token_embd: CudaSlice<half::f16>,
    pub output_norm: CudaSlice<half::f16>,
    pub output: QuantWeight,
    /// DiffusionGemma self-conditioning weights (None for non-diffusion models)
    pub sc: Option<crate::weights::ScWeights>,
    pub rope_freq_factors: Option<CudaSlice<f32>>,
    pub per_layer_token_embd: Option<QuantWeight>,
    pub per_layer_model_proj: Option<QuantWeight>,
    pub per_layer_proj_norm: Option<CudaSlice<half::f16>>,

    pub layer_norms: Vec<MoeStreamingNorms>,
    pub resident_layers: Vec<MoeLayerWeights>,
    pub n_resident: usize,

    pub host_layer_data: PinnedHostSlice<u8>,
    pub host_layer_offsets: Vec<MoeHostLayout>,

    pub shell_a: MoeLayerWeights,
    pub shell_b: MoeLayerWeights,
    pub dma_stream: Arc<CudaStream>,
    pub expert_dma_stream: Arc<CudaStream>,

    /// Expert DMA slots for streamed MoE experts.
    pub expert_slots: Vec<ExpertDmaSlot>,
    pub prev_selected_experts: Vec<Vec<usize>>,
    pub expert_weight_buf: CudaSlice<f32>,
    pub expert_gate_up_batch_out: CudaSlice<half::f16>,
    pub expert_act_batch_in: CudaSlice<half::f16>,
    pub expert_down_batch_out: CudaSlice<half::f16>,
    // GPU router: persistent gate weights pre-dequanted to F16 (~20 MB) + scratch buffers
    pub router_gates: Vec<CudaSlice<half::f16>>, // [n_layers] — gate weights as f16 on GPU
    pub router_norm_buf: CudaSlice<half::f16>,   // [dim] f16 — normed+scaled hidden for router
    pub router_logits_buf: CudaSlice<half::f16>, // [n_experts] f16 — router logits
    pub router_topk_ids: CudaSlice<i32>,         // [top_k] i32 — selected expert indices
    pub router_topk_weights: CudaSlice<f32>,     // [top_k] f32 — renormalized weights
    /// Generation counter for expert cache eviction. Incremented each decode token.
    /// Slots from the current generation (loaded this token) are preferred eviction
    /// targets over stale cross-token entries.
    pub expert_cache_generation: u64,
}

/// Reserve VRAM for KV cache (~880 MB), scratch (~300 MB), cuBLAS (~32 MB), headroom.
const POST_LOAD_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;

impl MoeModelWeights {
    /// Load Gemma4 MoE weights.
    /// Works in both fully-resident mode (`n_streamed = 0`) and streaming mode.
    pub fn load(
        gguf: &GgufFile,
        config: &ModelConfig,
        plan: &crate::vram_plan::StreamingPlan,
        alloc: &VramAllocator,
        dequant: &DequantKernels,
        gpu_idx: usize,
    ) -> Result<Self, LoadError> {
        let stream = Arc::clone(alloc.stream(gpu_idx));
        let n_layers = config.n_layers as usize;
        let n_resident = plan.n_resident as usize;
        let n_streamed = plan.n_streamed as usize;

        info!(
            arch = "gemma4_moe",
            layers = n_layers,
            resident = n_resident,
            streamed = n_streamed,
            "loading MoE weights"
        );

        // Global tensors
        let token_embd = upload_and_dequant(gguf, "token_embd.weight", alloc, dequant, gpu_idx)?;
        let output_norm = upload_and_dequant(gguf, "output_norm.weight", alloc, dequant, gpu_idx)?;

        // Output: try output.weight, fallback to tied embeddings
        let output = if gguf.find_tensor("output.weight").is_some() {
            upload_quantized(gguf, "output.weight", alloc, gpu_idx)?
        } else {
            info!("output.weight not found, using tied embeddings as f16 fallback");
            upload_quantized(gguf, "token_embd.weight", alloc, gpu_idx)?
        };

        // DiffusionGemma self-conditioning weights (global gated MLP)
        let sc = if gguf.find_tensor("self_cond_gate.weight").is_some() {
            info!("loading DiffusionGemma self-conditioning weights");
            Some(crate::weights::ScWeights {
                pre_norm: upload_and_dequant(
                    gguf,
                    "self_cond_pre_norm.weight",
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                gate: upload_quantized(gguf, "self_cond_gate.weight", alloc, gpu_idx)?,
                up: upload_quantized(gguf, "self_cond_up.weight", alloc, gpu_idx)?,
                down: upload_quantized(gguf, "self_cond_down.weight", alloc, gpu_idx)?,
            })
        } else {
            None
        };

        let rope_freq_factors = if gguf.find_tensor("rope_freqs.weight").is_some() {
            let (_ti, host_data) = gguf
                .tensor_data_by_name("rope_freqs.weight")
                .map_err(|_| LoadError::MissingTensor("rope_freqs.weight".into()))?;
            let n = host_data.len() / 4;
            let host_f32: Vec<f32> = host_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut gpu_buf = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            stream
                .memcpy_htod(&host_f32, &mut gpu_buf)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            Some(gpu_buf)
        } else {
            None
        };
        let per_layer_token_embd = if config.embd_per_layer.is_some()
            && gguf.find_tensor("per_layer_token_embd.weight").is_some()
        {
            Some(upload_quantized(
                gguf,
                "per_layer_token_embd.weight",
                alloc,
                gpu_idx,
            )?)
        } else {
            None
        };
        let per_layer_model_proj = if config.embd_per_layer.is_some()
            && gguf.find_tensor("per_layer_model_proj.weight").is_some()
        {
            Some(upload_quantized(
                gguf,
                "per_layer_model_proj.weight",
                alloc,
                gpu_idx,
            )?)
        } else {
            None
        };
        let per_layer_proj_norm = if config.embd_per_layer.is_some()
            && gguf.find_tensor("per_layer_proj_norm.weight").is_some()
        {
            Some(upload_and_dequant(
                gguf,
                "per_layer_proj_norm.weight",
                alloc,
                dequant,
                gpu_idx,
            )?)
        } else {
            None
        };

        // Load all layer norms to GPU (always resident)
        let mut layer_norms = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layer_norms.push(load_layer_norms(gguf, i, alloc, dequant, gpu_idx)?);
        }

        // Load resident layers fully to GPU
        let mut resident_layers = Vec::with_capacity(n_resident);
        for i in 0..n_resident {
            resident_layers.push(load_layer_to_gpu(gguf, i, config, alloc, dequant, gpu_idx)?);
        }

        // Plan streamed layers in host RAM first so we can allocate pinned memory once.
        let mut host_layer_offsets = Vec::with_capacity(n_streamed);
        let mut total_host_bytes = 0usize;
        if n_streamed > 0 {
            for i in n_resident..n_layers {
                let layout = plan_layer_host_layout(gguf, i, total_host_bytes)?;
                total_host_bytes += layout.size;
                host_layer_offsets.push(layout);
            }
            info!(
                host_mb = total_host_bytes / (1024 * 1024),
                "MoE streamed layer layout planned"
            );
        } else {
            info!("MoE fully resident mode: no streamed layer host layout");
        }

        // CUDA pinned host allocators generally reject size=0.
        // In fully-resident mode `n_streamed == 0`, so we keep a tiny dummy buffer.
        let pinned_bytes = total_host_bytes.max(1);
        let mut host_layer_data = unsafe { stream.context().alloc_pinned::<u8>(pinned_bytes) }
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        if n_streamed > 0 {
            let host_slice = host_layer_data
                .as_mut_slice()
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            for (host_idx, layer_idx) in (n_resident..n_layers).enumerate() {
                info!(
                    layer = layer_idx,
                    "loading MoE streamed layer to pinned host RAM"
                );
                let layout = &host_layer_offsets[host_idx];
                fill_layer_host_data(
                    gguf,
                    layer_idx,
                    &mut host_slice[layout.offset..layout.offset + layout.size],
                    layout,
                )?;
            }
            info!(
                host_mb = total_host_bytes / (1024 * 1024),
                "MoE streamed layer data loaded into pinned host RAM"
            );
        } else {
            info!("MoE fully resident mode: skipped streamed host buffer fill");
        }

        // Allocate double-buffer shells
        let shell_a = allocate_shell(config, gguf, &stream)?;
        let shell_b = allocate_shell(config, gguf, &stream)?;
        let dma_stream = stream
            .context()
            .new_stream()
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let expert_dma_stream = stream
            .context()
            .new_stream()
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;

        // Expert DMA scratch: one slot per routed expert, sized to actual quantized slices.
        let expert_gate_up_bytes = host_layer_offsets
            .iter()
            .map(|l| l.moe_gate_up_exps.size / config.n_experts as usize)
            .max()
            .unwrap_or(64)
            .max(64);
        let expert_down_bytes = host_layer_offsets
            .iter()
            .map(|l| l.moe_down_exps.size / config.n_experts as usize)
            .max()
            .unwrap_or(64)
            .max(64);
        let mk_expert_qw = |size: usize| -> Result<QuantWeight, LoadError> {
            let data = stream
                .alloc_zeros::<u8>(size)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            Ok(QuantWeight {
                data,
                quant_type: GgmlType::F16,
                n_elements: 0,
            })
        };
        let n_expert_slots =
            (config.n_experts_per_tok as usize * 4).max(config.n_experts_per_tok as usize);
        let mut expert_slots = Vec::with_capacity(n_expert_slots);
        for _ in 0..n_expert_slots {
            expert_slots.push(ExpertDmaSlot {
                gate_up: mk_expert_qw(expert_gate_up_bytes)?,
                down: mk_expert_qw(expert_down_bytes)?,
                cached_streamed_idx: None,
                cached_expert_id: None,
                ready: None,
                age: 0,
                load_generation: 0,
            });
        }
        let expert_ff = config.expert_ff_dim as usize;
        let d = config.dim as usize;
        let expert_weight_buf = stream
            .alloc_zeros::<f32>(config.n_experts_per_tok as usize)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let expert_gate_up_batch_out = stream
            .alloc_zeros::<half::f16>(config.n_experts_per_tok as usize * expert_ff * 2)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let expert_act_batch_in = stream
            .alloc_zeros::<half::f16>(config.n_experts_per_tok as usize * expert_ff)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let expert_down_batch_out = stream
            .alloc_zeros::<half::f16>(config.n_experts_per_tok as usize * d)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        // GPU router: upload all gate weights as F16 permanently (~20 MB for 30 layers)
        let mut router_gates: Vec<CudaSlice<half::f16>> = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let name = format!("blk.{layer}.ffn_gate_inp.weight");
            let qw = upload_quantized(gguf, &name, alloc, gpu_idx)?;
            // Dequant F32 → F16 once at load time
            let n_elems = qw.n_elements as usize;
            let mut f16_buf = stream
                .alloc_zeros::<half::f16>(n_elems)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            dequant
                .dequant(&qw.data, &mut f16_buf, qw.n_elements, qw.quant_type)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            drop(qw); // free F32 GPU buffer
            router_gates.push(f16_buf);
        }
        info!(
            n_layers,
            gate_mb = router_gates.iter().map(|g| g.len() * 2).sum::<usize>() / (1024 * 1024),
            "GPU router: gate weights uploaded (F16)"
        );
        // GPU router scratch (tiny: ~6KB total)
        let router_norm_buf = stream
            .alloc_zeros::<half::f16>(d)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let router_logits_buf = stream
            .alloc_zeros::<half::f16>(config.n_experts as usize)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let router_topk_ids = stream
            .alloc_zeros::<i32>(config.n_experts_per_tok as usize)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let router_topk_weights = stream
            .alloc_zeros::<f32>(config.n_experts_per_tok as usize)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;

        let mut model = Self {
            token_embd,
            output_norm,
            output,
            sc,
            rope_freq_factors,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            layer_norms,
            resident_layers,
            n_resident,
            host_layer_data,
            host_layer_offsets,
            shell_a,
            shell_b,
            dma_stream,
            expert_dma_stream,
            expert_slots,
            prev_selected_experts: vec![Vec::new(); n_layers],
            expert_weight_buf,
            expert_gate_up_batch_out,
            expert_act_batch_in,
            expert_down_batch_out,
            router_gates,
            router_norm_buf,
            router_logits_buf,
            router_topk_ids,
            router_topk_weights,
            expert_cache_generation: 0,
        };

        model.refill_resident_layers(
            gguf,
            config,
            alloc,
            dequant,
            gpu_idx,
            POST_LOAD_RESERVE_BYTES,
        )?;

        info!(
            n_resident = model.n_resident,
            n_streamed = model.host_layer_offsets.len(),
            "MoE weights loaded"
        );

        Ok(model)
    }

    fn refill_resident_layers(
        &mut self,
        gguf: &GgufFile,
        config: &ModelConfig,
        alloc: &VramAllocator,
        dequant: &DequantKernels,
        gpu_idx: usize,
        reserve_bytes: u64,
    ) -> Result<(), LoadError> {
        let stream = alloc.stream(gpu_idx);
        let mut added = 0usize;

        while self.n_resident < config.n_layers as usize && !self.host_layer_offsets.is_empty() {
            stream
                .synchronize()
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            let next_layer = self.n_resident;
            let needed = resident_layer_gpu_bytes(gguf, next_layer)?;
            let (free, _) = stream
                .context()
                .mem_get_info()
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            let free = free as u64;

            if free <= reserve_bytes + needed {
                break;
            }

            info!(
                layer = next_layer,
                free_mb = free / (1024 * 1024),
                needed_mb = needed / (1024 * 1024),
                reserve_mb = reserve_bytes / (1024 * 1024),
                "promoting streamed MoE layer to resident"
            );

            let layer = load_layer_to_gpu(gguf, next_layer, config, alloc, dequant, gpu_idx)?;
            self.resident_layers.push(layer);
            self.n_resident += 1;
            self.host_layer_offsets.remove(0);
            stream
                .synchronize()
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            added += 1;
        }

        if added > 0 {
            stream
                .synchronize()
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            let (free, _) = stream
                .context()
                .mem_get_info()
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            info!(
                added,
                resident = self.n_resident,
                streamed = self.host_layer_offsets.len(),
                free_mb = (free as u64) / (1024 * 1024),
                "post-load resident refill complete"
            );
        }

        Ok(())
    }

    /// Expand expert cache slots to fill available VRAM headroom.
    /// Called after KV cache + scratch are allocated, so we use only truly free VRAM.
    /// More slots = higher cache hit rate between consecutive tokens.
    pub fn expand_expert_cache(
        &mut self,
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
    ) -> Result<(), LoadError> {
        if self.host_layer_offsets.is_empty() {
            return Ok(()); // no streamed layers, no expert DMA needed
        }

        let (free, _) = stream
            .context()
            .mem_get_info()
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;

        // Reserve headroom for runtime allocations
        let headroom = 128 * 1024 * 1024usize; // 128 MB
        let available = if free > headroom {
            free - headroom
        } else {
            return Ok(());
        };

        let expert_gate_up_bytes = self
            .host_layer_offsets
            .iter()
            .map(|l| l.moe_gate_up_exps.size / config.n_experts as usize)
            .max()
            .unwrap_or(0)
            .max(64);
        let expert_down_bytes = self
            .host_layer_offsets
            .iter()
            .map(|l| l.moe_down_exps.size / config.n_experts as usize)
            .max()
            .unwrap_or(0)
            .max(64);

        let slot_bytes = expert_gate_up_bytes + expert_down_bytes;
        if slot_bytes == 0 {
            return Ok(());
        }

        // Target: cache all streamed-layer experts between tokens
        let n_streamed = self.host_layer_offsets.len();
        let target = n_streamed * config.n_experts_per_tok as usize;
        let additional = target.saturating_sub(self.expert_slots.len());
        let can_afford = available / slot_bytes;
        let to_add = additional.min(can_afford);

        if to_add == 0 {
            return Ok(());
        }

        for _ in 0..to_add {
            let gate_up_data = stream
                .alloc_zeros::<u8>(expert_gate_up_bytes)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            let down_data = stream
                .alloc_zeros::<u8>(expert_down_bytes)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            self.expert_slots.push(ExpertDmaSlot {
                gate_up: QuantWeight {
                    data: gate_up_data,
                    quant_type: GgmlType::F16,
                    n_elements: 0,
                },
                down: QuantWeight {
                    data: down_data,
                    quant_type: GgmlType::F16,
                    n_elements: 0,
                },
                cached_streamed_idx: None,
                cached_expert_id: None,
                ready: None,
                age: 0,
                load_generation: 0,
            });
        }

        info!(
            slots = self.expert_slots.len(),
            added = to_add,
            target,
            vram_mb = to_add * slot_bytes / (1024 * 1024),
            "expanded expert cache for cross-token reuse"
        );
        Ok(())
    }
}

fn resident_layer_gpu_bytes(gguf: &GgufFile, layer: usize) -> Result<u64, LoadError> {
    let pfx = format!("blk.{layer}");
    let quant_names = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
        "inp_gate.weight",
        "proj.weight",
    ];
    let f16_names = [
        "attn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "post_attention_norm.weight",
        "ffn_norm.weight",
        "post_ffw_norm.weight",
        "ffn_gate_inp.scale",
        "pre_ffw_norm_2.weight",
        "post_ffw_norm_1.weight",
        "post_ffw_norm_2.weight",
        "post_norm.weight",
    ];

    let mut bytes = 0u64;

    for name in quant_names {
        if let Some(t) = gguf.find_tensor(&format!("{pfx}.{name}")) {
            bytes += t.data_size();
        }
    }
    for name in f16_names {
        if let Some(t) = gguf.find_tensor(&format!("{pfx}.{name}")) {
            bytes += t.n_elements() * 2;
        }
    }

    Ok(bytes)
}

// ─── Forward pass ───────────────────────────────────────────────────────────

use crate::arch::gemma4_common::PerLayerEmbeddings;
use crate::forward::{ScratchBuffers, gemm_q, project_last_logits};
use crate::kv_cache::KvCache;
use chew_kernel::{GpuKernels, KernelError};

fn compute_router_topk_cpu(
    hidden_host: &[f32],
    layer: &MoeLayerWeights,
    config: &ModelConfig,
    seq_len: u32,
    top_k: usize,
) -> Vec<Vec<(usize, f32)>> {
    let dim = config.dim as usize;
    let n_experts = config.n_experts as usize;
    let inv_sqrt_dim = 1.0f32 / (config.dim as f32).sqrt();

    let mut result = Vec::with_capacity(seq_len as usize);

    for tok in 0..seq_len as usize {
        let x = &hidden_host[tok * dim..(tok + 1) * dim];
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rms = (mean_sq + config.rms_norm_eps).sqrt();

        let mut logits = vec![0.0f32; n_experts];
        for (expert, logit) in logits.iter_mut().enumerate() {
            let row = &layer.moe_gate_host[expert * dim..(expert + 1) * dim];
            let mut acc = 0.0f32;
            for i in 0..dim {
                let v = (x[i] / rms) * inv_sqrt_dim * layer.moe_gate_scale_host[i];
                acc += v * row[i];
            }
            if let Some(cap) = config.router_logit_softcap {
                acc = (acc / cap).tanh() * cap;
            }
            *logit = acc;
        }

        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_l: Vec<f32> = logits.iter().map(|x| (x - max_l).exp()).collect();
        let sum_l: f32 = exp_l.iter().sum();
        let probs: Vec<f32> = exp_l.iter().map(|x| x / sum_l).collect();

        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();
        let sel_sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        let selected = selected
            .into_iter()
            .map(|(expert, weight)| (expert, weight / sel_sum.max(f32::MIN_POSITIVE)))
            .collect();
        result.push(selected);
    }

    result
}

/// GEMM without GEMV path — always uses dequant+cuBLAS.
/// Avoids CUDA_ERROR_MISALIGNED_ADDRESS that GEMV has with MoE weights.
fn gemm_no_gemv(
    kernels: &mut GpuKernels,
    a: &CudaSlice<half::f16>,
    w: &QuantWeight,
    c: &mut CudaSlice<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), KernelError> {
    kernels.gemm.matmul_dequant(
        a,
        &w.data,
        w.quant_type,
        w.n_elements,
        c,
        m,
        n,
        k,
        &kernels.dequant,
    )
}

/// Run the MoE forward pass for all layers (streaming mode).
///
/// This handles:
/// - Shared KV (V=K) for full-attention layers
/// - Per-layer n_kv_heads
/// - MoE expert routing with top-k selection
/// - Shared dense FFN + expert FFN combination
pub fn forward_moe_streaming(
    hidden: &mut CudaSlice<f32>,
    moe_weights: &mut MoeModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    pe: Option<&PerLayerEmbeddings>,
    stream: &Arc<CudaStream>,
    diffusion: Option<&crate::arch::diffusion_gemma::DiffusionAttn>,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;
    let stream_ref = Arc::clone(stream);
    // Save buffer for token 0's MoE output: the per-token accumulator reuses
    // attn_mha_out[0..dim], and the write-back only fires for tok>0, so token 0
    // would otherwise be clobbered by later tokens (prefill MoE bug).
    let mut moe_pos0_save = if seq_len > 1 {
        Some(
            stream
                .alloc_zeros::<half::f16>(config.dim as usize)
                .map_err(|e| KernelError::Launch(e.to_string()))?,
        )
    } else {
        None
    };
    let profile = std::env::var("CHEW_PROFILE").is_ok();
    let debug_decode = std::env::var("CHEW_DEBUG_DECODE").is_ok();
    let sync_after_logits =
        profile || debug_decode || std::env::var("CHEW_SYNC_AFTER_LOGITS").is_ok();
    let mut t_stream = 0u128;
    let mut t_norm = 0u128;
    let mut t_gemm = 0u128;
    let mut t_gemm_qkv = 0u128;
    let mut t_gemm_attn_out = 0u128;
    let mut t_gemm_ffn_upgate = 0u128;
    let mut t_gemm_ffn_down = 0u128;
    let mut t_gemm_pe = 0u128;
    let mut t_gemm_logits = 0u128;
    let mut t_rope = 0u128;
    let mut t_kv = 0u128;
    let mut t_swa_mha = 0u128;
    let mut t_full_mha = 0u128;
    let mut t_router = 0u128;
    let mut t_expert = 0u128;
    let mut t_expert_wait = 0u128;
    let mut t_expert_wait_hit = 0u128;
    let mut t_expert_wait_miss = 0u128;
    let mut t_expert_wait_first = 0u128;
    let mut t_expert_wait_later = 0u128;
    let t_expert_load = 0u128;
    let mut t_expert_gate_up = 0u128;
    let mut t_expert_split = 0u128;
    let mut t_expert_act = 0u128;
    let mut t_expert_down = 0u128;
    let mut t_expert_accum = 0u128;
    let mut t_add = 0u128;
    let mut t_prefetch_submit = 0u128;
    let mut t_prefetch_htod = 0u128;
    let mut t_prefetch_dtod = 0u128;
    let mut t_prefetch_norms = 0u128;
    let mut t_prefetch_event = 0u128;
    let mut expert_cache_hits = 0u32;
    let mut expert_cache_misses = 0u32;
    let mut expert_overlap_hits = 0u32;
    let mut expert_overlap_total = 0u32;
    macro_rules! timed {
        ($accum:expr, $body:expr) => {{
            if profile {
                let _ = stream_ref.synchronize();
            }
            let _t0 = std::time::Instant::now();
            let _r = $body;
            if profile {
                let _ = stream_ref.synchronize();
                $accum += _t0.elapsed().as_micros();
            }
            _r
        }};
    }
    let max_layers = std::env::var("CHEW_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(config.n_layers as usize);
    let n_layers = max_layers.min(config.n_layers as usize);

    let mut shell_ready: [Option<CudaEvent>; 2] = [None, None];
    let mut shell_done: [Option<CudaEvent>; 2] = [None, None];

    if n_layers > moe_weights.n_resident {
        let t0 = std::time::Instant::now();
        let (ev, htod, dtod, norms, event_us) = enqueue_shell_prefetch(moe_weights, 0, 0, None)?;
        shell_ready[0] = Some(ev);
        if profile {
            t_prefetch_htod += htod;
            t_prefetch_dtod += dtod;
            t_prefetch_norms += norms;
            t_prefetch_event += event_us;
        }
        if profile {
            t_prefetch_submit += t0.elapsed().as_micros();
        }
    }
    if n_layers > moe_weights.n_resident + 1 {
        let t0 = std::time::Instant::now();
        let (ev, htod, dtod, norms, event_us) = enqueue_shell_prefetch(moe_weights, 1, 1, None)?;
        shell_ready[1] = Some(ev);
        if profile {
            t_prefetch_htod += htod;
            t_prefetch_dtod += dtod;
            t_prefetch_norms += norms;
            t_prefetch_event += event_us;
        }
        if profile {
            t_prefetch_submit += t0.elapsed().as_micros();
        }
    }

    // Debug: check hidden at very start of decode forward
    if seq_len == 1 && debug_decode {
        let mut h = vec![0.0f32; 4];
        stream
            .memcpy_dtoh(&hidden.slice(0..4), &mut h)
            .map_err(|e| KernelError::Launch(e.to_string()))?;
        info!(?h, "MoE forward START hidden (decode)");
    }

    let _generation_unused = moe_weights.expert_cache_generation; // kept for struct compat

    // Raw pointers for GPU router buffers — disjoint from layer weights, safe to use
    // while `layer: &MoeLayerWeights` borrows resident_layers/shells.
    let router_gates_ptr = moe_weights.router_gates.as_ptr(); // persistent gate weights for all layers
    let router_norm_ptr = &mut moe_weights.router_norm_buf as *mut CudaSlice<half::f16>;
    let router_logits_ptr = &mut moe_weights.router_logits_buf as *mut CudaSlice<half::f16>;
    let router_ids_ptr = &mut moe_weights.router_topk_ids as *mut CudaSlice<i32>;
    let router_weights_ptr = &mut moe_weights.router_topk_weights as *mut CudaSlice<f32>;
    // Decode GPU router is now the default fast path. Set CHEW_NO_GPU_ROUTER_DECODE=1
    // to force the conservative CPU router for debugging.
    let decode_gpu_router = seq_len == 1 && std::env::var("CHEW_NO_GPU_ROUTER_DECODE").is_err();
    // Prefill GPU router is opt-in for now: per-token GPU routing can regress
    // vs. the vectorized CPU path on some setups.
    // GPU router for prefill: default ON for diffusion (the CPU router runs
    // per-token and dominated the step), opt-in otherwise.
    let prefill_gpu_router = seq_len > 1
        && (diffusion.is_some() || std::env::var("CHEW_GPU_ROUTER_PREFILL").is_ok());
    let prefill_top_k_default = config.n_experts_per_tok as usize;
    let prefill_top_k = if seq_len > 1 {
        match std::env::var("CHEW_MOE_PREFILL_TOPK") {
            Ok(raw) => match raw.parse::<usize>() {
                Ok(k) if (1..=prefill_top_k_default).contains(&k) => {
                    if k != prefill_top_k_default {
                        info!(
                            prefill_top_k = k,
                            default_top_k = prefill_top_k_default,
                            "using reduced MoE top-k for prefill"
                        );
                    }
                    k
                }
                _ => prefill_top_k_default,
            },
            Err(_) => prefill_top_k_default,
        }
    } else {
        prefill_top_k_default
    };
    let prefill_top_k_u32 = prefill_top_k as u32;
    let use_fused_full_attn = std::env::var("CHEW_NO_FULL_ATTN_FUSED").is_err();
    let use_dual_qkv_gemv = std::env::var("CHEW_NO_MOE_GEMV_DUAL_QKV").is_err();
    let use_dual_ffn_gemv = std::env::var("CHEW_NO_MOE_GEMV_DUAL_FFN").is_err();
    let use_batched_resident_experts = std::env::var("CHEW_MOE_BATCHED_EXPERTS").is_ok();
    let mut router_hidden_tmp = if prefill_gpu_router {
        Some(
            stream
                .alloc_zeros::<f32>(config.dim as usize)
                .map_err(|e| KernelError::Launch(e.to_string()))?,
        )
    } else {
        None
    };

    for layer_idx in 0..n_layers {
        let hd = config.layer_head_dim(layer_idx);
        let kv_heads = config.layer_kv_heads(layer_idx);
        let has_kv = config.has_kv(layer_idx);
        let rope_theta = config.layer_rope_theta(layer_idx);
        let is_swa = config.is_swa(layer_idx);
        let attn_window = config.layer_attention_window(layer_idx);

        // Get layer weights — either from resident or streamed via shell
        let streamed_layout = if layer_idx < moe_weights.n_resident {
            None
        } else {
            let host_idx = layer_idx - moe_weights.n_resident;
            Some(moe_weights.host_layer_offsets[host_idx].clone())
        };

        let layer: &MoeLayerWeights = if layer_idx < moe_weights.n_resident {
            &moe_weights.resident_layers[layer_idx]
        } else {
            let host_idx = layer_idx - moe_weights.n_resident;
            let slot = host_idx % 2;
            if let Some(ready) = &shell_ready[slot] {
                timed!(t_stream, stream.wait(ready))
                    .map_err(|e| KernelError::Launch(e.to_string()))?;
            } else {
                let t0 = std::time::Instant::now();
                let (ev, htod, dtod, norms, event_us) =
                    enqueue_shell_prefetch(moe_weights, host_idx, slot, shell_done[slot].as_ref())?;
                shell_ready[slot] = Some(ev);
                if profile {
                    t_prefetch_htod += htod;
                    t_prefetch_dtod += dtod;
                    t_prefetch_norms += norms;
                    t_prefetch_event += event_us;
                }
                if profile {
                    t_prefetch_submit += t0.elapsed().as_micros();
                }
                if let Some(ready) = &shell_ready[slot] {
                    timed!(t_stream, stream.wait(ready))
                        .map_err(|e| KernelError::Launch(e.to_string()))?;
                }
            }

            let shell_ref = if slot == 0 {
                &moe_weights.shell_a
            } else {
                &moe_weights.shell_b
            };
            shell_ref
        };

        if debug_decode {
            info!(
                layer_idx,
                hd,
                kv_heads,
                is_swa,
                is_resident = layer_idx < moe_weights.n_resident,
                q_dim = config.n_heads * hd,
                kv_dim = kv_heads * hd,
                "MoE layer start"
            );
        }

        // Decode debug: sync after every op to find MISALIGNED source
        let dbg_sync = debug_decode && seq_len == 1 && layer_idx == 0;
        macro_rules! sync_check {
            ($label:expr) => {
                if dbg_sync {
                    if let Err(e) = stream.synchronize() {
                        tracing::error!(%e, label = $label, "CUDA ERROR in decode L0");
                        return Err(KernelError::Launch(format!("{}: {e}", $label)));
                    }
                }
            }
        }

        let is_resident = layer_idx < moe_weights.n_resident;

        // ── 1. Attention norm: f32 hidden → f16 norm_out ──
        if seq_len == 1 && is_resident {
            let x_q8 = kernels.gemv.x_q8_mut();
            timed!(
                t_norm,
                kernels.ops.rms_norm_f32in_q8(
                    hidden,
                    &layer.attn_norm,
                    &mut scratch.norm_out,
                    x_q8,
                    seq_len,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
        } else {
            timed!(
                t_norm,
                kernels.ops.rms_norm_f32in(
                    hidden,
                    &layer.attn_norm,
                    &mut scratch.norm_out,
                    seq_len,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
        }
        sync_check!("attn_norm");


        // ── 2. QKV projections ──
        let q_dim = config.n_heads * hd;
        let kv_dim = kv_heads * hd;

        // GEMV for resident layers, cuBLAS for streamed (shell alignment issues)
        if seq_len == 1 && !is_resident {
            timed!(
                t_gemm,
                kernels.gemv.quantize_input(&scratch.norm_out, config.dim)
            )?;
        }
        let gemm = |k: &mut GpuKernels,
                    a: &CudaSlice<half::f16>,
                    w: &QuantWeight,
                    c: &mut CudaSlice<half::f16>,
                    m: u32,
                    n: u32,
                    kk: u32|
         -> Result<(), KernelError> {
            if is_resident {
                gemm_q(k, a, w, c, m, n, kk)
            } else {
                gemm_no_gemv(k, a, w, c, m, n, kk)
            }
        };
        let _g0 = t_gemm;
        timed!(
            t_gemm,
            gemm(
                kernels,
                &scratch.norm_out,
                &layer.attn_q,
                &mut scratch.q,
                seq_len,
                q_dim,
                config.dim
            )
        )?;
        if profile {
            t_gemm_qkv += t_gemm - _g0;
        }
        sync_check!("Q_proj");
        if layer.has_attn_v && seq_len == 1 && is_resident && use_dual_qkv_gemv {
            let attn_v = layer.attn_v.as_ref().expect("attn_v storage missing");
            if layer.attn_k.quant_type == attn_v.quant_type {
                let _g0 = t_gemm;
                let used = timed!(
                    t_gemm,
                    kernels.gemv.gemv_dual(
                        &layer.attn_k.data,
                        &attn_v.data,
                        &mut scratch.k,
                        &mut scratch.v,
                        kv_dim,
                        config.dim,
                        layer.attn_k.quant_type,
                    )
                )?;
                if !used {
                    timed!(
                        t_gemm,
                        gemm(
                            kernels,
                            &scratch.norm_out,
                            &layer.attn_k,
                            &mut scratch.k,
                            seq_len,
                            kv_dim,
                            config.dim
                        )
                    )?;
                    timed!(
                        t_gemm,
                        gemm(
                            kernels,
                            &scratch.norm_out,
                            attn_v,
                            &mut scratch.v,
                            seq_len,
                            kv_dim,
                            config.dim
                        )
                    )?;
                }
                if profile {
                    t_gemm_qkv += t_gemm - _g0;
                }
            } else {
                let _g0 = t_gemm;
                timed!(
                    t_gemm,
                    gemm(
                        kernels,
                        &scratch.norm_out,
                        &layer.attn_k,
                        &mut scratch.k,
                        seq_len,
                        kv_dim,
                        config.dim
                    )
                )?;
                timed!(
                    t_gemm,
                    gemm(
                        kernels,
                        &scratch.norm_out,
                        attn_v,
                        &mut scratch.v,
                        seq_len,
                        kv_dim,
                        config.dim
                    )
                )?;
                if profile {
                    t_gemm_qkv += t_gemm - _g0;
                }
            }
        } else {
            let _g0 = t_gemm;
            timed!(
                t_gemm,
                gemm(
                    kernels,
                    &scratch.norm_out,
                    &layer.attn_k,
                    &mut scratch.k,
                    seq_len,
                    kv_dim,
                    config.dim
                )
            )?;
            if profile {
                t_gemm_qkv += t_gemm - _g0;
            }
            sync_check!("K_proj");

            if layer.has_attn_v {
                let attn_v = layer.attn_v.as_ref().expect("attn_v storage missing");
                let _g0 = t_gemm;
                timed!(
                    t_gemm,
                    gemm(
                        kernels,
                        &scratch.norm_out,
                        attn_v,
                        &mut scratch.v,
                        seq_len,
                        kv_dim,
                        config.dim
                    )
                )?;
                if profile {
                    t_gemm_qkv += t_gemm - _g0;
                }
            } else {
                let n = (seq_len * kv_dim) as usize;
                let k_view = scratch.k.slice(0..n);
                let mut v_view = scratch.v.slice_mut(0..n);
                timed!(t_kv, stream.memcpy_dtod(&k_view, &mut v_view))
                    .map_err(|e| KernelError::Launch(e.to_string()))?;
            }
        }
        sync_check!("V_proj");
        // Debug: check inputs/outputs for decode
        if debug_decode && layer_idx == 0 && seq_len == 1 {
            let mut h = vec![0.0f32; 4];
            stream.memcpy_dtoh(&hidden.slice(0..4), &mut h).ok();
            info!(?h, "L0 decode hidden input");
            let mut n = vec![half::f16::ZERO; 4];
            stream
                .memcpy_dtoh(&scratch.norm_out.slice(0..4), &mut n)
                .ok();
            info!(?n, "L0 decode norm_out (after attn_norm)");
        }
        if debug_decode && layer_idx == 0 && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.v.slice(0..4), &mut d).ok();
            info!(?d, "L0 decode V first 4");
            stream.memcpy_dtoh(&scratch.q.slice(0..4), &mut d).ok();
            info!(?d, "L0 decode Q first 4");
        }
        // ── 3. QK norms ──
        {
            let src_ptr = &scratch.q as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.q as *mut CudaSlice<half::f16>;
            unsafe {
                timed!(
                    t_norm,
                    kernels.ops.rms_norm(
                        &*src_ptr,
                        &layer.attn_q_norm,
                        &mut *dst_ptr,
                        seq_len * config.n_heads,
                        hd,
                        config.rms_norm_eps,
                    )
                )?;
            }
        }
        {
            let src_ptr = &scratch.k as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.k as *mut CudaSlice<half::f16>;
            unsafe {
                timed!(
                    t_norm,
                    kernels.ops.rms_norm(
                        &*src_ptr,
                        &layer.attn_k_norm,
                        &mut *dst_ptr,
                        seq_len * kv_heads,
                        hd,
                        config.rms_norm_eps,
                    )
                )?;
            }
        }
        // V norm (no weight)
        {
            let src_ptr = &scratch.v as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.v as *mut CudaSlice<half::f16>;
            unsafe {
                timed!(
                    t_norm,
                    kernels.ops.rms_norm_no_weight(
                        &*src_ptr,
                        &mut *dst_ptr,
                        seq_len * kv_heads,
                        hd,
                        config.rms_norm_eps,
                    )
                )?;
            }
        }

        sync_check!("QKV_norms");
        // ── 4. RoPE ──
        timed!(t_rope, {
            kernels
                .ops
                .rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
            kernels
                .ops
                .rope_neox(&mut scratch.k, seq_len, kv_heads, hd, pos, rope_theta)
        })?;

        sync_check!("RoPE");
        if debug_decode && matches!(layer_idx, 5 | 11) && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.q.slice(0..4), &mut d).ok();
            info!(layer_idx, ?d, "decode Q after RoPE");
            stream.memcpy_dtoh(&scratch.k.slice(0..4), &mut d).ok();
            info!(layer_idx, ?d, "decode K after RoPE");
            stream.memcpy_dtoh(&scratch.v.slice(0..4), &mut d).ok();
            info!(layer_idx, ?d, "decode V after norm");
        }
        // ── 5. KV cache write ──
        let kv_source = config.kv_source_layer(layer_idx);
        if has_kv {
            let kv_elems = seq_len * kv_dim;
            {
                let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
                timed!(
                    t_kv,
                    kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)
                )?;
            }
            {
                let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
                timed!(
                    t_kv,
                    kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)
                )?;
            }
        }

        sync_check!("KV_write");
        // ── 6. Multi-Head Attention ──
        {
            let k_full = kv_cache.k_full(kv_source, total_kv_len);
            let v_full = kv_cache.v_full(kv_source, total_kv_len);
            if debug_decode && matches!(layer_idx, 5 | 11) && seq_len == 1 {
                let mut d = vec![half::f16::ZERO; 4];
                stream.memcpy_dtoh(&k_full.slice(0..4), &mut d).ok();
                info!(layer_idx, kv_source, ?d, "decode K cache first 4");
                stream.memcpy_dtoh(&v_full.slice(0..4), &mut d).ok();
                info!(layer_idx, kv_source, ?d, "decode V cache first 4");
            }
            if let Some(diff) = diffusion {
                // Non-causal region-aware attention: an explicit mask fully
                // determines attendance (causality/window are off). Per-layer
                // mask: SWA layers clip prompt keys, global layers see all.
                let mask = if is_swa {
                    &diff.swa_mask
                } else {
                    &diff.global_mask
                };
                // Mask is [seq_len, total_kv_len]: square [n_tokens²] in UNIFIED
                // mode, rectangular [C, P+C] in DECODE (prefix-KV) mode.
                let mask_view = mask.slice(0..(seq_len * total_kv_len) as usize);
                timed!(
                    t_full_mha,
                    kernels.ops.mha_naive_masked(
                        &scratch.q,
                        &k_full,
                        &v_full,
                        &mask_view,
                        &mut scratch.attn_mha_out,
                        hd,
                        config.n_heads,
                        kv_heads,
                        seq_len,
                        total_kv_len,
                        config.attention_scale,
                        config.attn_logit_softcap.unwrap_or(0.0),
                    )
                )?;
            } else if !is_swa {
                if use_fused_full_attn {
                    timed!(
                        t_full_mha,
                        kernels.ops.mha_fused_scaled(
                            &scratch.q,
                            &k_full,
                            &v_full,
                            &mut scratch.attn_mha_out,
                            hd,
                            config.n_heads,
                            kv_heads,
                            seq_len,
                            total_kv_len,
                            pos,
                            0,
                            config.attention_scale,
                            config.attn_logit_softcap.unwrap_or(0.0),
                        )
                    )?;
                } else {
                    timed!(
                        t_full_mha,
                        kernels.ops.mha_naive(
                            &scratch.q,
                            &k_full,
                            &v_full,
                            &mut scratch.attn_mha_out,
                            hd,
                            config.n_heads,
                            kv_heads,
                            seq_len,
                            total_kv_len,
                            pos,
                            0,
                            config.attention_scale,
                            config.attn_logit_softcap.unwrap_or(0.0),
                        )
                    )?;
                }
            } else {
                timed!(
                    t_swa_mha,
                    kernels.ops.mha_fused_scaled(
                        &scratch.q,
                        &k_full,
                        &v_full,
                        &mut scratch.attn_mha_out,
                        hd,
                        config.n_heads,
                        kv_heads,
                        seq_len,
                        total_kv_len,
                        pos,
                        attn_window,
                        config.attention_scale,
                        config.attn_logit_softcap.unwrap_or(0.0),
                    )
                )?;
            }
        }

        if debug_decode && matches!(layer_idx, 0 | 5 | 11) && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream
                .memcpy_dtoh(&scratch.attn_mha_out.slice(0..4), &mut d)
                .ok();
            info!(layer_idx, ?d, "decode MHA out first 4");
        }

        // ── 7. Output projection ──
        if seq_len == 1 && is_resident {
            timed!(
                t_gemm,
                kernels.gemv.quantize_input(&scratch.attn_mha_out, q_dim)
            )?;
        }
        let _g0 = t_gemm;
        timed!(
            t_gemm,
            gemm(
                kernels,
                &scratch.attn_mha_out,
                &layer.attn_output,
                &mut scratch.attn_out,
                seq_len,
                config.dim,
                q_dim
            )
        )?;
        if profile {
            t_gemm_attn_out += t_gemm - _g0;
        }

        // ── 8. Post-attention norm + residual ──
        if debug_decode && matches!(layer_idx, 0 | 5 | 11) && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream
                .memcpy_dtoh(&scratch.attn_out.slice(0..4), &mut d)
                .ok();
            info!(layer_idx, ?d, "decode attn_out before post_norm");
        }
        timed!(
            t_add,
            kernels.ops.post_norm_add(
                hidden,
                &scratch.attn_out,
                &layer.post_attention_norm,
                &mut scratch.norm_out,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )
        )?;
        if debug_decode && matches!(layer_idx, 0 | 5) && seq_len == 1 {
            let mut d = vec![0.0f32; 4];
            stream.memcpy_dtoh(&hidden.slice(0..4), &mut d).ok();
            info!(layer_idx, ?d, "decode hidden after post_attn_norm");
        }

        // ══════════════════════════════════════════════════════════════
        // FFN: Shared MLP + MoE run in PARALLEL on attn_out (= hidden)
        // Following llama.cpp gemma4-iswa.cpp exactly
        //
        // OPTIMIZATION: Router + expert DMA enqueue run BEFORE the shared FFN.
        // The router only needs `hidden` (post-attention), not the FFN output.
        // This lets expert DMA overlap with ~5ms of shared FFN compute.
        // ══════════════════════════════════════════════════════════════

        // ── PHASE A: Router + Expert DMA enqueue (before shared FFN) ──
        let expert_ff = config.expert_ff_dim;
        let n_experts = config.n_experts;
        let streamed_idx = if layer_idx >= moe_weights.n_resident {
            Some(layer_idx - moe_weights.n_resident)
        } else {
            None
        };

        let selected_per_token = if decode_gpu_router {
            // Decode router: mirror llama.cpp more closely with explicit
            // norm -> scale -> gate_scale mul -> matmul -> softmax_topk.
            let gate_weights = unsafe { &*router_gates_ptr.add(layer_idx) };
            let router_norm = unsafe { &mut *router_norm_ptr };
            let router_logits = unsafe { &mut *router_logits_ptr };
            let out_ids = unsafe { &mut *router_ids_ptr };
            let out_weights = unsafe { &mut *router_weights_ptr };
            let inv_sqrt_dim = 1.0f32 / (config.dim as f32).sqrt();
            let softcap = config.router_logit_softcap.unwrap_or(0.0);
            timed!(
                t_router,
                kernels.ops.rms_norm_f32in_no_weight(
                    hidden,
                    router_norm,
                    1,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
            {
                let src_ptr = router_norm as *const CudaSlice<half::f16>;
                let dst_ptr = router_norm as *mut CudaSlice<half::f16>;
                unsafe {
                    timed!(
                        t_router,
                        kernels
                            .ops
                            .scale_f16(&*src_ptr, &mut *dst_ptr, config.dim, inv_sqrt_dim)
                    )?;
                }
            }
            {
                let src_ptr = router_norm as *const CudaSlice<half::f16>;
                let dst_ptr = router_norm as *mut CudaSlice<half::f16>;
                unsafe {
                    timed!(
                        t_router,
                        kernels.ops.mul_f16(
                            &*src_ptr,
                            &layer.moe_gate_scale,
                            &mut *dst_ptr,
                            config.dim,
                        )
                    )?;
                }
            }
            timed!(
                t_router,
                kernels.gemm.matmul_f16(
                    router_norm,
                    gate_weights,
                    router_logits,
                    1,
                    config.n_experts,
                    config.dim,
                )
            )?;
            timed!(
                t_router,
                kernels.ops.softmax_topk(
                    router_logits,
                    out_ids,
                    out_weights,
                    config.n_experts,
                    config.n_experts_per_tok,
                    softcap,
                )
            )?;
            // Small D2H copy (top_k ints + top_k floats) — no full-stream sync
            let top_k = config.n_experts_per_tok as usize;
            let mut ids_host = vec![0i32; top_k];
            let mut weights_host = vec![0.0f32; top_k];
            stream
                .memcpy_dtoh(&out_ids.slice(..top_k), &mut ids_host)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            stream
                .memcpy_dtoh(&out_weights.slice(..top_k), &mut weights_host)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            let selected: Vec<(usize, f32)> = ids_host
                .iter()
                .zip(weights_host.iter())
                .map(|(&id, &w)| (id as usize, w))
                .collect();
            vec![selected]
        } else if seq_len == 1 {
            let mut hidden_host = vec![0.0f32; config.dim as usize];
            stream
                .memcpy_dtoh(&hidden.slice(..), &mut hidden_host)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            timed!(
                t_router,
                Ok::<_, KernelError>(compute_router_topk_cpu(
                    &hidden_host,
                    layer,
                    config,
                    1,
                    config.n_experts_per_tok as usize,
                ))
            )?
        } else if prefill_gpu_router {
            // GPU router for prefill: run per token to avoid large CPU routing cost.
            let gate_weights = unsafe { &*router_gates_ptr.add(layer_idx) };
            let out_ids = unsafe { &mut *router_ids_ptr };
            let out_weights = unsafe { &mut *router_weights_ptr };
            let inv_sqrt_dim = 1.0f32 / (config.dim as f32).sqrt();
            let softcap = config.router_logit_softcap.unwrap_or(0.0);
            let top_k = prefill_top_k;
            let d = config.dim as usize;
            let hidden_tmp = router_hidden_tmp
                .as_mut()
                .expect("prefill GPU router temp buffer not allocated");
            let mut selected_all = Vec::with_capacity(seq_len as usize);
            let mut ids_host = vec![0i32; top_k];
            let mut weights_host = vec![0.0f32; top_k];
            for tok in 0..seq_len as usize {
                let src = hidden.slice(tok * d..(tok + 1) * d);
                let mut dst = hidden_tmp.slice_mut(..d);
                timed!(t_add, stream.memcpy_dtod(&src, &mut dst))
                    .map_err(|e| KernelError::Launch(e.to_string()))?;
                timed!(
                    t_router,
                    kernels.ops.fused_moe_router(
                        hidden_tmp,
                        &layer.moe_gate_scale,
                        gate_weights,
                        out_ids,
                        out_weights,
                        config.dim,
                        config.n_experts,
                        prefill_top_k_u32,
                        config.rms_norm_eps,
                        inv_sqrt_dim,
                        softcap,
                    )
                )?;
                stream
                    .memcpy_dtoh(&out_ids.slice(..top_k), &mut ids_host)
                    .map_err(|e| KernelError::Launch(e.to_string()))?;
                stream
                    .memcpy_dtoh(&out_weights.slice(..top_k), &mut weights_host)
                    .map_err(|e| KernelError::Launch(e.to_string()))?;
                let selected: Vec<(usize, f32)> = ids_host
                    .iter()
                    .zip(weights_host.iter())
                    .map(|(&id, &w)| (id as usize, w))
                    .collect();
                selected_all.push(selected);
            }
            selected_all
        } else {
            // CPU router for prefill (batch > 1), currently default.
            let mut hidden_host = vec![0.0f32; (seq_len * config.dim) as usize];
            stream
                .memcpy_dtoh(&hidden.slice(..), &mut hidden_host)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            timed!(
                t_router,
                Ok::<_, KernelError>(compute_router_topk_cpu(
                    &hidden_host,
                    layer,
                    config,
                    seq_len,
                    prefill_top_k,
                ))
            )?
        };
        if seq_len == 1 {
            let current: Vec<usize> = selected_per_token[0]
                .iter()
                .map(|(expert_id, _)| *expert_id)
                .collect();
            let prev = &moe_weights.prev_selected_experts[layer_idx];
            if !prev.is_empty() {
                for expert_id in current.iter() {
                    if prev.contains(expert_id) {
                        expert_overlap_hits += 1;
                    }
                }
                expert_overlap_total += current.len() as u32;
            }
            moe_weights.prev_selected_experts[layer_idx] = current;
        } else {
            moe_weights.prev_selected_experts[layer_idx].clear();
        }

        // For DECODE (seq_len==1): reserve + enqueue DMA now, overlaps with shared FFN (Phase A/B/C)
        // For PREFILL (seq_len>1): defer reservation to per-token expert loop (HEAD-style interlocked)
        // to avoid cross-token slot stomping.
        let mut decode_selected_slots: Vec<usize> = Vec::new();
        let mut decode_selected_cache_hits: Vec<bool> = Vec::new();
        if seq_len == 1 {
            let dma_stream = Arc::clone(&moe_weights.expert_dma_stream);
            let host_layer_data = &moe_weights.host_layer_data;
            let expert_slots = &mut moe_weights.expert_slots;
            let selected = &selected_per_token[0];
            if let (Some(layout), Some(streamed_idx)) = (&streamed_layout, streamed_idx) {
                let mut misses = Vec::with_capacity(selected.len());
                for &(expert_id, _) in selected.iter() {
                    let mut cache_hit = false;
                    let slot_idx = reserve_streamed_expert_slot(
                        expert_slots,
                        streamed_idx,
                        expert_id,
                        &decode_selected_slots,
                        &mut cache_hit,
                    )?;
                    if cache_hit {
                        expert_cache_hits += 1;
                    } else {
                        expert_cache_misses += 1;
                    }
                    if !cache_hit {
                        misses.push((expert_id, slot_idx));
                    }
                    decode_selected_slots.push(slot_idx);
                    decode_selected_cache_hits.push(cache_hit);
                }
                misses.sort_unstable_by_key(|&(expert_id, _)| expert_id);
                for (expert_id, slot_idx) in misses {
                    let ev = enqueue_streamed_expert_prefetch(
                        &dma_stream,
                        host_layer_data,
                        expert_slots,
                        layout,
                        expert_id,
                        slot_idx,
                        n_experts,
                        None,
                    )?;
                    expert_slots[slot_idx].ready = Some(ev);
                }
            }
        }

        // ── PHASE B: Shared MLP (expert DMA overlaps!) ──
        // norm(attn_out) → gate+up → GELU → down → post_ffw_norm_1
        if seq_len == 1 && is_resident {
            let x_q8 = kernels.gemv.x_q8_mut();
            timed!(
                t_norm,
                kernels.ops.rms_norm_f32in_q8(
                    hidden,
                    &layer.ffn_norm,
                    &mut scratch.norm_out,
                    x_q8,
                    seq_len,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
        } else {
            timed!(
                t_norm,
                kernels.ops.rms_norm_f32in(
                    hidden,
                    &layer.ffn_norm,
                    &mut scratch.norm_out,
                    seq_len,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
        }

        if seq_len == 1 && !is_resident {
            timed!(
                t_gemm,
                kernels.gemv.quantize_input(&scratch.norm_out, config.dim)
            )?;
        }
        let _g0 = t_gemm;
        if seq_len == 1
            && is_resident
            && use_dual_ffn_gemv
            && layer.ffn_gate.quant_type == layer.ffn_up.quant_type
        {
            let used = timed!(
                t_gemm,
                kernels.gemv.gemv_dual(
                    &layer.ffn_gate.data,
                    &layer.ffn_up.data,
                    &mut scratch.ffn_gate_out,
                    &mut scratch.ffn_up_out,
                    config.ff_dim,
                    config.dim,
                    layer.ffn_gate.quant_type,
                )
            )?;
            if !used {
                timed!(
                    t_gemm,
                    gemm(
                        kernels,
                        &scratch.norm_out,
                        &layer.ffn_gate,
                        &mut scratch.ffn_gate_out,
                        seq_len,
                        config.ff_dim,
                        config.dim
                    )
                )?;
                timed!(
                    t_gemm,
                    gemm(
                        kernels,
                        &scratch.norm_out,
                        &layer.ffn_up,
                        &mut scratch.ffn_up_out,
                        seq_len,
                        config.ff_dim,
                        config.dim
                    )
                )?;
            }
        } else {
            timed!(
                t_gemm,
                gemm(
                    kernels,
                    &scratch.norm_out,
                    &layer.ffn_gate,
                    &mut scratch.ffn_gate_out,
                    seq_len,
                    config.ff_dim,
                    config.dim
                )
            )?;
            timed!(
                t_gemm,
                gemm(
                    kernels,
                    &scratch.norm_out,
                    &layer.ffn_up,
                    &mut scratch.ffn_up_out,
                    seq_len,
                    config.ff_dim,
                    config.dim
                )
            )?;
        }
        if profile {
            t_gemm_ffn_upgate += t_gemm - _g0;
        }

        timed!(
            t_norm,
            kernels.ops.gelu(
                &scratch.ffn_gate_out,
                &scratch.ffn_up_out,
                &mut scratch.ffn_silu_out,
                seq_len * config.ff_dim,
            )
        )?;

        if seq_len == 1 && is_resident {
            timed!(
                t_gemm,
                kernels
                    .gemv
                    .quantize_input(&scratch.ffn_silu_out, config.ff_dim)
            )?;
        }
        let _g0 = t_gemm;
        timed!(
            t_gemm,
            gemm(
                kernels,
                &scratch.ffn_silu_out,
                &layer.ffn_down,
                &mut scratch.ffn_out,
                seq_len,
                config.dim,
                config.ff_dim
            )
        )?;
        if profile {
            t_gemm_ffn_down += t_gemm - _g0;
        }

        // RMSNorm shared MLP output (post_ffw_norm_1): ffn_out → attn_out → ffn_out
        timed!(
            t_norm,
            kernels.ops.rms_norm(
                &scratch.ffn_out,
                &layer.post_ffw_norm_1,
                &mut scratch.attn_out,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )
        )?;
        timed!(
            t_add,
            stream.memcpy_dtod(&scratch.attn_out, &mut scratch.ffn_out)
        )
        .map_err(|e| KernelError::Launch(e.to_string()))?;
        // ffn_out now holds cur_mlp (normed shared MLP output)
        if debug_decode && matches!(layer_idx, 0 | 5) && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream
                .memcpy_dtoh(&scratch.ffn_out.slice(0..4), &mut d)
                .ok();
            info!(layer_idx, ?d, "decode cur_mlp (normed shared FFN)");
            let mut h = vec![0.0f32; 4];
            stream.memcpy_dtoh(&hidden.slice(0..4), &mut h).ok();
            info!(layer_idx, ?h, "decode hidden before MoE");
        }

        // ── PHASE C: MoE input norm + Expert compute ──
        {
            {
                let expert_slots = &mut moe_weights.expert_slots;
                let dma_stream_c = Arc::clone(&moe_weights.expert_dma_stream);
                let host_layer_data_c = &moe_weights.host_layer_data;

                // MoE FFN input norm: pre_ffw_norm_2(hidden) → norm_out
                timed!(
                    t_norm,
                    kernels.ops.rms_norm_f32in(
                        hidden,
                        &layer.pre_ffw_norm_2,
                        &mut scratch.norm_out,
                        seq_len,
                        config.dim,
                        config.rms_norm_eps,
                    )
                )?;

                for tok in 0..seq_len as usize {
                    let d = config.dim as usize;
                    if seq_len > 1 && tok > 0 {
                        let src_ptr = &scratch.norm_out as *const CudaSlice<half::f16>;
                        let dst_ptr = &mut scratch.norm_out as *mut CudaSlice<half::f16>;
                        unsafe {
                            let src = (*src_ptr).slice(tok * d..(tok + 1) * d);
                            let mut dst = (*dst_ptr).slice_mut(0..d);
                            timed!(t_add, stream.memcpy_dtod(&src, &mut dst))
                                .map_err(|e| KernelError::Launch(e.to_string()))?;
                        }
                    }
                    let selected = &selected_per_token[tok];

                    // For prefill: reserve+DMA per-token (HEAD-style interlocked)
                    // For decode: use pre-reserved slots from Phase A
                    let (selected_slots_owned, selected_cache_hits_owned);
                    let (selected_slots, selected_cache_hits): (&[usize], &[bool]) = if seq_len > 1
                    {
                        let mut slots = Vec::with_capacity(selected.len());
                        let mut hits = Vec::with_capacity(selected.len());
                        if let (Some(layout), Some(streamed_idx)) = (&streamed_layout, streamed_idx)
                        {
                            let mut misses = Vec::with_capacity(selected.len());
                            for &(expert_id, _) in selected.iter() {
                                let mut cache_hit = false;
                                let slot_idx = reserve_streamed_expert_slot(
                                    expert_slots,
                                    streamed_idx,
                                    expert_id,
                                    &slots,
                                    &mut cache_hit,
                                )?;
                                if cache_hit {
                                    expert_cache_hits += 1;
                                } else {
                                    expert_cache_misses += 1;
                                }
                                if !cache_hit {
                                    misses.push((expert_id, slot_idx));
                                }
                                slots.push(slot_idx);
                                hits.push(cache_hit);
                            }
                            misses.sort_unstable_by_key(|&(eid, _)| eid);
                            for (eid, sid) in misses {
                                let ev = enqueue_streamed_expert_prefetch(
                                    &dma_stream_c,
                                    host_layer_data_c,
                                    expert_slots,
                                    layout,
                                    eid,
                                    sid,
                                    n_experts,
                                    None,
                                )?;
                                expert_slots[sid].ready = Some(ev);
                            }
                        }
                        selected_slots_owned = slots;
                        selected_cache_hits_owned = hits;
                        (&selected_slots_owned, &selected_cache_hits_owned)
                    } else {
                        (&decode_selected_slots[..], &decode_selected_cache_hits[..])
                    };

                    let resident_local = streamed_layout.is_none();
                    if resident_local && use_batched_resident_experts {
                        let batch = selected.len() as u32;
                        let fused_dim = expert_ff * 2;

                        let mut weights_host = Vec::with_capacity(batch as usize);
                        let mut gate_up_elements = 0u32;
                        let mut down_elements = 0u32;
                        let mut gate_up_qtype = GgmlType::F16;
                        let mut down_qtype = GgmlType::F16;
                        let mut gate_up_views = Vec::with_capacity(batch as usize);
                        let mut down_views = Vec::with_capacity(batch as usize);

                        for &(expert_id, weight) in selected.iter() {
                            let mut w_norm = weight;
                            if let Some(&exp_scale) = layer.moe_down_scale.get(expert_id) {
                                w_norm *= exp_scale;
                            }
                            weights_host.push(w_norm);

                            let resident_gu_info = expert_slice_info(
                                &layer.moe_gate_up_exps,
                                expert_id as u32,
                                n_experts,
                            );
                            let resident_dn_info = expert_slice_info(
                                &layer.moe_down_exps,
                                expert_id as u32,
                                n_experts,
                            );
                            gate_up_elements = resident_gu_info.n_elements;
                            down_elements = resident_dn_info.n_elements;
                            gate_up_qtype = resident_gu_info.quant_type;
                            down_qtype = resident_dn_info.quant_type;
                            gate_up_views.push(layer.moe_gate_up_exps.data.slice(
                                resident_gu_info.byte_offset
                                    ..resident_gu_info.byte_offset + resident_gu_info.byte_size,
                            ));
                            down_views.push(layer.moe_down_exps.data.slice(
                                resident_dn_info.byte_offset
                                    ..resident_dn_info.byte_offset + resident_dn_info.byte_size,
                            ));
                        }

                        {
                            let mut w_dst =
                                moe_weights.expert_weight_buf.slice_mut(0..batch as usize);
                            stream
                                .memcpy_htod(&weights_host, &mut w_dst)
                                .map_err(|e| KernelError::Launch(e.to_string()))?;
                        }

                        let _e0 = t_expert;
                        timed!(
                            t_expert,
                            kernels.gemm.matmul_dequant_strided_batched(
                                &scratch.norm_out,
                                &gate_up_views,
                                gate_up_qtype,
                                gate_up_elements,
                                &mut moe_weights.expert_gate_up_batch_out,
                                1,
                                fused_dim,
                                config.dim,
                                &kernels.dequant,
                            )
                        )?;
                        if profile {
                            t_expert_gate_up += t_expert - _e0;
                        }

                        let _e0 = t_expert;
                        timed!(
                            t_expert,
                            kernels.ops.gelu_split_batch(
                                &moe_weights.expert_gate_up_batch_out,
                                &mut moe_weights.expert_act_batch_in,
                                expert_ff,
                                batch,
                            )
                        )?;
                        if profile {
                            t_expert_split += t_expert - _e0;
                            t_expert_act += 0;
                        }

                        let _e0 = t_expert;
                        timed!(
                            t_expert,
                            kernels.gemm.matmul_dequant_strided_batched_a_strided(
                                &moe_weights.expert_act_batch_in,
                                &down_views,
                                down_qtype,
                                down_elements,
                                &mut moe_weights.expert_down_batch_out,
                                1,
                                config.dim,
                                expert_ff,
                                expert_ff as i64,
                                &kernels.dequant,
                            )
                        )?;
                        if profile {
                            t_expert_down += t_expert - _e0;
                        }

                        let _e0 = t_expert;
                        timed!(
                            t_expert,
                            kernels.ops.weighted_sum_rows_f16(
                                &moe_weights.expert_down_batch_out,
                                &moe_weights.expert_weight_buf,
                                &mut scratch.attn_mha_out,
                                config.dim,
                                batch,
                            )
                        )?;
                        if profile {
                            t_expert_accum += t_expert - _e0;
                        }
                    } else {
                        let mut first_expert = true;
                        for (expert_pos, &(expert_id, weight)) in selected.iter().enumerate() {
                            let mut w_norm = weight;
                            if let Some(&exp_scale) = layer.moe_down_scale.get(expert_id) {
                                w_norm *= exp_scale;
                            }

                            let slot_idx = if streamed_layout.is_some() {
                                selected_slots[expert_pos]
                            } else {
                                expert_pos
                            };
                            if streamed_layout.is_some() {
                                if let Some(ready) = expert_slots[slot_idx].ready.as_ref() {
                                    let _e0 = t_expert;
                                    timed!(t_expert, stream.wait(ready))
                                        .map_err(|e| KernelError::Launch(e.to_string()))?;
                                    if profile {
                                        let waited = t_expert - _e0;
                                        t_expert_wait += waited;
                                        if selected_cache_hits
                                            .get(expert_pos)
                                            .copied()
                                            .unwrap_or(false)
                                        {
                                            t_expert_wait_hit += waited;
                                        } else {
                                            t_expert_wait_miss += waited;
                                        }
                                        if expert_pos == 0 {
                                            t_expert_wait_first += waited;
                                        } else {
                                            t_expert_wait_later += waited;
                                        }
                                    }
                                }
                            }

                            let fused_dim = expert_ff * 2;
                            let _e0 = t_expert;
                            if streamed_layout.is_some() {
                                let gate_up_qw = &expert_slots[slot_idx].gate_up;
                                timed!(
                                    t_expert,
                                    kernels.gemm.matmul_dequant(
                                        &scratch.norm_out,
                                        &gate_up_qw.data,
                                        gate_up_qw.quant_type,
                                        gate_up_qw.n_elements,
                                        &mut scratch.ffn_gate_out,
                                        1,
                                        fused_dim,
                                        config.dim,
                                        &kernels.dequant,
                                    )
                                )?;
                            } else {
                                let resident_gu_info = expert_slice_info(
                                    &layer.moe_gate_up_exps,
                                    expert_id as u32,
                                    n_experts,
                                );
                                let resident_gate_up = layer.moe_gate_up_exps.data.slice(
                                    resident_gu_info.byte_offset
                                        ..resident_gu_info.byte_offset + resident_gu_info.byte_size,
                                );
                                timed!(
                                    t_expert,
                                    kernels.gemm.matmul_dequant_view(
                                        &scratch.norm_out,
                                        &resident_gate_up,
                                        resident_gu_info.quant_type,
                                        resident_gu_info.n_elements,
                                        &mut scratch.ffn_gate_out,
                                        1,
                                        fused_dim,
                                        config.dim,
                                        &kernels.dequant,
                                    )
                                )?;
                            }
                            if profile {
                                t_expert_gate_up += t_expert - _e0;
                            }

                            let _e0 = t_expert;
                            timed!(
                                t_expert,
                                kernels.ops.gelu_split_batch(
                                    &scratch.ffn_gate_out,
                                    &mut scratch.ffn_silu_out,
                                    expert_ff,
                                    1,
                                )
                            )?;
                            if profile {
                                t_expert_split += t_expert - _e0;
                                t_expert_act += 0;
                            }

                            let _e0 = t_expert;
                            if streamed_layout.is_some() {
                                let down_qw = &expert_slots[slot_idx].down;
                                timed!(
                                    t_expert,
                                    kernels.gemm.matmul_dequant(
                                        &scratch.ffn_silu_out,
                                        &down_qw.data,
                                        down_qw.quant_type,
                                        down_qw.n_elements,
                                        &mut scratch.attn_out,
                                        1,
                                        config.dim,
                                        expert_ff,
                                        &kernels.dequant,
                                    )
                                )?;
                            } else {
                                let resident_dn_info = expert_slice_info(
                                    &layer.moe_down_exps,
                                    expert_id as u32,
                                    n_experts,
                                );
                                let resident_down = layer.moe_down_exps.data.slice(
                                    resident_dn_info.byte_offset
                                        ..resident_dn_info.byte_offset + resident_dn_info.byte_size,
                                );
                                timed!(
                                    t_expert,
                                    kernels.gemm.matmul_dequant_view(
                                        &scratch.ffn_silu_out,
                                        &resident_down,
                                        resident_dn_info.quant_type,
                                        resident_dn_info.n_elements,
                                        &mut scratch.attn_out,
                                        1,
                                        config.dim,
                                        expert_ff,
                                        &kernels.dequant,
                                    )
                                )?;
                            }
                            if profile {
                                t_expert_down += t_expert - _e0;
                            }

                            if first_expert {
                                let _e0 = t_expert;
                                timed!(
                                    t_expert,
                                    kernels.ops.scale_f16(
                                        &scratch.attn_out,
                                        &mut scratch.attn_mha_out,
                                        config.dim,
                                        w_norm
                                    )
                                )?;
                                if profile {
                                    t_expert_accum += t_expert - _e0;
                                }
                                first_expert = false;
                            } else {
                                let src_ptr = &scratch.attn_out as *const CudaSlice<half::f16>;
                                let dst_ptr = &mut scratch.attn_out as *mut CudaSlice<half::f16>;
                                let _e0 = t_expert;
                                unsafe {
                                    timed!(
                                        t_expert,
                                        kernels.ops.scale_f16(
                                            &*src_ptr,
                                            &mut *dst_ptr,
                                            config.dim,
                                            w_norm
                                        )
                                    )?;
                                }
                                let a = &scratch.attn_out as *const CudaSlice<half::f16>;
                                let b = &scratch.attn_mha_out as *const CudaSlice<half::f16>;
                                let c = &mut scratch.attn_mha_out as *mut CudaSlice<half::f16>;
                                unsafe {
                                    timed!(
                                        t_expert,
                                        kernels.ops.add_f16(&*a, &*b, &mut *c, config.dim)
                                    )?;
                                }
                                if profile {
                                    t_expert_accum += t_expert - _e0;
                                }
                            }
                        }
                    }

                    if seq_len > 1 && tok > 0 {
                        let d = config.dim as usize;
                        let src_ptr = &scratch.attn_mha_out as *const CudaSlice<half::f16>;
                        let dst_ptr = &mut scratch.attn_mha_out as *mut CudaSlice<half::f16>;
                        unsafe {
                            let src = (*src_ptr).slice(0..d);
                            let mut dst = (*dst_ptr).slice_mut(tok * d..(tok + 1) * d);
                            timed!(t_add, stream.memcpy_dtod(&src, &mut dst))
                                .map_err(|e| KernelError::Launch(e.to_string()))?;
                        }
                    } else if seq_len > 1 {
                        // tok == 0: stash before later tokens overwrite [0..dim]
                        let d = config.dim as usize;
                        if let Some(save) = moe_pos0_save.as_mut() {
                            stream
                                .memcpy_dtod(&scratch.attn_mha_out.slice(0..d), save)
                                .map_err(|e| KernelError::Launch(e.to_string()))?;
                        }
                    }
                }
                if seq_len > 1 {
                    // restore token 0's MoE output to position 0
                    let d = config.dim as usize;
                    if let Some(save) = moe_pos0_save.as_ref() {
                        let dst_ptr = &mut scratch.attn_mha_out as *mut CudaSlice<half::f16>;
                        unsafe {
                            let mut dst = (*dst_ptr).slice_mut(0..d);
                            stream
                                .memcpy_dtod(save, &mut dst)
                                .map_err(|e| KernelError::Launch(e.to_string()))?;
                        }
                    }
                }

                // RMSNorm cur_moe (post_ffw_norm_2): attn_mha_out → norm_out
                timed!(
                    t_norm,
                    kernels.ops.rms_norm(
                        &scratch.attn_mha_out,
                        &layer.post_ffw_norm_2,
                        &mut scratch.norm_out,
                        seq_len,
                        config.dim,
                        config.rms_norm_eps,
                    )
                )?;

                // Combine: cur = cur_mlp + cur_moe
                timed!(
                    t_add,
                    kernels.ops.add_f16(
                        &scratch.norm_out,
                        &scratch.ffn_out,
                        &mut scratch.attn_out,
                        seq_len * config.dim
                    )
                )?;
                // Copy result back to ffn_out for post-FFN norm
                timed!(
                    t_add,
                    stream.memcpy_dtod(&scratch.attn_out, &mut scratch.ffn_out)
                )
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            } // close MoE inner block
        } // close MoE outer block

        // ── 10. Post-FFN norm (shared) + residual ──
        // cur = post_ffw_norm(cur_mlp + cur_moe)
        // hidden = cur + attn_out (residual)
        timed!(
            t_add,
            kernels.ops.post_norm_add(
                hidden,
                &scratch.ffn_out,
                &layer.post_ffw_norm,
                &mut scratch.norm_out,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )
        )?;

        if let (Some(inp_gate), Some(proj), Some(post_norm), Some(pe_data), Some(epl)) = (
            layer.inp_gate.as_ref(),
            layer.proj.as_ref(),
            layer.post_norm.as_ref(),
            pe,
            config.embd_per_layer,
        ) {
            let pe_gate = scratch
                .pe_gate_out
                .as_mut()
                .expect("pe_gate_out not allocated");
            let pe_proj = scratch
                .pe_proj_out
                .as_mut()
                .expect("pe_proj_out not allocated");
            let n_elems_pe = (seq_len * config.dim) as usize;
            {
                let mut norm_view = scratch.norm_out.slice_mut(0..n_elems_pe);
                kernels
                    .ops
                    .copy_f32_to_f16(hidden, &mut norm_view, seq_len * config.dim)?;
            }

            if seq_len == 1 && is_resident {
                timed!(
                    t_gemm,
                    kernels.gemv.quantize_input(&scratch.norm_out, config.dim)
                )?;
            }
            let _g0 = t_gemm;
            timed!(
                t_gemm,
                gemm(
                    kernels,
                    &scratch.norm_out,
                    inp_gate,
                    pe_gate,
                    seq_len,
                    epl,
                    config.dim
                )
            )?;
            if profile {
                t_gemm_pe += t_gemm - _g0;
            }

            {
                let src_ptr = pe_gate as *const CudaSlice<half::f16>;
                let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
                unsafe {
                    timed!(
                        t_norm,
                        kernels
                            .ops
                            .gelu_act(&*src_ptr, &mut *dst_ptr, seq_len * epl)
                    )?;
                }
            }

            {
                let src_ptr = pe_gate as *const CudaSlice<half::f16>;
                let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
                let layer_off = (layer_idx as u32) * epl;
                unsafe {
                    timed!(
                        t_add,
                        kernels.ops.pe_strided_mul(
                            &*src_ptr,
                            &pe_data.data,
                            &mut *dst_ptr,
                            epl,
                            pe_data.row_width,
                            layer_off,
                            seq_len,
                        )
                    )?;
                }
            }

            if seq_len == 1 && is_resident {
                timed!(t_gemm, kernels.gemv.quantize_input(pe_gate, epl))?;
            }
            let _g0 = t_gemm;
            timed!(
                t_gemm,
                gemm(kernels, pe_gate, proj, pe_proj, seq_len, config.dim, epl)
            )?;
            if profile {
                t_gemm_pe += t_gemm - _g0;
            }

            timed!(
                t_add,
                kernels.ops.post_norm_add(
                    hidden,
                    pe_proj,
                    post_norm,
                    &mut scratch.norm_out,
                    seq_len,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
        }

        // ── 11. Layer output scale ──
        if let Some(scale) = layer.layer_output_scale {
            timed!(
                t_add,
                kernels
                    .ops
                    .scale_f32_inplace(hidden, seq_len * config.dim, scale)
            )?;
        }

        // Diffusion debug: per-layer hidden stats (works for seq_len>1 too).
        // Hidden state check after layer
        if debug_decode && seq_len == 1 && matches!(layer_idx, 0 | 1 | 5 | 11 | 17 | 23 | 29) {
            let mut d = vec![0.0f32; 8];
            stream
                .memcpy_dtoh(&hidden.slice(0..8), &mut d)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            let has_nan = d.iter().any(|x| x.is_nan());
            let maxv = d.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            info!(layer_idx, has_nan, maxv, ?d, "decode hidden after layer");
        }

        if layer_idx >= moe_weights.n_resident {
            let host_idx = layer_idx - moe_weights.n_resident;
            let slot = host_idx % 2;
            shell_done[slot] = Some(
                stream
                    .record_event(None)
                    .map_err(|e| KernelError::Launch(e.to_string()))?,
            );
            let next_streamed = host_idx + 2;
            if next_streamed < moe_weights.host_layer_offsets.len() {
                let t0 = std::time::Instant::now();
                let (ev, htod, dtod, norms, event_us) = enqueue_shell_prefetch(
                    moe_weights,
                    next_streamed,
                    slot,
                    shell_done[slot].as_ref(),
                )?;
                shell_ready[slot] = Some(ev);
                if profile {
                    t_prefetch_htod += htod;
                    t_prefetch_dtod += dtod;
                    t_prefetch_norms += norms;
                    t_prefetch_event += event_us;
                }
                if profile {
                    t_prefetch_submit += t0.elapsed().as_micros();
                }
            }
        }
    }

    if debug_decode {
        info!(
            pos_before_advance = kv_cache.pos(),
            seq_len, "kv_cache before advance"
        );
    }

    // ── Final: output norm + logits ──
    if seq_len == 1 {
        let x_q8 = kernels.gemv.x_q8_mut();
        timed!(
            t_norm,
            kernels.ops.rms_norm_f32in_q8(
                hidden,
                &moe_weights.output_norm,
                &mut scratch.norm_out,
                x_q8,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )
        )?;
    } else {
        timed!(
            t_norm,
            kernels.ops.rms_norm_f32in(
                hidden,
                &moe_weights.output_norm,
                &mut scratch.norm_out,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )
        )?;
    }

    // Diffusion: the caller projects ALL canvas positions and resets the KV
    // cache itself. scratch.norm_out now holds the final-normed hidden for every
    // position; skip the autoregressive last-logit projection + advance.
    if diffusion.is_some() {
        if profile {
            tracing::info!(
                kv_len = total_kv_len,
                expert_us = t_expert,
                expert_gate_up_us = t_expert_gate_up,
                expert_down_us = t_expert_down,
                expert_accum_us = t_expert_accum,
                router_us = t_router,
                gemm_qkv_us = t_gemm_qkv,
                full_mha_us = t_full_mha,
                swa_mha_us = t_swa_mha,
                norm_us = t_norm,
                "PROFILE diffusion canvas step"
            );
        }
        return Ok(());
    }

    let _g0 = t_gemm;
    timed!(
        t_gemm,
        project_last_logits(
            kernels,
            stream,
            &scratch.norm_out,
            &mut scratch.attn_out,
            &moe_weights.output,
            &mut scratch.logits,
            seq_len,
            config.vocab_size,
            config.dim
        )
    )?;
    if profile {
        t_gemm_logits += t_gemm - _g0;
    }

    // Optional sync after logit GEMM for debug/profiling only.
    // In normal decode this must stay async to avoid per-token host stalls.
    if sync_after_logits {
        stream.synchronize().map_err(|e| {
            tracing::error!(%e, "CUDA error after logit GEMM");
            KernelError::Launch(e.to_string())
        })?;
    }

    // Logit softcap
    if let Some(cap) = config.logit_softcap {
        kernels
            .ops
            .logit_softcap_inplace(&mut scratch.logits, config.vocab_size, cap)?;
    }

    // Advance KV cache position
    kv_cache.advance(seq_len);

    if profile {
        let total = t_stream
            + t_norm
            + t_gemm
            + t_rope
            + t_kv
            + t_swa_mha
            + t_full_mha
            + t_router
            + t_expert
            + t_add;
        info!(
            stream_us = t_stream,
            norm_us = t_norm,
            gemm_us = t_gemm,
            gemm_qkv_us = t_gemm_qkv,
            gemm_attn_out_us = t_gemm_attn_out,
            gemm_ffn_upgate_us = t_gemm_ffn_upgate,
            gemm_ffn_down_us = t_gemm_ffn_down,
            gemm_pe_us = t_gemm_pe,
            gemm_logits_us = t_gemm_logits,
            rope_us = t_rope,
            kv_us = t_kv,
            swa_mha_us = t_swa_mha,
            full_mha_us = t_full_mha,
            router_us = t_router,
            expert_us = t_expert,
            expert_wait_us = t_expert_wait,
            expert_wait_hit_us = t_expert_wait_hit,
            expert_wait_miss_us = t_expert_wait_miss,
            expert_wait_first_us = t_expert_wait_first,
            expert_wait_later_us = t_expert_wait_later,
            expert_load_us = t_expert_load,
            expert_gate_up_us = t_expert_gate_up,
            expert_split_us = t_expert_split,
            expert_act_us = t_expert_act,
            expert_down_us = t_expert_down,
            expert_accum_us = t_expert_accum,
            add_us = t_add,
            prefetch_submit_us = t_prefetch_submit,
            prefetch_htod_us = t_prefetch_htod,
            prefetch_dtod_us = t_prefetch_dtod,
            prefetch_norms_us = t_prefetch_norms,
            prefetch_event_us = t_prefetch_event,
            expert_cache_hits,
            expert_cache_misses,
            expert_overlap_hits,
            expert_overlap_total,
            total_us = total,
            kv_len = total_kv_len,
            resident = moe_weights.n_resident,
            "PROFILE decode step (gemma4_moe)"
        );
    } else if seq_len == 1 && (expert_cache_hits > 0 || expert_cache_misses > 0) {
        let total = expert_cache_hits + expert_cache_misses;
        let hit_pct = if total > 0 {
            expert_cache_hits as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        tracing::debug!(
            expert_cache_hits,
            expert_cache_misses,
            hit_pct = format_args!("{hit_pct:.1}%"),
            slots = moe_weights.expert_slots.len(),
            gen = moe_weights.expert_cache_generation,
            expert_overlap_hits,
            expert_overlap_total,
            "expert cache stats"
        );
    }

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

pub fn expert_slice_info_slot(slot: &TensorSlot, expert_idx: u32, n_experts: u32) -> ExpertSlice {
    let expert_bytes = slot.size / n_experts as usize;
    let expert_elements = slot.n_elements / n_experts;

    ExpertSlice {
        byte_offset: expert_idx as usize * expert_bytes,
        byte_size: expert_bytes,
        n_elements: expert_elements,
        quant_type: slot.quant_type,
    }
}

fn quant_nbytes(n_elements: u32, quant_type: GgmlType) -> usize {
    let bs = quant_type.block_size() as usize;
    let bb = quant_type.block_bytes() as usize;
    n_elements.div_ceil(bs as u32) as usize * bb
}

fn reserve_streamed_expert_slot(
    expert_slots: &mut [ExpertDmaSlot],
    streamed_idx: usize,
    expert_id: usize,
    protected_slots: &[usize],
    cache_hit: &mut bool,
) -> Result<usize, KernelError> {
    let now = expert_slots
        .iter()
        .map(|s| s.age)
        .max()
        .unwrap_or(0)
        .wrapping_add(1);

    // Cache hit — update age and return
    for (slot_idx, slot) in expert_slots.iter_mut().enumerate() {
        if slot.cached_streamed_idx == Some(streamed_idx)
            && slot.cached_expert_id == Some(expert_id)
        {
            slot.age = now;
            *cache_hit = true;
            return Ok(slot_idx);
        }
    }

    // Eviction: 1) Empty, 2) Stale same-layer (safe to evict), 3) Global LRU
    let mut victim = None;
    let mut victim_age = u64::MAX;

    // Priority 1: empty slot
    for (slot_idx, slot) in expert_slots.iter().enumerate() {
        if protected_slots.contains(&slot_idx) {
            continue;
        }
        if slot.cached_streamed_idx.is_none() {
            victim = Some(slot_idx);
            break;
        }
    }

    // Priority 2: stale same-layer entry (different expert, guaranteed not needed this token)
    if victim.is_none() {
        for (slot_idx, slot) in expert_slots.iter().enumerate() {
            if protected_slots.contains(&slot_idx) {
                continue;
            }
            if slot.cached_streamed_idx == Some(streamed_idx)
                && slot.cached_expert_id != Some(expert_id)
                && slot.age < victim_age
            {
                victim_age = slot.age;
                victim = Some(slot_idx);
            }
        }
    }

    // Priority 3: global LRU fallback
    if victim.is_none() {
        victim_age = u64::MAX;
        for (slot_idx, slot) in expert_slots.iter().enumerate() {
            if protected_slots.contains(&slot_idx) {
                continue;
            }
            if slot.age < victim_age {
                victim_age = slot.age;
                victim = Some(slot_idx);
            }
        }
    }

    let slot_idx = victim.unwrap_or(0);
    let slot = &mut expert_slots[slot_idx];
    slot.cached_streamed_idx = Some(streamed_idx);
    slot.cached_expert_id = Some(expert_id);
    slot.ready = None;
    slot.age = now;
    *cache_hit = false;
    Ok(slot_idx)
}

fn enqueue_streamed_expert_prefetch(
    dma_stream: &Arc<CudaStream>,
    host_layer_data: &PinnedHostSlice<u8>,
    expert_slots: &mut [ExpertDmaSlot],
    layout: &MoeHostLayout,
    expert_id: usize,
    slot_idx: usize,
    n_experts: u32,
    wait_on: Option<&CudaEvent>,
) -> Result<CudaEvent, KernelError> {
    if let Some(done) = wait_on {
        dma_stream
            .wait(done)
            .map_err(|e| KernelError::Launch(e.to_string()))?;
    }

    let host = host_layer_data
        .as_slice()
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    let slot = &mut expert_slots[slot_idx];

    let gu = expert_slice_info_slot(&layout.moe_gate_up_exps, expert_id as u32, n_experts);
    let gu_base = layout.offset + layout.moe_gate_up_exps.off + gu.byte_offset;
    let gu_src = &host[gu_base..gu_base + gu.byte_size];
    dma_stream
        .memcpy_htod(gu_src, &mut slot.gate_up.data.slice_mut(0..gu.byte_size))
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    slot.gate_up.quant_type = gu.quant_type;
    slot.gate_up.n_elements = gu.n_elements;

    let dn = expert_slice_info_slot(&layout.moe_down_exps, expert_id as u32, n_experts);
    let dn_base = layout.offset + layout.moe_down_exps.off + dn.byte_offset;
    let dn_src = &host[dn_base..dn_base + dn.byte_size];
    dma_stream
        .memcpy_htod(dn_src, &mut slot.down.data.slice_mut(0..dn.byte_size))
        .map_err(|e| KernelError::Launch(e.to_string()))?;
    slot.down.quant_type = dn.quant_type;
    slot.down.n_elements = dn.n_elements;

    dma_stream
        .record_event(None)
        .map_err(|e| KernelError::Launch(e.to_string()))
}
