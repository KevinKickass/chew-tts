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
    /// Router weights: [dim, n_experts]
    pub moe_gate: QuantWeight,
    /// Router scale: [dim] — applied to rms_norm(attn_out) / sqrt(dim) before router
    pub moe_gate_scale: CudaSlice<half::f16>,
    /// Fused gate+up for all experts: [dim, expert_ff_dim*2, n_experts]
    pub moe_gate_up_exps: QuantWeight,
    /// Down projection for all experts: [expert_ff_dim, dim, n_experts]
    pub moe_down_exps: QuantWeight,
    /// MoE norms
    pub pre_ffw_norm_2: CudaSlice<half::f16>,
    pub post_ffw_norm_1: CudaSlice<half::f16>,
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
    pub moe_gate_scale: CudaSlice<half::f16>,
    pub pre_ffw_norm_2: CudaSlice<half::f16>,
    pub post_ffw_norm_1: CudaSlice<half::f16>,
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
    let moe_gate_scale = upload_and_dequant(gguf, &format!("{pfx}.ffn_gate_inp.scale"), alloc, dequant, gpu_idx)?;
    let moe_gate_up_exps = upload_quantized(gguf, &format!("{pfx}.ffn_gate_up_exps.weight"), alloc, gpu_idx)?;
    let moe_down_exps = upload_quantized(gguf, &format!("{pfx}.ffn_down_exps.weight"), alloc, gpu_idx)?;

    // MoE norms
    let pre_ffw_norm_2 = upload_and_dequant(gguf, &format!("{pfx}.pre_ffw_norm_2.weight"), alloc, dequant, gpu_idx)?;
    let post_ffw_norm_1 = upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm_1.weight"), alloc, dequant, gpu_idx)?;
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
        moe_gate_scale,
        moe_gate_up_exps,
        moe_down_exps,
        pre_ffw_norm_2,
        post_ffw_norm_1,
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
        moe_gate_scale: upload_and_dequant(gguf, &format!("{pfx}.ffn_gate_inp.scale"), alloc, dequant, gpu_idx)?,
        pre_ffw_norm_2: upload_and_dequant(gguf, &format!("{pfx}.pre_ffw_norm_2.weight"), alloc, dequant, gpu_idx)?,
        post_ffw_norm_1: upload_and_dequant(gguf, &format!("{pfx}.post_ffw_norm_1.weight"), alloc, dequant, gpu_idx)?,
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
/// Sizes are computed from actual GGUF tensor byte sizes (worst case across layers).
pub fn allocate_shell(
    config: &ModelConfig,
    gguf: &GgufFile,
    stream: &Arc<CudaStream>,
) -> Result<MoeLayerWeights, LoadError> {
    let d = config.dim as usize;
    let hd = config.max_head_dim as usize;

    let alloc_u8 = |size: usize| -> Result<CudaSlice<u8>, LoadError> {
        stream.alloc_zeros::<u8>(size.max(64)) // minimum 64 bytes
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

    // Compute max byte size for each tensor role across all layers
    let tensor_names = [
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "ffn_gate_inp.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight",
    ];
    let mut max_sizes = [0usize; 10];
    for layer in 0..config.n_layers {
        for (i, name) in tensor_names.iter().enumerate() {
            let full_name = format!("blk.{layer}.{name}");
            if let Some(ti) = gguf.find_tensor(&full_name) {
                max_sizes[i] = max_sizes[i].max(ti.data_size() as usize);
            }
        }
    }

    info!(
        attn_q_mb = max_sizes[0] / (1024*1024),
        gate_up_exps_mb = max_sizes[8] / (1024*1024),
        down_exps_mb = max_sizes[9] / (1024*1024),
        "MoE shell sizes from GGUF"
    );

    Ok(MoeLayerWeights {
        attn_norm: alloc_f16(d)?,
        attn_q: mk_qw(max_sizes[0])?,          // attn_q
        attn_k: mk_qw(max_sizes[1])?,          // attn_k
        attn_v: Some(mk_qw(max_sizes[2].max(64))?), // attn_v (0 for full-attn layers)
        attn_output: mk_qw(max_sizes[3])?,     // attn_output
        attn_q_norm: alloc_f16(hd)?,
        attn_k_norm: alloc_f16(hd)?,
        post_attention_norm: alloc_f16(d)?,
        ffn_norm: alloc_f16(d)?,
        ffn_gate: mk_qw(max_sizes[4])?,        // ffn_gate (shared)
        ffn_up: mk_qw(max_sizes[5])?,          // ffn_up (shared)
        ffn_down: mk_qw(max_sizes[6])?,        // ffn_down (shared)
        post_ffw_norm: alloc_f16(d)?,
        moe_gate: mk_qw(max_sizes[7])?,        // ffn_gate_inp (router)
        moe_gate_scale: alloc_f16(d)?,
        moe_gate_up_exps: mk_qw(max_sizes[8])?, // ffn_gate_up_exps (3D)
        moe_down_exps: mk_qw(max_sizes[9])?,   // ffn_down_exps (3D)
        pre_ffw_norm_2: alloc_f16(d)?,
        post_ffw_norm_1: alloc_f16(d)?,
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
    stream.memcpy_dtod(&norms.moe_gate_scale, &mut shell.moe_gate_scale).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.pre_ffw_norm_2, &mut shell.pre_ffw_norm_2).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.post_ffw_norm_1, &mut shell.post_ffw_norm_1).map_err(|e| e.to_string())?;
    stream.memcpy_dtod(&norms.post_ffw_norm_2, &mut shell.post_ffw_norm_2).map_err(|e| e.to_string())?;
    Ok(())
}

// ─── Full model container ────────────────────────────────────────────────────

/// All weights for a Gemma 4 MoE model in streaming mode.
pub struct MoeModelWeights {
    pub token_embd: CudaSlice<half::f16>,
    pub output_norm: CudaSlice<half::f16>,
    pub output: QuantWeight,
    pub rope_freq_factors: Option<CudaSlice<f32>>,

    pub layer_norms: Vec<MoeStreamingNorms>,
    pub resident_layers: Vec<MoeLayerWeights>,
    pub n_resident: usize,

    pub host_layer_data: Vec<u8>,
    pub host_layer_offsets: Vec<MoeHostLayout>,

    pub shell_a: MoeLayerWeights,
    pub shell_b: MoeLayerWeights,

    /// Scratch buffer for expert weight slicing (single expert, reused)
    pub expert_scratch: QuantWeight,
}

impl MoeModelWeights {
    /// Load the full MoE model with streaming support.
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

        info!(arch = "gemma4_moe", layers = n_layers, resident = n_resident, streamed = n_streamed,
            "loading MoE streaming weights");

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

        let rope_freq_factors = if gguf.find_tensor("rope_freqs.weight").is_some() {
            let (_ti, host_data) = gguf.tensor_data_by_name("rope_freqs.weight")
                .map_err(|_| LoadError::MissingTensor("rope_freqs.weight".into()))?;
            let n = host_data.len() / 4;
            let host_f32: Vec<f32> = host_data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut gpu_buf = stream.alloc_zeros::<f32>(n)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            stream.memcpy_htod(&host_f32, &mut gpu_buf)
                .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
            Some(gpu_buf)
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

        // Load streamed layers to host RAM
        let mut host_layer_data = Vec::new();
        let mut host_layer_offsets = Vec::with_capacity(n_streamed);
        for i in n_resident..n_layers {
            info!(layer = i, "loading MoE streamed layer to host RAM");
            host_layer_offsets.push(load_layer_to_host(gguf, i, &mut host_layer_data)?);
        }
        info!(host_mb = host_layer_data.len() / (1024 * 1024), "MoE streamed layer data loaded");

        // Allocate double-buffer shells
        let shell_a = allocate_shell(config, gguf, &stream)?;
        let shell_b = allocate_shell(config, gguf, &stream)?;

        // Expert scratch: big enough for largest expert slice
        // gate_up_exps per expert: dim * expert_ff_dim * 2 * 2 bytes (f16 upper bound)
        let expert_ff = config.expert_ff_dim as usize;
        let d = config.dim as usize;
        let expert_gate_up_bytes = d * expert_ff * 2 * 2; // fused gate+up, f16 size as upper bound
        let expert_down_bytes = expert_ff * d * 2;
        let expert_scratch_size = expert_gate_up_bytes.max(expert_down_bytes);
        let expert_scratch_data = stream.alloc_zeros::<u8>(expert_scratch_size)
            .map_err(|e| LoadError::Vram(chew_vram::VramError::Alloc(e.to_string())))?;
        let expert_scratch = QuantWeight {
            data: expert_scratch_data,
            quant_type: GgmlType::F16,
            n_elements: 0,
        };

        info!(n_resident, n_streamed, "MoE streaming weights loaded");

        Ok(Self {
            token_embd,
            output_norm,
            output,
            rope_freq_factors,
            layer_norms,
            resident_layers,
            n_resident,
            host_layer_data,
            host_layer_offsets,
            shell_a,
            shell_b,
            expert_scratch,
        })
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────────

use crate::forward::ScratchBuffers;
use crate::kv_cache::KvCache;
use chew_kernel::{GpuKernels, KernelError};

/// GEMM without GEMV path — always uses dequant+cuBLAS.
/// Avoids CUDA_ERROR_MISALIGNED_ADDRESS that GEMV has with MoE weights.
fn gemm_no_gemv(
    kernels: &mut GpuKernels,
    a: &CudaSlice<half::f16>,
    w: &QuantWeight,
    c: &mut CudaSlice<half::f16>,
    m: u32, n: u32, k: u32,
) -> Result<(), KernelError> {
    kernels.gemm.matmul_dequant(a, &w.data, w.quant_type, w.n_elements, c, m, n, k, &kernels.dequant)
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
    stream: &Arc<CudaStream>,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;
    let max_layers = std::env::var("CHEW_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(config.n_layers as usize);
    let n_layers = max_layers.min(config.n_layers as usize);

    let mut use_shell_a = true;

    // Debug: check hidden at very start of decode forward
    if seq_len == 1 {
        let mut h = vec![0.0f32; 4];
        stream.memcpy_dtoh(&hidden.slice(0..4), &mut h)
            .map_err(|e| KernelError::Launch(e.to_string()))?;
        info!(?h, "MoE forward START hidden (decode)");
    }

    for layer_idx in 0..n_layers {
        let hd = config.layer_head_dim(layer_idx);
        let kv_heads = config.layer_kv_heads(layer_idx);
        let has_kv = config.has_kv(layer_idx);
        let rope_theta = config.layer_rope_theta(layer_idx);
        let is_swa = config.is_swa(layer_idx);

        // Get layer weights — either from resident or streamed via shell
        let layer: &MoeLayerWeights = if layer_idx < moe_weights.n_resident {
            &moe_weights.resident_layers[layer_idx]
        } else {
            // Upload streamed layer to shell
            let host_idx = layer_idx - moe_weights.n_resident;
            let layout = &moe_weights.host_layer_offsets[host_idx];
            let norms = &moe_weights.layer_norms[layer_idx];
            let shell = if use_shell_a { &mut moe_weights.shell_a } else { &mut moe_weights.shell_b };

            upload_to_shell(&moe_weights.host_layer_data, layout, shell, stream)
                .map_err(|e| KernelError::Launch(e))?;
            copy_norms_to_shell(norms, shell, stream)
                .map_err(|e| KernelError::Launch(e))?;
            stream.synchronize().map_err(|e| KernelError::Launch(e.to_string()))?;

            let shell_ref = if use_shell_a { &moe_weights.shell_a } else { &moe_weights.shell_b };
            use_shell_a = !use_shell_a;
            shell_ref
        };

        info!(layer_idx, hd, kv_heads, is_swa, q_dim = config.n_heads * hd, kv_dim = kv_heads * hd, "MoE layer start");

        // Decode debug: sync after every op to find MISALIGNED source
        let dbg_sync = seq_len == 1 && layer_idx == 0;
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

        // ── 1. Attention norm: f32 hidden → f16 norm_out ──
        kernels.ops.rms_norm_f32in(
            hidden, &layer.attn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
        sync_check!("attn_norm");

        // ── 2. QKV projections ──
        let q_dim = config.n_heads * hd;
        let kv_dim = kv_heads * hd;

        // NOTE: skip quantize_input for MoE — GEMV has alignment issues with MoE weights
        // All projections use dequant+cuBLAS fallback path instead of GEMV
        gemm_no_gemv(kernels, &scratch.norm_out, &layer.attn_q, &mut scratch.q,
            seq_len, q_dim, config.dim)?;
        sync_check!("Q_proj");
        gemm_no_gemv(kernels, &scratch.norm_out, &layer.attn_k, &mut scratch.k,
            seq_len, kv_dim, config.dim)?;
        sync_check!("K_proj");

        // V = K for shared-KV layers (full attention), or separate V
        if let Some(ref attn_v) = layer.attn_v {
            gemm_no_gemv(kernels, &scratch.norm_out, attn_v, &mut scratch.v,
                seq_len, kv_dim, config.dim)?;
        } else {
            // Shared KV: V = K — copy only kv_dim * seq_len elements
            let n = (seq_len * kv_dim) as usize;
            let k_view = scratch.k.slice(0..n);
            let mut v_view = scratch.v.slice_mut(0..n);
            stream.memcpy_dtod(&k_view, &mut v_view)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
        }

        // Debug: check inputs/outputs for decode
        if layer_idx == 0 && seq_len == 1 {
            let mut h = vec![0.0f32; 4];
            stream.memcpy_dtoh(&hidden.slice(0..4), &mut h).ok();
            info!(?h, "L0 decode hidden input");
            let mut n = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.norm_out.slice(0..4), &mut n).ok();
            info!(?n, "L0 decode norm_out (after attn_norm)");
        }
        if layer_idx == 0 && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.v.slice(0..4), &mut d).ok();
            info!(?d, "L0 decode V first 4");
            stream.memcpy_dtoh(&scratch.q.slice(0..4), &mut d).ok();
            info!(?d, "L0 decode Q first 4");
        }

        sync_check!("V_proj");
        // ── 3. QK norms ──
        {
            let src_ptr = &scratch.q as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.q as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm(
                    &*src_ptr, &layer.attn_q_norm, &mut *dst_ptr,
                    seq_len * config.n_heads, hd, config.rms_norm_eps,
                )?;
            }
        }
        {
            let src_ptr = &scratch.k as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.k as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm(
                    &*src_ptr, &layer.attn_k_norm, &mut *dst_ptr,
                    seq_len * kv_heads, hd, config.rms_norm_eps,
                )?;
            }
        }
        // V norm (no weight)
        {
            let src_ptr = &scratch.v as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.v as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm_no_weight(
                    &*src_ptr, &mut *dst_ptr,
                    seq_len * kv_heads, hd, config.rms_norm_eps,
                )?;
            }
        }

        sync_check!("QKV_norms");
        // ── 4. RoPE ──
        if !is_swa {
            if let Some(ref ff) = moe_weights.rope_freq_factors {
                kernels.ops.rope_neox_freqs(&mut scratch.q, ff, seq_len, config.n_heads, hd, pos, rope_theta)?;
                kernels.ops.rope_neox_freqs(&mut scratch.k, ff, seq_len, kv_heads, hd, pos, rope_theta)?;
            } else {
                kernels.ops.rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
                kernels.ops.rope_neox(&mut scratch.k, seq_len, kv_heads, hd, pos, rope_theta)?;
            }
        } else {
            kernels.ops.rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
            kernels.ops.rope_neox(&mut scratch.k, seq_len, kv_heads, hd, pos, rope_theta)?;
        }

        sync_check!("RoPE");
        // ── 5. KV cache write ──
        let kv_source = config.kv_source_layer(layer_idx);
        if has_kv {
            let kv_elems = seq_len * kv_dim;
            {
                let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
                kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
            }
            {
                let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
                kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
            }
        }

        sync_check!("KV_write");
        // ── 6. Multi-Head Attention ──
        {
            let k_full = kv_cache.k_full(kv_source, total_kv_len);
            let v_full = kv_cache.v_full(kv_source, total_kv_len);
            kernels.ops.mha_fused_scaled(
                &scratch.q, &k_full, &v_full, &mut scratch.attn_mha_out,
                hd, config.n_heads, kv_heads,
                seq_len, total_kv_len, pos,
                config.attention_scale,
            )?;
        }

        if layer_idx == 0 && seq_len == 1 {
            let mut d = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.attn_mha_out.slice(0..4), &mut d).ok();
            info!(?d, "L0 decode MHA out first 4");
        }

        // ── 7. Output projection ──
        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.attn_mha_out, q_dim)?;
        }
        gemm_no_gemv(kernels, &scratch.attn_mha_out, &layer.attn_output, &mut scratch.attn_out,
            seq_len, config.dim, q_dim)?;

        // ── 8. Post-attention norm + residual ──
        if layer_idx == 0 {
            let mut d = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.attn_out.slice(0..4), &mut d).ok();
            info!(?d, "L0 attn_out before post_norm");
        }
        kernels.ops.post_norm_add(
            hidden, &scratch.attn_out, &layer.post_attention_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;
        if layer_idx == 0 {
            let mut d = vec![0.0f32; 4];
            stream.memcpy_dtoh(&hidden.slice(0..4), &mut d).ok();
            info!(?d, "L0 hidden after post_attn_norm");
        }

        // ══════════════════════════════════════════════════════════════
        // FFN: Shared MLP + MoE run in PARALLEL on attn_out (= hidden)
        // Following llama.cpp gemma4-iswa.cpp exactly
        // ══════════════════════════════════════════════════════════════

        // ── 9a. Shared MLP: norm(attn_out) → gate+up → GELU → down → post_ffw_norm_1 ──
        kernels.ops.rms_norm_f32in(
            hidden, &layer.ffn_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;

        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }
        gemm_no_gemv(kernels, &scratch.norm_out, &layer.ffn_gate, &mut scratch.ffn_gate_out,
            seq_len, config.ff_dim, config.dim)?;
        gemm_no_gemv(kernels, &scratch.norm_out, &layer.ffn_up, &mut scratch.ffn_up_out,
            seq_len, config.ff_dim, config.dim)?;

        kernels.ops.gelu(
            &scratch.ffn_gate_out, &scratch.ffn_up_out, &mut scratch.ffn_silu_out,
            seq_len * config.ff_dim,
        )?;

        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        }
        // cur_mlp → ffn_out
        gemm_no_gemv(kernels, &scratch.ffn_silu_out, &layer.ffn_down, &mut scratch.ffn_out,
            seq_len, config.dim, config.ff_dim)?;

        // RMSNorm the shared MLP output (post_ffw_norm_1)
        {
            let src_ptr = &scratch.ffn_out as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.ffn_out as *mut CudaSlice<half::f16>;
            unsafe {
                kernels.ops.rms_norm(
                    &*src_ptr, &layer.post_ffw_norm_1, &mut *dst_ptr,
                    seq_len, config.dim, config.rms_norm_eps,
                )?;
            }
        }
        // ffn_out now holds cur_mlp (normed shared MLP output)
        if layer_idx == 0 {
            let mut d = vec![half::f16::ZERO; 4];
            stream.memcpy_dtoh(&scratch.ffn_out.slice(0..4), &mut d).ok();
            info!(?d, "L0 cur_mlp (normed shared FFN)");
            let mut h = vec![0.0f32; 4];
            stream.memcpy_dtoh(&hidden.slice(0..4), &mut h).ok();
            info!(?h, "L0 hidden before MoE");
        }

        // ── 9b. MoE Router ──
        {
        // Router operates on attn_out (= hidden), NOT on the MLP output
        // llama.cpp: tmp = rms_norm(attn_out) * (1/sqrt(n_embd)) * gate_inp_scale
        //            logits = tmp @ gate_inp
        {
            let expert_ff = config.expert_ff_dim;
            let n_experts = config.n_experts;
            let top_k = config.n_experts_per_tok;
            let dim_scale = 1.0 / (config.dim as f32).sqrt();

            // Router: rms_norm(attn_out) * (1/sqrt(dim)) * gate_scale → router input
            // Use norm_out as temp, attn_mha_out as final router input
            // Step 1: f32 hidden → f16 into norm_out
            kernels.ops.rms_norm_f32in(
                hidden, &layer.attn_norm, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
            // Note: this applies attn_norm weights, but llama.cpp does weightless rms_norm.
            // For now use weighted norm — the gate_scale will compensate somewhat.
            // TODO: proper weightless f32→f16 rms_norm kernel

            // Step 2: scale by 1/sqrt(dim) into attn_mha_out
            kernels.ops.scale_f16(
                &scratch.norm_out, &mut scratch.attn_mha_out,
                seq_len * config.dim, dim_scale,
            )?;

            // Step 3: broadcast multiply by gate_inp_scale [dim] → norm_out
            kernels.ops.mul_f16_broadcast(
                &scratch.attn_mha_out, &layer.moe_gate_scale, &mut scratch.norm_out,
                seq_len * config.dim, config.dim,
            )?;

            // Router GEMM: router_input (norm_out) @ gate_inp → logits [seq_len, n_experts]
            if seq_len == 1 {
                kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
            }
            // Reuse attn_out for logits (n_experts=128 < dim=2816, fits)
            gemm_no_gemv(kernels, &scratch.norm_out, &layer.moe_gate, &mut scratch.attn_out,
                seq_len, n_experts, config.dim)?;

            // ── 9c. MoE FFN input (separate norm on attn_out) ──
            // cur_moe_input = pre_ffw_norm_2(attn_out)
            kernels.ops.rms_norm_f32in(
                hidden, &layer.pre_ffw_norm_2, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;
            // norm_out now holds cur_moe input

            // ── 9d. Expert selection (CPU for decode) ──
            let n_exp = n_experts as usize;
            let mut router_logits_host = vec![half::f16::ZERO; n_exp * seq_len as usize];
            stream.memcpy_dtoh(&scratch.attn_out.slice(0..router_logits_host.len()), &mut router_logits_host)
                .map_err(|e| KernelError::Launch(e.to_string()))?;

            // For prefill: only route last token through experts (sufficient for next-token prediction)
            // For decode: single token, so tok=0
            // TODO: full per-token expert routing for proper prefill
            let tok = (seq_len as usize) - 1; // last token only
            {
                let tok_logits = &router_logits_host[tok * n_exp..(tok + 1) * n_exp];
                let logits_f32: Vec<f32> = tok_logits.iter().map(|x| x.to_f32()).collect();
                let max_l = logits_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_l: Vec<f32> = logits_f32.iter().map(|x| (x - max_l).exp()).collect();
                let sum_l: f32 = exp_l.iter().sum();
                let probs: Vec<f32> = exp_l.iter().map(|x| x / sum_l).collect();

                let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k as usize).collect();
                let sel_sum: f32 = selected.iter().map(|(_, w)| w).sum();

                // ── 9e. Expert computation ──
                let mut first_expert = true;
                for &(expert_id, weight) in &selected {
                    let w_norm = weight / sel_sum;

                    // Copy expert gate_up slice to scratch
                    let gu_info = expert_slice_info(&layer.moe_gate_up_exps, expert_id as u32, n_experts);
                    let gu_end = gu_info.byte_offset + gu_info.byte_size;
                    let gu_total = layer.moe_gate_up_exps.data.len();
                    if gu_end > gu_total {
                        tracing::error!(expert_id, byte_offset=gu_info.byte_offset, byte_size=gu_info.byte_size, gu_end, gu_total, n_experts, "gate_up_exps OOB!");
                        return Err(KernelError::Launch(format!("gate_up_exps OOB: {gu_end} > {gu_total}")));
                    }
                    stream.memcpy_dtod(
                        &layer.moe_gate_up_exps.data.slice(gu_info.byte_offset..gu_end),
                        &mut moe_weights.expert_scratch.data.slice_mut(0..gu_info.byte_size),
                    ).map_err(|e| KernelError::Launch(e.to_string()))?;
                    moe_weights.expert_scratch.quant_type = gu_info.quant_type;
                    moe_weights.expert_scratch.n_elements = gu_info.n_elements;

                    // norm_out @ expert_gate_up → [1, expert_ff*2]
                    let fused_dim = expert_ff * 2;
                    // Force dequant+cuBLAS path for expert GEMMs (GEMV can misalign on scratch)
                    kernels.gemm.matmul_dequant(
                        &scratch.norm_out, &moe_weights.expert_scratch.data,
                        moe_weights.expert_scratch.quant_type, moe_weights.expert_scratch.n_elements,
                        &mut scratch.ffn_gate_out, 1, fused_dim, config.dim, &kernels.dequant,
                    )?;

                    // Split gate/up and GELU
                    let n = expert_ff as usize;
                    stream.memcpy_dtod(
                        &scratch.ffn_gate_out.slice(0..n),
                        &mut scratch.ffn_silu_out.slice_mut(0..n),
                    ).map_err(|e| KernelError::Launch(e.to_string()))?;
                    stream.memcpy_dtod(
                        &scratch.ffn_gate_out.slice(n..n*2),
                        &mut scratch.ffn_up_out.slice_mut(0..n),
                    ).map_err(|e| KernelError::Launch(e.to_string()))?;
                    kernels.ops.gelu(
                        &scratch.ffn_silu_out, &scratch.ffn_up_out, &mut scratch.ffn_gate_out,
                        expert_ff,
                    )?;

                    // Copy expert down slice to scratch
                    let dn_info = expert_slice_info(&layer.moe_down_exps, expert_id as u32, n_experts);
                    stream.memcpy_dtod(
                        &layer.moe_down_exps.data.slice(dn_info.byte_offset..dn_info.byte_offset + dn_info.byte_size),
                        &mut moe_weights.expert_scratch.data.slice_mut(0..dn_info.byte_size),
                    ).map_err(|e| KernelError::Launch(e.to_string()))?;
                    moe_weights.expert_scratch.quant_type = dn_info.quant_type;
                    moe_weights.expert_scratch.n_elements = dn_info.n_elements;

                    // expert_result @ down → [1, dim]
                    kernels.gemm.matmul_dequant(
                        &scratch.ffn_gate_out, &moe_weights.expert_scratch.data,
                        moe_weights.expert_scratch.quant_type, moe_weights.expert_scratch.n_elements,
                        &mut scratch.attn_out, 1, config.dim, expert_ff, &kernels.dequant,
                    )?;

                    // Weighted accumulate into attn_mha_out (reuse as moe accumulator)
                    if first_expert {
                        kernels.ops.scale_f16(&scratch.attn_out, &mut scratch.attn_mha_out,
                            config.dim, w_norm)?;
                        first_expert = false;
                    } else {
                        let src_ptr = &scratch.attn_out as *const CudaSlice<half::f16>;
                        let dst_ptr = &mut scratch.attn_out as *mut CudaSlice<half::f16>;
                        unsafe { kernels.ops.scale_f16(&*src_ptr, &mut *dst_ptr, config.dim, w_norm)?; }
                        let a = &scratch.attn_out as *const CudaSlice<half::f16>;
                        let b = &scratch.attn_mha_out as *const CudaSlice<half::f16>;
                        let c = &mut scratch.attn_mha_out as *mut CudaSlice<half::f16>;
                        unsafe { kernels.ops.add_f16(&*a, &*b, &mut *c, config.dim)?; }
                    }
                }
            }
            // attn_mha_out now holds cur_moe (weighted expert sum)

            // Sync to catch any pending CUDA errors from expert loop
            stream.synchronize().map_err(|e| {
                tracing::error!(%e, "CUDA error after expert loop");
                KernelError::Launch(e.to_string())
            })?;

            // RMSNorm cur_moe (post_ffw_norm_2): attn_mha_out → norm_out
            kernels.ops.rms_norm(
                &scratch.attn_mha_out, &layer.post_ffw_norm_2, &mut scratch.norm_out,
                seq_len, config.dim, config.rms_norm_eps,
            )?;

            // ── 9f. Combine: cur = cur_mlp + cur_moe ──
            // ffn_out = cur_mlp, norm_out = normed cur_moe
            // Result into ffn_out
            kernels.ops.add_f16(&scratch.norm_out, &scratch.ffn_out, &mut scratch.attn_out,
                seq_len * config.dim)?;
            // Copy result back to ffn_out for post-FFN norm
            stream.memcpy_dtod(&scratch.attn_out, &mut scratch.ffn_out)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            } // close MoE inner block
        } // close MoE outer block

        // ── 10. Post-FFN norm (shared) + residual ──
        // cur = post_ffw_norm(cur_mlp + cur_moe)
        // hidden = cur + attn_out (residual)
        kernels.ops.post_norm_add(
            hidden, &scratch.ffn_out, &layer.post_ffw_norm, &mut scratch.norm_out,
            seq_len, config.dim, config.rms_norm_eps,
        )?;

        // ── 11. Layer output scale ──
        if let Some(scale) = layer.layer_output_scale {
            kernels.ops.scale_f32_inplace(hidden, seq_len * config.dim, scale)?;
        }

        // Hidden state check after layer
        {
            let mut d = vec![0.0f32; 8];
            stream.memcpy_dtoh(&hidden.slice(0..8), &mut d)
                .map_err(|e| KernelError::Launch(e.to_string()))?;
            let has_nan = d.iter().any(|x| x.is_nan());
            let maxv = d.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            info!(layer_idx, has_nan, maxv, ?d, "hidden after layer");
        }
    }

    // ── Final: output norm + logits ──
    kernels.ops.rms_norm_f32in(
        hidden, &moe_weights.output_norm, &mut scratch.norm_out,
        seq_len, config.dim, config.rms_norm_eps,
    )?;

    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }
    gemm_no_gemv(kernels, &scratch.norm_out, &moe_weights.output, &mut scratch.logits,
        seq_len, config.vocab_size, config.dim)?;

    // Sync after logit GEMM to catch errors
    stream.synchronize().map_err(|e| {
        tracing::error!(%e, "CUDA error after logit GEMM");
        KernelError::Launch(e.to_string())
    })?;

    // Logit softcap
    if let Some(cap) = config.logit_softcap {
        kernels.ops.logit_softcap_inplace(&mut scratch.logits, seq_len * config.vocab_size, cap)?;
    }

    // Advance KV cache position
    kv_cache.advance(seq_len);

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
