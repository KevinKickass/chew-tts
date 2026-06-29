use crate::arch::gemma4_common::PerLayerEmbeddings;
use crate::config::ModelConfig;
use crate::forward::{ScratchBuffers, gemm_q};
use crate::kv_cache::KvCache;
use crate::weights::{LayerWeights, StreamingWeights};
use chew_kernel::{GpuKernels, KernelError};
use cudarc::driver::CudaSlice;

pub fn forward_layer_llama(
    hidden: &mut CudaSlice<f32>,
    layer: &LayerWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    layer_idx: usize,
    next_attn_norm: Option<&CudaSlice<half::f16>>,
    n_elems: u32,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;

    if seq_len == 1 && layer_idx > 0 {}
    gemm_q(
        kernels,
        &scratch.norm_out,
        &layer.attn_q,
        &mut scratch.q,
        seq_len,
        config.n_heads * config.head_dim,
        config.dim,
    )?;
    if seq_len == 1 && layer.attn_k.quant_type == layer.attn_v.quant_type {
        let nk = config.n_kv_heads * config.head_dim;
        let used = kernels.gemv.gemv_dual(
            &layer.attn_k.data,
            &layer.attn_v.data,
            &mut scratch.k,
            &mut scratch.v,
            nk,
            config.dim,
            layer.attn_k.quant_type,
        )?;
        if !used {
            gemm_q(
                kernels,
                &scratch.norm_out,
                &layer.attn_k,
                &mut scratch.k,
                seq_len,
                config.n_kv_heads * config.head_dim,
                config.dim,
            )?;
            gemm_q(
                kernels,
                &scratch.norm_out,
                &layer.attn_v,
                &mut scratch.v,
                seq_len,
                config.n_kv_heads * config.head_dim,
                config.dim,
            )?;
        }
    } else {
        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.attn_k,
            &mut scratch.k,
            seq_len,
            config.n_kv_heads * config.head_dim,
            config.dim,
        )?;
        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.attn_v,
            &mut scratch.v,
            seq_len,
            config.n_kv_heads * config.head_dim,
            config.dim,
        )?;
    }

    kernels.ops.rope(
        &mut scratch.q,
        seq_len,
        config.n_heads,
        config.head_dim,
        pos,
        config.rope_theta,
    )?;
    kernels.ops.rope(
        &mut scratch.k,
        seq_len,
        config.n_kv_heads,
        config.head_dim,
        pos,
        config.rope_theta,
    )?;

    let kv_elems = seq_len * config.n_kv_heads * config.head_dim;
    {
        let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
        kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
    }
    {
        let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
        kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
    }

    {
        let k_full = kv_cache.k_full(layer_idx, total_kv_len);
        let v_full = kv_cache.v_full(layer_idx, total_kv_len);
        kernels.ops.mha_fused(
            &scratch.q,
            &k_full,
            &v_full,
            &mut scratch.attn_mha_out,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            seq_len,
            total_kv_len,
            pos,
        )?;
    }

    if seq_len == 1 {
        kernels
            .gemv
            .quantize_input(&scratch.attn_mha_out, config.n_heads * config.head_dim)?;
    }
    gemm_q(
        kernels,
        &scratch.attn_mha_out,
        &layer.attn_output,
        &mut scratch.attn_out,
        seq_len,
        config.dim,
        config.n_heads * config.head_dim,
    )?;

    if seq_len == 1 {
        let x_q8 = kernels.gemv.x_q8_mut();
        kernels.ops.fused_add_rmsnorm_q8(
            hidden,
            &scratch.attn_out,
            &layer.ffn_norm,
            &mut scratch.norm_out,
            x_q8,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
    } else {
        kernels.ops.fused_add_rmsnorm(
            hidden,
            &scratch.attn_out,
            &layer.ffn_norm,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
    }

    if seq_len == 1 && layer.ffn_gate.quant_type == layer.ffn_up.quant_type {
        let used = kernels.gemv.gemv_dual(
            &layer.ffn_gate.data,
            &layer.ffn_up.data,
            &mut scratch.ffn_gate_out,
            &mut scratch.ffn_up_out,
            config.ff_dim,
            config.dim,
            layer.ffn_gate.quant_type,
        )?;
        if !used {
            gemm_q(
                kernels,
                &scratch.norm_out,
                &layer.ffn_gate,
                &mut scratch.ffn_gate_out,
                seq_len,
                config.ff_dim,
                config.dim,
            )?;
            gemm_q(
                kernels,
                &scratch.norm_out,
                &layer.ffn_up,
                &mut scratch.ffn_up_out,
                seq_len,
                config.ff_dim,
                config.dim,
            )?;
        }
    } else {
        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.ffn_gate,
            &mut scratch.ffn_gate_out,
            seq_len,
            config.ff_dim,
            config.dim,
        )?;
        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.ffn_up,
            &mut scratch.ffn_up_out,
            seq_len,
            config.ff_dim,
            config.dim,
        )?;
    }

    kernels.ops.silu(
        &scratch.ffn_gate_out,
        &scratch.ffn_up_out,
        &mut scratch.ffn_silu_out,
        seq_len * config.ff_dim,
    )?;

    if seq_len == 1 {
        kernels
            .gemv
            .quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
    }
    gemm_q(
        kernels,
        &scratch.ffn_silu_out,
        &layer.ffn_down,
        &mut scratch.ffn_out,
        seq_len,
        config.dim,
        config.ff_dim,
    )?;

    if let Some(next_norm) = next_attn_norm {
        if seq_len == 1 {
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.fused_add_rmsnorm_q8(
                hidden,
                &scratch.ffn_out,
                next_norm,
                &mut scratch.norm_out,
                x_q8,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.fused_add_rmsnorm(
                hidden,
                &scratch.ffn_out,
                next_norm,
                &mut scratch.norm_out,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )?;
        }
    } else {
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.ffn_out, n_elems)?;
    }

    Ok(())
}

pub fn forward_layer_gemma4(
    hidden: &mut CudaSlice<f32>,
    layer: &LayerWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
    layer_idx: usize,
    pos: u32,
    total_kv_len: u32,
    pe: Option<&PerLayerEmbeddings>,
    sw: &StreamingWeights,
) -> Result<(), KernelError> {
    let hd = config.layer_head_dim(layer_idx);
    let has_kv = config.has_kv(layer_idx);
    let rope_theta = config.layer_rope_theta(layer_idx);

    kernels.ops.rms_norm_f32in(
        hidden,
        &layer.attn_norm,
        &mut scratch.norm_out,
        seq_len,
        config.dim,
        config.rms_norm_eps,
    )?;

    let kv_heads = config.layer_kv_heads(layer_idx);
    let q_dim = config.n_heads * hd;
    let kv_dim = kv_heads * hd;

    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }

    gemm_q(
        kernels,
        &scratch.norm_out,
        &layer.attn_q,
        &mut scratch.q,
        seq_len,
        q_dim,
        config.dim,
    )?;
    gemm_q(
        kernels,
        &scratch.norm_out,
        &layer.attn_k,
        &mut scratch.k,
        seq_len,
        kv_dim,
        config.dim,
    )?;
    gemm_q(
        kernels,
        &scratch.norm_out,
        &layer.attn_v,
        &mut scratch.v,
        seq_len,
        kv_dim,
        config.dim,
    )?;

    if let Some(ref q_norm) = layer.attn_q_norm {
        let src_ptr = &scratch.q as *const CudaSlice<half::f16>;
        let dst_ptr = &mut scratch.q as *mut CudaSlice<half::f16>;
        unsafe {
            kernels.ops.rms_norm(
                &*src_ptr,
                q_norm,
                &mut *dst_ptr,
                seq_len * config.n_heads,
                hd,
                config.rms_norm_eps,
            )?;
        }
    }
    if let Some(ref k_norm) = layer.attn_k_norm {
        let src_ptr = &scratch.k as *const CudaSlice<half::f16>;
        let dst_ptr = &mut scratch.k as *mut CudaSlice<half::f16>;
        unsafe {
                kernels.ops.rms_norm(
                    &*src_ptr,
                    k_norm,
                    &mut *dst_ptr,
                    seq_len * kv_heads,
                    hd,
                    config.rms_norm_eps,
                )?;
        }
    }

    {
        let src_ptr = &scratch.v as *const CudaSlice<half::f16>;
        let dst_ptr = &mut scratch.v as *mut CudaSlice<half::f16>;
        unsafe {
                kernels.ops.rms_norm_no_weight(
                    &*src_ptr,
                    &mut *dst_ptr,
                    seq_len * kv_heads,
                    hd,
                    config.rms_norm_eps,
                )?;
        }
    }

    let is_swa = config.is_swa(layer_idx);
    let attn_window = config.layer_attention_window(layer_idx);
    if !is_swa {
        if let Some(ref ff) = sw.rope_freq_factors {
            kernels.ops.rope_neox_freqs(
                &mut scratch.q,
                ff,
                seq_len,
                config.n_heads,
                hd,
                pos,
                rope_theta,
            )?;
            kernels.ops.rope_neox_freqs(
                &mut scratch.k,
                ff,
                seq_len,
                kv_heads,
                hd,
                pos,
                rope_theta,
            )?;
        } else {
            kernels
                .ops
                .rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
            kernels.ops.rope_neox(
                &mut scratch.k,
                seq_len,
                kv_heads,
                hd,
                pos,
                rope_theta,
            )?;
        }
    } else {
        kernels
            .ops
            .rope_neox(&mut scratch.q, seq_len, config.n_heads, hd, pos, rope_theta)?;
        kernels.ops.rope_neox(
            &mut scratch.k,
            seq_len,
            kv_heads,
            hd,
            pos,
            rope_theta,
        )?;
    }

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

    {
        let k_full = kv_cache.k_full(kv_source, total_kv_len);
        let v_full = kv_cache.v_full(kv_source, total_kv_len);
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
        )?;
    }

    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.attn_mha_out, q_dim)?;
    }
    gemm_q(
        kernels,
        &scratch.attn_mha_out,
        &layer.attn_output,
        &mut scratch.attn_out,
        seq_len,
        config.dim,
        q_dim,
    )?;

    if let Some(ref pan) = layer.post_attention_norm {
        kernels.ops.post_norm_add(
            hidden,
            &scratch.attn_out,
            pan,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
    } else {
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.attn_out, seq_len * config.dim)?;
    }

    kernels.ops.rms_norm_f32in(
        hidden,
        &layer.ffn_norm,
        &mut scratch.norm_out,
        seq_len,
        config.dim,
        config.rms_norm_eps,
    )?;

    if seq_len == 1 {
        kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
    }
    gemm_q(
        kernels,
        &scratch.norm_out,
        &layer.ffn_gate,
        &mut scratch.ffn_gate_out,
        seq_len,
        config.ff_dim,
        config.dim,
    )?;
    gemm_q(
        kernels,
        &scratch.norm_out,
        &layer.ffn_up,
        &mut scratch.ffn_up_out,
        seq_len,
        config.ff_dim,
        config.dim,
    )?;

    kernels.ops.gelu(
        &scratch.ffn_gate_out,
        &scratch.ffn_up_out,
        &mut scratch.ffn_silu_out,
        seq_len * config.ff_dim,
    )?;

    if seq_len == 1 {
        kernels
            .gemv
            .quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
    }
    gemm_q(
        kernels,
        &scratch.ffn_silu_out,
        &layer.ffn_down,
        &mut scratch.ffn_out,
        seq_len,
        config.dim,
        config.ff_dim,
    )?;

    if let Some(ref pfn) = layer.post_ffw_norm {
        kernels.ops.post_norm_add(
            hidden,
            &scratch.ffn_out,
            pfn,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
    } else {
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.ffn_out, seq_len * config.dim)?;
    }

    if let (Some(inp_gate), Some(proj), Some(post_norm), Some(pe_data), Some(epl)) = (
        &layer.inp_gate,
        &layer.proj,
        &layer.post_norm,
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

        if seq_len == 1 {
            kernels.gemv.quantize_input(&scratch.norm_out, config.dim)?;
        }
        gemm_q(
            kernels,
            &scratch.norm_out,
            inp_gate,
            pe_gate,
            seq_len,
            epl,
            config.dim,
        )?;

        {
            let src_ptr = pe_gate as *const CudaSlice<half::f16>;
            let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
            unsafe {
                kernels
                    .ops
                    .gelu_act(&*src_ptr, &mut *dst_ptr, seq_len * epl)?;
            }
        }

        {
            let src_ptr = pe_gate as *const CudaSlice<half::f16>;
            let dst_ptr = pe_gate as *mut CudaSlice<half::f16>;
            let layer_off = (layer_idx as u32) * epl;
            unsafe {
                kernels.ops.pe_strided_mul(
                    &*src_ptr,
                    &pe_data.data,
                    &mut *dst_ptr,
                    epl,
                    pe_data.row_width,
                    layer_off,
                    seq_len,
                )?;
            }
        }

        if seq_len == 1 {
            kernels.gemv.quantize_input(pe_gate, epl)?;
        }
        gemm_q(kernels, pe_gate, proj, pe_proj, seq_len, config.dim, epl)?;

        kernels.ops.post_norm_add(
            hidden,
            pe_proj,
            post_norm,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
    }

    if let Some(scale) = layer.layer_output_scale {
        if (scale - 1.0).abs() > 1e-6 {
            kernels
                .ops
                .scale_f32_inplace(hidden, seq_len * config.dim, scale)?;
        }
    }

    Ok(())
}
