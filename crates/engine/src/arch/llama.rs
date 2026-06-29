use crate::config::ModelConfig;
use crate::forward::{ScratchBuffers, gemm_q, project_last_logits};
use crate::kv_cache::KvCache;
use crate::weights::ModelWeights;
use chew_kernel::{GpuKernels, KernelError};
use cudarc::driver::CudaSlice;
use std::sync::Arc;
use tracing::info;

/// Dense Llama-style forward pass.
pub fn forward(
    hidden: &mut CudaSlice<f32>,
    weights: &ModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    kv_cache: &mut KvCache,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
) -> Result<(), KernelError> {
    let pos = kv_cache.pos();
    let total_kv_len = pos + seq_len;
    let n_elems = seq_len * config.dim;

    let stream_ref = Arc::clone(kernels.ops.stream());

    let profile = std::env::var("CHEW_PROFILE").is_ok() && seq_len == 1;
    let mut t_norm = 0u128;
    let mut t_gemm = 0u128;
    let mut t_rope = 0u128;
    let mut t_kv = 0u128;
    let mut t_mha = 0u128;
    let mut t_silu = 0u128;
    let mut t_add = 0u128;
    macro_rules! timed {
        ($accum:ident, $body:expr) => {{
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
    if n_layers < config.n_layers as usize {
        info!(
            n_layers,
            total = config.n_layers,
            "DEBUG: running limited layers"
        );
    }

    if n_layers > 0 {
        if seq_len == 1 {
            let x_q8 = kernels.gemv.x_q8_mut();
            kernels.ops.rms_norm_f32in_q8(
                hidden,
                &weights.layers[0].attn_norm,
                &mut scratch.norm_out,
                x_q8,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )?;
        } else {
            kernels.ops.rms_norm_f32in(
                hidden,
                &weights.layers[0].attn_norm,
                &mut scratch.norm_out,
                seq_len,
                config.dim,
                config.rms_norm_eps,
            )?;
        }
    }

    for layer_idx in 0..n_layers {
        let layer = &weights.layers[layer_idx];

        if seq_len == 1 && layer_idx > 0 {
        } else if seq_len > 1 {
        }

        timed!(
            t_gemm,
            gemm_q(
                kernels,
                &scratch.norm_out,
                &layer.attn_q,
                &mut scratch.q,
                seq_len,
                config.n_heads * config.head_dim,
                config.dim
            )
        )?;
        if seq_len == 1 && layer.attn_k.quant_type == layer.attn_v.quant_type {
            let nk = config.n_kv_heads * config.head_dim;
            let used = timed!(
                t_gemm,
                kernels.gemv.gemv_dual(
                    &layer.attn_k.data,
                    &layer.attn_v.data,
                    &mut scratch.k,
                    &mut scratch.v,
                    nk,
                    config.dim,
                    layer.attn_k.quant_type,
                )
            )?;
            if !used {
                timed!(
                    t_gemm,
                    gemm_q(
                        kernels,
                        &scratch.norm_out,
                        &layer.attn_k,
                        &mut scratch.k,
                        seq_len,
                        config.n_kv_heads * config.head_dim,
                        config.dim
                    )
                )?;
                timed!(
                    t_gemm,
                    gemm_q(
                        kernels,
                        &scratch.norm_out,
                        &layer.attn_v,
                        &mut scratch.v,
                        seq_len,
                        config.n_kv_heads * config.head_dim,
                        config.dim
                    )
                )?;
            }
        } else {
            timed!(
                t_gemm,
                gemm_q(
                    kernels,
                    &scratch.norm_out,
                    &layer.attn_k,
                    &mut scratch.k,
                    seq_len,
                    config.n_kv_heads * config.head_dim,
                    config.dim
                )
            )?;
            timed!(
                t_gemm,
                gemm_q(
                    kernels,
                    &scratch.norm_out,
                    &layer.attn_v,
                    &mut scratch.v,
                    seq_len,
                    config.n_kv_heads * config.head_dim,
                    config.dim
                )
            )?;
        }

        timed!(t_rope, {
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
            )
        })?;

        let kv_elems = seq_len * config.n_kv_heads * config.head_dim;
        {
            let mut k_cache = kv_cache.k_mut(layer_idx, seq_len);
            kernels.ops.copy_f16(&scratch.k, &mut k_cache, kv_elems)?;
        }
        {
            let mut v_cache = kv_cache.v_mut(layer_idx, seq_len);
            kernels.ops.copy_f16(&scratch.v, &mut v_cache, kv_elems)?;
        }

        timed!(t_mha, {
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
            )
        })?;

        if seq_len == 1 {
            kernels
                .gemv
                .quantize_input(&scratch.attn_mha_out, config.n_heads * config.head_dim)?;
        }
        timed!(
            t_gemm,
            gemm_q(
                kernels,
                &scratch.attn_mha_out,
                &layer.attn_output,
                &mut scratch.attn_out,
                seq_len,
                config.dim,
                config.n_heads * config.head_dim
            )
        )?;

        if seq_len == 1 {
            let x_q8 = kernels.gemv.x_q8_mut();
            timed!(
                t_add,
                kernels.ops.fused_add_rmsnorm_q8(
                    hidden,
                    &scratch.attn_out,
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
                t_add,
                kernels.ops.fused_add_rmsnorm(
                    hidden,
                    &scratch.attn_out,
                    &layer.ffn_norm,
                    &mut scratch.norm_out,
                    seq_len,
                    config.dim,
                    config.rms_norm_eps,
                )
            )?;
        }

        if seq_len == 1 && layer.ffn_gate.quant_type == layer.ffn_up.quant_type {
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
                    gemm_q(
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
                    gemm_q(
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
                gemm_q(
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
                gemm_q(
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

        timed!(
            t_silu,
            kernels.ops.silu(
                &scratch.ffn_gate_out,
                &scratch.ffn_up_out,
                &mut scratch.ffn_silu_out,
                seq_len * config.ff_dim,
            )
        )?;

        if seq_len == 1 {
            kernels
                .gemv
                .quantize_input(&scratch.ffn_silu_out, config.ff_dim)?;
        }
        timed!(
            t_gemm,
            gemm_q(
                kernels,
                &scratch.ffn_silu_out,
                &layer.ffn_down,
                &mut scratch.ffn_out,
                seq_len,
                config.dim,
                config.ff_dim
            )
        )?;

        if layer_idx + 1 < n_layers {
            if seq_len == 1 {
                let x_q8 = kernels.gemv.x_q8_mut();
                timed!(
                    t_add,
                    kernels.ops.fused_add_rmsnorm_q8(
                        hidden,
                        &scratch.ffn_out,
                        &weights.layers[layer_idx + 1].attn_norm,
                        &mut scratch.norm_out,
                        x_q8,
                        seq_len,
                        config.dim,
                        config.rms_norm_eps,
                    )
                )?;
            } else {
                timed!(
                    t_add,
                    kernels.ops.fused_add_rmsnorm(
                        hidden,
                        &scratch.ffn_out,
                        &weights.layers[layer_idx + 1].attn_norm,
                        &mut scratch.norm_out,
                        seq_len,
                        config.dim,
                        config.rms_norm_eps,
                    )
                )?;
            }
        } else {
            timed!(
                t_add,
                kernels
                    .ops
                    .add_inplace_f32_f16(hidden, &scratch.ffn_out, n_elems)
            )?;
        }
    }

    if profile {
        let total = t_norm + t_gemm + t_rope + t_kv + t_mha + t_silu + t_add;
        info!(
            gemv_us = t_gemm,
            mha_us = t_mha,
            norm_us = t_norm,
            rope_us = t_rope,
            kv_us = t_kv,
            silu_us = t_silu,
            add_us = t_add,
            total_us = total,
            kv_len = total_kv_len,
            "PROFILE decode step"
        );
    }

    kernels.ops.rms_norm_f32in(
        hidden,
        &weights.output_norm,
        &mut scratch.norm_out,
        seq_len,
        config.dim,
        config.rms_norm_eps,
    )?;

    project_last_logits(
        kernels,
        &stream_ref,
        &scratch.norm_out,
        &mut scratch.attn_out,
        &weights.output,
        &mut scratch.logits,
        seq_len,
        config.vocab_size,
        config.dim,
    )?;

    kv_cache.advance(seq_len);

    Ok(())
}
