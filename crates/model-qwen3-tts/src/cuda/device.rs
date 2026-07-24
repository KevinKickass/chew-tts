use super::{TalkerDecoderLayer, TalkerLayerKvCache, TalkerLayerScratch};
use crate::{QwenDType, TalkerConfig};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::CudaSlice;
use half::f16;

impl<T: QwenDType> TalkerDecoderLayer<T> {
    /// Execute a layer without moving the hidden state through host memory.
    pub fn forward_cached_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        seq_len: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
        cache: &mut TalkerLayerKvCache,
        scratch: &mut TalkerLayerScratch<T>,
    ) -> anyhow::Result<()> {
        let hidden_dim = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            hidden.len() >= seq_len * hidden_dim,
            "device hidden state is too small"
        );
        ensure!(
            scratch.max_tokens >= seq_len,
            "scratch capacity {} is smaller than {seq_len}",
            scratch.max_tokens
        );
        ensure!(
            cache.kv_dim == kv_dim,
            "KV cache geometry does not match the model"
        );
        ensure!(
            cache.position + seq_len <= cache.max_seq_len,
            "KV cache capacity {} exceeded by position {} + {seq_len}",
            cache.max_seq_len,
            cache.position
        );
        let rows = u32::try_from(seq_len).context("sequence length exceeds CUDA limits")?;
        let position = u32::try_from(cache.position).context("KV position exceeds CUDA limits")?;
        let total_kv_len =
            u32::try_from(cache.position + seq_len).context("KV length exceeds CUDA limits")?;

        T::rms_norm_f32in(
            kernels,
            hidden,
            &self.input_norm,
            &mut scratch.norm,
            rows,
            hidden_dim as u32,
            config.rms_norm_eps as f32,
        )?;
        if seq_len == 1 {
            T::gemv(
                kernels,
                &scratch.norm,
                &self.q_proj,
                &mut scratch.q_native,
                q_dim as u32,
                hidden_dim as u32,
            )?;
            T::gemv(
                kernels,
                &scratch.norm,
                &self.k_proj,
                &mut scratch.k_native,
                kv_dim as u32,
                hidden_dim as u32,
            )?;
            T::gemv(
                kernels,
                &scratch.norm,
                &self.v_proj,
                &mut scratch.v_native,
                kv_dim as u32,
                hidden_dim as u32,
            )?;
        } else {
            T::matmul(
                kernels,
                &scratch.norm,
                &self.q_proj,
                &mut scratch.q_native,
                rows,
                q_dim as u32,
                hidden_dim as u32,
            )?;
            T::matmul(
                kernels,
                &scratch.norm,
                &self.k_proj,
                &mut scratch.k_native,
                rows,
                kv_dim as u32,
                hidden_dim as u32,
            )?;
            T::matmul(
                kernels,
                &scratch.norm,
                &self.v_proj,
                &mut scratch.v_native,
                rows,
                kv_dim as u32,
                hidden_dim as u32,
            )?;
        }
        if let Some(bias) = &self.q_bias {
            T::add_bias(kernels, &mut scratch.q_native, bias, rows, q_dim as u32)?;
        }
        if let Some(bias) = &self.k_bias {
            T::add_bias(kernels, &mut scratch.k_native, bias, rows, kv_dim as u32)?;
        }
        if let Some(bias) = &self.v_bias {
            T::add_bias(kernels, &mut scratch.v_native, bias, rows, kv_dim as u32)?;
        }
        T::to_f16(
            kernels,
            &scratch.q_native,
            &mut scratch.q,
            rows * q_dim as u32,
        )?;
        T::to_f16(
            kernels,
            &scratch.k_native,
            &mut scratch.k,
            rows * kv_dim as u32,
        )?;
        T::to_f16(
            kernels,
            &scratch.v_native,
            &mut scratch.v,
            rows * kv_dim as u32,
        )?;

        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // The kernels support an aliased source/destination for per-head Q/K norm.
            unsafe {
                let q_in = &scratch.q as *const CudaSlice<f16>;
                let q_out = &mut scratch.q as *mut CudaSlice<f16>;
                kernels.ops.rms_norm(
                    &*q_in,
                    q_norm,
                    &mut *q_out,
                    rows * config.num_attention_heads as u32,
                    config.head_dim as u32,
                    config.rms_norm_eps as f32,
                )?;
                let k_in = &scratch.k as *const CudaSlice<f16>;
                let k_out = &mut scratch.k as *mut CudaSlice<f16>;
                kernels.ops.rms_norm(
                    &*k_in,
                    k_norm,
                    &mut *k_out,
                    rows * config.num_key_value_heads as u32,
                    config.head_dim as u32,
                    config.rms_norm_eps as f32,
                )?;
            }
        }

        if self.apply_rope {
            if let Some(factors) = &self.rope_factors {
                kernels.ops.rope_neox_freqs(
                    &mut scratch.q,
                    factors,
                    rows,
                    config.num_attention_heads as u32,
                    config.head_dim as u32,
                    position,
                    config.rope_theta as f32,
                )?;
                kernels.ops.rope_neox_freqs(
                    &mut scratch.k,
                    factors,
                    rows,
                    config.num_key_value_heads as u32,
                    config.head_dim as u32,
                    position,
                    config.rope_theta as f32,
                )?;
            } else {
                kernels.ops.rope_neox(
                    &mut scratch.q,
                    rows,
                    config.num_attention_heads as u32,
                    config.head_dim as u32,
                    position,
                    config.rope_theta as f32,
                )?;
                kernels.ops.rope_neox(
                    &mut scratch.k,
                    rows,
                    config.num_key_value_heads as u32,
                    config.head_dim as u32,
                    position,
                    config.rope_theta as f32,
                )?;
            }
        }

        let cache_offset = cache.position * kv_dim;
        let cache_end = cache_offset + seq_len * kv_dim;
        {
            let mut destination = cache.k.slice_mut(cache_offset..cache_end);
            kernels
                .ops
                .copy_f16(&scratch.k, &mut destination, rows * kv_dim as u32)?;
        }
        {
            let mut destination = cache.v.slice_mut(cache_offset..cache_end);
            kernels
                .ops
                .copy_f16(&scratch.v, &mut destination, rows * kv_dim as u32)?;
        }
        if self.causal_attention {
            kernels.ops.mha_fused(
                &scratch.q,
                &cache.k.slice(..cache_end),
                &cache.v.slice(..cache_end),
                &mut scratch.attention,
                config.head_dim as u32,
                config.num_attention_heads as u32,
                config.num_key_value_heads as u32,
                rows,
                total_kv_len,
                position,
            )?;
        } else {
            ensure!(
                position == 0 && total_kv_len == rows,
                "bidirectional attention cannot append to a KV cache"
            );
            kernels.ops.mha_naive_full(
                &scratch.q,
                &cache.k.slice(..cache_end),
                &cache.v.slice(..cache_end),
                &mut scratch.attention,
                config.head_dim as u32,
                config.num_attention_heads as u32,
                config.num_key_value_heads as u32,
                rows,
                total_kv_len,
                1.0 / (config.head_dim as f32).sqrt(),
                0.0,
            )?;
        }
        T::from_f16(
            kernels,
            &scratch.attention,
            &mut scratch.attention_native,
            rows * q_dim as u32,
        )?;
        if seq_len == 1 {
            T::gemv(
                kernels,
                &scratch.attention_native,
                &self.o_proj,
                &mut scratch.attention_out,
                hidden_dim as u32,
                q_dim as u32,
            )?;
        } else {
            T::matmul(
                kernels,
                &scratch.attention_native,
                &self.o_proj,
                &mut scratch.attention_out,
                rows,
                hidden_dim as u32,
                q_dim as u32,
            )?;
        }
        T::add_residual(
            kernels,
            hidden,
            &scratch.attention_out,
            rows * hidden_dim as u32,
        )?;

        T::rms_norm_f32in(
            kernels,
            hidden,
            &self.post_attention_norm,
            &mut scratch.norm,
            rows,
            hidden_dim as u32,
            config.rms_norm_eps as f32,
        )?;
        if seq_len == 1 {
            T::gemv_dual(
                kernels,
                &scratch.norm,
                &self.gate_proj,
                &self.up_proj,
                &mut scratch.gate,
                &mut scratch.up,
                intermediate as u32,
                hidden_dim as u32,
            )?;
        } else {
            T::matmul(
                kernels,
                &scratch.norm,
                &self.gate_proj,
                &mut scratch.gate,
                rows,
                intermediate as u32,
                hidden_dim as u32,
            )?;
            T::matmul(
                kernels,
                &scratch.norm,
                &self.up_proj,
                &mut scratch.up,
                rows,
                intermediate as u32,
                hidden_dim as u32,
            )?;
        }
        T::silu(
            kernels,
            &scratch.gate,
            &scratch.up,
            &mut scratch.activation,
            rows * intermediate as u32,
        )?;
        if seq_len == 1 {
            T::gemv(
                kernels,
                &scratch.activation,
                &self.down_proj,
                &mut scratch.mlp_out,
                hidden_dim as u32,
                intermediate as u32,
            )?;
        } else {
            T::matmul(
                kernels,
                &scratch.activation,
                &self.down_proj,
                &mut scratch.mlp_out,
                rows,
                hidden_dim as u32,
                intermediate as u32,
            )?;
        }
        T::add_residual(kernels, hidden, &scratch.mlp_out, rows * hidden_dim as u32)?;
        cache.position += seq_len;
        Ok(())
    }
}
