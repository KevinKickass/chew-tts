use crate::config::ModelConfig;
use crate::forward::{ScratchBuffers, gemm_q};
use crate::weights::{LoadError, QuantWeight, upload_and_dequant_any, upload_quantized_any};
use chew_gguf::GgufFile;
use chew_kernel::{GpuKernels, KernelError};
use chew_vram::VramAllocator;
use cudarc::driver::CudaSlice;

pub struct BertEmbeddings {
    pub word_embeddings: CudaSlice<half::f16>,
    pub position_embeddings: CudaSlice<half::f16>,
    pub token_type_embeddings: CudaSlice<half::f16>,
    pub norm_weight: CudaSlice<half::f16>,
    pub norm_bias: CudaSlice<half::f16>,
}

pub struct BertLayerWeights {
    pub attn_q: QuantWeight,
    pub attn_q_bias: CudaSlice<half::f16>,
    pub attn_k: QuantWeight,
    pub attn_k_bias: CudaSlice<half::f16>,
    pub attn_v: QuantWeight,
    pub attn_v_bias: CudaSlice<half::f16>,
    pub attn_output: QuantWeight,
    pub attn_output_bias: CudaSlice<half::f16>,
    pub attn_norm_weight: CudaSlice<half::f16>,
    pub attn_norm_bias: CudaSlice<half::f16>,
    pub ffn_gate: QuantWeight,
    pub ffn_gate_bias: CudaSlice<half::f16>,
    pub ffn_down: QuantWeight,
    pub ffn_down_bias: CudaSlice<half::f16>,
    pub ffn_norm_weight: CudaSlice<half::f16>,
    pub ffn_norm_bias: CudaSlice<half::f16>,
}

pub struct BertModelWeights {
    pub embeddings: BertEmbeddings,
    pub layers: Vec<BertLayerWeights>,
}

impl BertModelWeights {
    pub fn load(
        gguf: &GgufFile,
        config: &ModelConfig,
        alloc: &VramAllocator,
        dequant: &chew_kernel::DequantKernels,
        gpu_idx: usize,
    ) -> Result<Self, LoadError> {
        let embeddings = BertEmbeddings {
            word_embeddings: upload_and_dequant_any(
                gguf,
                &["token_embd.weight", "embeddings.word_embeddings.weight"],
                alloc,
                dequant,
                gpu_idx,
            )?,
            position_embeddings: upload_and_dequant_any(
                gguf,
                &[
                    "position_embd.weight",
                    "embeddings.position_embeddings.weight",
                ],
                alloc,
                dequant,
                gpu_idx,
            )?,
            token_type_embeddings: upload_and_dequant_any(
                gguf,
                &[
                    "token_types.weight",
                    "token_type_embd.weight",
                    "embeddings.token_type_embeddings.weight",
                ],
                alloc,
                dequant,
                gpu_idx,
            )?,
            norm_weight: upload_and_dequant_any(
                gguf,
                &[
                    "token_embd_norm.weight",
                    "embd_norm.weight",
                    "embeddings.LayerNorm.weight",
                ],
                alloc,
                dequant,
                gpu_idx,
            )?,
            norm_bias: upload_and_dequant_any(
                gguf,
                &[
                    "token_embd_norm.bias",
                    "embd_norm.bias",
                    "embeddings.LayerNorm.bias",
                ],
                alloc,
                dequant,
                gpu_idx,
            )?,
        };

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers {
            let q_w = format!("encoder.layer.{i}.attention.self.query.weight");
            let q_b = format!("encoder.layer.{i}.attention.self.query.bias");
            let k_w = format!("encoder.layer.{i}.attention.self.key.weight");
            let k_b = format!("encoder.layer.{i}.attention.self.key.bias");
            let v_w = format!("encoder.layer.{i}.attention.self.value.weight");
            let v_b = format!("encoder.layer.{i}.attention.self.value.bias");
            let ao_w = format!("encoder.layer.{i}.attention.output.dense.weight");
            let ao_b = format!("encoder.layer.{i}.attention.output.dense.bias");
            let an_w = format!("encoder.layer.{i}.attention.output.LayerNorm.weight");
            let an_b = format!("encoder.layer.{i}.attention.output.LayerNorm.bias");
            let fg_w = format!("encoder.layer.{i}.intermediate.dense.weight");
            let fg_b = format!("encoder.layer.{i}.intermediate.dense.bias");
            let fd_w = format!("encoder.layer.{i}.output.dense.weight");
            let fd_b = format!("encoder.layer.{i}.output.dense.bias");
            let fn_w = format!("encoder.layer.{i}.output.LayerNorm.weight");
            let fn_b = format!("encoder.layer.{i}.output.LayerNorm.bias");

            layers.push(BertLayerWeights {
                attn_q: upload_quantized_any(
                    gguf,
                    &[&format!("blk.{i}.attn_q.weight"), &q_w],
                    alloc,
                    gpu_idx,
                )?,
                attn_q_bias: upload_and_dequant_any(
                    gguf,
                    &[&format!("blk.{i}.attn_q.bias"), &q_b],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                attn_k: upload_quantized_any(
                    gguf,
                    &[&format!("blk.{i}.attn_k.weight"), &k_w],
                    alloc,
                    gpu_idx,
                )?,
                attn_k_bias: upload_and_dequant_any(
                    gguf,
                    &[&format!("blk.{i}.attn_k.bias"), &k_b],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                attn_v: upload_quantized_any(
                    gguf,
                    &[&format!("blk.{i}.attn_v.weight"), &v_w],
                    alloc,
                    gpu_idx,
                )?,
                attn_v_bias: upload_and_dequant_any(
                    gguf,
                    &[&format!("blk.{i}.attn_v.bias"), &v_b],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                attn_output: upload_quantized_any(
                    gguf,
                    &[&format!("blk.{i}.attn_output.weight"), &ao_w],
                    alloc,
                    gpu_idx,
                )?,
                attn_output_bias: upload_and_dequant_any(
                    gguf,
                    &[&format!("blk.{i}.attn_output.bias"), &ao_b],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                attn_norm_weight: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("blk.{i}.attn_output_norm.weight"),
                        &format!("blk.{i}.post_attention_norm.weight"),
                        &an_w,
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                attn_norm_bias: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("blk.{i}.attn_output_norm.bias"),
                        &format!("blk.{i}.post_attention_norm.bias"),
                        &an_b,
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ffn_gate: upload_quantized_any(
                    gguf,
                    &[
                        &format!("blk.{i}.ffn_up.weight"),
                        &format!("blk.{i}.ffn_gate.weight"),
                        &fg_w,
                    ],
                    alloc,
                    gpu_idx,
                )?,
                ffn_gate_bias: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("blk.{i}.ffn_up.bias"),
                        &format!("blk.{i}.ffn_gate.bias"),
                        &fg_b,
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ffn_down: upload_quantized_any(
                    gguf,
                    &[&format!("blk.{i}.ffn_down.weight"), &fd_w],
                    alloc,
                    gpu_idx,
                )?,
                ffn_down_bias: upload_and_dequant_any(
                    gguf,
                    &[&format!("blk.{i}.ffn_down.bias"), &fd_b],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ffn_norm_weight: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("blk.{i}.layer_output_norm.weight"),
                        &format!("blk.{i}.post_ffw_norm.weight"),
                        &fn_w,
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
                ffn_norm_bias: upload_and_dequant_any(
                    gguf,
                    &[
                        &format!("blk.{i}.layer_output_norm.bias"),
                        &format!("blk.{i}.post_ffw_norm.bias"),
                        &fn_b,
                    ],
                    alloc,
                    dequant,
                    gpu_idx,
                )?,
            });
        }

        Ok(Self { embeddings, layers })
    }
}

/// Post-LN BERT encoder forward for embedding models such as MiniLM.
///
/// `hidden` is the already-summed embedding input in f32:
/// word + position + token_type, before embedding LayerNorm.
pub fn forward(
    hidden: &mut CudaSlice<f32>,
    weights: &BertModelWeights,
    config: &ModelConfig,
    kernels: &mut GpuKernels,
    scratch: &mut ScratchBuffers,
    seq_len: u32,
) -> Result<(), KernelError> {
    let n_elems = seq_len * config.dim;
    let attn_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    kernels.ops.layer_norm_f32in(
        hidden,
        &weights.embeddings.norm_weight,
        &weights.embeddings.norm_bias,
        &mut scratch.norm_out,
        seq_len,
        config.dim,
        config.rms_norm_eps,
    )?;
    kernels
        .ops
        .copy_f16_to_f32(&scratch.norm_out, hidden, n_elems)?;

    for layer in &weights.layers {
        {
            let mut norm_view = scratch.norm_out.slice_mut(..);
            kernels
                .ops
                .copy_f32_to_f16(hidden, &mut norm_view, n_elems)?;
        }

        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.attn_q,
            &mut scratch.q,
            seq_len,
            attn_dim,
            config.dim,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut scratch.q, &layer.attn_q_bias, seq_len, attn_dim)?;

        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.attn_k,
            &mut scratch.k,
            seq_len,
            kv_dim,
            config.dim,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut scratch.k, &layer.attn_k_bias, seq_len, kv_dim)?;

        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.attn_v,
            &mut scratch.v,
            seq_len,
            kv_dim,
            config.dim,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut scratch.v, &layer.attn_v_bias, seq_len, kv_dim)?;

        let k_view = scratch.k.slice(0..(seq_len * kv_dim) as usize);
        let v_view = scratch.v.slice(0..(seq_len * kv_dim) as usize);
        kernels.ops.mha_naive_full(
            &scratch.q,
            &k_view,
            &v_view,
            &mut scratch.attn_mha_out,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            seq_len,
            seq_len,
            1.0 / (config.head_dim as f32).sqrt(),
            0.0,
        )?;

        gemm_q(
            kernels,
            &scratch.attn_mha_out,
            &layer.attn_output,
            &mut scratch.attn_out,
            seq_len,
            config.dim,
            attn_dim,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut scratch.attn_out,
            &layer.attn_output_bias,
            seq_len,
            config.dim,
        )?;
        kernels
            .ops
            .add_f32_f16(hidden, &scratch.attn_out, &mut scratch.residual, n_elems)?;
        kernels.ops.layer_norm_f32in(
            &scratch.residual,
            &layer.attn_norm_weight,
            &layer.attn_norm_bias,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
        kernels
            .ops
            .copy_f16_to_f32(&scratch.norm_out, hidden, n_elems)?;

        gemm_q(
            kernels,
            &scratch.norm_out,
            &layer.ffn_gate,
            &mut scratch.ffn_gate_out,
            seq_len,
            config.ff_dim,
            config.dim,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut scratch.ffn_gate_out,
            &layer.ffn_gate_bias,
            seq_len,
            config.ff_dim,
        )?;
        {
            let src_ptr = &scratch.ffn_gate_out as *const CudaSlice<half::f16>;
            let dst_ptr = &mut scratch.ffn_up_out as *mut CudaSlice<half::f16>;
            unsafe {
                kernels
                    .ops
                    .gelu_act(&*src_ptr, &mut *dst_ptr, seq_len * config.ff_dim)?;
            }
        }

        gemm_q(
            kernels,
            &scratch.ffn_up_out,
            &layer.ffn_down,
            &mut scratch.ffn_out,
            seq_len,
            config.dim,
            config.ff_dim,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut scratch.ffn_out,
            &layer.ffn_down_bias,
            seq_len,
            config.dim,
        )?;
        kernels
            .ops
            .add_f32_f16(hidden, &scratch.ffn_out, &mut scratch.residual, n_elems)?;
        kernels.ops.layer_norm_f32in(
            &scratch.residual,
            &layer.ffn_norm_weight,
            &layer.ffn_norm_bias,
            &mut scratch.norm_out,
            seq_len,
            config.dim,
            config.rms_norm_eps,
        )?;
        kernels
            .ops
            .copy_f16_to_f32(&scratch.norm_out, hidden, n_elems)?;
    }

    Ok(())
}
