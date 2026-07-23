use crate::{load_f16_tensor, load_f32_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;
const LATENT_DIM: usize = 512;
const QUANTIZERS: usize = 16;

struct CodecTransformerLayer {
    input_norm: CudaSlice<f16>,
    q_proj: CudaSlice<f16>,
    k_proj: CudaSlice<f16>,
    v_proj: CudaSlice<f16>,
    o_proj: CudaSlice<f16>,
    attention_scale: CudaSlice<f16>,
    post_attention_norm: CudaSlice<f16>,
    gate_proj: CudaSlice<f16>,
    up_proj: CudaSlice<f16>,
    down_proj: CudaSlice<f16>,
    mlp_scale: CudaSlice<f16>,
}

struct CodecUpsampleStage {
    transposed_weight: CudaSlice<f16>,
    transposed_bias: CudaSlice<f16>,
    depthwise_weight: CudaSlice<f16>,
    depthwise_bias: CudaSlice<f16>,
    norm_weight: CudaSlice<f16>,
    norm_bias: CudaSlice<f16>,
    pointwise_in_weight: CudaSlice<f16>,
    pointwise_in_bias: CudaSlice<f16>,
    pointwise_out_weight: CudaSlice<f16>,
    pointwise_out_bias: CudaSlice<f16>,
    gamma: CudaSlice<f16>,
}

/// Qwen 12-Hz codec codebooks and their 1x1 output projections.
pub struct CodecQuantizer {
    first_codebook: CudaSlice<f16>,
    rest_codebooks: Vec<CudaSlice<f16>>,
    first_output_proj: CudaSlice<f16>,
    rest_output_proj: CudaSlice<f16>,
    pre_conv_weight: CudaSlice<f16>,
    pre_conv_bias: CudaSlice<f16>,
    transformer_input_weight: CudaSlice<f16>,
    transformer_input_bias: CudaSlice<f16>,
    transformer_layers: Vec<CodecTransformerLayer>,
    transformer_norm: CudaSlice<f16>,
    transformer_output_weight: CudaSlice<f16>,
    transformer_output_bias: CudaSlice<f16>,
    upsample_stages: Vec<CodecUpsampleStage>,
}

impl CodecQuantizer {
    pub fn load(tokenizer_dir: impl AsRef<Path>, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let tokenizer_dir = tokenizer_dir.as_ref();
        let normalize_codebook = |prefix: &str| -> anyhow::Result<CudaSlice<f16>> {
            let sums =
                load_f32_tensor(tokenizer_dir, &format!("{prefix}._codebook.embedding_sum"))?;
            let usage =
                load_f32_tensor(tokenizer_dir, &format!("{prefix}._codebook.cluster_usage"))?;
            ensure!(
                sums.shape == [CODEBOOK_SIZE, CODEBOOK_DIM],
                "{prefix} embedding_sum has shape {:?}",
                sums.shape
            );
            ensure!(
                usage.shape == [CODEBOOK_SIZE],
                "{prefix} cluster_usage has shape {:?}",
                usage.shape
            );
            let normalized = sums
                .values
                .chunks_exact(CODEBOOK_DIM)
                .zip(&usage.values)
                .flat_map(|(row, usage)| {
                    let divisor = usage.max(1e-7);
                    row.iter().map(move |value| f16::from_f32(value / divisor))
                })
                .collect::<Vec<_>>();
            stream
                .clone_htod(&normalized)
                .with_context(|| format!("could not upload {prefix}"))
        };
        let load_projection = |name: &str| -> anyhow::Result<CudaSlice<f16>> {
            let tensor = load_f16_tensor(tokenizer_dir, name)?;
            ensure!(
                tensor.shape == [LATENT_DIM, CODEBOOK_DIM, 1],
                "{name} has shape {:?}",
                tensor.shape
            );
            stream
                .clone_htod(&tensor.values)
                .with_context(|| format!("could not upload {name}"))
        };

        let first_codebook = normalize_codebook("decoder.quantizer.rvq_first.vq.layers.0")?;
        let mut rest_codebooks = Vec::with_capacity(QUANTIZERS - 1);
        for index in 0..QUANTIZERS - 1 {
            rest_codebooks.push(normalize_codebook(&format!(
                "decoder.quantizer.rvq_rest.vq.layers.{index}"
            ))?);
        }
        let pre_conv_weight = load_f16_tensor(tokenizer_dir, "decoder.pre_conv.conv.weight")?;
        ensure!(
            pre_conv_weight.shape == [1024, LATENT_DIM, 3],
            "pre-conv weight has shape {:?}",
            pre_conv_weight.shape
        );
        let pre_conv_bias = load_f16_tensor(tokenizer_dir, "decoder.pre_conv.conv.bias")?;
        ensure!(
            pre_conv_bias.shape == [1024],
            "pre-conv bias has shape {:?}",
            pre_conv_bias.shape
        );
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let tensor = load_f16_tensor(tokenizer_dir, name)?;
            ensure!(
                tensor.shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                tensor.shape
            );
            Ok(stream.clone_htod(&tensor.values)?)
        };
        let transformer_input_weight =
            load("decoder.pre_transformer.input_proj.weight", &[512, 1024])?;
        let transformer_input_bias = load("decoder.pre_transformer.input_proj.bias", &[512])?;
        let mut transformer_layers = Vec::with_capacity(8);
        for layer in 0..8 {
            let prefix = format!("decoder.pre_transformer.layers.{layer}");
            transformer_layers.push(CodecTransformerLayer {
                input_norm: load(&format!("{prefix}.input_layernorm.weight"), &[512])?,
                q_proj: load(&format!("{prefix}.self_attn.q_proj.weight"), &[1024, 512])?,
                k_proj: load(&format!("{prefix}.self_attn.k_proj.weight"), &[1024, 512])?,
                v_proj: load(&format!("{prefix}.self_attn.v_proj.weight"), &[1024, 512])?,
                o_proj: load(&format!("{prefix}.self_attn.o_proj.weight"), &[512, 1024])?,
                attention_scale: load(&format!("{prefix}.self_attn_layer_scale.scale"), &[512])?,
                post_attention_norm: load(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[512],
                )?,
                gate_proj: load(&format!("{prefix}.mlp.gate_proj.weight"), &[1024, 512])?,
                up_proj: load(&format!("{prefix}.mlp.up_proj.weight"), &[1024, 512])?,
                down_proj: load(&format!("{prefix}.mlp.down_proj.weight"), &[512, 1024])?,
                mlp_scale: load(&format!("{prefix}.mlp_layer_scale.scale"), &[512])?,
            });
        }
        let mut upsample_stages = Vec::with_capacity(2);
        for stage in 0..2 {
            let prefix = format!("decoder.upsample.{stage}");
            upsample_stages.push(CodecUpsampleStage {
                transposed_weight: load(&format!("{prefix}.0.conv.weight"), &[1024, 1024, 2])?,
                transposed_bias: load(&format!("{prefix}.0.conv.bias"), &[1024])?,
                depthwise_weight: load(&format!("{prefix}.1.dwconv.conv.weight"), &[1024, 1, 7])?,
                depthwise_bias: load(&format!("{prefix}.1.dwconv.conv.bias"), &[1024])?,
                norm_weight: load(&format!("{prefix}.1.norm.weight"), &[1024])?,
                norm_bias: load(&format!("{prefix}.1.norm.bias"), &[1024])?,
                pointwise_in_weight: load(&format!("{prefix}.1.pwconv1.weight"), &[4096, 1024])?,
                pointwise_in_bias: load(&format!("{prefix}.1.pwconv1.bias"), &[4096])?,
                pointwise_out_weight: load(&format!("{prefix}.1.pwconv2.weight"), &[1024, 4096])?,
                pointwise_out_bias: load(&format!("{prefix}.1.pwconv2.bias"), &[1024])?,
                gamma: load(&format!("{prefix}.1.gamma"), &[1024])?,
            });
        }
        Ok(Self {
            first_codebook,
            rest_codebooks,
            first_output_proj: load_projection("decoder.quantizer.rvq_first.output_proj.weight")?,
            rest_output_proj: load_projection("decoder.quantizer.rvq_rest.output_proj.weight")?,
            pre_conv_weight: stream.clone_htod(&pre_conv_weight.values)?,
            pre_conv_bias: stream.clone_htod(&pre_conv_bias.values)?,
            transformer_input_weight,
            transformer_input_bias,
            transformer_layers,
            transformer_norm: load("decoder.pre_transformer.norm.weight", &[512])?,
            transformer_output_weight: load(
                "decoder.pre_transformer.output_proj.weight",
                &[1024, 512],
            )?,
            transformer_output_bias: load("decoder.pre_transformer.output_proj.bias", &[1024])?,
            upsample_stages,
        })
    }

    /// Decode one 16-codebook frame into the 512-channel codec latent.
    pub fn decode_frame(
        &self,
        codes: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            codes.len() == QUANTIZERS,
            "codec frame has {} codes, expected {QUANTIZERS}",
            codes.len()
        );
        let stream = Arc::clone(kernels.ops.stream());
        let first_id = codes[0].rem_euclid(CODEBOOK_SIZE as i32);
        let first_id_gpu = stream.clone_htod(&[first_id])?;
        let mut first_embed = stream.alloc_zeros::<f16>(CODEBOOK_DIM)?;
        kernels.ops.gather_rows_f16(
            &self.first_codebook,
            &first_id_gpu,
            &mut first_embed,
            1,
            CODEBOOK_DIM as u32,
        )?;

        let mut rest_accum = stream.alloc_zeros::<f32>(CODEBOOK_DIM)?;
        let mut rest_embed = stream.alloc_zeros::<f16>(CODEBOOK_DIM)?;
        for (index, (code, codebook)) in codes[1..].iter().zip(&self.rest_codebooks).enumerate() {
            ensure!(
                *code >= 0 && (*code as usize) < CODEBOOK_SIZE,
                "acoustic code {} at group {} is outside 0..{CODEBOOK_SIZE}",
                code,
                index + 1
            );
            let id = stream.clone_htod(&[*code])?;
            kernels
                .ops
                .gather_rows_f16(codebook, &id, &mut rest_embed, 1, CODEBOOK_DIM as u32)?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut rest_accum, &rest_embed, CODEBOOK_DIM as u32)?;
        }
        {
            let mut rest_embed_view = rest_embed.slice_mut(..);
            kernels
                .ops
                .copy_f32_to_f16(&rest_accum, &mut rest_embed_view, CODEBOOK_DIM as u32)?;
        }

        let mut first_projected = stream.alloc_zeros::<f16>(LATENT_DIM)?;
        let mut rest_projected = stream.alloc_zeros::<f16>(LATENT_DIM)?;
        let mut latent = stream.alloc_zeros::<f16>(LATENT_DIM)?;
        kernels.gemv.gemv_f16(
            &first_embed,
            &self.first_output_proj,
            &mut first_projected,
            LATENT_DIM as u32,
            CODEBOOK_DIM as u32,
        )?;
        kernels.gemv.gemv_f16(
            &rest_embed,
            &self.rest_output_proj,
            &mut rest_projected,
            LATENT_DIM as u32,
            CODEBOOK_DIM as u32,
        )?;
        kernels.ops.add_f16(
            &first_projected,
            &rest_projected,
            &mut latent,
            LATENT_DIM as u32,
        )?;
        stream.synchronize()?;

        let mut latent_host = vec![f16::ZERO; LATENT_DIM];
        stream.memcpy_dtoh(&latent, &mut latent_host)?;
        Ok(latent_host.into_iter().map(f16::to_f32).collect())
    }

    /// Decode and apply the codec's causal 512→1024 pre-convolution.
    pub fn decode_frame_preconv(
        &self,
        codes: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let latent = self.decode_frame(codes, kernels)?;
        let stream = Arc::clone(kernels.ops.stream());
        let latent_f16 = latent.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let latent_gpu = stream.clone_htod(&latent_f16)?;
        let mut output = stream.alloc_zeros::<f16>(1024)?;
        kernels.ops.conv1d_causal_f16(
            &latent_gpu,
            &self.pre_conv_weight,
            &self.pre_conv_bias,
            &mut output,
            LATENT_DIM as u32,
            1024,
            1,
            3,
            1,
            1,
        )?;
        stream.synchronize()?;
        let mut output_host = vec![f16::ZERO; 1024];
        stream.memcpy_dtoh(&output, &mut output_host)?;
        Ok(output_host.into_iter().map(f16::to_f32).collect())
    }

    /// Run quantizer, pre-conv, and the eight-layer codec transformer.
    pub fn decode_frame_transformer(
        &self,
        codes: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        const HIDDEN: usize = 512;
        const Q_DIM: usize = 1024;
        const FF_DIM: usize = 1024;
        let preconv = self.decode_frame_preconv(codes, kernels)?;
        let stream = Arc::clone(kernels.ops.stream());
        let preconv_f16 = preconv.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let preconv_gpu = stream.clone_htod(&preconv_f16)?;
        let mut projected = stream.alloc_zeros::<f16>(HIDDEN)?;
        kernels.gemv.gemv_f16(
            &preconv_gpu,
            &self.transformer_input_weight,
            &mut projected,
            HIDDEN as u32,
            1024,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.transformer_input_bias,
            1,
            HIDDEN as u32,
        )?;

        let mut hidden = stream.alloc_zeros::<f32>(HIDDEN)?;
        kernels
            .ops
            .copy_f16_to_f32(&projected, &mut hidden, HIDDEN as u32)?;
        let mut norm = stream.alloc_zeros::<f16>(HIDDEN)?;
        let mut q = stream.alloc_zeros::<f16>(Q_DIM)?;
        let mut k = stream.alloc_zeros::<f16>(Q_DIM)?;
        let mut v = stream.alloc_zeros::<f16>(Q_DIM)?;
        let mut attention = stream.alloc_zeros::<f16>(Q_DIM)?;
        let mut delta = stream.alloc_zeros::<f16>(HIDDEN)?;
        let mut scaled = stream.alloc_zeros::<f16>(HIDDEN)?;
        let mut gate = stream.alloc_zeros::<f16>(FF_DIM)?;
        let mut up = stream.alloc_zeros::<f16>(FF_DIM)?;
        let mut activation = stream.alloc_zeros::<f16>(FF_DIM)?;

        for layer in &self.transformer_layers {
            kernels.ops.rms_norm_f32in(
                &hidden,
                &layer.input_norm,
                &mut norm,
                1,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels
                .gemv
                .gemv_f16(&norm, &layer.q_proj, &mut q, Q_DIM as u32, HIDDEN as u32)?;
            kernels
                .gemv
                .gemv_f16(&norm, &layer.k_proj, &mut k, Q_DIM as u32, HIDDEN as u32)?;
            kernels
                .gemv
                .gemv_f16(&norm, &layer.v_proj, &mut v, Q_DIM as u32, HIDDEN as u32)?;
            kernels.ops.rope_neox(&mut q, 1, 16, 64, 0, 10_000.0)?;
            kernels.ops.rope_neox(&mut k, 1, 16, 64, 0, 10_000.0)?;
            kernels.ops.mha_fused(
                &q,
                &k.slice(..),
                &v.slice(..),
                &mut attention,
                64,
                16,
                16,
                1,
                1,
                0,
            )?;
            kernels.gemv.gemv_f16(
                &attention,
                &layer.o_proj,
                &mut delta,
                HIDDEN as u32,
                Q_DIM as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.attention_scale,
                &mut scaled,
                HIDDEN as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, HIDDEN as u32)?;

            kernels.ops.rms_norm_f32in(
                &hidden,
                &layer.post_attention_norm,
                &mut norm,
                1,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemv.gemv_dual_f16(
                &norm,
                &layer.gate_proj,
                &layer.up_proj,
                &mut gate,
                &mut up,
                FF_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .silu(&gate, &up, &mut activation, FF_DIM as u32)?;
            kernels.gemv.gemv_f16(
                &activation,
                &layer.down_proj,
                &mut delta,
                HIDDEN as u32,
                FF_DIM as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.mlp_scale,
                &mut scaled,
                HIDDEN as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, HIDDEN as u32)?;
        }

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.transformer_norm,
            &mut norm,
            1,
            HIDDEN as u32,
            1e-5,
        )?;
        let mut output = stream.alloc_zeros::<f16>(1024)?;
        kernels.gemv.gemv_f16(
            &norm,
            &self.transformer_output_weight,
            &mut output,
            1024,
            HIDDEN as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut output, &self.transformer_output_bias, 1, 1024)?;
        stream.synchronize()?;
        let mut output_host = vec![f16::ZERO; 1024];
        stream.memcpy_dtoh(&output, &mut output_host)?;
        Ok(output_host.into_iter().map(f16::to_f32).collect())
    }

    /// Run the codec front end through both 2x ConvNeXt upsampling stages.
    pub fn decode_frame_upsampled(
        &self,
        codes: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        const CHANNELS: usize = 1024;
        const EXPANDED: usize = 4096;
        let transformed = self.decode_frame_transformer(codes, kernels)?;
        let stream = Arc::clone(kernels.ops.stream());
        let transformed = transformed
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let mut hidden = stream.clone_htod(&transformed)?;
        let mut seq_len = 1usize;

        for stage in &self.upsample_stages {
            let output_len = seq_len * 2;
            let mut residual = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.conv_transpose1d_causal_f16(
                &hidden,
                &stage.transposed_weight,
                &stage.transposed_bias,
                &mut residual,
                CHANNELS as u32,
                CHANNELS as u32,
                seq_len as u32,
                2,
                2,
            )?;

            let mut depthwise = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.conv1d_causal_f16(
                &residual,
                &stage.depthwise_weight,
                &stage.depthwise_bias,
                &mut depthwise,
                CHANNELS as u32,
                CHANNELS as u32,
                output_len as u32,
                7,
                1,
                CHANNELS as u32,
            )?;
            let mut channel_last = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.transpose_f16(
                &depthwise,
                &mut channel_last,
                CHANNELS as u32,
                output_len as u32,
            )?;
            let mut channel_last_f32 = stream.alloc_zeros::<f32>(CHANNELS * output_len)?;
            kernels.ops.copy_f16_to_f32(
                &channel_last,
                &mut channel_last_f32,
                (CHANNELS * output_len) as u32,
            )?;
            let mut normalized = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.layer_norm_f32in(
                &channel_last_f32,
                &stage.norm_weight,
                &stage.norm_bias,
                &mut normalized,
                output_len as u32,
                CHANNELS as u32,
                1e-6,
            )?;

            let mut expanded = stream.alloc_zeros::<f16>(EXPANDED * output_len)?;
            kernels.gemm.matmul_f16(
                &normalized,
                &stage.pointwise_in_weight,
                &mut expanded,
                output_len as u32,
                EXPANDED as u32,
                CHANNELS as u32,
            )?;
            kernels.ops.add_bias_f16_inplace(
                &mut expanded,
                &stage.pointwise_in_bias,
                output_len as u32,
                EXPANDED as u32,
            )?;
            let mut activated = stream.alloc_zeros::<f16>(EXPANDED * output_len)?;
            kernels
                .ops
                .gelu_erf_f16(&expanded, &mut activated, (EXPANDED * output_len) as u32)?;
            let mut projected = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.gemm.matmul_f16(
                &activated,
                &stage.pointwise_out_weight,
                &mut projected,
                output_len as u32,
                CHANNELS as u32,
                EXPANDED as u32,
            )?;
            kernels.ops.add_bias_f16_inplace(
                &mut projected,
                &stage.pointwise_out_bias,
                output_len as u32,
                CHANNELS as u32,
            )?;
            let mut scaled = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.mul_f16_broadcast(
                &projected,
                &stage.gamma,
                &mut scaled,
                (CHANNELS * output_len) as u32,
                CHANNELS as u32,
            )?;
            let mut channel_first = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.transpose_f16(
                &scaled,
                &mut channel_first,
                output_len as u32,
                CHANNELS as u32,
            )?;
            let mut output = stream.alloc_zeros::<f16>(CHANNELS * output_len)?;
            kernels.ops.add_f16(
                &residual,
                &channel_first,
                &mut output,
                (CHANNELS * output_len) as u32,
            )?;
            hidden = output;
            seq_len = output_len;
        }

        stream.synchronize()?;
        let mut output_host = vec![f16::ZERO; CHANNELS * seq_len];
        stream.memcpy_dtoh(&hidden, &mut output_host)?;
        Ok(output_host.into_iter().map(f16::to_f32).collect())
    }
}
