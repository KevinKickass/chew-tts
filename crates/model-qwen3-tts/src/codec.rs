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

struct SnakeBetaWeights {
    alpha: CudaSlice<f16>,
    beta: CudaSlice<f16>,
}

struct CodecResidualUnit {
    first_activation: SnakeBetaWeights,
    first_weight: CudaSlice<f16>,
    first_bias: CudaSlice<f16>,
    second_activation: SnakeBetaWeights,
    second_weight: CudaSlice<f16>,
    second_bias: CudaSlice<f16>,
    dilation: usize,
}

struct CodecDecoderBlock {
    activation: SnakeBetaWeights,
    transposed_phase_weights: Vec<CudaSlice<f16>>,
    transposed_bias: CudaSlice<f16>,
    residual_units: Vec<CodecResidualUnit>,
    in_channels: usize,
    out_channels: usize,
    rate: usize,
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
    decoder_input_weight: CudaSlice<f16>,
    decoder_input_bias: CudaSlice<f16>,
    decoder_blocks: Vec<CodecDecoderBlock>,
    decoder_final_activation: SnakeBetaWeights,
    decoder_final_weight: CudaSlice<f16>,
    decoder_final_bias: CudaSlice<f16>,
}

struct CodecLayerKvCache {
    key: CudaSlice<f16>,
    value: CudaSlice<f16>,
}

/// Persistent causal state for the codec's eight-layer pre-transformer.
pub struct CodecTransformerSession {
    caches: Vec<CodecLayerKvCache>,
    latent_history: Vec<Vec<f32>>,
    position: usize,
    max_frames: usize,
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
        let load_snake = |prefix: &str, channels: usize| -> anyhow::Result<SnakeBetaWeights> {
            Ok(SnakeBetaWeights {
                alpha: load(&format!("{prefix}.alpha"), &[channels])?,
                beta: load(&format!("{prefix}.beta"), &[channels])?,
            })
        };
        let load_transposed_phases = |name: &str,
                                      in_channels: usize,
                                      out_channels: usize,
                                      rate: usize|
         -> anyhow::Result<Vec<CudaSlice<f16>>> {
            let tensor = load_f16_tensor(tokenizer_dir, name)?;
            let kernel_size = rate * 2;
            ensure!(
                tensor.shape == [in_channels, out_channels, kernel_size],
                "{name} has shape {:?}",
                tensor.shape
            );
            (0..rate)
                .map(|phase| {
                    let mut packed = vec![f16::ZERO; out_channels * in_channels * 2];
                    for output in 0..out_channels {
                        for input in 0..in_channels {
                            let source = (input * out_channels + output) * kernel_size;
                            let destination = (output * in_channels + input) * 2;
                            packed[destination] = tensor.values[source + phase + rate];
                            packed[destination + 1] = tensor.values[source + phase];
                        }
                    }
                    Ok(stream.clone_htod(&packed)?)
                })
                .collect()
        };
        let decoder_input_weight = load("decoder.decoder.0.conv.weight", &[1536, 1024, 7])?;
        let decoder_input_bias = load("decoder.decoder.0.conv.bias", &[1536])?;
        let mut decoder_blocks = Vec::with_capacity(4);
        for (block, (in_channels, out_channels, rate)) in [
            (1536usize, 768usize, 8usize),
            (768, 384, 5),
            (384, 192, 4),
            (192, 96, 3),
        ]
        .into_iter()
        .enumerate()
        {
            let prefix = format!("decoder.decoder.{}", block + 1);
            let mut residual_units = Vec::with_capacity(3);
            for (unit, dilation) in [1usize, 3, 9].into_iter().enumerate() {
                let residual_prefix = format!("{prefix}.block.{}", unit + 2);
                residual_units.push(CodecResidualUnit {
                    first_activation: load_snake(&format!("{residual_prefix}.act1"), out_channels)?,
                    first_weight: load(
                        &format!("{residual_prefix}.conv1.conv.weight"),
                        &[out_channels, out_channels, 7],
                    )?,
                    first_bias: load(
                        &format!("{residual_prefix}.conv1.conv.bias"),
                        &[out_channels],
                    )?,
                    second_activation: load_snake(
                        &format!("{residual_prefix}.act2"),
                        out_channels,
                    )?,
                    second_weight: load(
                        &format!("{residual_prefix}.conv2.conv.weight"),
                        &[out_channels, out_channels, 1],
                    )?,
                    second_bias: load(
                        &format!("{residual_prefix}.conv2.conv.bias"),
                        &[out_channels],
                    )?,
                    dilation,
                });
            }
            decoder_blocks.push(CodecDecoderBlock {
                activation: load_snake(&format!("{prefix}.block.0"), in_channels)?,
                transposed_phase_weights: load_transposed_phases(
                    &format!("{prefix}.block.1.conv.weight"),
                    in_channels,
                    out_channels,
                    rate,
                )?,
                transposed_bias: load(&format!("{prefix}.block.1.conv.bias"), &[out_channels])?,
                residual_units,
                in_channels,
                out_channels,
                rate,
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
            decoder_input_weight,
            decoder_input_bias,
            decoder_blocks,
            decoder_final_activation: load_snake("decoder.decoder.5", 96)?,
            decoder_final_weight: load("decoder.decoder.6.conv.weight", &[1, 96, 7])?,
            decoder_final_bias: load("decoder.decoder.6.conv.bias", &[1])?,
        })
    }

    pub fn start_transformer_session(
        &self,
        max_frames: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<CodecTransformerSession> {
        ensure!(max_frames > 0, "codec session must hold at least one frame");
        let caches = (0..self.transformer_layers.len())
            .map(|_| {
                Ok(CodecLayerKvCache {
                    key: stream.alloc_zeros::<f16>(1024 * max_frames)?,
                    value: stream.alloc_zeros::<f16>(1024 * max_frames)?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(CodecTransformerSession {
            caches,
            latent_history: Vec::with_capacity(2),
            position: 0,
            max_frames,
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

    /// Decode several codec frames jointly through the causal transformer.
    ///
    /// The returned tensor is channel-first `[1024, frames]`.
    pub fn decode_frames_transformer(
        &self,
        frames: &[Vec<i32>],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        const HIDDEN: usize = 512;
        const Q_DIM: usize = 1024;
        const FF_DIM: usize = 1024;
        ensure!(!frames.is_empty(), "at least one codec frame is required");
        let seq_len = frames.len();

        let mut latent_cf = vec![0.0f32; LATENT_DIM * seq_len];
        for (position, codes) in frames.iter().enumerate() {
            let latent = self.decode_frame(codes, kernels)?;
            for channel in 0..LATENT_DIM {
                latent_cf[channel * seq_len + position] = latent[channel];
            }
        }

        let stream = Arc::clone(kernels.ops.stream());
        let latent_f16 = latent_cf.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let latent_gpu = stream.clone_htod(&latent_f16)?;
        let mut preconv_cf = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels.ops.conv1d_causal_f16(
            &latent_gpu,
            &self.pre_conv_weight,
            &self.pre_conv_bias,
            &mut preconv_cf,
            LATENT_DIM as u32,
            1024,
            seq_len as u32,
            3,
            1,
            1,
        )?;
        let mut preconv_cl = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels
            .ops
            .transpose_f16(&preconv_cf, &mut preconv_cl, 1024, seq_len as u32)?;
        let mut projected = stream.alloc_zeros::<f16>(HIDDEN * seq_len)?;
        kernels.gemm.matmul_f16(
            &preconv_cl,
            &self.transformer_input_weight,
            &mut projected,
            seq_len as u32,
            HIDDEN as u32,
            1024,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.transformer_input_bias,
            seq_len as u32,
            HIDDEN as u32,
        )?;

        let hidden_elements = HIDDEN * seq_len;
        let q_elements = Q_DIM * seq_len;
        let ff_elements = FF_DIM * seq_len;
        let mut hidden = stream.alloc_zeros::<f32>(hidden_elements)?;
        kernels
            .ops
            .copy_f16_to_f32(&projected, &mut hidden, hidden_elements as u32)?;
        let mut norm = stream.alloc_zeros::<f16>(hidden_elements)?;
        let mut q = stream.alloc_zeros::<f16>(q_elements)?;
        let mut k = stream.alloc_zeros::<f16>(q_elements)?;
        let mut v = stream.alloc_zeros::<f16>(q_elements)?;
        let mut attention = stream.alloc_zeros::<f16>(q_elements)?;
        let mut delta = stream.alloc_zeros::<f16>(hidden_elements)?;
        let mut scaled = stream.alloc_zeros::<f16>(hidden_elements)?;
        let mut gate = stream.alloc_zeros::<f16>(ff_elements)?;
        let mut up = stream.alloc_zeros::<f16>(ff_elements)?;
        let mut activation = stream.alloc_zeros::<f16>(ff_elements)?;

        for layer in &self.transformer_layers {
            kernels.ops.rms_norm_f32in(
                &hidden,
                &layer.input_norm,
                &mut norm,
                seq_len as u32,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.q_proj,
                &mut q,
                seq_len as u32,
                Q_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.k_proj,
                &mut k,
                seq_len as u32,
                Q_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.v_proj,
                &mut v,
                seq_len as u32,
                Q_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .rope_neox(&mut q, seq_len as u32, 16, 64, 0, 10_000.0)?;
            kernels
                .ops
                .rope_neox(&mut k, seq_len as u32, 16, 64, 0, 10_000.0)?;
            kernels.ops.mha_fused(
                &q,
                &k.slice(..),
                &v.slice(..),
                &mut attention,
                64,
                16,
                16,
                seq_len as u32,
                seq_len as u32,
                0,
            )?;
            kernels.gemm.matmul_f16(
                &attention,
                &layer.o_proj,
                &mut delta,
                seq_len as u32,
                HIDDEN as u32,
                Q_DIM as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.attention_scale,
                &mut scaled,
                hidden_elements as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, hidden_elements as u32)?;

            kernels.ops.rms_norm_f32in(
                &hidden,
                &layer.post_attention_norm,
                &mut norm,
                seq_len as u32,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.gate_proj,
                &mut gate,
                seq_len as u32,
                FF_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.up_proj,
                &mut up,
                seq_len as u32,
                FF_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .silu(&gate, &up, &mut activation, ff_elements as u32)?;
            kernels.gemm.matmul_f16(
                &activation,
                &layer.down_proj,
                &mut delta,
                seq_len as u32,
                HIDDEN as u32,
                FF_DIM as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.mlp_scale,
                &mut scaled,
                hidden_elements as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, hidden_elements as u32)?;
        }

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.transformer_norm,
            &mut norm,
            seq_len as u32,
            HIDDEN as u32,
            1e-5,
        )?;
        let mut output_cl = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.transformer_output_weight,
            &mut output_cl,
            seq_len as u32,
            1024,
            HIDDEN as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut output_cl,
            &self.transformer_output_bias,
            seq_len as u32,
            1024,
        )?;
        let mut output_cf = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels
            .ops
            .transpose_f16(&output_cl, &mut output_cf, seq_len as u32, 1024)?;
        stream.synchronize()?;
        let mut output_host = vec![f16::ZERO; 1024 * seq_len];
        stream.memcpy_dtoh(&output_cf, &mut output_host)?;
        Ok(output_host.into_iter().map(f16::to_f32).collect())
    }

    /// Decode a new frame chunk while retaining exact transformer KV state.
    ///
    /// The returned tensor is channel-first `[1024, new_frames]`.
    pub fn decode_frames_transformer_session(
        &self,
        frames: &[Vec<i32>],
        session: &mut CodecTransformerSession,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        const HIDDEN: usize = 512;
        const Q_DIM: usize = 1024;
        const FF_DIM: usize = 1024;
        ensure!(!frames.is_empty(), "at least one codec frame is required");
        let seq_len = frames.len();
        ensure!(
            session.position + seq_len <= session.max_frames,
            "codec session exceeds {} frames",
            session.max_frames
        );

        let mut new_latents = Vec::with_capacity(seq_len);
        for codes in frames {
            new_latents.push(self.decode_frame(codes, kernels)?);
        }
        let history_len = session.latent_history.len();
        let input_len = history_len + seq_len;
        let mut latent_cf = vec![0.0f32; LATENT_DIM * input_len];
        for (position, latent) in session
            .latent_history
            .iter()
            .chain(&new_latents)
            .enumerate()
        {
            for channel in 0..LATENT_DIM {
                latent_cf[channel * input_len + position] = latent[channel];
            }
        }

        let stream = Arc::clone(kernels.ops.stream());
        let latent_f16 = latent_cf.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let latent_gpu = stream.clone_htod(&latent_f16)?;
        let mut preconv_cf = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels.ops.conv1d_causal_offset_f16(
            &latent_gpu,
            &self.pre_conv_weight,
            &self.pre_conv_bias,
            &mut preconv_cf,
            LATENT_DIM as u32,
            1024,
            input_len as u32,
            seq_len as u32,
            history_len as u32,
            3,
            1,
            1,
        )?;
        let mut preconv_cl = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels
            .ops
            .transpose_f16(&preconv_cf, &mut preconv_cl, 1024, seq_len as u32)?;
        let mut projected = stream.alloc_zeros::<f16>(HIDDEN * seq_len)?;
        kernels.gemm.matmul_f16(
            &preconv_cl,
            &self.transformer_input_weight,
            &mut projected,
            seq_len as u32,
            HIDDEN as u32,
            1024,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.transformer_input_bias,
            seq_len as u32,
            HIDDEN as u32,
        )?;

        let hidden_elements = HIDDEN * seq_len;
        let q_elements = Q_DIM * seq_len;
        let ff_elements = FF_DIM * seq_len;
        let mut hidden = stream.alloc_zeros::<f32>(hidden_elements)?;
        kernels
            .ops
            .copy_f16_to_f32(&projected, &mut hidden, hidden_elements as u32)?;
        let mut norm = stream.alloc_zeros::<f16>(hidden_elements)?;
        let mut q = stream.alloc_zeros::<f16>(q_elements)?;
        let mut k = stream.alloc_zeros::<f16>(q_elements)?;
        let mut v = stream.alloc_zeros::<f16>(q_elements)?;
        let mut attention = stream.alloc_zeros::<f16>(q_elements)?;
        let mut delta = stream.alloc_zeros::<f16>(hidden_elements)?;
        let mut scaled = stream.alloc_zeros::<f16>(hidden_elements)?;
        let mut gate = stream.alloc_zeros::<f16>(ff_elements)?;
        let mut up = stream.alloc_zeros::<f16>(ff_elements)?;
        let mut activation = stream.alloc_zeros::<f16>(ff_elements)?;
        let cache_start = session.position * Q_DIM;
        let cache_end = (session.position + seq_len) * Q_DIM;

        for ((layer, cache), layer_index) in self
            .transformer_layers
            .iter()
            .zip(&mut session.caches)
            .zip(0usize..)
        {
            kernels.ops.rms_norm_f32in(
                &hidden,
                &layer.input_norm,
                &mut norm,
                seq_len as u32,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.q_proj,
                &mut q,
                seq_len as u32,
                Q_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.k_proj,
                &mut k,
                seq_len as u32,
                Q_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.v_proj,
                &mut v,
                seq_len as u32,
                Q_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.ops.rope_neox(
                &mut q,
                seq_len as u32,
                16,
                64,
                session.position as u32,
                10_000.0,
            )?;
            kernels.ops.rope_neox(
                &mut k,
                seq_len as u32,
                16,
                64,
                session.position as u32,
                10_000.0,
            )?;
            kernels.ops.copy_f16(
                &k,
                &mut cache.key.slice_mut(cache_start..cache_end),
                q_elements as u32,
            )?;
            kernels.ops.copy_f16(
                &v,
                &mut cache.value.slice_mut(cache_start..cache_end),
                q_elements as u32,
            )?;
            kernels.ops.mha_fused(
                &q,
                &cache.key.slice(..cache_end),
                &cache.value.slice(..cache_end),
                &mut attention,
                64,
                16,
                16,
                seq_len as u32,
                (session.position + seq_len) as u32,
                session.position as u32,
            )?;
            kernels.gemm.matmul_f16(
                &attention,
                &layer.o_proj,
                &mut delta,
                seq_len as u32,
                HIDDEN as u32,
                Q_DIM as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.attention_scale,
                &mut scaled,
                hidden_elements as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, hidden_elements as u32)?;
            kernels.ops.rms_norm_f32in(
                &hidden,
                &layer.post_attention_norm,
                &mut norm,
                seq_len as u32,
                HIDDEN as u32,
                1e-5,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.gate_proj,
                &mut gate,
                seq_len as u32,
                FF_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels.gemm.matmul_f16(
                &norm,
                &layer.up_proj,
                &mut up,
                seq_len as u32,
                FF_DIM as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .silu(&gate, &up, &mut activation, ff_elements as u32)?;
            kernels.gemm.matmul_f16(
                &activation,
                &layer.down_proj,
                &mut delta,
                seq_len as u32,
                HIDDEN as u32,
                FF_DIM as u32,
            )?;
            kernels.ops.mul_f16_broadcast(
                &delta,
                &layer.mlp_scale,
                &mut scaled,
                hidden_elements as u32,
                HIDDEN as u32,
            )?;
            kernels
                .ops
                .add_inplace_f32_f16(&mut hidden, &scaled, hidden_elements as u32)
                .with_context(|| format!("codec transformer layer {layer_index} failed"))?;
        }

        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.transformer_norm,
            &mut norm,
            seq_len as u32,
            HIDDEN as u32,
            1e-5,
        )?;
        let mut output_cl = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.transformer_output_weight,
            &mut output_cl,
            seq_len as u32,
            1024,
            HIDDEN as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut output_cl,
            &self.transformer_output_bias,
            seq_len as u32,
            1024,
        )?;
        let mut output_cf = stream.alloc_zeros::<f16>(1024 * seq_len)?;
        kernels
            .ops
            .transpose_f16(&output_cl, &mut output_cf, seq_len as u32, 1024)?;
        stream.synchronize()?;
        let mut output_host = vec![f16::ZERO; 1024 * seq_len];
        stream.memcpy_dtoh(&output_cf, &mut output_host)?;

        session.position += seq_len;
        let mut combined = session
            .latent_history
            .drain(..)
            .chain(new_latents)
            .collect::<Vec<_>>();
        if combined.len() > 2 {
            combined.drain(..combined.len() - 2);
        }
        session.latent_history = combined;
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

    /// Jointly transform and 4x-upsample several codec frames.
    ///
    /// The returned tensor is channel-first `[1024, frames * 4]`.
    pub fn decode_frames_upsampled(
        &self,
        frames: &[Vec<i32>],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let transformed = self.decode_frames_transformer(frames, kernels)?;
        self.decode_transformed_upsampled(&transformed, frames.len(), kernels)
    }

    /// Jointly 4x-upsample already transformed codec frames.
    ///
    /// `transformed` must be channel-first `[1024, frame_count]`.
    pub fn decode_transformed_upsampled(
        &self,
        transformed: &[f32],
        frame_count: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        const CHANNELS: usize = 1024;
        const EXPANDED: usize = 4096;
        anyhow::ensure!(
            transformed.len() == CHANNELS * frame_count,
            "transformed codec tensor has {} values, expected {} for {frame_count} frame(s)",
            transformed.len(),
            CHANNELS * frame_count
        );
        let stream = Arc::clone(kernels.ops.stream());
        let transformed = transformed
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let mut hidden = stream.clone_htod(&transformed)?;
        let mut seq_len = frame_count;

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

    /// Decode one complete 12.5-Hz codec frame into 1,920 PCM samples at 24 kHz.
    pub fn decode_frame_audio(
        &self,
        codes: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let upsampled = self.decode_frame_upsampled(codes, kernels)?;
        self.decode_upsampled_audio(upsampled, 1, kernels)
    }

    /// Decode multiple codec frames into one continuous 24-kHz waveform.
    pub fn decode_frames_audio(
        &self,
        frames: &[Vec<i32>],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let upsampled = self.decode_frames_upsampled(frames, kernels)?;
        self.decode_upsampled_audio(upsampled, frames.len(), kernels)
    }

    /// Decode already transformed codec frames into one continuous 24-kHz waveform.
    ///
    /// `transformed` must be channel-first `[1024, frame_count]`.
    pub fn decode_transformed_audio(
        &self,
        transformed: &[f32],
        frame_count: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let upsampled = self.decode_transformed_upsampled(transformed, frame_count, kernels)?;
        self.decode_upsampled_audio(upsampled, frame_count, kernels)
    }

    fn decode_upsampled_audio(
        &self,
        upsampled: Vec<f32>,
        frame_count: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let upsampled = upsampled.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let upsampled = stream.clone_htod(&upsampled)?;
        let mut seq_len = frame_count * 4;
        let mut hidden = stream.alloc_zeros::<f16>(1536 * seq_len)?;
        conv1d_causal_gemm(
            &upsampled,
            &self.decoder_input_weight,
            &self.decoder_input_bias,
            &mut hidden,
            1024,
            1536,
            seq_len as u32,
            7,
            1,
            kernels,
        )?;

        for block in &self.decoder_blocks {
            let mut activated = stream.alloc_zeros::<f16>(block.in_channels * seq_len)?;
            kernels.ops.snake_beta_f16(
                &hidden,
                &block.activation.alpha,
                &block.activation.beta,
                &mut activated,
                block.in_channels as u32,
                seq_len as u32,
            )?;
            let output_len = seq_len * block.rate;
            let mut output = stream.alloc_zeros::<f16>(block.out_channels * output_len)?;
            conv_transpose1d_causal_gemm(
                &activated,
                &block.transposed_phase_weights,
                &block.transposed_bias,
                &mut output,
                block.in_channels as u32,
                block.out_channels as u32,
                seq_len as u32,
                block.rate as u32,
                kernels,
            )?;

            for unit in &block.residual_units {
                let mut first_activation =
                    stream.alloc_zeros::<f16>(block.out_channels * output_len)?;
                kernels.ops.snake_beta_f16(
                    &output,
                    &unit.first_activation.alpha,
                    &unit.first_activation.beta,
                    &mut first_activation,
                    block.out_channels as u32,
                    output_len as u32,
                )?;
                let mut first_conv = stream.alloc_zeros::<f16>(block.out_channels * output_len)?;
                conv1d_causal_gemm(
                    &first_activation,
                    &unit.first_weight,
                    &unit.first_bias,
                    &mut first_conv,
                    block.out_channels as u32,
                    block.out_channels as u32,
                    output_len as u32,
                    7,
                    unit.dilation as u32,
                    kernels,
                )?;
                let mut second_activation =
                    stream.alloc_zeros::<f16>(block.out_channels * output_len)?;
                kernels.ops.snake_beta_f16(
                    &first_conv,
                    &unit.second_activation.alpha,
                    &unit.second_activation.beta,
                    &mut second_activation,
                    block.out_channels as u32,
                    output_len as u32,
                )?;
                let mut second_conv = stream.alloc_zeros::<f16>(block.out_channels * output_len)?;
                conv1d_causal_gemm(
                    &second_activation,
                    &unit.second_weight,
                    &unit.second_bias,
                    &mut second_conv,
                    block.out_channels as u32,
                    block.out_channels as u32,
                    output_len as u32,
                    1,
                    1,
                    kernels,
                )?;
                let mut residual = stream.alloc_zeros::<f16>(block.out_channels * output_len)?;
                kernels.ops.add_f16(
                    &output,
                    &second_conv,
                    &mut residual,
                    (block.out_channels * output_len) as u32,
                )?;
                output = residual;
            }
            hidden = output;
            seq_len = output_len;
        }

        ensure!(
            seq_len == frame_count * 1920,
            "codec produced {seq_len} samples for {frame_count} frames"
        );
        let mut final_activation = stream.alloc_zeros::<f16>(96 * seq_len)?;
        kernels.ops.snake_beta_f16(
            &hidden,
            &self.decoder_final_activation.alpha,
            &self.decoder_final_activation.beta,
            &mut final_activation,
            96,
            seq_len as u32,
        )?;
        let mut waveform = stream.alloc_zeros::<f16>(seq_len)?;
        conv1d_causal_gemm(
            &final_activation,
            &self.decoder_final_weight,
            &self.decoder_final_bias,
            &mut waveform,
            96,
            1,
            seq_len as u32,
            7,
            1,
            kernels,
        )?;
        let mut clamped = stream.alloc_zeros::<f16>(seq_len)?;
        kernels
            .ops
            .clamp_f16(&waveform, &mut clamped, seq_len as u32, -1.0, 1.0)?;
        stream.synchronize()?;
        let mut output_host = vec![f16::ZERO; seq_len];
        stream.memcpy_dtoh(&clamped, &mut output_host)?;
        Ok(output_host.into_iter().map(f16::to_f32).collect())
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d_causal_gemm(
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    in_channels: u32,
    out_channels: u32,
    seq_len: u32,
    kernel_size: u32,
    dilation: u32,
    kernels: &mut GpuKernels,
) -> anyhow::Result<()> {
    let stream = Arc::clone(kernels.ops.stream());
    let width = in_channels * kernel_size;
    let mut unfolded = stream.alloc_zeros::<f16>((seq_len * width) as usize)?;
    kernels.ops.unfold_causal_f16(
        input,
        &mut unfolded,
        in_channels,
        seq_len,
        kernel_size,
        dilation,
    )?;
    let mut channel_last = stream.alloc_zeros::<f16>((seq_len * out_channels) as usize)?;
    kernels.gemm.matmul_f16(
        &unfolded,
        weight,
        &mut channel_last,
        seq_len,
        out_channels,
        width,
    )?;
    kernels
        .ops
        .add_bias_f16_inplace(&mut channel_last, bias, seq_len, out_channels)?;
    kernels
        .ops
        .transpose_f16(&channel_last, output, seq_len, out_channels)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_causal_gemm(
    input: &CudaSlice<f16>,
    phase_weights: &[CudaSlice<f16>],
    bias: &CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    in_channels: u32,
    out_channels: u32,
    input_len: u32,
    stride: u32,
    kernels: &mut GpuKernels,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        phase_weights.len() == stride as usize,
        "transposed convolution has {} phases, expected {stride}",
        phase_weights.len()
    );
    let stream = Arc::clone(kernels.ops.stream());
    let width = in_channels * 2;
    let mut unfolded = stream.alloc_zeros::<f16>((input_len * width) as usize)?;
    kernels
        .ops
        .unfold_causal_f16(input, &mut unfolded, in_channels, input_len, 2, 1)?;
    let mut phase_output = stream.alloc_zeros::<f16>((input_len * out_channels) as usize)?;
    for (phase, weight) in phase_weights.iter().enumerate() {
        kernels.gemm.matmul_f16(
            &unfolded,
            weight,
            &mut phase_output,
            input_len,
            out_channels,
            width,
        )?;
        kernels.ops.scatter_conv_transpose_phase_f16(
            &phase_output,
            bias,
            output,
            input_len,
            out_channels,
            stride,
            phase as u32,
        )?;
    }
    Ok(())
}
