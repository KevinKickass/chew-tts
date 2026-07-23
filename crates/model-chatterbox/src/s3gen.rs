use crate::{ChatterboxConditioning, ChatterboxFlowEstimator};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_safetensors::MappedSafetensors;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

pub const S3_HIDDEN_SIZE: usize = 512;
pub const S3_INTERMEDIATE_SIZE: usize = 2_048;
pub const S3_ATTENTION_HEADS: usize = 8;
pub const S3_HEAD_DIM: usize = 64;
pub const S3_VOCAB_SIZE: usize = 6_561;
pub const S3_MEL_BINS: usize = 80;

pub struct ChatterboxS3Encoder {
    token_embedding: CudaSlice<f16>,
    input_weight: CudaSlice<f16>,
    input_bias: CudaSlice<f16>,
    input_norm_weight: CudaSlice<f16>,
    input_norm_bias: CudaSlice<f16>,
    lookahead1_weight: CudaSlice<f16>,
    lookahead1_bias: CudaSlice<f16>,
    lookahead2_weight: CudaSlice<f16>,
    lookahead2_bias: CudaSlice<f16>,
    layers: Vec<ChatterboxS3ConformerLayer>,
    upsample_weight: CudaSlice<f16>,
    upsample_bias: CudaSlice<f16>,
    up_input_weight: CudaSlice<f16>,
    up_input_bias: CudaSlice<f16>,
    up_input_norm_weight: CudaSlice<f16>,
    up_input_norm_bias: CudaSlice<f16>,
    up_layers: Vec<ChatterboxS3ConformerLayer>,
    final_norm_weight: CudaSlice<f16>,
    final_norm_bias: CudaSlice<f16>,
    projection_weight: CudaSlice<f16>,
    projection_bias: CudaSlice<f16>,
}

pub struct ChatterboxS3Flow {
    encoder: ChatterboxS3Encoder,
    estimator: ChatterboxFlowEstimator,
    speaker_weight: Vec<f32>,
    speaker_bias: Vec<f32>,
}

/// One inference-mode block from S3Gen's UpsampleConformerEncoder.
///
/// The V3 checkpoint disables the optional macaron FFN and convolution
/// module, so the actual block is relative MHA followed by a SiLU FFN.
pub struct ChatterboxS3ConformerLayer {
    norm_mha_weight: CudaSlice<f16>,
    norm_mha_bias: CudaSlice<f16>,
    q_weight: CudaSlice<f16>,
    q_bias: CudaSlice<f16>,
    k_weight: CudaSlice<f16>,
    k_bias: CudaSlice<f16>,
    v_weight: CudaSlice<f16>,
    v_bias: CudaSlice<f16>,
    pos_weight: CudaSlice<f16>,
    pos_bias_u: CudaSlice<f16>,
    pos_bias_v: CudaSlice<f16>,
    out_weight: CudaSlice<f16>,
    out_bias: CudaSlice<f16>,
    norm_ff_weight: CudaSlice<f16>,
    norm_ff_bias: CudaSlice<f16>,
    ff1_weight: CudaSlice<f16>,
    ff1_bias: CudaSlice<f16>,
    ff2_weight: CudaSlice<f16>,
    ff2_bias: CudaSlice<f16>,
}

impl ChatterboxS3ConformerLayer {
    pub fn load(
        model_dir: &Path,
        upsampled: bool,
        layer_index: usize,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let limit = if upsampled { 4 } else { 6 };
        ensure!(
            layer_index < limit,
            "S3Gen Conformer layer {layer_index} is outside 0..{limit}"
        );
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let group = if upsampled {
            "flow.encoder.up_encoders"
        } else {
            "flow.encoder.encoders"
        };
        let prefix = format!("{group}.{layer_index}");
        let load = |suffix: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let (shape, values) = weights
                .tensor_f16(&name)
                .with_context(|| format!("could not load Chatterbox S3Gen {name}"))?;
            ensure!(
                shape == expected,
                "Chatterbox S3Gen {name} has shape {shape:?}, expected {expected:?}"
            );
            Ok(stream.clone_htod(&values)?)
        };
        Ok(Self {
            norm_mha_weight: load("norm_mha.weight", &[S3_HIDDEN_SIZE])?,
            norm_mha_bias: load("norm_mha.bias", &[S3_HIDDEN_SIZE])?,
            q_weight: load(
                "self_attn.linear_q.weight",
                &[S3_HIDDEN_SIZE, S3_HIDDEN_SIZE],
            )?,
            q_bias: load("self_attn.linear_q.bias", &[S3_HIDDEN_SIZE])?,
            k_weight: load(
                "self_attn.linear_k.weight",
                &[S3_HIDDEN_SIZE, S3_HIDDEN_SIZE],
            )?,
            k_bias: load("self_attn.linear_k.bias", &[S3_HIDDEN_SIZE])?,
            v_weight: load(
                "self_attn.linear_v.weight",
                &[S3_HIDDEN_SIZE, S3_HIDDEN_SIZE],
            )?,
            v_bias: load("self_attn.linear_v.bias", &[S3_HIDDEN_SIZE])?,
            pos_weight: load(
                "self_attn.linear_pos.weight",
                &[S3_HIDDEN_SIZE, S3_HIDDEN_SIZE],
            )?,
            pos_bias_u: load("self_attn.pos_bias_u", &[S3_ATTENTION_HEADS, S3_HEAD_DIM])?,
            pos_bias_v: load("self_attn.pos_bias_v", &[S3_ATTENTION_HEADS, S3_HEAD_DIM])?,
            out_weight: load(
                "self_attn.linear_out.weight",
                &[S3_HIDDEN_SIZE, S3_HIDDEN_SIZE],
            )?,
            out_bias: load("self_attn.linear_out.bias", &[S3_HIDDEN_SIZE])?,
            norm_ff_weight: load("norm_ff.weight", &[S3_HIDDEN_SIZE])?,
            norm_ff_bias: load("norm_ff.bias", &[S3_HIDDEN_SIZE])?,
            ff1_weight: load(
                "feed_forward.w_1.weight",
                &[S3_INTERMEDIATE_SIZE, S3_HIDDEN_SIZE],
            )?,
            ff1_bias: load("feed_forward.w_1.bias", &[S3_INTERMEDIATE_SIZE])?,
            ff2_weight: load(
                "feed_forward.w_2.weight",
                &[S3_HIDDEN_SIZE, S3_INTERMEDIATE_SIZE],
            )?,
            ff2_bias: load("feed_forward.w_2.bias", &[S3_HIDDEN_SIZE])?,
        })
    }

    pub fn forward(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "S3Gen sequence must be non-empty");
        ensure!(
            hidden_host.len() == seq_len * S3_HIDDEN_SIZE,
            "S3Gen input has {} values, expected {}",
            hidden_host.len(),
            seq_len * S3_HIDDEN_SIZE
        );
        let stream = Arc::clone(kernels.ops.stream());
        let elements = seq_len * S3_HIDDEN_SIZE;
        let mut hidden = stream.clone_htod(hidden_host)?;
        self.forward_device(&mut hidden, seq_len, kernels)?;
        stream.synchronize()?;
        let mut output = vec![0.0; elements];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
    }

    fn forward_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        let stream = Arc::clone(kernels.ops.stream());
        let elements = seq_len * S3_HIDDEN_SIZE;
        let mut norm = stream.alloc_zeros::<f16>(elements)?;
        kernels.ops.layer_norm_f32in(
            hidden,
            &self.norm_mha_weight,
            &self.norm_mha_bias,
            &mut norm,
            seq_len as u32,
            S3_HIDDEN_SIZE as u32,
            1e-12,
        )?;

        let mut q = stream.alloc_zeros::<f16>(elements)?;
        let mut k = stream.alloc_zeros::<f16>(elements)?;
        let mut v = stream.alloc_zeros::<f16>(elements)?;
        for (weight, bias, output) in [
            (&self.q_weight, &self.q_bias, &mut q),
            (&self.k_weight, &self.k_bias, &mut k),
            (&self.v_weight, &self.v_bias, &mut v),
        ] {
            kernels.gemm.matmul_f16(
                &norm,
                weight,
                output,
                seq_len as u32,
                S3_HIDDEN_SIZE as u32,
                S3_HIDDEN_SIZE as u32,
            )?;
            kernels.ops.add_bias_f16_inplace(
                output,
                bias,
                seq_len as u32,
                S3_HIDDEN_SIZE as u32,
            )?;
        }

        let pos_host = espnet_relative_positions(seq_len);
        let pos_input = stream.clone_htod(&pos_host)?;
        let mut pos = stream.alloc_zeros::<f16>((2 * seq_len - 1) * S3_HIDDEN_SIZE)?;
        kernels.gemm.matmul_f16(
            &pos_input,
            &self.pos_weight,
            &mut pos,
            (2 * seq_len - 1) as u32,
            S3_HIDDEN_SIZE as u32,
            S3_HIDDEN_SIZE as u32,
        )?;
        let mut attention = stream.alloc_zeros::<f16>(elements)?;
        kernels.ops.mha_relative_full(
            &q,
            &k,
            &v,
            &pos,
            &self.pos_bias_u,
            &self.pos_bias_v,
            &mut attention,
            S3_HEAD_DIM as u32,
            S3_ATTENTION_HEADS as u32,
            seq_len as u32,
        )?;
        let mut attention_out = stream.alloc_zeros::<f16>(elements)?;
        kernels.gemm.matmul_f16(
            &attention,
            &self.out_weight,
            &mut attention_out,
            seq_len as u32,
            S3_HIDDEN_SIZE as u32,
            S3_HIDDEN_SIZE as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut attention_out,
            &self.out_bias,
            seq_len as u32,
            S3_HIDDEN_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &attention_out, elements as u32)?;

        kernels.ops.layer_norm_f32in(
            hidden,
            &self.norm_ff_weight,
            &self.norm_ff_bias,
            &mut norm,
            seq_len as u32,
            S3_HIDDEN_SIZE as u32,
            1e-12,
        )?;
        let mut ff1 = stream.alloc_zeros::<f16>(seq_len * S3_INTERMEDIATE_SIZE)?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.ff1_weight,
            &mut ff1,
            seq_len as u32,
            S3_INTERMEDIATE_SIZE as u32,
            S3_HIDDEN_SIZE as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut ff1,
            &self.ff1_bias,
            seq_len as u32,
            S3_INTERMEDIATE_SIZE as u32,
        )?;
        let mut activated = stream.alloc_zeros::<f16>(seq_len * S3_INTERMEDIATE_SIZE)?;
        kernels.ops.silu_act_f16(
            &ff1,
            &mut activated,
            (seq_len * S3_INTERMEDIATE_SIZE) as u32,
        )?;
        let mut ff2 = stream.alloc_zeros::<f16>(elements)?;
        kernels.gemm.matmul_f16(
            &activated,
            &self.ff2_weight,
            &mut ff2,
            seq_len as u32,
            S3_HIDDEN_SIZE as u32,
            S3_INTERMEDIATE_SIZE as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut ff2,
            &self.ff2_bias,
            seq_len as u32,
            S3_HIDDEN_SIZE as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &ff2, elements as u32)?;
        Ok(())
    }
}

impl ChatterboxS3Encoder {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let load = |name: &str, shape: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let (actual, values) = weights.tensor_f16(name)?;
            ensure!(
                actual == shape,
                "{name}: got {actual:?}, expected {shape:?}"
            );
            Ok(stream.clone_htod(&values)?)
        };
        Ok(Self {
            token_embedding: load("flow.input_embedding.weight", &[6561, 512])?,
            input_weight: load("flow.encoder.embed.out.0.weight", &[512, 512])?,
            input_bias: load("flow.encoder.embed.out.0.bias", &[512])?,
            input_norm_weight: load("flow.encoder.embed.out.1.weight", &[512])?,
            input_norm_bias: load("flow.encoder.embed.out.1.bias", &[512])?,
            lookahead1_weight: load(
                "flow.encoder.pre_lookahead_layer.conv1.weight",
                &[512, 512, 4],
            )?,
            lookahead1_bias: load("flow.encoder.pre_lookahead_layer.conv1.bias", &[512])?,
            lookahead2_weight: load(
                "flow.encoder.pre_lookahead_layer.conv2.weight",
                &[512, 512, 3],
            )?,
            lookahead2_bias: load("flow.encoder.pre_lookahead_layer.conv2.bias", &[512])?,
            layers: (0..6)
                .map(|i| ChatterboxS3ConformerLayer::load(model_dir, false, i, stream))
                .collect::<anyhow::Result<_>>()?,
            upsample_weight: load("flow.encoder.up_layer.conv.weight", &[512, 512, 5])?,
            upsample_bias: load("flow.encoder.up_layer.conv.bias", &[512])?,
            up_input_weight: load("flow.encoder.up_embed.out.0.weight", &[512, 512])?,
            up_input_bias: load("flow.encoder.up_embed.out.0.bias", &[512])?,
            up_input_norm_weight: load("flow.encoder.up_embed.out.1.weight", &[512])?,
            up_input_norm_bias: load("flow.encoder.up_embed.out.1.bias", &[512])?,
            up_layers: (0..4)
                .map(|i| ChatterboxS3ConformerLayer::load(model_dir, true, i, stream))
                .collect::<anyhow::Result<_>>()?,
            final_norm_weight: load("flow.encoder.after_norm.weight", &[512])?,
            final_norm_bias: load("flow.encoder.after_norm.bias", &[512])?,
            projection_weight: load("flow.encoder_proj.weight", &[S3_MEL_BINS, 512])?,
            projection_bias: load("flow.encoder_proj.bias", &[S3_MEL_BINS])?,
        })
    }

    /// Prompt and generated speech tokens to frame-major [2*T, 80] flow input.
    pub fn encode(&self, tokens: &[i32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "S3Gen token sequence is empty");
        ensure!(
            tokens
                .iter()
                .all(|x| *x >= 0 && (*x as usize) < S3_VOCAB_SIZE),
            "S3Gen token is outside vocabulary"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let ids = stream.clone_htod(tokens)?;
        let mut embeddings = stream.alloc_zeros::<f16>(tokens.len() * 512)?;
        kernels.ops.gather_rows_f16(
            &self.token_embedding,
            &ids,
            &mut embeddings,
            tokens.len() as u32,
            512,
        )?;
        let mut hidden = linear_norm_scale(
            &embeddings,
            tokens.len(),
            &self.input_weight,
            &self.input_bias,
            &self.input_norm_weight,
            &self.input_norm_bias,
            kernels,
        )?;
        self.prelookahead(&mut hidden, tokens.len(), kernels)?;
        for layer in &self.layers {
            layer.forward_device(&mut hidden, tokens.len(), kernels)?;
        }

        let up_len = tokens.len() * 2;
        let mut sequence = stream.alloc_zeros::<f16>(tokens.len() * 512)?;
        kernels.ops.copy_f32_to_f16(
            &hidden,
            &mut sequence.slice_mut(..),
            (tokens.len() * 512) as u32,
        )?;
        let mut channels = stream.alloc_zeros::<f16>(tokens.len() * 512)?;
        kernels
            .ops
            .transpose_f16(&sequence, &mut channels, tokens.len() as u32, 512)?;
        let mut repeated = stream.alloc_zeros::<f16>(up_len * 512)?;
        kernels
            .ops
            .repeat_interleave_f16(&channels, &mut repeated, 512, tokens.len() as u32, 2)?;
        let mut convolved = stream.alloc_zeros::<f16>(up_len * 512)?;
        kernels.ops.conv1d_padded_f16(
            &repeated,
            &self.upsample_weight,
            &self.upsample_bias,
            &mut convolved,
            512,
            512,
            up_len as u32,
            5,
            4,
        )?;
        let mut up_sequence = stream.alloc_zeros::<f16>(up_len * 512)?;
        kernels
            .ops
            .transpose_f16(&convolved, &mut up_sequence, 512, up_len as u32)?;
        hidden = linear_norm_scale(
            &up_sequence,
            up_len,
            &self.up_input_weight,
            &self.up_input_bias,
            &self.up_input_norm_weight,
            &self.up_input_norm_bias,
            kernels,
        )?;
        for layer in &self.up_layers {
            layer.forward_device(&mut hidden, up_len, kernels)?;
        }
        let mut normalized = stream.alloc_zeros::<f16>(up_len * 512)?;
        kernels.ops.layer_norm_f32in(
            &hidden,
            &self.final_norm_weight,
            &self.final_norm_bias,
            &mut normalized,
            up_len as u32,
            512,
            1e-5,
        )?;
        let mut projected = stream.alloc_zeros::<f16>(up_len * S3_MEL_BINS)?;
        kernels.gemm.matmul_f16(
            &normalized,
            &self.projection_weight,
            &mut projected,
            up_len as u32,
            S3_MEL_BINS as u32,
            512,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.projection_bias,
            up_len as u32,
            S3_MEL_BINS as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; up_len * S3_MEL_BINS];
        stream.memcpy_dtoh(&projected, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    fn prelookahead(
        &self,
        hidden: &mut CudaSlice<f32>,
        seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        let stream = Arc::clone(kernels.ops.stream());
        let n = seq_len * 512;
        let mut sequence = stream.alloc_zeros::<f16>(n)?;
        kernels
            .ops
            .copy_f32_to_f16(hidden, &mut sequence.slice_mut(..), n as u32)?;
        let mut channels = stream.alloc_zeros::<f16>(n)?;
        kernels
            .ops
            .transpose_f16(&sequence, &mut channels, seq_len as u32, 512)?;
        let mut first = stream.alloc_zeros::<f16>(n)?;
        kernels.ops.conv1d_padded_f16(
            &channels,
            &self.lookahead1_weight,
            &self.lookahead1_bias,
            &mut first,
            512,
            512,
            seq_len as u32,
            4,
            0,
        )?;
        let mut activated = stream.alloc_zeros::<f16>(n)?;
        kernels
            .ops
            .leaky_relu_f16(&first, &mut activated, n as u32, 0.01)?;
        let mut second = stream.alloc_zeros::<f16>(n)?;
        kernels.ops.conv1d_padded_f16(
            &activated,
            &self.lookahead2_weight,
            &self.lookahead2_bias,
            &mut second,
            512,
            512,
            seq_len as u32,
            3,
            2,
        )?;
        let mut delta = stream.alloc_zeros::<f16>(n)?;
        kernels
            .ops
            .transpose_f16(&second, &mut delta, 512, seq_len as u32)?;
        kernels.ops.add_inplace_f32_f16(hidden, &delta, n as u32)?;
        Ok(())
    }
}

fn linear_norm_scale(
    input: &CudaSlice<f16>,
    seq_len: usize,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    norm_weight: &CudaSlice<f16>,
    norm_bias: &CudaSlice<f16>,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f32>> {
    let stream = Arc::clone(kernels.ops.stream());
    let n = seq_len * 512;
    let mut linear = stream.alloc_zeros::<f16>(n)?;
    kernels
        .gemm
        .matmul_f16(input, weight, &mut linear, seq_len as u32, 512, 512)?;
    kernels
        .ops
        .add_bias_f16_inplace(&mut linear, bias, seq_len as u32, 512)?;
    let mut linear_f32 = stream.alloc_zeros::<f32>(n)?;
    kernels
        .ops
        .copy_f16_to_f32(&linear, &mut linear_f32, n as u32)?;
    let mut normalized = stream.alloc_zeros::<f16>(n)?;
    kernels.ops.layer_norm_f32in(
        &linear_f32,
        norm_weight,
        norm_bias,
        &mut normalized,
        seq_len as u32,
        512,
        1e-5,
    )?;
    let mut scaled = stream.alloc_zeros::<f16>(n)?;
    kernels
        .ops
        .scale_f16(&normalized, &mut scaled, n as u32, 512.0f32.sqrt())?;
    let mut output = stream.alloc_zeros::<f32>(n)?;
    kernels
        .ops
        .copy_f16_to_f32(&scaled, &mut output, n as u32)?;
    Ok(output)
}

impl ChatterboxS3Flow {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let (weight_shape, speaker_weight) =
            weights.tensor_f32("flow.spk_embed_affine_layer.weight")?;
        let (bias_shape, speaker_bias) = weights.tensor_f32("flow.spk_embed_affine_layer.bias")?;
        ensure!(
            weight_shape == [80, 192],
            "invalid S3Gen speaker projection"
        );
        ensure!(bias_shape == [80], "invalid S3Gen speaker bias");
        Ok(Self {
            encoder: ChatterboxS3Encoder::load(model_dir, stream)?,
            estimator: ChatterboxFlowEstimator::load(model_dir, stream)?,
            speaker_weight,
            speaker_bias,
        })
    }

    /// Generate the new (non-reference) mel frames with the official cosine
    /// Euler schedule and CFG rate 0.7.
    pub fn generate_mel(
        &self,
        generated_tokens: &[i32],
        conditioning: &ChatterboxConditioning,
        steps: usize,
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(steps > 0, "CFM step count must be non-zero");
        ensure!(!generated_tokens.is_empty(), "no generated speech tokens");
        ensure!(
            generated_tokens
                .iter()
                .all(|token| *token >= 0 && (*token as usize) < S3_VOCAB_SIZE),
            "generated speech token is outside S3Gen vocabulary"
        );
        let mut tokens = conditioning.s3_prompt_tokens.clone();
        tokens.extend_from_slice(generated_tokens);
        let mean = self.encoder.encode(&tokens, kernels)?;
        let frames = tokens.len() * 2;
        let prompt_frames = conditioning.s3_prompt_feature_frames;
        ensure!(
            prompt_frames <= frames
                && conditioning.s3_prompt_features.len() == prompt_frames * S3_MEL_BINS,
            "invalid S3Gen prompt feature geometry"
        );
        let speaker = project_normalized_speaker(
            &conditioning.s3_embedding,
            &self.speaker_weight,
            &self.speaker_bias,
        )?;
        let mut condition = vec![0.0f32; frames * S3_MEL_BINS];
        condition[..conditioning.s3_prompt_features.len()]
            .copy_from_slice(&conditioning.s3_prompt_features);
        let mut state = gaussian_noise(frames * S3_MEL_BINS, seed);
        for step in 0..steps {
            let t0 = 1.0 - ((step as f32 / steps as f32) * 0.5 * std::f32::consts::PI).cos();
            let t1 = 1.0 - (((step + 1) as f32 / steps as f32) * 0.5 * std::f32::consts::PI).cos();
            let mut conditional = Vec::with_capacity(frames * 320);
            let mut unconditional = Vec::with_capacity(frames * 320);
            for frame in 0..frames {
                let range = frame * S3_MEL_BINS..(frame + 1) * S3_MEL_BINS;
                conditional.extend_from_slice(&state[range.clone()]);
                conditional.extend_from_slice(&mean[range.clone()]);
                conditional.extend_from_slice(&speaker);
                conditional.extend_from_slice(&condition[range.clone()]);
                unconditional.extend_from_slice(&state[range]);
                unconditional.extend(std::iter::repeat_n(0.0, 240));
            }
            let (velocity_cond, velocity_uncond) =
                self.estimator
                    .forward_cfg(&conditional, &unconditional, frames, t0, kernels)?;
            let dt = t1 - t0;
            for ((value, cond), uncond) in state.iter_mut().zip(velocity_cond).zip(velocity_uncond)
            {
                *value += dt * (1.7 * cond - 0.7 * uncond);
            }
        }
        Ok(state[prompt_frames * S3_MEL_BINS..].to_vec())
    }
}

fn project_normalized_speaker(
    embedding: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> anyhow::Result<Vec<f32>> {
    ensure!(
        embedding.len() == 192,
        "S3Gen speaker embedding must have 192 values"
    );
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    ensure!(
        norm > 0.0 && norm.is_finite(),
        "invalid S3Gen speaker embedding norm"
    );
    Ok((0..80)
        .map(|row| {
            bias[row]
                + embedding
                    .iter()
                    .enumerate()
                    .map(|(column, value)| weight[row * 192 + column] * value / norm)
                    .sum::<f32>()
        })
        .collect())
}

fn gaussian_noise(count: usize, mut state: u64) -> Vec<f32> {
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)) as f32;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        output.push(radius * angle.cos());
        if output.len() < count {
            output.push(radius * angle.sin());
        }
    }
    output
}

fn espnet_relative_positions(seq_len: usize) -> Vec<f16> {
    let mut result = Vec::with_capacity((2 * seq_len - 1) * S3_HIDDEN_SIZE);
    for position in (0..seq_len)
        .rev()
        .map(|value| value as f32)
        .chain((1..seq_len).map(|value| -(value as f32)))
    {
        for channel in 0..S3_HIDDEN_SIZE / 2 {
            let frequency =
                (-((10_000.0f32).ln()) * (2 * channel) as f32 / S3_HIDDEN_SIZE as f32).exp();
            result.push(f16::from_f32((position * frequency).sin()));
            result.push(f16::from_f32((position * frequency).cos()));
        }
    }
    result
}
