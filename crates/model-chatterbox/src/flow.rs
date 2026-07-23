use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_safetensors::MappedSafetensors;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const DIM: usize = 256;
const ATTENTION_DIM: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const FF_DIM: usize = 1_024;
const TIME_INPUT_DIM: usize = 320;

pub struct ChatterboxFlowTimeEmbedding {
    linear1_weight: CudaSlice<f16>,
    linear1_bias: CudaSlice<f16>,
    linear2_weight: CudaSlice<f16>,
    linear2_bias: CudaSlice<f16>,
}

pub struct ChatterboxFlowEstimator {
    time: ChatterboxFlowTimeEmbedding,
    down_resnet: ChatterboxFlowResnetBlock,
    down_transformers: Vec<ChatterboxFlowTransformerBlock>,
    downsample_weight: CudaSlice<f16>,
    downsample_bias: CudaSlice<f16>,
    mid: Vec<(
        ChatterboxFlowResnetBlock,
        Vec<ChatterboxFlowTransformerBlock>,
    )>,
    up_resnet: ChatterboxFlowResnetBlock,
    up_transformers: Vec<ChatterboxFlowTransformerBlock>,
    upsample_weight: CudaSlice<f16>,
    upsample_bias: CudaSlice<f16>,
    final_weight: CudaSlice<f16>,
    final_bias: CudaSlice<f16>,
    final_norm_weight: CudaSlice<f16>,
    final_norm_bias: CudaSlice<f16>,
    projection_weight: CudaSlice<f16>,
    projection_bias: CudaSlice<f16>,
}

pub struct ChatterboxFlowResnetBlock {
    input_channels: usize,
    conv1_weight: CudaSlice<f16>,
    conv1_bias: CudaSlice<f16>,
    norm1_weight: CudaSlice<f16>,
    norm1_bias: CudaSlice<f16>,
    time_weight: CudaSlice<f16>,
    time_bias: CudaSlice<f16>,
    conv2_weight: CudaSlice<f16>,
    conv2_bias: CudaSlice<f16>,
    norm2_weight: CudaSlice<f16>,
    norm2_bias: CudaSlice<f16>,
    residual_weight: CudaSlice<f16>,
    residual_bias: CudaSlice<f16>,
}

/// The repeated BasicTransformerBlock in the S3Gen conditional-flow U-Net.
pub struct ChatterboxFlowTransformerBlock {
    norm1_weight: CudaSlice<f16>,
    norm1_bias: CudaSlice<f16>,
    q_weight: CudaSlice<f16>,
    k_weight: CudaSlice<f16>,
    v_weight: CudaSlice<f16>,
    out_weight: CudaSlice<f16>,
    out_bias: CudaSlice<f16>,
    norm3_weight: CudaSlice<f16>,
    norm3_bias: CudaSlice<f16>,
    ff1_weight: CudaSlice<f16>,
    ff1_bias: CudaSlice<f16>,
    ff2_weight: CudaSlice<f16>,
    ff2_bias: CudaSlice<f16>,
}

struct ChatterboxFlowTransformerScratch {
    norm: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    attention: CudaSlice<f16>,
    q_packed: CudaSlice<f16>,
    k_packed: CudaSlice<f16>,
    v_transposed: CudaSlice<f16>,
    attention_scores: CudaSlice<f16>,
    attention_packed: CudaSlice<f16>,
    batch_frames: usize,
    delta: CudaSlice<f16>,
    ff: CudaSlice<f16>,
    activated: CudaSlice<f16>,
}

impl ChatterboxFlowTransformerScratch {
    fn new(seq_len: usize, batch_frames: usize, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        ensure!(
            batch_frames > 0 && seq_len.is_multiple_of(batch_frames),
            "invalid flow attention batch geometry"
        );
        let batches = seq_len / batch_frames;
        let head_batches = batches * HEADS;
        Ok(Self {
            norm: stream.alloc_zeros(seq_len * DIM)?,
            q: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            k: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            v: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            attention: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            q_packed: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            k_packed: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            v_transposed: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            attention_scores: stream.alloc_zeros(head_batches * batch_frames * batch_frames)?,
            attention_packed: stream.alloc_zeros(seq_len * ATTENTION_DIM)?,
            batch_frames,
            delta: stream.alloc_zeros(seq_len * DIM)?,
            ff: stream.alloc_zeros(seq_len * FF_DIM)?,
            activated: stream.alloc_zeros(seq_len * FF_DIM)?,
        })
    }
}

impl ChatterboxFlowTransformerBlock {
    /// `prefix` addresses a block such as
    /// `flow.decoder.estimator.mid_blocks.0.1.0`.
    pub fn load(model_dir: &Path, prefix: &str, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let load = |suffix: &str, shape: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let (actual, values) = weights
                .tensor_f16(&name)
                .with_context(|| format!("could not load Chatterbox flow {name}"))?;
            ensure!(
                actual == shape,
                "{name}: got {actual:?}, expected {shape:?}"
            );
            Ok(stream.clone_htod(&values)?)
        };
        Ok(Self {
            norm1_weight: load("norm1.weight", &[DIM])?,
            norm1_bias: load("norm1.bias", &[DIM])?,
            q_weight: load("attn1.to_q.weight", &[ATTENTION_DIM, DIM])?,
            k_weight: load("attn1.to_k.weight", &[ATTENTION_DIM, DIM])?,
            v_weight: load("attn1.to_v.weight", &[ATTENTION_DIM, DIM])?,
            out_weight: load("attn1.to_out.0.weight", &[DIM, ATTENTION_DIM])?,
            out_bias: load("attn1.to_out.0.bias", &[DIM])?,
            norm3_weight: load("norm3.weight", &[DIM])?,
            norm3_bias: load("norm3.bias", &[DIM])?,
            ff1_weight: load("ff.net.0.proj.weight", &[FF_DIM, DIM])?,
            ff1_bias: load("ff.net.0.proj.bias", &[FF_DIM])?,
            ff2_weight: load("ff.net.2.weight", &[DIM, FF_DIM])?,
            ff2_bias: load("ff.net.2.bias", &[DIM])?,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            seq_len > 0 && input.len() == seq_len * DIM,
            "invalid flow block input"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let n = seq_len * DIM;
        let mut hidden = stream.clone_htod(input)?;
        let mut scratch = ChatterboxFlowTransformerScratch::new(seq_len, seq_len, &stream)?;
        self.forward_device(&mut hidden, seq_len, &mut scratch, kernels)?;
        stream.synchronize()?;
        let mut output = vec![0.0; n];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
    }

    fn forward_device(
        &self,
        hidden: &mut CudaSlice<f32>,
        seq_len: usize,
        scratch: &mut ChatterboxFlowTransformerScratch,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<()> {
        let n = seq_len * DIM;
        kernels.ops.layer_norm_f32in(
            hidden,
            &self.norm1_weight,
            &self.norm1_bias,
            &mut scratch.norm,
            seq_len as u32,
            DIM as u32,
            1e-5,
        )?;
        for (weight, output) in [
            (&self.q_weight, &mut scratch.q),
            (&self.k_weight, &mut scratch.k),
            (&self.v_weight, &mut scratch.v),
        ] {
            kernels.gemm.matmul_f16(
                &scratch.norm,
                weight,
                output,
                seq_len as u32,
                ATTENTION_DIM as u32,
                DIM as u32,
            )?;
        }
        let batches = seq_len / scratch.batch_frames;
        let head_batches = batches * HEADS;
        kernels.ops.attention_pack_qkv_f16(
            &scratch.q,
            &scratch.k,
            &scratch.v,
            &mut scratch.q_packed,
            &mut scratch.k_packed,
            &mut scratch.v_transposed,
            seq_len as u32,
            scratch.batch_frames as u32,
            HEADS as u32,
            HEAD_DIM as u32,
        )?;
        kernels.gemm.matmul_f16_strided_batched(
            &scratch.q_packed,
            &scratch.k_packed,
            &mut scratch.attention_scores,
            scratch.batch_frames as u32,
            scratch.batch_frames as u32,
            HEAD_DIM as u32,
            head_batches as u32,
        )?;
        kernels.ops.softmax_rows_scaled_f16_inplace(
            &mut scratch.attention_scores,
            (head_batches * scratch.batch_frames) as u32,
            scratch.batch_frames as u32,
            1.0 / (HEAD_DIM as f32).sqrt(),
        )?;
        kernels.gemm.matmul_f16_strided_batched(
            &scratch.attention_scores,
            &scratch.v_transposed,
            &mut scratch.attention_packed,
            scratch.batch_frames as u32,
            HEAD_DIM as u32,
            scratch.batch_frames as u32,
            head_batches as u32,
        )?;
        kernels.ops.attention_unpack_f16(
            &scratch.attention_packed,
            &mut scratch.attention,
            seq_len as u32,
            scratch.batch_frames as u32,
            HEADS as u32,
            HEAD_DIM as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.attention,
            &self.out_weight,
            &mut scratch.delta,
            seq_len as u32,
            DIM as u32,
            ATTENTION_DIM as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut scratch.delta,
            &self.out_bias,
            seq_len as u32,
            DIM as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.delta, n as u32)?;

        kernels.ops.layer_norm_f32in(
            hidden,
            &self.norm3_weight,
            &self.norm3_bias,
            &mut scratch.norm,
            seq_len as u32,
            DIM as u32,
            1e-5,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.norm,
            &self.ff1_weight,
            &mut scratch.ff,
            seq_len as u32,
            FF_DIM as u32,
            DIM as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut scratch.ff,
            &self.ff1_bias,
            seq_len as u32,
            FF_DIM as u32,
        )?;
        kernels.ops.gelu_erf_f16(
            &scratch.ff,
            &mut scratch.activated,
            (seq_len * FF_DIM) as u32,
        )?;
        kernels.gemm.matmul_f16(
            &scratch.activated,
            &self.ff2_weight,
            &mut scratch.delta,
            seq_len as u32,
            DIM as u32,
            FF_DIM as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut scratch.delta,
            &self.ff2_bias,
            seq_len as u32,
            DIM as u32,
        )?;
        kernels
            .ops
            .add_inplace_f32_f16(hidden, &scratch.delta, n as u32)?;
        Ok(())
    }
}

impl ChatterboxFlowTimeEmbedding {
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
            linear1_weight: load(
                "flow.decoder.estimator.time_mlp.linear_1.weight",
                &[FF_DIM, TIME_INPUT_DIM],
            )?,
            linear1_bias: load("flow.decoder.estimator.time_mlp.linear_1.bias", &[FF_DIM])?,
            linear2_weight: load(
                "flow.decoder.estimator.time_mlp.linear_2.weight",
                &[FF_DIM, FF_DIM],
            )?,
            linear2_bias: load("flow.decoder.estimator.time_mlp.linear_2.bias", &[FF_DIM])?,
        })
    }

    pub fn forward(&self, timestep: f32, kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let output = self.forward_device(timestep, kernels)?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; FF_DIM];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    fn forward_device(
        &self,
        timestep: f32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let stream = Arc::clone(kernels.ops.stream());
        let half_dim = TIME_INPUT_DIM / 2;
        let exponent_scale = 10_000.0f32.ln() / (half_dim - 1) as f32;
        let angles = (0..half_dim)
            .map(|index| 1_000.0 * timestep * (-(index as f32) * exponent_scale).exp())
            .collect::<Vec<_>>();
        let input = angles
            .iter()
            .map(|value| f16::from_f32(value.sin()))
            .chain(angles.iter().map(|value| f16::from_f32(value.cos())))
            .collect::<Vec<_>>();
        let input = stream.clone_htod(&input)?;
        let mut first = stream.alloc_zeros::<f16>(FF_DIM)?;
        kernels.gemv.gemv_f16(
            &input,
            &self.linear1_weight,
            &mut first,
            FF_DIM as u32,
            TIME_INPUT_DIM as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut first, &self.linear1_bias, 1, FF_DIM as u32)?;
        let mut activated = stream.alloc_zeros::<f16>(FF_DIM)?;
        kernels
            .ops
            .silu_act_f16(&first, &mut activated, FF_DIM as u32)?;
        let mut output = stream.alloc_zeros::<f16>(FF_DIM)?;
        kernels.gemv.gemv_f16(
            &activated,
            &self.linear2_weight,
            &mut output,
            FF_DIM as u32,
            FF_DIM as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut output, &self.linear2_bias, 1, FF_DIM as u32)?;
        Ok(output)
    }
}

impl ChatterboxFlowResnetBlock {
    pub fn load(model_dir: &Path, prefix: &str, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(model_dir.join("s3gen_v3.safetensors"))?;
        let conv1_name = format!("{prefix}.block1.block.0.weight");
        let (conv1_shape, conv1_values) = weights.tensor_f16(&conv1_name)?;
        ensure!(
            conv1_shape.len() == 3 && conv1_shape[0] == DIM && conv1_shape[2] == 3,
            "{conv1_name} has invalid shape {conv1_shape:?}"
        );
        let input_channels = conv1_shape[1];
        let load = |suffix: &str, shape: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let name = format!("{prefix}.{suffix}");
            let (actual, values) = weights.tensor_f16(&name)?;
            ensure!(
                actual == shape,
                "{name}: got {actual:?}, expected {shape:?}"
            );
            Ok(stream.clone_htod(&values)?)
        };
        Ok(Self {
            input_channels,
            conv1_weight: stream.clone_htod(&conv1_values)?,
            conv1_bias: load("block1.block.0.bias", &[DIM])?,
            norm1_weight: load("block1.block.2.weight", &[DIM])?,
            norm1_bias: load("block1.block.2.bias", &[DIM])?,
            time_weight: load("mlp.1.weight", &[DIM, FF_DIM])?,
            time_bias: load("mlp.1.bias", &[DIM])?,
            conv2_weight: load("block2.block.0.weight", &[DIM, DIM, 3])?,
            conv2_bias: load("block2.block.0.bias", &[DIM])?,
            norm2_weight: load("block2.block.2.weight", &[DIM])?,
            norm2_bias: load("block2.block.2.bias", &[DIM])?,
            residual_weight: load("res_conv.weight", &[DIM, input_channels, 1])?,
            residual_bias: load("res_conv.bias", &[DIM])?,
        })
    }

    /// Frame-major input and output. `time_embedding` is the estimator's
    /// already-expanded 1024-value timestep vector.
    pub fn forward(
        &self,
        input: &[f32],
        seq_len: usize,
        time_embedding: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            input.len() == seq_len * self.input_channels && time_embedding.len() == FF_DIM,
            "invalid flow ResNet input"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let input = stream.clone_htod(input)?;
        let time = time_embedding
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let time = stream.clone_htod(&time)?;
        let output = self.forward_device(&input, seq_len, seq_len, &time, kernels)?;
        stream.synchronize()?;
        let mut host = vec![0.0; seq_len * DIM];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host)
    }

    fn forward_device(
        &self,
        input: &CudaSlice<f32>,
        seq_len: usize,
        batch_frames: usize,
        time_embedding: &CudaSlice<f16>,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<CudaSlice<f32>> {
        let stream = Arc::clone(kernels.ops.stream());
        let mut input_f16 = stream.alloc_zeros::<f16>(seq_len * self.input_channels)?;
        kernels.ops.copy_f32_to_f16(
            input,
            &mut input_f16.slice_mut(..),
            (seq_len * self.input_channels) as u32,
        )?;
        let mut input_cf = stream.alloc_zeros::<f16>(seq_len * self.input_channels)?;
        kernels.ops.transpose_f16(
            &input_f16,
            &mut input_cf,
            seq_len as u32,
            self.input_channels as u32,
        )?;
        let first_cf = causal_conv_channel_first_f16(
            &input_cf,
            &self.conv1_weight,
            &self.conv1_bias,
            self.input_channels,
            DIM,
            seq_len,
            batch_frames,
            3,
            1,
            kernels,
        )?;
        let mut first = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .transpose_f16(&first_cf, &mut first, DIM as u32, seq_len as u32)?;
        first = norm_mish(
            &first,
            seq_len,
            &self.norm1_weight,
            &self.norm1_bias,
            kernels,
        )?;

        let mut time_mish = stream.alloc_zeros::<f16>(FF_DIM)?;
        kernels
            .ops
            .mish_f16(time_embedding, &mut time_mish, FF_DIM as u32)?;
        let mut time_out = stream.alloc_zeros::<f16>(DIM)?;
        kernels.gemv.gemv_f16(
            &time_mish,
            &self.time_weight,
            &mut time_out,
            DIM as u32,
            FF_DIM as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut time_out, &self.time_bias, 1, DIM as u32)?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut first, &time_out, seq_len as u32, DIM as u32)?;

        let mut first_cf = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .transpose_f16(&first, &mut first_cf, seq_len as u32, DIM as u32)?;
        let second_cf = causal_conv_channel_first_f16(
            &first_cf,
            &self.conv2_weight,
            &self.conv2_bias,
            DIM,
            DIM,
            seq_len,
            batch_frames,
            3,
            1,
            kernels,
        )?;
        let mut second = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .transpose_f16(&second_cf, &mut second, DIM as u32, seq_len as u32)?;
        second = norm_mish(
            &second,
            seq_len,
            &self.norm2_weight,
            &self.norm2_bias,
            kernels,
        )?;

        let residual_cf = causal_conv_channel_first_f16(
            &input_cf,
            &self.residual_weight,
            &self.residual_bias,
            self.input_channels,
            DIM,
            seq_len,
            batch_frames,
            1,
            1,
            kernels,
        )?;
        let mut residual = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .transpose_f16(&residual_cf, &mut residual, DIM as u32, seq_len as u32)?;
        let mut output = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .add_f16(&second, &residual, &mut output, (seq_len * DIM) as u32)?;
        let mut output_f32 = stream.alloc_zeros::<f32>(seq_len * DIM)?;
        kernels
            .ops
            .copy_f16_to_f32(&output, &mut output_f32, (seq_len * DIM) as u32)?;
        Ok(output_f32)
    }
}

impl ChatterboxFlowEstimator {
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
        let transformer_group = |group: &str, stream: &Arc<CudaStream>| -> anyhow::Result<Vec<_>> {
            (0..4)
                .map(|index| {
                    ChatterboxFlowTransformerBlock::load(
                        model_dir,
                        &format!("{group}.{index}"),
                        stream,
                    )
                })
                .collect()
        };
        let mut mid = Vec::with_capacity(12);
        for index in 0..12 {
            let base = format!("flow.decoder.estimator.mid_blocks.{index}");
            mid.push((
                ChatterboxFlowResnetBlock::load(model_dir, &format!("{base}.0"), stream)?,
                transformer_group(&format!("{base}.1"), stream)?,
            ));
        }
        Ok(Self {
            time: ChatterboxFlowTimeEmbedding::load(model_dir, stream)?,
            down_resnet: ChatterboxFlowResnetBlock::load(
                model_dir,
                "flow.decoder.estimator.down_blocks.0.0",
                stream,
            )?,
            down_transformers: transformer_group("flow.decoder.estimator.down_blocks.0.1", stream)?,
            downsample_weight: load(
                "flow.decoder.estimator.down_blocks.0.2.weight",
                &[DIM, DIM, 3],
            )?,
            downsample_bias: load("flow.decoder.estimator.down_blocks.0.2.bias", &[DIM])?,
            mid,
            up_resnet: ChatterboxFlowResnetBlock::load(
                model_dir,
                "flow.decoder.estimator.up_blocks.0.0",
                stream,
            )?,
            up_transformers: transformer_group("flow.decoder.estimator.up_blocks.0.1", stream)?,
            upsample_weight: load(
                "flow.decoder.estimator.up_blocks.0.2.weight",
                &[DIM, DIM, 3],
            )?,
            upsample_bias: load("flow.decoder.estimator.up_blocks.0.2.bias", &[DIM])?,
            final_weight: load(
                "flow.decoder.estimator.final_block.block.0.weight",
                &[DIM, DIM, 3],
            )?,
            final_bias: load("flow.decoder.estimator.final_block.block.0.bias", &[DIM])?,
            final_norm_weight: load("flow.decoder.estimator.final_block.block.2.weight", &[DIM])?,
            final_norm_bias: load("flow.decoder.estimator.final_block.block.2.bias", &[DIM])?,
            projection_weight: load("flow.decoder.estimator.final_proj.weight", &[80, DIM, 1])?,
            projection_bias: load("flow.decoder.estimator.final_proj.bias", &[80])?,
        })
    }

    /// One conditional-flow velocity evaluation. Input is frame-major
    /// [frames, 320] = noisy mel, encoder mean, speaker, and prompt condition.
    pub fn forward(
        &self,
        input: &[f32],
        frames: usize,
        timestep: f32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        self.forward_batched(input, frames, frames, timestep, kernels)
    }

    /// Evaluate the conditional and unconditional CFG branches as one GPU
    /// batch while keeping attention and causal convolutions isolated.
    pub fn forward_cfg(
        &self,
        conditional: &[f32],
        unconditional: &[f32],
        frames: usize,
        timestep: f32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            conditional.len() == frames * 320 && unconditional.len() == frames * 320,
            "flow CFG estimator expects two [frames, 320] inputs"
        );
        let mut input = Vec::with_capacity(conditional.len() + unconditional.len());
        input.extend_from_slice(conditional);
        input.extend_from_slice(unconditional);
        let output = self.forward_batched(&input, frames * 2, frames, timestep, kernels)?;
        let split = frames * 80;
        Ok((output[..split].to_vec(), output[split..].to_vec()))
    }

    fn forward_batched(
        &self,
        input: &[f32],
        total_frames: usize,
        batch_frames: usize,
        timestep: f32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            batch_frames > 0
                && total_frames.is_multiple_of(batch_frames)
                && input.len() == total_frames * 320,
            "flow estimator expects a whole batch of [frames, 320] inputs"
        );
        let stream = Arc::clone(kernels.ops.stream());
        let input = stream.clone_htod(input)?;
        let time = self.time.forward_device(timestep, kernels)?;
        let mut transformer_scratch =
            ChatterboxFlowTransformerScratch::new(total_frames, batch_frames, &stream)?;
        let mut hidden =
            self.down_resnet
                .forward_device(&input, total_frames, batch_frames, &time, kernels)?;
        for block in &self.down_transformers {
            block.forward_device(&mut hidden, total_frames, &mut transformer_scratch, kernels)?;
        }
        let mut skip = stream.alloc_zeros::<f16>(total_frames * DIM)?;
        kernels.ops.copy_f32_to_f16(
            &hidden,
            &mut skip.slice_mut(..),
            (total_frames * DIM) as u32,
        )?;
        hidden = causal_conv_frame_major(
            &hidden,
            total_frames,
            batch_frames,
            DIM,
            DIM,
            &self.downsample_weight,
            &self.downsample_bias,
            3,
            kernels,
        )?;
        for (resnet, transformers) in &self.mid {
            hidden = resnet.forward_device(&hidden, total_frames, batch_frames, &time, kernels)?;
            for block in transformers {
                block.forward_device(
                    &mut hidden,
                    total_frames,
                    &mut transformer_scratch,
                    kernels,
                )?;
            }
        }
        let mut joined = stream.alloc_zeros::<f32>(total_frames * DIM * 2)?;
        kernels.ops.concat_f32_f16_rows(
            &hidden,
            &skip,
            &mut joined,
            total_frames as u32,
            256,
            256,
        )?;
        hidden =
            self.up_resnet
                .forward_device(&joined, total_frames, batch_frames, &time, kernels)?;
        for block in &self.up_transformers {
            block.forward_device(&mut hidden, total_frames, &mut transformer_scratch, kernels)?;
        }
        hidden = causal_conv_frame_major(
            &hidden,
            total_frames,
            batch_frames,
            DIM,
            DIM,
            &self.upsample_weight,
            &self.upsample_bias,
            3,
            kernels,
        )?;
        let final_conv = causal_conv_frame_major_f16(
            &hidden,
            total_frames,
            batch_frames,
            DIM,
            DIM,
            &self.final_weight,
            &self.final_bias,
            3,
            kernels,
        )?;
        let final_hidden = norm_mish(
            &final_conv,
            total_frames,
            &self.final_norm_weight,
            &self.final_norm_bias,
            kernels,
        )?;
        let mut output = stream.alloc_zeros::<f16>(total_frames * 80)?;
        kernels.gemm.matmul_f16(
            &final_hidden,
            &self.projection_weight,
            &mut output,
            total_frames as u32,
            80,
            DIM as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut output,
            &self.projection_bias,
            total_frames as u32,
            80,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; total_frames * 80];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }
}

#[allow(clippy::too_many_arguments)]
fn causal_conv_frame_major(
    input: &CudaSlice<f32>,
    frames: usize,
    batch_frames: usize,
    in_channels: usize,
    out_channels: usize,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    kernel: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f32>> {
    let f16 = causal_conv_frame_major_f16(
        input,
        frames,
        batch_frames,
        in_channels,
        out_channels,
        weight,
        bias,
        kernel,
        kernels,
    )?;
    let stream = Arc::clone(kernels.ops.stream());
    let mut output = stream.alloc_zeros::<f32>(frames * out_channels)?;
    kernels
        .ops
        .copy_f16_to_f32(&f16, &mut output, (frames * out_channels) as u32)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn causal_conv_frame_major_f16(
    input: &CudaSlice<f32>,
    frames: usize,
    batch_frames: usize,
    in_channels: usize,
    out_channels: usize,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    kernel: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let mut sequence = stream.alloc_zeros::<f16>(frames * in_channels)?;
    kernels.ops.copy_f32_to_f16(
        input,
        &mut sequence.slice_mut(..),
        (frames * in_channels) as u32,
    )?;
    let mut channels = stream.alloc_zeros::<f16>(frames * in_channels)?;
    kernels
        .ops
        .transpose_f16(&sequence, &mut channels, frames as u32, in_channels as u32)?;
    let convolved = causal_conv_channel_first_f16(
        &channels,
        weight,
        bias,
        in_channels,
        out_channels,
        frames,
        batch_frames,
        kernel,
        1,
        kernels,
    )?;
    let mut output = stream.alloc_zeros::<f16>(frames * out_channels)?;
    kernels
        .ops
        .transpose_f16(&convolved, &mut output, out_channels as u32, frames as u32)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn causal_conv_channel_first_f16(
    input: &CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    in_channels: usize,
    out_channels: usize,
    frames: usize,
    batch_frames: usize,
    kernel: usize,
    dilation: usize,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let width = in_channels * kernel;
    let mut unfolded = stream.alloc_zeros::<f16>(frames * width)?;
    kernels.ops.unfold_causal_batched_f16(
        input,
        &mut unfolded,
        in_channels as u32,
        frames as u32,
        batch_frames as u32,
        kernel as u32,
        dilation as u32,
    )?;
    let mut rows = stream.alloc_zeros::<f16>(frames * out_channels)?;
    kernels.gemm.matmul_f16(
        &unfolded,
        weight,
        &mut rows,
        frames as u32,
        out_channels as u32,
        width as u32,
    )?;
    kernels
        .ops
        .add_bias_f16_inplace(&mut rows, bias, frames as u32, out_channels as u32)?;
    let mut output = stream.alloc_zeros::<f16>(frames * out_channels)?;
    kernels
        .ops
        .transpose_f16(&rows, &mut output, frames as u32, out_channels as u32)?;
    Ok(output)
}

fn norm_mish(
    input: &CudaSlice<f16>,
    seq_len: usize,
    weight: &CudaSlice<f16>,
    bias: &CudaSlice<f16>,
    kernels: &mut GpuKernels,
) -> anyhow::Result<CudaSlice<f16>> {
    let stream = Arc::clone(kernels.ops.stream());
    let n = seq_len * DIM;
    let mut input_f32 = stream.alloc_zeros::<f32>(n)?;
    kernels
        .ops
        .copy_f16_to_f32(input, &mut input_f32, n as u32)?;
    let mut normalized = stream.alloc_zeros::<f16>(n)?;
    kernels.ops.layer_norm_f32in(
        &input_f32,
        weight,
        bias,
        &mut normalized,
        seq_len as u32,
        DIM as u32,
        1e-5,
    )?;
    let mut output = stream.alloc_zeros::<f16>(n)?;
    kernels.ops.mish_f16(&normalized, &mut output, n as u32)?;
    Ok(output)
}
