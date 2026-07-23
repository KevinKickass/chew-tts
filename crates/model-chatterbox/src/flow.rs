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
        let mut norm = stream.alloc_zeros::<f16>(n)?;
        kernels.ops.layer_norm_f32in(
            &hidden,
            &self.norm1_weight,
            &self.norm1_bias,
            &mut norm,
            seq_len as u32,
            DIM as u32,
            1e-5,
        )?;
        let mut q = stream.alloc_zeros::<f16>(seq_len * ATTENTION_DIM)?;
        let mut k = stream.alloc_zeros::<f16>(seq_len * ATTENTION_DIM)?;
        let mut v = stream.alloc_zeros::<f16>(seq_len * ATTENTION_DIM)?;
        for (weight, output) in [
            (&self.q_weight, &mut q),
            (&self.k_weight, &mut k),
            (&self.v_weight, &mut v),
        ] {
            kernels.gemm.matmul_f16(
                &norm,
                weight,
                output,
                seq_len as u32,
                ATTENTION_DIM as u32,
                DIM as u32,
            )?;
        }
        let mut attention = stream.alloc_zeros::<f16>(seq_len * ATTENTION_DIM)?;
        kernels.ops.mha_naive_full(
            &q,
            &k.slice(..),
            &v.slice(..),
            &mut attention,
            HEAD_DIM as u32,
            HEADS as u32,
            HEADS as u32,
            seq_len as u32,
            seq_len as u32,
            1.0 / (HEAD_DIM as f32).sqrt(),
            0.0,
        )?;
        let mut delta = stream.alloc_zeros::<f16>(n)?;
        kernels.gemm.matmul_f16(
            &attention,
            &self.out_weight,
            &mut delta,
            seq_len as u32,
            DIM as u32,
            ATTENTION_DIM as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut delta, &self.out_bias, seq_len as u32, DIM as u32)?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &delta, n as u32)?;

        kernels.ops.layer_norm_f32in(
            &hidden,
            &self.norm3_weight,
            &self.norm3_bias,
            &mut norm,
            seq_len as u32,
            DIM as u32,
            1e-5,
        )?;
        let mut ff = stream.alloc_zeros::<f16>(seq_len * FF_DIM)?;
        kernels.gemm.matmul_f16(
            &norm,
            &self.ff1_weight,
            &mut ff,
            seq_len as u32,
            FF_DIM as u32,
            DIM as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut ff, &self.ff1_bias, seq_len as u32, FF_DIM as u32)?;
        let mut activated = stream.alloc_zeros::<f16>(seq_len * FF_DIM)?;
        kernels
            .ops
            .gelu_erf_f16(&ff, &mut activated, (seq_len * FF_DIM) as u32)?;
        kernels.gemm.matmul_f16(
            &activated,
            &self.ff2_weight,
            &mut delta,
            seq_len as u32,
            DIM as u32,
            FF_DIM as u32,
        )?;
        kernels
            .ops
            .add_bias_f16_inplace(&mut delta, &self.ff2_bias, seq_len as u32, DIM as u32)?;
        kernels
            .ops
            .add_inplace_f32_f16(&mut hidden, &delta, n as u32)?;
        stream.synchronize()?;
        let mut output = vec![0.0; n];
        stream.memcpy_dtoh(&hidden, &mut output)?;
        Ok(output)
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
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; FF_DIM];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
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
        let input_f16 = input.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        let input_f16 = stream.clone_htod(&input_f16)?;
        let mut input_cf = stream.alloc_zeros::<f16>(input.len())?;
        kernels.ops.transpose_f16(
            &input_f16,
            &mut input_cf,
            seq_len as u32,
            self.input_channels as u32,
        )?;
        let mut first_cf = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels.ops.conv1d_causal_f16(
            &input_cf,
            &self.conv1_weight,
            &self.conv1_bias,
            &mut first_cf,
            self.input_channels as u32,
            DIM as u32,
            seq_len as u32,
            3,
            1,
            1,
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

        let time = time_embedding
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let mut time = stream.clone_htod(&time)?;
        let mut time_mish = stream.alloc_zeros::<f16>(FF_DIM)?;
        kernels.ops.mish_f16(&time, &mut time_mish, FF_DIM as u32)?;
        std::mem::swap(&mut time, &mut time_mish);
        let mut time_out = stream.alloc_zeros::<f16>(DIM)?;
        kernels.gemv.gemv_f16(
            &time,
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
        let mut second_cf = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels.ops.conv1d_causal_f16(
            &first_cf,
            &self.conv2_weight,
            &self.conv2_bias,
            &mut second_cf,
            DIM as u32,
            DIM as u32,
            seq_len as u32,
            3,
            1,
            1,
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

        let mut residual_cf = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels.ops.conv1d_causal_f16(
            &input_cf,
            &self.residual_weight,
            &self.residual_bias,
            &mut residual_cf,
            self.input_channels as u32,
            DIM as u32,
            seq_len as u32,
            1,
            1,
            1,
        )?;
        let mut residual = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .transpose_f16(&residual_cf, &mut residual, DIM as u32, seq_len as u32)?;
        let mut output = stream.alloc_zeros::<f16>(seq_len * DIM)?;
        kernels
            .ops
            .add_f16(&second, &residual, &mut output, (seq_len * DIM) as u32)?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; seq_len * DIM];
        stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }
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
