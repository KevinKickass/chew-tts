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
