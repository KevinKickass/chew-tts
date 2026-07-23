use crate::{TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

/// Native text/codec embeddings and projections surrounding the talker stack.
pub struct TalkerFrontend {
    text_embedding: CudaSlice<f16>,
    text_fc1_weight: CudaSlice<f16>,
    text_fc1_bias: CudaSlice<f16>,
    text_fc2_weight: CudaSlice<f16>,
    text_fc2_bias: CudaSlice<f16>,
    codec_embedding: CudaSlice<f16>,
    codec_head: CudaSlice<f16>,
    text_vocab_size: usize,
    text_hidden_size: usize,
    hidden_size: usize,
    codec_vocab_size: usize,
}

impl TalkerFrontend {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let load = |name: &str, expected: &[usize]| -> anyhow::Result<CudaSlice<f16>> {
            let tensor = load_f16_tensor(model_dir, name)
                .with_context(|| format!("could not load {name}"))?;
            ensure!(
                tensor.shape == expected,
                "{name} has shape {:?}, expected {expected:?}",
                tensor.shape
            );
            Ok(stream.clone_htod(&tensor.values)?)
        };
        let text_hidden = config.text_hidden_size;
        let hidden = config.hidden_size;
        Ok(Self {
            text_embedding: load(
                "talker.model.text_embedding.weight",
                &[config.text_vocab_size, text_hidden],
            )?,
            text_fc1_weight: load(
                "talker.text_projection.linear_fc1.weight",
                &[text_hidden, text_hidden],
            )?,
            text_fc1_bias: load("talker.text_projection.linear_fc1.bias", &[text_hidden])?,
            text_fc2_weight: load(
                "talker.text_projection.linear_fc2.weight",
                &[hidden, text_hidden],
            )?,
            text_fc2_bias: load("talker.text_projection.linear_fc2.bias", &[hidden])?,
            codec_embedding: load(
                "talker.model.codec_embedding.weight",
                &[config.vocab_size, hidden],
            )?,
            codec_head: load("talker.codec_head.weight", &[config.vocab_size, hidden])?,
            text_vocab_size: config.text_vocab_size,
            text_hidden_size: text_hidden,
            hidden_size: hidden,
            codec_vocab_size: config.vocab_size,
        })
    }

    /// Embed and project text token IDs into talker hidden states.
    pub fn project_text_tokens(
        &self,
        token_ids: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(!token_ids.is_empty(), "at least one text token is required");
        for token in token_ids {
            ensure!(
                *token >= 0 && (*token as usize) < self.text_vocab_size,
                "text token {token} is outside 0..{}",
                self.text_vocab_size
            );
        }
        let rows = token_ids.len();
        let stream = Arc::clone(kernels.ops.stream());
        let ids = stream.clone_htod(token_ids)?;
        let mut embeddings = stream.alloc_zeros::<f16>(rows * self.text_hidden_size)?;
        kernels.ops.gather_rows_f16(
            &self.text_embedding,
            &ids,
            &mut embeddings,
            rows as u32,
            self.text_hidden_size as u32,
        )?;
        let mut fc1 = stream.alloc_zeros::<f16>(rows * self.text_hidden_size)?;
        kernels.gemm.matmul_f16(
            &embeddings,
            &self.text_fc1_weight,
            &mut fc1,
            rows as u32,
            self.text_hidden_size as u32,
            self.text_hidden_size as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut fc1,
            &self.text_fc1_bias,
            rows as u32,
            self.text_hidden_size as u32,
        )?;
        let mut activated = stream.alloc_zeros::<f16>(rows * self.text_hidden_size)?;
        kernels
            .ops
            .silu_act_f16(&fc1, &mut activated, (rows * self.text_hidden_size) as u32)?;
        let mut projected = stream.alloc_zeros::<f16>(rows * self.hidden_size)?;
        kernels.gemm.matmul_f16(
            &activated,
            &self.text_fc2_weight,
            &mut projected,
            rows as u32,
            self.hidden_size as u32,
            self.text_hidden_size as u32,
        )?;
        kernels.ops.add_bias_f16_inplace(
            &mut projected,
            &self.text_fc2_bias,
            rows as u32,
            self.hidden_size as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; rows * self.hidden_size];
        stream.memcpy_dtoh(&projected, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    /// Look up talker codec embeddings for semantic or control tokens.
    pub fn codec_embeddings(
        &self,
        token_ids: &[i32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            !token_ids.is_empty(),
            "at least one codec token is required"
        );
        for token in token_ids {
            ensure!(
                *token >= 0 && (*token as usize) < self.codec_vocab_size,
                "codec token {token} is outside 0..{}",
                self.codec_vocab_size
            );
        }
        let stream = Arc::clone(kernels.ops.stream());
        let ids = stream.clone_htod(token_ids)?;
        let mut embeddings = stream.alloc_zeros::<f16>(token_ids.len() * self.hidden_size)?;
        kernels.ops.gather_rows_f16(
            &self.codec_embedding,
            &ids,
            &mut embeddings,
            token_ids.len() as u32,
            self.hidden_size as u32,
        )?;
        stream.synchronize()?;
        let mut host = vec![f16::ZERO; token_ids.len() * self.hidden_size];
        stream.memcpy_dtoh(&embeddings, &mut host)?;
        Ok(host.into_iter().map(f16::to_f32).collect())
    }

    /// Project one normalized talker hidden state and return its argmax token.
    pub fn semantic_argmax(&self, hidden: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<i32> {
        ensure!(
            hidden.len() == self.hidden_size,
            "talker hidden has {} values, expected {}",
            hidden.len(),
            self.hidden_size
        );
        let stream = Arc::clone(kernels.ops.stream());
        let hidden = hidden
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let hidden = stream.clone_htod(&hidden)?;
        let mut logits = stream.alloc_zeros::<f16>(self.codec_vocab_size)?;
        let mut token = stream.alloc_zeros::<i32>(1)?;
        kernels.gemv.gemv_f16(
            &hidden,
            &self.codec_head,
            &mut logits,
            self.codec_vocab_size as u32,
            self.hidden_size as u32,
        )?;
        kernels
            .ops
            .argmax_f16(&logits, &mut token, self.codec_vocab_size as u32)?;
        stream.synchronize()?;
        let mut host = [0i32];
        stream.memcpy_dtoh(&token, &mut host)?;
        Ok(host[0])
    }
}
