use crate::cuda::{TalkerDecoderLayer, TalkerLayerKvCache, TalkerLayerScratch};
use crate::{CodePredictorConfig, TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The five-layer multi-codebook decoder used after every talker step.
pub struct CodePredictorTransformer {
    layers: Vec<TalkerDecoderLayer>,
    final_norm: CudaSlice<f16>,
    geometry: TalkerConfig,
}

impl CodePredictorTransformer {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &CodePredictorConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let geometry = predictor_geometry(config);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("talker.code_predictor.model.layers.{layer_index}");
            layers.push(
                TalkerDecoderLayer::load_from_prefix(model_dir, &prefix, &geometry, stream)
                    .with_context(|| {
                        format!("could not load code predictor layer {layer_index}")
                    })?,
            );
        }
        let norm = load_f16_tensor(model_dir, "talker.code_predictor.model.norm.weight")
            .context("could not load code predictor final norm")?;
        ensure!(
            norm.shape == [config.hidden_size],
            "code predictor final norm has shape {:?}, expected [{}]",
            norm.shape,
            config.hidden_size
        );
        let final_norm = stream
            .clone_htod(&norm.values)
            .context("could not upload code predictor final norm")?;
        Ok(Self {
            layers,
            final_norm,
            geometry,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Execute one prepared code-predictor sequence.
    pub fn forward_hidden(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        max_seq_len: usize,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            hidden_host.len() == seq_len * self.geometry.hidden_size,
            "hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * self.geometry.hidden_size
        );
        ensure!(max_seq_len >= seq_len, "KV cache is smaller than the input");

        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = TalkerLayerScratch::allocate(seq_len, &self.geometry, &stream)?;
        let mut caches = (0..self.layers.len())
            .map(|_| TalkerLayerKvCache::allocate(max_seq_len, &self.geometry, &stream))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for (layer, cache) in self.layers.iter().zip(&mut caches) {
            layer.forward_cached_device(
                &mut hidden,
                seq_len,
                &self.geometry,
                kernels,
                cache,
                &mut scratch,
            )?;
        }
        kernels.ops.rms_norm_f32in(
            &hidden,
            &self.final_norm,
            &mut scratch.norm,
            seq_len as u32,
            self.geometry.hidden_size as u32,
            self.geometry.rms_norm_eps as f32,
        )?;
        stream.synchronize()?;

        let output_len = seq_len * self.geometry.hidden_size;
        let mut output = vec![f16::ZERO; output_len];
        stream.memcpy_dtoh(&scratch.norm.slice(..output_len), &mut output)?;
        Ok(output.into_iter().map(f16::to_f32).collect())
    }
}

fn predictor_geometry(config: &CodePredictorConfig) -> TalkerConfig {
    TalkerConfig {
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        vocab_size: config.vocab_size,
        text_vocab_size: 0,
        text_hidden_size: config.hidden_size,
        num_code_groups: config.num_code_groups,
        max_position_embeddings: config.max_position_embeddings,
        rope_theta: config.rope_theta,
        rms_norm_eps: config.rms_norm_eps,
        code_predictor_config: config.clone(),
        codec_language_id: HashMap::new(),
    }
}
