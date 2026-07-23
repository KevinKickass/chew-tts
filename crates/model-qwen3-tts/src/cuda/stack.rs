use super::{TalkerDecoderLayer, TalkerLayerKvCache, TalkerLayerScratch};
use crate::{TalkerConfig, load_f16_tensor};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use std::path::Path;
use std::sync::Arc;

/// GPU-resident Qwen3-TTS talker transformer.
pub struct TalkerTransformer {
    layers: Vec<TalkerDecoderLayer>,
    final_norm: CudaSlice<f16>,
}

impl TalkerTransformer {
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: &TalkerConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(
                TalkerDecoderLayer::load(model_dir, layer_index, config, stream)
                    .with_context(|| format!("could not load talker layer {layer_index}"))?,
            );
        }
        let norm = load_f16_tensor(model_dir, "talker.model.norm.weight")
            .context("could not load talker final norm")?;
        ensure!(
            norm.shape == [config.hidden_size],
            "talker final norm has shape {:?}, expected [{}]",
            norm.shape,
            config.hidden_size
        );
        let final_norm = stream
            .clone_htod(&norm.values)
            .context("could not upload talker final norm")?;
        Ok(Self { layers, final_norm })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Correctness-first stack execution from prepared talker embeddings.
    ///
    /// Hidden state and scratch stay on the GPU for the complete stack. This
    /// method creates a fresh cache; a persistent generation session follows.
    pub fn forward_hidden(
        &self,
        hidden_host: &[f32],
        seq_len: usize,
        max_seq_len: usize,
        config: &TalkerConfig,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(seq_len > 0, "sequence length must be non-zero");
        ensure!(
            hidden_host.len() == seq_len * config.hidden_size,
            "hidden input has {} values, expected {}",
            hidden_host.len(),
            seq_len * config.hidden_size
        );
        ensure!(
            max_seq_len >= seq_len,
            "maximum sequence length {max_seq_len} is below prompt length {seq_len}"
        );

        let stream = Arc::clone(kernels.ops.stream());
        let mut hidden = stream.clone_htod(hidden_host)?;
        let mut scratch = TalkerLayerScratch::allocate(seq_len, config, &stream)?;
        let mut caches = (0..self.layers.len())
            .map(|_| TalkerLayerKvCache::allocate(max_seq_len, config, &stream))
            .collect::<anyhow::Result<Vec<_>>>()?;

        for (layer, cache) in self.layers.iter().zip(&mut caches) {
            layer.forward_cached_device(
                &mut hidden,
                seq_len,
                config,
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
            config.hidden_size as u32,
            config.rms_norm_eps as f32,
        )?;
        stream.synchronize()?;

        let mut output_f16 = vec![f16::ZERO; seq_len * config.hidden_size];
        stream.memcpy_dtoh(
            &scratch.norm.slice(..seq_len * config.hidden_size),
            &mut output_f16,
        )?;
        Ok(output_f16.into_iter().map(f16::to_f32).collect())
    }
}
