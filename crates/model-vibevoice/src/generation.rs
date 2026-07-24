use crate::VibeVoiceConfig;
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, QwenDType, load_bf16_tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

struct Connector {
    fc1: CudaSlice<bf16>,
    fc1_bias: CudaSlice<bf16>,
    norm: CudaSlice<bf16>,
    fc2: CudaSlice<bf16>,
    fc2_bias: CudaSlice<bf16>,
}

struct EosClassifier {
    fc1: CudaSlice<bf16>,
    fc1_bias: CudaSlice<bf16>,
    fc2: CudaSlice<bf16>,
    fc2_bias: CudaSlice<bf16>,
}

/// Small non-transformer pieces required by the realtime generation loop.
pub struct VibeVoiceGenerationWeights {
    embeddings: Vec<bf16>,
    text_type: Vec<f32>,
    speech_type: Vec<f32>,
    connector: Connector,
    eos: EosClassifier,
    hidden: usize,
    vocab: usize,
    latent: usize,
    scaling: f32,
    bias: f32,
    stream: Arc<CudaStream>,
}

impl VibeVoiceGenerationWeights {
    pub fn load(
        model_dir: &Path,
        config: &VibeVoiceConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let hidden = config.decoder_config.hidden_size;
        let latent = config.acoustic_vae_dim;
        let embeddings = load_host(
            model_dir,
            "model.language_model.embed_tokens.weight",
            &[config.decoder_config.vocab_size, hidden],
        )?;
        let input_types = load_host(model_dir, "model.tts_input_types.weight", &[2, hidden])?;
        let connector = Connector {
            fc1: load_device(
                model_dir,
                "model.acoustic_connector.fc1.weight",
                &[hidden, latent],
                stream,
            )?,
            fc1_bias: load_device(
                model_dir,
                "model.acoustic_connector.fc1.bias",
                &[hidden],
                stream,
            )?,
            norm: load_device(
                model_dir,
                "model.acoustic_connector.norm.weight",
                &[hidden],
                stream,
            )?,
            fc2: load_device(
                model_dir,
                "model.acoustic_connector.fc2.weight",
                &[hidden, hidden],
                stream,
            )?,
            fc2_bias: load_device(
                model_dir,
                "model.acoustic_connector.fc2.bias",
                &[hidden],
                stream,
            )?,
        };
        let eos = EosClassifier {
            fc1: load_device(
                model_dir,
                "tts_eos_classifier.fc1.weight",
                &[hidden, hidden],
                stream,
            )?,
            fc1_bias: load_device(model_dir, "tts_eos_classifier.fc1.bias", &[hidden], stream)?,
            fc2: load_device(
                model_dir,
                "tts_eos_classifier.fc2.weight",
                &[1, hidden],
                stream,
            )?,
            fc2_bias: load_device(model_dir, "tts_eos_classifier.fc2.bias", &[1], stream)?,
        };
        let scaling = load_scalar(model_dir, "model.speech_scaling_factor")?;
        let bias = load_scalar(model_dir, "model.speech_bias_factor")?;
        ensure!(
            scaling.is_finite() && scaling.abs() > f32::EPSILON && bias.is_finite(),
            "invalid VibeVoice speech scaling"
        );
        Ok(Self {
            embeddings,
            text_type: input_types[..hidden]
                .iter()
                .copied()
                .map(bf16::to_f32)
                .collect(),
            speech_type: input_types[hidden..]
                .iter()
                .copied()
                .map(bf16::to_f32)
                .collect(),
            connector,
            eos,
            hidden,
            vocab: config.decoder_config.vocab_size,
            latent,
            scaling,
            bias,
            stream: Arc::clone(stream),
        })
    }

    pub fn embed_text(&self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "VibeVoice text token window is empty");
        let mut output = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            ensure!(
                (id as usize) < self.vocab,
                "VibeVoice token {id} is out of range"
            );
            let start = id as usize * self.hidden;
            output.extend(
                self.embeddings[start..start + self.hidden]
                    .iter()
                    .copied()
                    .map(bf16::to_f32),
            );
        }
        Ok(output)
    }

    pub fn add_text_type(&self, values: &mut [f32]) -> anyhow::Result<()> {
        self.add_type(values, &self.text_type)
    }

    pub fn add_speech_type(&self, values: &mut [f32]) -> anyhow::Result<()> {
        self.add_type(values, &self.speech_type)
    }

    fn add_type(&self, values: &mut [f32], kind: &[f32]) -> anyhow::Result<()> {
        ensure!(
            values.len().is_multiple_of(self.hidden),
            "VibeVoice typed hidden-state size mismatch"
        );
        for row in values.chunks_exact_mut(self.hidden) {
            for (value, addition) in row.iter_mut().zip(kind) {
                *value += addition;
            }
        }
        Ok(())
    }

    pub fn connect_latent(
        &self,
        latent: &[f32],
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            latent.len() == self.latent,
            "VibeVoice latent size mismatch"
        );
        let input = self.stream.clone_htod(
            &latent
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut first = self.stream.alloc_zeros::<bf16>(self.hidden)?;
        Bf16::gemv(
            kernels,
            &input,
            &self.connector.fc1,
            &mut first,
            self.hidden as u32,
            self.latent as u32,
        )?;
        kernels.ops.add_bias_bf16_inplace(
            &mut first,
            &self.connector.fc1_bias,
            1,
            self.hidden as u32,
        )?;
        let mut first_f32 = self.stream.alloc_zeros::<f32>(self.hidden)?;
        kernels
            .ops
            .copy_bf16_to_f32(&first, &mut first_f32, self.hidden as u32)?;
        let mut norm = self.stream.alloc_zeros::<bf16>(self.hidden)?;
        kernels.ops.rms_norm_f32in_bf16(
            &first_f32,
            &self.connector.norm,
            &mut norm,
            1,
            self.hidden as u32,
            1e-6,
        )?;
        let mut output = self.stream.alloc_zeros::<bf16>(self.hidden)?;
        Bf16::gemv(
            kernels,
            &norm,
            &self.connector.fc2,
            &mut output,
            self.hidden as u32,
            self.hidden as u32,
        )?;
        kernels.ops.add_bias_bf16_inplace(
            &mut output,
            &self.connector.fc2_bias,
            1,
            self.hidden as u32,
        )?;
        self.stream.synchronize()?;
        let mut host = vec![bf16::ZERO; self.hidden];
        self.stream.memcpy_dtoh(&output, &mut host)?;
        Ok(host.into_iter().map(bf16::to_f32).collect())
    }

    pub fn eos_probability(&self, hidden: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<f32> {
        ensure!(
            hidden.len() == self.hidden,
            "VibeVoice EOS hidden size mismatch"
        );
        let input = self.stream.clone_htod(
            &hidden
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let mut first = self.stream.alloc_zeros::<bf16>(self.hidden)?;
        Bf16::gemv(
            kernels,
            &input,
            &self.eos.fc1,
            &mut first,
            self.hidden as u32,
            self.hidden as u32,
        )?;
        kernels
            .ops
            .add_bias_bf16_inplace(&mut first, &self.eos.fc1_bias, 1, self.hidden as u32)?;
        let mut activated = self.stream.alloc_zeros::<bf16>(self.hidden)?;
        kernels
            .ops
            .relu_bf16(&first, &mut activated, self.hidden as u32)?;
        let mut output = self.stream.alloc_zeros::<bf16>(1)?;
        Bf16::gemv(
            kernels,
            &activated,
            &self.eos.fc2,
            &mut output,
            1,
            self.hidden as u32,
        )?;
        kernels
            .ops
            .add_bias_bf16_inplace(&mut output, &self.eos.fc2_bias, 1, 1)?;
        self.stream.synchronize()?;
        let mut host = [bf16::ZERO];
        self.stream.memcpy_dtoh(&output, &mut host)?;
        Ok(1.0 / (1.0 + (-host[0].to_f32()).exp()))
    }

    pub fn decoder_latent(&self, generated: &[f32]) -> anyhow::Result<Vec<f32>> {
        ensure!(
            generated.len() == self.latent,
            "VibeVoice latent size mismatch"
        );
        Ok(generated
            .iter()
            .map(|value| value / self.scaling - self.bias)
            .collect())
    }
}

fn load_host(model_dir: &Path, name: &str, shape: &[usize]) -> anyhow::Result<Vec<bf16>> {
    let tensor =
        load_bf16_tensor(model_dir, name).with_context(|| format!("could not load {name}"))?;
    ensure!(tensor.shape == shape, "{name} has shape {:?}", tensor.shape);
    Ok(tensor.values)
}

fn load_device(
    model_dir: &Path,
    name: &str,
    shape: &[usize],
    stream: &Arc<CudaStream>,
) -> anyhow::Result<CudaSlice<bf16>> {
    Ok(stream.clone_htod(&load_host(model_dir, name, shape)?)?)
}

fn load_scalar(model_dir: &Path, name: &str) -> anyhow::Result<f32> {
    let tensor = load_bf16_tensor(model_dir, name)?;
    ensure!(
        tensor.shape.is_empty() && tensor.values.len() == 1,
        "{name} is not a scalar"
    );
    Ok(tensor.values[0].to_f32())
}
