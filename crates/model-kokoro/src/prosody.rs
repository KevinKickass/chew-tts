use crate::{KokoroAlbert, KokoroBiLstm, KokoroCheckpoint, KokoroConfig, KokoroVoice};
use anyhow::ensure;
use chew_kernel::GpuKernels;
use cudarc::driver::CudaStream;
use std::path::Path;
use std::sync::Arc;

struct AdaLayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// Duration/prosody state before the F0/noise and decoder branches.
pub struct KokoroProsody {
    /// Duration-encoder output aligned to acoustic frames, frame-major `[A,640]`.
    pub aligned: Vec<f32>,
    pub acoustic_frames: usize,
    pub durations: Vec<usize>,
    pub decoder_style: Vec<f32>,
    pub predictor_style: Vec<f32>,
}

pub struct KokoroProsodyFrontend {
    albert: KokoroAlbert,
    duration_lstms: Vec<KokoroBiLstm>,
    duration_norms: Vec<AdaLayerNorm>,
    duration_lstm: KokoroBiLstm,
    duration_weight: Vec<f32>,
    duration_bias: Vec<f32>,
}

impl KokoroProsodyFrontend {
    pub fn load(model_dir: &Path, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
        let mut duration_lstms = Vec::new();
        let mut duration_norms = Vec::new();
        for (slot, index) in [0, 2, 4].into_iter().enumerate() {
            duration_lstms.push(KokoroBiLstm::load(
                &checkpoint,
                "predictor",
                &format!("module.text_encoder.lstms.{index}"),
                640,
                256,
                stream,
            )?);
            let prefix = format!("module.text_encoder.lstms.{}.fc", index + 1);
            let (weight_shape, weight) =
                checkpoint.tensor_f32("predictor", &format!("{prefix}.weight"))?;
            let (bias_shape, bias) =
                checkpoint.tensor_f32("predictor", &format!("{prefix}.bias"))?;
            ensure!(
                weight_shape == [1024, 128] && bias_shape == [1024],
                "invalid Kokoro duration AdaLayerNorm {slot}"
            );
            duration_norms.push(AdaLayerNorm { weight, bias });
        }
        let (duration_weight_shape, duration_weight) =
            checkpoint.tensor_f32("predictor", "module.duration_proj.linear_layer.weight")?;
        let (duration_bias_shape, duration_bias) =
            checkpoint.tensor_f32("predictor", "module.duration_proj.linear_layer.bias")?;
        ensure!(
            duration_weight_shape == [50, 512] && duration_bias_shape == [50],
            "invalid Kokoro duration projection"
        );
        Ok(Self {
            albert: KokoroAlbert::load(model_dir, stream)?,
            duration_lstms,
            duration_norms,
            duration_lstm: KokoroBiLstm::load(
                &checkpoint,
                "predictor",
                "module.lstm",
                640,
                256,
                stream,
            )?,
            duration_weight,
            duration_bias,
        })
    }

    pub fn predict(
        &self,
        ids: &[usize],
        voice: &KokoroVoice,
        speed: f32,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<KokoroProsody> {
        ensure!(
            speed.is_finite() && speed > 0.0,
            "Kokoro speed must be positive"
        );
        ensure!(
            ids.len() >= 3,
            "Kokoro input requires boundary and phoneme tokens"
        );
        let config_phonemes = ids.len() - 2;
        let style = voice.style_for_phoneme_count(config_phonemes)?;
        ensure!(
            style.len() == 256,
            "Kokoro voice style must have 256 values"
        );
        let decoder_style = style[..128].to_vec();
        let predictor_style = style[128..].to_vec();
        let frames = ids.len();
        let mut hidden = self.albert.encode(ids, kernels)?;
        for (lstm, norm) in self.duration_lstms.iter().zip(&self.duration_norms) {
            let mut styled = Vec::with_capacity(frames * 640);
            for frame in 0..frames {
                styled.extend_from_slice(&hidden[frame * 512..(frame + 1) * 512]);
                styled.extend_from_slice(&predictor_style);
            }
            hidden = lstm.forward(&styled, frames, kernels)?;
            norm.forward_inplace(&mut hidden, frames, &predictor_style);
        }
        let mut duration_input = Vec::with_capacity(frames * 640);
        for frame in 0..frames {
            duration_input.extend_from_slice(&hidden[frame * 512..(frame + 1) * 512]);
            duration_input.extend_from_slice(&predictor_style);
        }
        let duration_hidden = self
            .duration_lstm
            .forward(&duration_input, frames, kernels)?;
        let durations = (0..frames)
            .map(|frame| {
                let row = &duration_hidden[frame * 512..(frame + 1) * 512];
                let total = (0..50)
                    .map(|output| {
                        let value = self.duration_bias[output]
                            + row
                                .iter()
                                .enumerate()
                                .map(|(input, value)| {
                                    self.duration_weight[output * 512 + input] * value
                                })
                                .sum::<f32>();
                        1.0 / (1.0 + (-value).exp())
                    })
                    .sum::<f32>()
                    / speed;
                total.round().max(1.0) as usize
            })
            .collect::<Vec<_>>();
        let acoustic_frames = durations.iter().sum();
        let mut aligned = Vec::with_capacity(acoustic_frames * 640);
        for (frame, duration) in durations.iter().copied().enumerate() {
            let row = &duration_input[frame * 640..(frame + 1) * 640];
            for _ in 0..duration {
                aligned.extend_from_slice(row);
            }
        }
        Ok(KokoroProsody {
            aligned,
            acoustic_frames,
            durations,
            decoder_style,
            predictor_style,
        })
    }
}

impl AdaLayerNorm {
    fn forward_inplace(&self, hidden: &mut [f32], frames: usize, style: &[f32]) {
        let affine = (0..1024)
            .map(|output| {
                self.bias[output]
                    + style
                        .iter()
                        .enumerate()
                        .map(|(input, value)| self.weight[output * 128 + input] * value)
                        .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for frame in 0..frames {
            let row = &mut hidden[frame * 512..(frame + 1) * 512];
            let mean = row.iter().sum::<f32>() / 512.0;
            let variance = row
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f32>()
                / 512.0;
            let inverse = 1.0 / (variance + 1e-5).sqrt();
            for channel in 0..512 {
                row[channel] = (1.0 + affine[channel]) * (row[channel] - mean) * inverse
                    + affine[512 + channel];
            }
        }
    }
}

pub fn load_default_voice(model_dir: &Path, config: &KokoroConfig) -> anyhow::Result<KokoroVoice> {
    KokoroVoice::load(&model_dir.join("voices/af_heart.pt"), config)
}
