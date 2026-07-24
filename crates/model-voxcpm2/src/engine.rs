use crate::projections::Bf16Linear;
use crate::{
    VoxCpm2AudioDecoder, VoxCpm2AudioEncoder, VoxCpm2Config, VoxCpm2FlowDecoder,
    VoxCpm2Projections, VoxCpm2TransformerBackbones,
};
use anyhow::{Context, ensure};
use chew_kernel::GpuKernels;
use chew_model_qwen3_tts::{Bf16, QwenDType, load_bf16_tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const AUDIO_START_TOKEN: u32 = 101;

pub struct VoxCpm2Engine {
    backbones: VoxCpm2TransformerBackbones,
    projections: VoxCpm2Projections,
    flow: VoxCpm2FlowDecoder,
    decoder: VoxCpm2AudioDecoder,
    encoder: VoxCpm2AudioEncoder,
    local_input: Bf16Linear,
    special_token: Vec<f32>,
    embeddings: CudaSlice<Bf16>,
    tokenizer: Tokenizer,
    hidden: usize,
    local_hidden: usize,
    feature_dim: usize,
    patch_size: usize,
    stream: Arc<CudaStream>,
}

pub struct VoxCpm2Generation {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub patches: usize,
    pub elapsed: Duration,
}

impl VoxCpm2Engine {
    pub fn load(
        model_dir: &Path,
        config: &VoxCpm2Config,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let backbones = VoxCpm2TransformerBackbones::load(model_dir, config, stream)?;
        let projections = VoxCpm2Projections::load(model_dir, config, stream)?;
        let flow = VoxCpm2FlowDecoder::load(model_dir, config, stream)?;
        let decoder = VoxCpm2AudioDecoder::load(model_dir, config, stream)?;
        let encoder = VoxCpm2AudioEncoder::load(model_dir, config, stream)?;
        let local_hidden = config.encoder_config.hidden_dim;
        let local_input = Bf16Linear::load(
            model_dir,
            "feat_encoder.in_proj",
            config.feat_dim,
            local_hidden,
            true,
            stream,
        )?;
        let special = load_bf16_tensor(model_dir, "feat_encoder.special_token")
            .context("could not load VoxCPM2 local-encoder special token")?;
        ensure!(
            special.values.len() == local_hidden,
            "VoxCPM2 special-token geometry disagrees"
        );
        let embedding = load_bf16_tensor(model_dir, "base_lm.embed_tokens.weight")
            .context("could not load VoxCPM2 text embeddings")?;
        ensure!(
            embedding.shape == [config.lm_config.vocab_size, config.lm_config.hidden_size],
            "VoxCPM2 text-embedding geometry disagrees"
        );
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("could not load VoxCPM2 tokenizer: {error}"))?;
        Ok(Self {
            backbones,
            projections,
            flow,
            decoder,
            encoder,
            local_input,
            special_token: special.values.into_iter().map(Bf16::to_f32).collect(),
            embeddings: stream.clone_htod(&embedding.values)?,
            tokenizer,
            hidden: config.lm_config.hidden_size,
            local_hidden,
            feature_dim: config.feat_dim,
            patch_size: config.patch_size,
            stream: Arc::clone(stream),
        })
    }

    /// Native zero-shot VoxCPM2 synthesis. Reference-audio encoding is added
    /// separately; this path already exercises the production autoregressive
    /// LM, residual LM, local encoder, flow decoder, and 48-kHz AudioVAE.
    pub fn generate_zero_shot(
        &self,
        text: &str,
        min_patches: usize,
        max_patches: usize,
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<VoxCpm2Generation> {
        self.generate(text, None, min_patches, max_patches, seed, kernels)
    }

    pub fn generate_with_reference(
        &self,
        text: &str,
        reference_16khz: &[f32],
        min_patches: usize,
        max_patches: usize,
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<VoxCpm2Generation> {
        ensure!(
            reference_16khz.len() >= 2_560,
            "VoxCPM2 reference audio must be at least 160 ms"
        );
        self.generate(
            text,
            Some(reference_16khz),
            min_patches,
            max_patches,
            seed,
            kernels,
        )
    }

    fn generate(
        &self,
        text: &str,
        reference_16khz: Option<&[f32]>,
        min_patches: usize,
        max_patches: usize,
        seed: u64,
        kernels: &mut GpuKernels,
    ) -> anyhow::Result<VoxCpm2Generation> {
        ensure!(!text.trim().is_empty(), "VoxCPM2 text is empty");
        ensure!(
            max_patches > min_patches,
            "VoxCPM2 patch limit must exceed its minimum"
        );
        let started = Instant::now();
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("VoxCPM2 tokenization failed: {error}"))?;
        let mut text_ids = encoded.get_ids().to_vec();
        text_ids.push(AUDIO_START_TOKEN);
        let reference_patches = if let Some(audio) = reference_16khz {
            let patch_samples = 640 * self.patch_size;
            let mut padded = audio.to_vec();
            padded.resize(audio.len().div_ceil(patch_samples) * patch_samples, 0.0);
            let latents = self.encoder.encode(&padded, kernels)?;
            ensure!(
                latents
                    .len()
                    .is_multiple_of(self.patch_size * self.feature_dim),
                "VoxCPM2 reference latent geometry disagrees"
            );
            Some(latents)
        } else {
            None
        };
        let reference_count = reference_patches.as_ref().map_or(0, |values| {
            values.len() / (self.patch_size * self.feature_dim)
        });
        let mut ids = Vec::with_capacity(text_ids.len() + reference_count + 2);
        if reference_count > 0 {
            ids.push(103);
            ids.resize(ids.len() + reference_count, 0);
            ids.push(104);
        }
        ids.extend_from_slice(&text_ids);
        ensure!(!ids.is_empty(), "VoxCPM2 tokenizer returned no tokens");
        let text_embeddings = self.embed(&ids, kernels)?;
        let mut combined = text_embeddings.clone();
        let mut feature_embeddings = vec![0.0f32; ids.len() * self.hidden];
        let mut audio_mask = vec![false; ids.len()];
        if let Some(reference) = &reference_patches {
            for (index, patch) in reference
                .chunks_exact(self.patch_size * self.feature_dim)
                .enumerate()
            {
                let row = index + 1;
                let local = self.encode_patch(patch, kernels)?;
                combined[row * self.hidden..(row + 1) * self.hidden].copy_from_slice(&local);
                feature_embeddings[row * self.hidden..(row + 1) * self.hidden]
                    .copy_from_slice(&local);
                audio_mask[row] = true;
            }
        }
        let capacity = ids.len() + max_patches;
        let mut session =
            self.backbones
                .start_autoregressive_session(capacity, ids.len(), &self.stream)?;
        let mut base_prompt =
            self.backbones
                .base_forward(&mut session, &combined, ids.len(), kernels)?;
        for (row, is_audio) in audio_mask.iter().copied().enumerate() {
            if is_audio {
                let quantized = self.projections.quantize(
                    &base_prompt[row * self.hidden..(row + 1) * self.hidden],
                    kernels,
                )?;
                base_prompt[row * self.hidden..(row + 1) * self.hidden].copy_from_slice(&quantized);
            }
        }
        let mut residual_input = Vec::with_capacity(ids.len() * self.hidden * 2);
        for (index, row) in base_prompt.chunks_exact(self.hidden).enumerate() {
            residual_input.extend_from_slice(row);
            residual_input.extend_from_slice(
                &feature_embeddings[index * self.hidden..(index + 1) * self.hidden],
            );
        }
        let residual_input = self.projections.fuse(&residual_input, ids.len(), kernels)?;
        let residual_prompt =
            self.backbones
                .residual_forward(&mut session, &residual_input, ids.len(), kernels)?;
        let mut lm_hidden = last_row(&base_prompt, self.hidden);
        let mut residual_hidden = last_row(&residual_prompt, self.hidden);
        let mut previous_patch = vec![0.0; self.patch_size * self.feature_dim];
        let mut generated = Vec::with_capacity(max_patches * self.patch_size * self.feature_dim);
        let mut patches = 0usize;
        for patch_index in 0..max_patches {
            let mu = self
                .projections
                .flow_condition(&lm_hidden, &residual_hidden, kernels)?;
            let patch = self.flow.generate_patch(
                &self.backbones,
                &mu,
                &previous_patch,
                seed.wrapping_add((patch_index as u64 + 1) * 0x9e37_79b9),
                kernels,
            )?;
            let local = self.encode_patch(&patch, kernels)?;
            generated.extend_from_slice(&patch);
            previous_patch = patch;
            patches += 1;
            if patch_index > min_patches && self.projections.should_stop(&lm_hidden, kernels)? {
                break;
            }
            lm_hidden = self
                .backbones
                .base_forward(&mut session, &local, 1, kernels)?;
            lm_hidden = self.projections.quantize(&lm_hidden, kernels)?;
            let mut fused = Vec::with_capacity(self.hidden * 2);
            fused.extend_from_slice(&lm_hidden);
            fused.extend_from_slice(&local);
            let fused = self.projections.fuse(&fused, 1, kernels)?;
            residual_hidden = self
                .backbones
                .residual_forward(&mut session, &fused, 1, kernels)?;
        }
        let audio = self.decoder.decode(&generated, kernels)?;
        Ok(VoxCpm2Generation {
            audio,
            sample_rate: 48_000,
            patches,
            elapsed: started.elapsed(),
        })
    }

    fn embed(&self, ids: &[u32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let ids = ids
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let device_ids = self.stream.clone_htod(&ids)?;
        let mut output = self.stream.alloc_zeros::<Bf16>(ids.len() * self.hidden)?;
        Bf16::gather(
            kernels,
            &self.embeddings,
            &device_ids,
            &mut output,
            ids.len() as u32,
            self.hidden as u32,
        )?;
        self.download(&output)
    }

    fn encode_patch(&self, patch: &[f32], kernels: &mut GpuKernels) -> anyhow::Result<Vec<f32>> {
        let input = patch
            .iter()
            .copied()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let input = self.stream.clone_htod(&input)?;
        let projected = self
            .local_input
            .forward_native(&input, self.patch_size, kernels)?;
        let projected = self.download(&projected)?;
        let mut sequence = Vec::with_capacity((self.patch_size + 1) * self.local_hidden);
        sequence.extend_from_slice(&self.special_token);
        sequence.extend_from_slice(&projected);
        let encoded = self
            .backbones
            .encoder_forward(&sequence, self.patch_size + 1, kernels)?;
        self.projections
            .encode_local(&encoded[..self.local_hidden], 1, kernels)
    }

    fn download(&self, output: &CudaSlice<Bf16>) -> anyhow::Result<Vec<f32>> {
        self.stream.synchronize()?;
        let mut host = vec![Bf16::zero(); output.len()];
        self.stream.memcpy_dtoh(output, &mut host)?;
        Ok(host.into_iter().map(Bf16::to_f32).collect())
    }
}

fn last_row(values: &[f32], width: usize) -> Vec<f32> {
    values[values.len() - width..].to_vec()
}
