use crate::audio_input::{decode_wav, resample};
use crate::voice_design::{SynthesisOutput, SynthesisRequest};
use anyhow::{Context, ensure};
use chew_model_voxcpm2::{VoxCpm2Engine as NativeVoxCpm2Engine, inspect_model};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct VoxCpm2Engine {
    engine: NativeVoxCpm2Engine,
    kernels: chew_kernel::GpuKernels,
    max_patches: usize,
    pub load_elapsed: Duration,
    pub vram_bytes: u64,
}

impl VoxCpm2Engine {
    pub fn load(model_dir: &Path, gpu: usize, max_patches: usize) -> anyhow::Result<Self> {
        ensure!(max_patches > 2, "maximum VoxCPM2 patch count must exceed 2");
        let inspection = inspect_model(model_dir)?;
        let allocator = chew_vram::VramAllocator::init()?;
        ensure!(
            gpu < allocator.gpu_count(),
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
        let free_before = allocator.free_bytes(gpu)?;
        let stream = Arc::clone(allocator.stream(gpu));
        let lm = &inspection.config.lm_config;
        let mut kernels = chew_kernel::GpuKernels::load(
            &stream,
            lm.intermediate_size * lm.hidden_size,
            lm.intermediate_size,
        )?;
        let started = Instant::now();
        let engine = NativeVoxCpm2Engine::load(model_dir, &inspection.config, &stream)?;
        let _warmup = engine.generate_zero_shot(".", 2, 4, 0x564f_5843_504d_3201, &mut kernels)?;
        kernels.ops.stream().synchronize()?;
        let load_elapsed = started.elapsed();
        let free_loaded = allocator.free_bytes(gpu)?;
        Ok(Self {
            engine,
            kernels,
            max_patches,
            load_elapsed,
            vram_bytes: free_before.saturating_sub(free_loaded),
        })
    }

    pub fn model_id(&self) -> &'static str {
        "tts-studio"
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest) -> anyhow::Result<SynthesisOutput> {
        ensure!(!request.text.trim().is_empty(), "text must not be empty");
        let max_patches = request
            .max_frames
            .min(self.max_patches)
            .min(estimated_frame_capacity(request));
        ensure!(max_patches > 2, "max_frames must exceed 2 for VoxCPM2");

        let generation = if let Some(reference) = request.reference_audio_wav.as_deref() {
            let prompt_started = Instant::now();
            let (samples, sample_rate) =
                decode_wav(reference).context("could not decode VoxCPM2 reference audio")?;
            let reference = resample(&samples, sample_rate, 16_000);
            let prompt_elapsed = prompt_started.elapsed();
            let generation = self.engine.generate_with_reference(
                &request.text,
                &reference,
                2,
                max_patches,
                request.seed,
                &mut self.kernels,
            )?;
            (generation, prompt_elapsed)
        } else {
            (
                self.engine.generate_zero_shot(
                    &request.text,
                    2,
                    max_patches,
                    request.seed,
                    &mut self.kernels,
                )?,
                Duration::ZERO,
            )
        };
        let (generation, prompt_elapsed) = generation;
        ensure!(
            generation.audio.iter().all(|sample| sample.is_finite()),
            "VoxCPM2 generated non-finite audio"
        );
        Ok(SynthesisOutput {
            samples: generation.audio,
            sample_rate: generation.sample_rate,
            generated_frames: generation.patches,
            prompt_elapsed: prompt_elapsed + generation.prompt_elapsed,
            generation_elapsed: generation.generation_elapsed,
            codec_elapsed: generation.codec_elapsed,
        })
    }
}

fn estimated_frame_capacity(request: &SynthesisRequest) -> usize {
    let characters = request.text.chars().count();
    if matches!(request.language.as_str(), "chinese" | "japanese" | "korean") {
        characters.saturating_mul(2).saturating_add(24)
    } else {
        characters.div_ceil(2).saturating_add(32)
    }
}
