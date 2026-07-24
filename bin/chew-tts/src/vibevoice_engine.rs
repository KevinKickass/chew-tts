use crate::voice_design::{SynthesisOutput, SynthesisRequest, load_qwen_tokenizer};
use anyhow::ensure;
use chew_model_vibevoice::{
    VibeVoiceAcousticDecoder, VibeVoiceBackbones, VibeVoiceConfig, VibeVoiceDecoderState,
    VibeVoiceDiffusionHead, VibeVoiceGenerationWeights, VibeVoicePrompt, VibeVoiceScheduler,
    inspect_model,
};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const SAMPLE_RATE: u32 = 24_000;
const DIFFUSION_STEPS: usize = 5;
const CFG_SCALE: f32 = 1.5;

pub struct VibeVoiceEngine {
    model_dir: PathBuf,
    config: VibeVoiceConfig,
    tokenizer: Tokenizer,
    prompts: HashMap<PathBuf, VibeVoicePrompt>,
    backbones: VibeVoiceBackbones,
    diffusion: VibeVoiceDiffusionHead,
    decoder: VibeVoiceAcousticDecoder,
    generation: VibeVoiceGenerationWeights,
    kernels: chew_kernel::GpuKernels,
    stream: Arc<CudaStream>,
    max_frames: usize,
    pub load_elapsed: Duration,
    pub vram_bytes: u64,
}

impl VibeVoiceEngine {
    pub fn load(model_dir: &Path, gpu: usize, max_frames: usize) -> anyhow::Result<Self> {
        ensure!(
            max_frames > 0,
            "maximum VibeVoice frame count must be non-zero"
        );
        let inspection = inspect_model(model_dir)?;
        let tokenizer = load_qwen_tokenizer(model_dir)?;
        let allocator = chew_vram::VramAllocator::init()?;
        ensure!(
            gpu < allocator.gpu_count(),
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
        let free_before = allocator.free_bytes(gpu)?;
        let stream = Arc::clone(allocator.stream(gpu));
        let decoder_config = &inspection.config.decoder_config;
        let kernels = chew_kernel::GpuKernels::load(
            &stream,
            decoder_config.intermediate_size * decoder_config.hidden_size,
            decoder_config.intermediate_size,
        )?;
        let started = Instant::now();
        let backbones = VibeVoiceBackbones::load(model_dir, &inspection.config, &stream)?;
        let diffusion = VibeVoiceDiffusionHead::load(model_dir, &inspection.config, &stream)?;
        let decoder = VibeVoiceAcousticDecoder::load(model_dir, &inspection.config, &stream)?;
        let generation = VibeVoiceGenerationWeights::load(model_dir, &inspection.config, &stream)?;
        let mut engine = Self {
            model_dir: model_dir.to_owned(),
            config: inspection.config,
            tokenizer,
            prompts: HashMap::new(),
            backbones,
            diffusion,
            decoder,
            generation,
            kernels,
            stream,
            max_frames,
            load_elapsed: Duration::ZERO,
            vram_bytes: 0,
        };
        let _warmup = engine.synthesize(&SynthesisRequest {
            text: ".".into(),
            voice: "alloy".into(),
            instruction: None,
            reference_audio_wav: None,
            reference_text: None,
            language: "english".into(),
            speed: 1.0,
            max_frames: 1,
            seed: 0x5649_4245_564f_4943,
            temperature: 1.0,
            top_k: 1,
            chunk_frames: 1,
            chunk_context: 0,
        })?;
        engine.stream.synchronize()?;
        engine.load_elapsed = started.elapsed();
        let free_loaded = allocator.free_bytes(gpu)?;
        engine.vram_bytes = free_before.saturating_sub(free_loaded);
        Ok(engine)
    }

    pub fn model_id(&self) -> &'static str {
        "tts-realtime"
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest) -> anyhow::Result<SynthesisOutput> {
        self.synthesize_inner(request, None)?
            .ok_or_else(|| anyhow::anyhow!("VibeVoice synthesis stopped before completion"))
    }

    pub fn synthesize_streaming(
        &mut self,
        request: &SynthesisRequest,
        mut emit: impl FnMut(SynthesisOutput) -> bool,
    ) -> anyhow::Result<()> {
        self.synthesize_inner(request, Some(&mut emit))?;
        Ok(())
    }

    fn synthesize_inner(
        &mut self,
        request: &SynthesisRequest,
        mut emit: Option<&mut dyn FnMut(SynthesisOutput) -> bool>,
    ) -> anyhow::Result<Option<SynthesisOutput>> {
        ensure!(!request.text.trim().is_empty(), "text must not be empty");
        let max_frames = request
            .max_frames
            .min(self.max_frames)
            .min(estimated_frame_capacity(request));
        ensure!(max_frames > 0, "max_frames must be positive");
        let encoded = self
            .tokenizer
            .encode(format!("{}\n", request.text.trim()), false)
            .map_err(|error| anyhow::anyhow!("could not tokenize VibeVoice text: {error}"))?;
        let text_ids = encoded.get_ids();
        ensure!(!text_ids.is_empty(), "VibeVoice text produced no tokens");

        let prompt_path = self.voice_path(&request.voice, &request.language)?;
        if !self.prompts.contains_key(&prompt_path) {
            let prompt = VibeVoicePrompt::load(&prompt_path, &self.config)?;
            self.prompts.insert(prompt_path.clone(), prompt);
        }
        let prompt_started = Instant::now();
        let prompt = self.prompts.get(&prompt_path).expect("prompt was inserted");
        let max_new_tokens = text_ids.len() + max_frames + 8;
        let mut session =
            self.backbones
                .start_prompt_session(prompt, max_new_tokens, 5, &self.stream)?;
        let prompt_elapsed = prompt_started.elapsed();

        let generation_started = Instant::now();
        let mut previous_emit = generation_started;
        let mut rng = VibeVoiceRng::new(request.seed);
        let mut decoder_state = VibeVoiceDecoderState::default();
        let mut decoder_latents = Vec::new();
        let mut samples = Vec::new();
        let mut speech_frames = 0usize;
        let mut text_offset = 0usize;
        let mut codec_elapsed = Duration::ZERO;
        let mut eos = 0.0f32;
        while speech_frames < max_frames {
            if text_offset < text_ids.len() {
                let end = (text_offset + 5).min(text_ids.len());
                let ids = &text_ids[text_offset..end];
                let embeddings = self.generation.embed_text(ids)?;
                self.backbones.push_text(
                    &mut session,
                    &embeddings,
                    ids.len(),
                    |hidden| self.generation.add_text_type(hidden),
                    &mut self.kernels,
                )?;
                text_offset = end;
            }
            for _ in 0..6 {
                if speech_frames >= max_frames {
                    break;
                }
                let mut speech = (0..self.config.acoustic_vae_dim)
                    .map(|_| rng.normal())
                    .collect::<Vec<_>>();
                let mut scheduler =
                    VibeVoiceScheduler::new(&self.config.diffusion_head_config, DIFFUSION_STEPS)?;
                for timestep in scheduler.timesteps().to_vec() {
                    let (positive, negative) = self.diffusion.forward_cfg(
                        &speech,
                        timestep as f32,
                        session.positive_condition(),
                        session.negative_condition(),
                        &mut self.kernels,
                    )?;
                    let guided = positive
                        .iter()
                        .zip(&negative)
                        .map(|(&positive, &negative)| negative + CFG_SCALE * (positive - negative))
                        .collect::<Vec<_>>();
                    speech = scheduler.step(&guided, timestep, &speech)?;
                }
                let decoder_latent = self.generation.decoder_latent(&speech)?;
                speech_frames += 1;
                if let Some(callback) = emit.as_mut() {
                    let codec_started = Instant::now();
                    let chunk = self.decoder.decode_streaming(
                        &decoder_latent,
                        &mut decoder_state,
                        &mut self.kernels,
                    )?;
                    let frame_codec_elapsed = codec_started.elapsed();
                    codec_elapsed += frame_codec_elapsed;
                    ensure!(
                        chunk.iter().all(|sample| sample.is_finite()),
                        "VibeVoice generated non-finite audio"
                    );
                    let now = Instant::now();
                    let frame_elapsed = now.duration_since(previous_emit);
                    previous_emit = now;
                    let keep_going = callback(SynthesisOutput {
                        samples: chunk,
                        sample_rate: SAMPLE_RATE,
                        generated_frames: 1,
                        prompt_elapsed: if speech_frames == 1 {
                            prompt_elapsed
                        } else {
                            Duration::ZERO
                        },
                        generation_elapsed: frame_elapsed.saturating_sub(frame_codec_elapsed),
                        codec_elapsed: frame_codec_elapsed,
                    });
                    if !keep_going {
                        return Ok(None);
                    }
                } else {
                    decoder_latents.extend(decoder_latent);
                }
                let acoustic = self.generation.connect_latent(&speech, &mut self.kernels)?;
                self.backbones.push_speech(
                    &mut session,
                    acoustic,
                    |hidden| self.generation.add_speech_type(hidden),
                    &mut self.kernels,
                )?;
                eos = self
                    .generation
                    .eos_probability(session.positive_condition(), &mut self.kernels)?;
                if text_offset == text_ids.len() && eos > 0.5 {
                    break;
                }
            }
            if text_offset == text_ids.len() && eos > 0.5 {
                break;
            }
        }
        if emit.is_none() {
            let codec_started = Instant::now();
            samples = self.decoder.decode(&decoder_latents, &mut self.kernels)?;
            codec_elapsed = codec_started.elapsed();
        }
        ensure!(
            samples.iter().all(|sample| sample.is_finite()),
            "VibeVoice generated non-finite audio"
        );
        if emit.is_some() {
            return Ok(None);
        }
        Ok(Some(SynthesisOutput {
            samples,
            sample_rate: SAMPLE_RATE,
            generated_frames: speech_frames,
            prompt_elapsed,
            generation_elapsed: generation_started.elapsed().saturating_sub(codec_elapsed),
            codec_elapsed,
        }))
    }

    fn voice_path(&self, voice: &str, language: &str) -> anyhow::Result<PathBuf> {
        let voice = voice.trim();
        let preset = match voice.to_ascii_lowercase().as_str() {
            "alloy" | "nova" | "shimmer" if language == "german" => "de-Spk1_woman",
            "echo" | "fable" | "onyx" if language == "german" => "de-Spk0_man",
            "alloy" | "nova" | "shimmer" => "en-Emma_woman",
            "echo" | "fable" | "onyx" => "en-Carter_man",
            _ => voice.strip_suffix(".safetensors").unwrap_or(voice),
        };
        ensure!(
            !preset.is_empty()
                && preset
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "invalid VibeVoice preset name"
        );
        let path = self
            .model_dir
            .join("voices")
            .join(format!("{preset}.safetensors"));
        ensure!(
            path.is_file(),
            "VibeVoice preset {preset:?} is unavailable in {}",
            self.model_dir.join("voices").display()
        );
        Ok(path)
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

struct VibeVoiceRng {
    state: u64,
    spare: Option<f32>,
}

impl VibeVoiceRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            spare: None,
        }
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) as f32 + 0.5) / (1u32 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().max(1e-7).ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}
