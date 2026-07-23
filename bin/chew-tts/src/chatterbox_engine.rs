use crate::voice_design::{SynthesisOutput, SynthesisRequest};
use anyhow::ensure;
use chew_model_chatterbox::{
    ChatterboxConditioning, ChatterboxHiFT, ChatterboxS3Flow, ChatterboxT3Frontend,
    ChatterboxT3Transformer, ChatterboxTokenizer, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    STOP_SPEECH_TOKEN,
};
use cudarc::driver::CudaStream;
use half::f16;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const CFG_WEIGHT: f32 = 0.5;
const FLOW_STEPS: usize = 10;

pub struct ChatterboxEngine {
    stream: Arc<CudaStream>,
    kernels: chew_kernel::GpuKernels,
    tokenizer: ChatterboxTokenizer,
    conditioning: ChatterboxConditioning,
    frontend: ChatterboxT3Frontend,
    transformer: ChatterboxT3Transformer,
    flow: ChatterboxS3Flow,
    hift: ChatterboxHiFT,
    max_tokens: usize,
    pub load_elapsed: Duration,
    pub vram_bytes: u64,
}

impl ChatterboxEngine {
    pub fn load(model_dir: &Path, gpu: usize, max_tokens: usize) -> anyhow::Result<Self> {
        ensure!(max_tokens > 0, "maximum token count must be non-zero");
        let allocator = chew_vram::VramAllocator::init()?;
        ensure!(
            gpu < allocator.gpu_count(),
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
        let free_before = allocator.free_bytes(gpu)?;
        let stream = Arc::clone(allocator.stream(gpu));
        let kernels = chew_kernel::GpuKernels::load(
            &stream,
            HIDDEN_SIZE * INTERMEDIATE_SIZE,
            INTERMEDIATE_SIZE,
        )?;
        let started = Instant::now();
        let tokenizer = ChatterboxTokenizer::load(model_dir)?;
        let conditioning = ChatterboxConditioning::load(&model_dir.join("conds.pt"))?;
        let frontend = ChatterboxT3Frontend::load(model_dir, &stream)?;
        let transformer = ChatterboxT3Transformer::load(model_dir, &stream)?;
        let flow = ChatterboxS3Flow::load(model_dir, &stream)?;
        let hift = ChatterboxHiFT::load(model_dir, &stream)?;
        stream.synchronize()?;
        let load_elapsed = started.elapsed();
        let vram_bytes = free_before.saturating_sub(allocator.free_bytes(gpu)?);
        Ok(Self {
            stream,
            kernels,
            tokenizer,
            conditioning,
            frontend,
            transformer,
            flow,
            hift,
            max_tokens,
            load_elapsed,
            vram_bytes,
        })
    }

    pub fn model_id(&self) -> &'static str {
        "tts-expressive"
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest) -> anyhow::Result<SynthesisOutput> {
        ensure!(!request.text.trim().is_empty(), "text must not be empty");
        ensure!(
            request.max_frames > 0 && request.max_frames <= self.max_tokens,
            "max_frames must be between 1 and {}",
            self.max_tokens
        );
        ensure!(
            request.reference_audio_wav.is_none(),
            "native Chatterbox reference-audio conditioning is not implemented yet"
        );
        let text_tokens = self
            .tokenizer
            .encode(&request.text, chatterbox_language(&request.language))?;
        let prompt_started = Instant::now();
        let prefix =
            self.frontend
                .build_prefix(&text_tokens, &self.conditioning, &mut self.kernels)?;
        let capacity = prefix.tokens + request.max_frames;
        let mut conditional =
            self.transformer
                .start_session(capacity, prefix.tokens, &self.stream)?;
        let mut unconditional =
            self.transformer
                .start_session(capacity, prefix.tokens, &self.stream)?;
        let mut conditional_hidden = self.transformer.forward_session(
            &mut conditional,
            &prefix.conditional,
            prefix.tokens,
            &mut self.kernels,
        )?;
        let mut unconditional_hidden = self.transformer.forward_session(
            &mut unconditional,
            &prefix.unconditional,
            prefix.tokens,
            &mut self.kernels,
        )?;
        let prompt_elapsed = prompt_started.elapsed();

        let generation_started = Instant::now();
        let mut random = request.seed.max(1);
        let mut generated = Vec::new();
        for position in 0..request.max_frames {
            let conditional_logits = self.frontend.speech_logits(
                &conditional_hidden[conditional_hidden.len() - HIDDEN_SIZE..],
                &mut self.kernels,
            )?;
            let unconditional_logits = self.frontend.speech_logits(
                &unconditional_hidden[unconditional_hidden.len() - HIDDEN_SIZE..],
                &mut self.kernels,
            )?;
            let token = sample_cfg(
                &conditional_logits,
                &unconditional_logits,
                CFG_WEIGHT,
                request.temperature,
                request.top_k,
                &mut random,
            )? as i32;
            generated.push(token);
            if token == STOP_SPEECH_TOKEN as i32 {
                break;
            }
            let embedding =
                self.frontend
                    .speech_embedding(token, position + 1, &mut self.kernels)?;
            conditional_hidden = self.transformer.forward_session(
                &mut conditional,
                &embedding,
                1,
                &mut self.kernels,
            )?;
            unconditional_hidden = self.transformer.forward_session(
                &mut unconditional,
                &embedding,
                1,
                &mut self.kernels,
            )?;
        }
        let generation_elapsed = generation_started.elapsed();
        let generated = generated
            .into_iter()
            .take_while(|token| *token != STOP_SPEECH_TOKEN as i32)
            .collect::<Vec<_>>();
        ensure!(
            !generated.is_empty(),
            "Chatterbox generated no speech tokens"
        );
        ensure!(
            generated.len() < request.max_frames,
            "Chatterbox reached the {}-token safety limit before EOS",
            request.max_frames
        );

        drop(conditional_hidden);
        drop(unconditional_hidden);
        drop(conditional);
        drop(unconditional);
        let codec_started = Instant::now();
        let mel = self.flow.generate_mel(
            &generated,
            &self.conditioning,
            FLOW_STEPS,
            request.seed.wrapping_add(1),
            &mut self.kernels,
        )?;
        let mut samples = self.hift.synthesize(
            &mel,
            mel.len() / 80,
            request.seed.wrapping_add(2),
            &mut self.kernels,
        )?;
        apply_leading_fade(&mut samples);
        self.stream.synchronize()?;
        let codec_elapsed = codec_started.elapsed();
        Ok(SynthesisOutput {
            samples,
            generated_frames: generated.len(),
            prompt_elapsed,
            generation_elapsed,
            codec_elapsed,
        })
    }
}

fn chatterbox_language(language: &str) -> &str {
    match language {
        "english" => "en",
        "german" => "de",
        "french" => "fr",
        "spanish" => "es",
        "italian" => "it",
        other => other,
    }
}

fn sample_cfg(
    conditional: &[f16],
    unconditional: &[f16],
    cfg_weight: f32,
    temperature: f32,
    top_k: usize,
    random: &mut u64,
) -> anyhow::Result<usize> {
    ensure!(
        conditional.len() == unconditional.len() && !conditional.is_empty(),
        "invalid Chatterbox logits"
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "invalid temperature"
    );
    let k = top_k.clamp(1, conditional.len());
    let mut best = Vec::<(f32, usize)>::with_capacity(k);
    for (token, (conditional, unconditional)) in conditional.iter().zip(unconditional).enumerate() {
        let conditional = conditional.to_f32();
        let value = conditional + cfg_weight * (conditional - unconditional.to_f32());
        let position = best.partition_point(|(other, _)| *other > value);
        if position < k {
            best.insert(position, (value, token));
            best.truncate(k);
        }
    }
    if k == 1 {
        return Ok(best[0].1);
    }
    let maximum = best[0].0;
    let mut total = 0.0f32;
    for (value, _) in &mut best {
        *value = ((*value - maximum) / temperature).exp();
        total += *value;
    }
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    let unit = (*random >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    let mut target = unit as f32 * total;
    let fallback = best.last().map_or(0, |(_, token)| *token);
    for (probability, token) in best {
        if target <= probability {
            return Ok(token);
        }
        target -= probability;
    }
    Ok(fallback)
}

fn apply_leading_fade(samples: &mut [f32]) {
    const SILENCE: usize = 480;
    const FADE: usize = 480;
    for sample in samples.iter_mut().take(SILENCE) {
        *sample = 0.0;
    }
    for (index, sample) in samples.iter_mut().skip(SILENCE).take(FADE).enumerate() {
        let phase = index as f32 / FADE as f32;
        *sample *= 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();
    }
}
