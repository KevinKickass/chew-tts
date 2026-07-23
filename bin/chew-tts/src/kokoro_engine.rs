use crate::voice_design::{SynthesisOutput, SynthesisRequest};
use anyhow::{Context, ensure};
use chew_model_kokoro::{
    KokoroConfig, KokoroDecoderFrontend, KokoroF0Noise, KokoroGenerator, KokoroProsodyFrontend,
    KokoroTextEncoder, KokoroVoice,
};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct KokoroEngine {
    model_dir: PathBuf,
    config: KokoroConfig,
    stream: Arc<CudaStream>,
    kernels: chew_kernel::GpuKernels,
    prosody: KokoroProsodyFrontend,
    text: KokoroTextEncoder,
    f0_noise: KokoroF0Noise,
    decoder: KokoroDecoderFrontend,
    generator: KokoroGenerator,
    voices: HashMap<String, KokoroVoice>,
    pub load_elapsed: Duration,
    pub vram_bytes: u64,
}

impl KokoroEngine {
    pub fn load(model_dir: &Path, gpu: usize) -> anyhow::Result<Self> {
        let config = KokoroConfig::load(model_dir)?;
        let allocator = chew_vram::VramAllocator::init()?;
        ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
        let free_before = allocator.free_bytes(gpu)?;
        let stream = Arc::clone(allocator.stream(gpu));
        let kernels = chew_kernel::GpuKernels::load(&stream, 768 * 2_048, 2_048)?;
        let started = Instant::now();
        let prosody = KokoroProsodyFrontend::load(model_dir, &stream)?;
        let text = KokoroTextEncoder::load(model_dir, &stream)?;
        let f0_noise = KokoroF0Noise::load(model_dir, &stream)?;
        let decoder = KokoroDecoderFrontend::load(model_dir, &stream)?;
        let generator = KokoroGenerator::load(model_dir, &stream)?;
        stream.synchronize()?;
        let load_elapsed = started.elapsed();
        let vram_bytes = free_before.saturating_sub(allocator.free_bytes(gpu)?);
        // Load the default eagerly so readiness also verifies voice storage.
        let default = KokoroVoice::load(&model_dir.join("voices/af_heart.pt"), &config)?;
        let mut voices = HashMap::new();
        voices.insert("af_heart".into(), default);
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            config,
            stream,
            kernels,
            prosody,
            text,
            f0_noise,
            decoder,
            generator,
            voices,
            load_elapsed,
            vram_bytes,
        })
    }

    pub fn model_id(&self) -> &'static str {
        "tts-fast"
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest) -> anyhow::Result<SynthesisOutput> {
        ensure!(!request.text.trim().is_empty(), "text must not be empty");
        let phonemes = if request.language == "ipa" || request.language == "phonemes" {
            request.text.clone()
        } else {
            phonemize(&request.text, &request.language)?
        };
        let tokens = self.config.tokenize_phonemes(&phonemes)?;
        ensure!(
            tokens.skipped_phonemes <= tokens.phoneme_count / 4 + 2,
            "phonemizer produced too many unsupported symbols"
        );
        let voice_name = normalize_voice(&request.voice);
        if !self.voices.contains_key(&voice_name) {
            let path = self
                .model_dir
                .join("voices")
                .join(format!("{voice_name}.pt"));
            ensure!(
                path.is_file(),
                "unsupported Kokoro voice {:?}",
                request.voice
            );
            self.voices
                .insert(voice_name.clone(), KokoroVoice::load(&path, &self.config)?);
        }
        let voice = self
            .voices
            .get(&voice_name)
            .context("voice cache failure")?;
        let started = Instant::now();
        let prosody = self
            .prosody
            .predict(&tokens.ids, voice, request.speed, &mut self.kernels)?;
        let asr = self
            .text
            .encode_aligned(&tokens.ids, &prosody.durations, &mut self.kernels)?;
        let (f0, noise) = self.f0_noise.predict(
            &prosody.aligned,
            prosody.acoustic_frames,
            &prosody.predictor_style,
            &mut self.kernels,
        )?;
        let latent = self.decoder.decode(
            &asr,
            &f0,
            &noise,
            prosody.acoustic_frames,
            &prosody.decoder_style,
            &mut self.kernels,
        )?;
        let samples = self.generator.synthesize(
            &latent,
            &f0,
            f0.len(),
            &prosody.decoder_style,
            request.seed,
            &mut self.kernels,
        )?;
        self.stream.synchronize()?;
        let elapsed = started.elapsed();
        Ok(SynthesisOutput {
            samples,
            generated_frames: f0.len(),
            prompt_elapsed: Duration::ZERO,
            generation_elapsed: elapsed,
            codec_elapsed: Duration::ZERO,
        })
    }
}

fn normalize_voice(voice: &str) -> String {
    let voice = voice.trim().to_ascii_lowercase().replace('-', "_");
    match voice.as_str() {
        "" | "alloy" | "default" | "heart" => "af_heart".into(),
        _ => voice,
    }
}

fn phonemize(text: &str, language: &str) -> anyhow::Result<String> {
    let voice = match language {
        "en" | "en-us" | "english" => "en-us",
        "en-gb" => "en-gb",
        "de" | "de-de" | "german" => "de",
        "fr" | "fr-fr" | "french" => "fr-fr",
        "es" | "es-es" | "spanish" => "es",
        "it" | "it-it" | "italian" => "it",
        other => other,
    };
    let output = Command::new("espeak-ng")
        .args(["-q", "--ipa=3", "-v", voice, text])
        .output()
        .with_context(|| "could not run espeak-ng; install it or submit language=ipa")?;
    ensure!(
        output.status.success(),
        "espeak-ng failed for language {language:?}"
    );
    let phonemes = String::from_utf8(output.stdout)
        .context("espeak-ng returned invalid UTF-8")?
        .replace(['\u{200d}', '\n', '\r'], "");
    ensure!(
        !phonemes.trim().is_empty(),
        "phonemizer returned no phonemes"
    );
    Ok(phonemes)
}
