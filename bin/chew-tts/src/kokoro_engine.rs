use crate::voice_design::{SynthesisOutput, SynthesisRequest};
use anyhow::{Context, ensure};
use chew_model_kokoro::{
    KokoroConfig, KokoroDecoderFrontend, KokoroF0Noise, KokoroGenerator, KokoroProsodyFrontend,
    KokoroTextEncoder, KokoroVoice,
};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
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
        // Pay the dynamic-library initialization cost during readiness rather
        // than on the first customer request.
        let _ = espeak_api();
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
        let prompt_started = Instant::now();
        let prosody = self
            .prosody
            .predict(&tokens.ids, voice, request.speed, &mut self.kernels)?;
        let asr = self
            .text
            .encode_aligned(&tokens.ids, &prosody.durations, &mut self.kernels)?;
        let prompt_elapsed = prompt_started.elapsed();
        let generation_started = Instant::now();
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
        let generation_elapsed = generation_started.elapsed();
        let codec_started = Instant::now();
        let samples = self.generator.synthesize(
            &latent,
            &f0,
            f0.len(),
            &prosody.decoder_style,
            request.seed,
            &mut self.kernels,
        )?;
        self.stream.synchronize()?;
        let codec_elapsed = codec_started.elapsed();
        Ok(SynthesisOutput {
            samples,
            sample_rate: 24_000,
            generated_frames: f0.len(),
            prompt_elapsed,
            generation_elapsed,
            codec_elapsed,
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
    if let Some(result) = phonemize_in_process(text, voice) {
        return result;
    }
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

type EspeakInitialize = unsafe extern "C" fn(i32, i32, *const c_char, i32) -> i32;
type EspeakSetVoice = unsafe extern "C" fn(*const c_char) -> i32;
type EspeakTextToPhonemes = unsafe extern "C" fn(*mut *const c_void, i32, i32) -> *const c_char;

struct EspeakApi {
    _handle: *mut c_void,
    set_voice: EspeakSetVoice,
    text_to_phonemes: EspeakTextToPhonemes,
}

// eSpeak-NG keeps voice and translator state globally. Serializing this tiny
// frontend operation makes it safe across multiple persistent GPU workers.
unsafe impl Send for EspeakApi {}

static ESPEAK: OnceLock<Option<Mutex<EspeakApi>>> = OnceLock::new();

fn espeak_api() -> Option<&'static Mutex<EspeakApi>> {
    ESPEAK
        .get_or_init(|| EspeakApi::load().map(Mutex::new))
        .as_ref()
}

fn phonemize_in_process(text: &str, voice: &str) -> Option<anyhow::Result<String>> {
    let api = espeak_api()?;
    Some((|| {
        let api = api
            .lock()
            .map_err(|_| anyhow::anyhow!("eSpeak-NG phonemizer lock was poisoned"))?;
        let voice = CString::new(voice).context("language contains a NUL byte")?;
        let status = unsafe { (api.set_voice)(voice.as_ptr()) };
        ensure!(
            status == 0,
            "eSpeak-NG does not support the selected language"
        );
        let input = CString::new(text).context("text contains a NUL byte")?;
        let mut cursor = input.as_ptr().cast::<c_void>();
        let mut output = String::new();
        while !cursor.is_null() {
            let chunk = unsafe { (api.text_to_phonemes)(&mut cursor, 1, 2) };
            ensure!(!chunk.is_null(), "eSpeak-NG phonemizer returned no output");
            output.push_str(
                unsafe { CStr::from_ptr(chunk) }
                    .to_str()
                    .context("eSpeak-NG returned invalid UTF-8")?,
            );
        }
        let output = output.replace(['\u{200d}', '\n', '\r'], "");
        ensure!(!output.trim().is_empty(), "phonemizer returned no phonemes");
        Ok(output)
    })())
}

impl EspeakApi {
    fn load() -> Option<Self> {
        #[cfg(unix)]
        unsafe {
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            const RTLD_NOW: i32 = 2;
            let mut handle = std::ptr::null_mut();
            for library in [c"libespeak-ng.so.1", c"libespeak-ng.so"] {
                handle = dlopen(library.as_ptr(), RTLD_NOW);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                return None;
            }
            let initialize = dlsym(handle, c"espeak_Initialize".as_ptr());
            let set_voice = dlsym(handle, c"espeak_SetVoiceByName".as_ptr());
            let text_to_phonemes = dlsym(handle, c"espeak_TextToPhonemes".as_ptr());
            if initialize.is_null() || set_voice.is_null() || text_to_phonemes.is_null() {
                return None;
            }
            let initialize: EspeakInitialize = std::mem::transmute(initialize);
            let set_voice: EspeakSetVoice = std::mem::transmute(set_voice);
            let text_to_phonemes: EspeakTextToPhonemes = std::mem::transmute(text_to_phonemes);
            // AUDIO_OUTPUT_SYNCHRONOUS; no audio is generated by TextToPhonemes.
            if initialize(2, 0, std::ptr::null(), 0) < 0 {
                return None;
            }
            Some(Self {
                _handle: handle,
                set_voice,
                text_to_phonemes,
            })
        }
        #[cfg(not(unix))]
        {
            None
        }
    }
}
