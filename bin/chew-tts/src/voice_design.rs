use crate::audio_input::{decode_wav, resample, speaker_mel};
use anyhow::{Context, ensure};
use chew_model_qwen3_tts::{
    CodePredictorGenerationSession, CodePredictorTransformer, CodecEncoder, CodecQuantizer,
    CodecTransformerSession, ModelType, SemanticSamplingSession, SpeakerEncoder, TalkerConfig,
    TalkerFrontend, TalkerTransformer, inspect_model,
};
use cudarc::driver::CudaStream;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

pub struct VoiceDesignEngine {
    model_type: ModelType,
    config: TalkerConfig,
    tokenizer: Tokenizer,
    stream: Arc<CudaStream>,
    kernels: chew_kernel::GpuKernels,
    talker: TalkerTransformer,
    frontend: TalkerFrontend,
    predictor: CodePredictorTransformer,
    predictor_session: CodePredictorGenerationSession,
    semantic_session: SemanticSamplingSession,
    codec: CodecQuantizer,
    speaker_encoder: Option<SpeakerEncoder>,
    codec_encoder: Option<CodecEncoder>,
    reference_cache: HashMap<[u8; 32], CachedReference>,
    max_frames: usize,
    pub load_elapsed: Duration,
    pub vram_bytes: u64,
}

#[derive(Clone)]
struct CachedReference {
    samples: Vec<f32>,
    speaker_embedding: Vec<f32>,
    codes: Option<Vec<Vec<i32>>>,
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub voice: String,
    pub instruction: Option<String>,
    pub reference_audio_wav: Option<Vec<u8>>,
    pub reference_text: Option<String>,
    pub language: String,
    pub max_frames: usize,
    pub seed: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub chunk_frames: usize,
    pub chunk_context: usize,
}

pub struct SynthesisOutput {
    pub samples: Vec<f32>,
    pub generated_frames: usize,
    pub prompt_elapsed: Duration,
    pub generation_elapsed: Duration,
    pub codec_elapsed: Duration,
}

impl SynthesisOutput {
    pub fn inference_elapsed(&self) -> Duration {
        self.prompt_elapsed + self.generation_elapsed + self.codec_elapsed
    }
}

impl VoiceDesignEngine {
    pub fn load(model_dir: &Path, gpu: usize, max_frames: usize) -> anyhow::Result<Self> {
        ensure!(max_frames > 0, "maximum frame count must be non-zero");
        let inspection = inspect_model(model_dir)?;
        let model_type = inspection.config.tts_model_type;
        let config = inspection.config.talker_config;
        let tokenizer = load_qwen_tokenizer(model_dir)?;
        let allocator = chew_vram::VramAllocator::init()?;
        ensure!(
            gpu < allocator.gpu_count(),
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
        let free_before = allocator.free_bytes(gpu)?;
        let stream = Arc::clone(allocator.stream(gpu));
        let max_matrix = (config.intermediate_size * config.hidden_size)
            .max(config.text_hidden_size * config.text_hidden_size);
        let max_vector = config.intermediate_size.max(config.text_hidden_size);
        let kernels = chew_kernel::GpuKernels::load(&stream, max_matrix, max_vector)?;

        let started = Instant::now();
        let talker = TalkerTransformer::load(model_dir, &config, &stream)?;
        let frontend = TalkerFrontend::load(model_dir, &config, &stream)?;
        let predictor =
            CodePredictorTransformer::load(model_dir, &config.code_predictor_config, &stream)?;
        let predictor_session = predictor.start_generation_session(&stream)?;
        let semantic_session = frontend.start_semantic_sampling_session(max_frames, &stream)?;
        let codec = CodecQuantizer::load(model_dir.join("speech_tokenizer"), &stream)?;
        let speaker_encoder = if model_type == ModelType::Base {
            Some(SpeakerEncoder::load(model_dir, &stream)?)
        } else {
            None
        };
        let codec_encoder = if model_type == ModelType::Base {
            Some(CodecEncoder::load(
                &model_dir.join("speech_tokenizer"),
                &stream,
            )?)
        } else {
            None
        };
        let load_elapsed = started.elapsed();
        let free_loaded = allocator.free_bytes(gpu)?;

        // Compile/load kernels before the server reports readiness.
        kernels.ops.stream().synchronize()?;
        Ok(Self {
            model_type,
            config,
            tokenizer,
            stream,
            kernels,
            talker,
            frontend,
            predictor,
            predictor_session,
            semantic_session,
            codec,
            speaker_encoder,
            codec_encoder,
            reference_cache: HashMap::new(),
            max_frames,
            load_elapsed,
            vram_bytes: free_before.saturating_sub(free_loaded),
        })
    }

    pub fn model_id(&self) -> &'static str {
        match self.model_type {
            ModelType::Base => "tts-multilingual-base",
            ModelType::CustomVoice => "tts-multilingual",
            ModelType::VoiceDesign => "tts-voice-design",
        }
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest) -> anyhow::Result<SynthesisOutput> {
        ensure!(!request.text.trim().is_empty(), "text must not be empty");
        ensure!(
            request.max_frames > 0 && request.max_frames <= self.max_frames,
            "max_frames must be between 1 and {}",
            self.max_frames
        );
        ensure!(
            request.temperature.is_finite() && request.temperature > 0.0,
            "temperature must be finite and positive"
        );
        ensure!(
            (1..=64).contains(&request.top_k),
            "top_k must be between 1 and 64"
        );
        let language_key = request.language.to_ascii_lowercase();
        let language_codec_id = self
            .config
            .codec_language_id
            .get(&language_key)
            .copied()
            .with_context(|| {
                let mut supported = self
                    .config
                    .codec_language_id
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>();
                supported.sort();
                format!(
                    "unsupported language {:?}; supported: {supported:?}",
                    request.language
                )
            })? as i32;
        let encode = |value: &str| -> anyhow::Result<Vec<i32>> {
            Ok(self
                .tokenizer
                .encode(value, false)
                .map_err(|error| anyhow::anyhow!("could not encode text: {error}"))?
                .get_ids()
                .iter()
                .map(|id| *id as i32)
                .collect())
        };
        let text_ids = encode(&request.text)?;
        let instruction_ids = request
            .instruction
            .as_deref()
            .filter(|instruction| !instruction.trim().is_empty())
            .map(|instruction| encode(&format!("<|im_start|>user\n{instruction}<|im_end|>\n")))
            .transpose()?;
        ensure!(
            self.model_type != ModelType::VoiceDesign || instruction_ids.is_some(),
            "instruction must not be empty for a VoiceDesign model"
        );
        let wants_icl = self.model_type == ModelType::Base
            && request
                .reference_text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty());
        let mut base_codes = None;
        let speaker_embedding = match self.model_type {
            ModelType::CustomVoice => {
                let speaker = normalize_speaker_name(&request.voice);
                let speaker_id = self.config.spk_id.get(&speaker).copied().with_context(|| {
                    let mut supported = self.config.spk_id.keys().cloned().collect::<Vec<_>>();
                    supported.sort();
                    format!(
                        "unsupported voice {:?}; supported: {supported:?}",
                        request.voice
                    )
                })?;
                Some(
                    self.frontend
                        .codec_embeddings(&[speaker_id as i32], &mut self.kernels)?,
                )
            }
            ModelType::Base => {
                let wav = request
                    .reference_audio_wav
                    .as_deref()
                    .context("reference_audio is required for a Base model")?;
                let digest: [u8; 32] = Sha256::digest(wav).into();
                let cached = self.reference_cache.get(&digest).cloned();
                let mut reference = if let Some(cached) = cached {
                    cached
                } else {
                    let (samples, sample_rate) = decode_wav(wav)?;
                    let samples = resample(&samples, sample_rate, 24_000);
                    ensure!(
                        samples.len() <= 60 * 24_000,
                        "reference audio exceeds 60 seconds"
                    );
                    let (mel, frames) = speaker_mel(&samples)?;
                    let speaker_embedding = self
                        .speaker_encoder
                        .as_ref()
                        .context("Base speaker encoder was not loaded")?
                        .encode_mel(&mel, frames, &mut self.kernels)?;
                    CachedReference {
                        samples,
                        speaker_embedding,
                        codes: None,
                    }
                };
                if wants_icl && reference.codes.is_none() {
                    reference.codes = Some(
                        self.codec_encoder
                            .as_ref()
                            .context("Base codec encoder was not loaded")?
                            .encode(&reference.samples, &mut self.kernels)?,
                    );
                }
                if !self.reference_cache.contains_key(&digest)
                    && self.reference_cache.len() >= 8
                    && let Some(oldest) = self.reference_cache.keys().next().copied()
                {
                    self.reference_cache.remove(&oldest);
                }
                base_codes = reference.codes.clone();
                let embedding = reference.speaker_embedding.clone();
                self.reference_cache.insert(digest, reference);
                Some(embedding)
            }
            ModelType::VoiceDesign => None,
        };
        let mut seed = request.seed;

        let prompt_started = Instant::now();
        let inputs = if wants_icl {
            let reference_text_ids =
                encode(request.reference_text.as_deref().expect("checked above"))?;
            let reference_codes = base_codes
                .as_deref()
                .context("Base reference codec frames are unavailable")?;
            self.frontend.build_icl_inputs(
                &text_ids,
                &reference_text_ids,
                reference_codes,
                instruction_ids.as_deref(),
                language_codec_id,
                speaker_embedding
                    .as_deref()
                    .context("Base speaker embedding is unavailable")?,
                &self.predictor,
                &mut self.kernels,
            )?
        } else {
            self.frontend.build_conditioned_inputs(
                &text_ids,
                instruction_ids.as_deref(),
                language_codec_id,
                speaker_embedding.as_deref(),
                &mut self.kernels,
            )?
        };
        let max_seq_len = inputs.prefill_tokens + request.max_frames;
        ensure!(
            max_seq_len <= self.config.max_position_embeddings,
            "prompt plus generation is {max_seq_len} tokens, model limit is {}",
            self.config.max_position_embeddings
        );
        let mut talker_session = self.talker.start_session(
            max_seq_len,
            inputs.prefill_tokens,
            &self.config,
            &self.stream,
        )?;
        let normalized = self.talker.forward_session(
            &mut talker_session,
            &inputs.prefill,
            inputs.prefill_tokens,
            &self.config,
            &mut self.kernels,
        )?;
        let mut last_hidden = normalized[normalized.len() - self.config.hidden_size..].to_vec();
        let mut generated_semantics = Vec::with_capacity(request.max_frames);
        let mut semantic = self.frontend.semantic_speech_sample_with_session(
            &mut self.semantic_session,
            &last_hidden,
            &generated_semantics,
            request.temperature,
            request.top_k,
            1.05,
            &mut seed,
            &mut self.kernels,
        )?;
        generated_semantics.push(semantic);
        let prompt_elapsed = prompt_started.elapsed();

        let generation_started = Instant::now();
        let mut pending = Vec::with_capacity(request.chunk_frames.max(1));
        let mut transformed_context = Vec::with_capacity(1024 * request.chunk_context);
        let mut codec_session = if request.chunk_frames > 0 {
            Some(
                self.codec
                    .start_transformer_session(request.max_frames, &self.stream)?,
            )
        } else {
            None
        };
        let mut all_frames = if request.chunk_frames == 0 {
            Vec::with_capacity(request.max_frames)
        } else {
            Vec::new()
        };
        let mut samples = Vec::new();
        let mut codec_elapsed = Duration::ZERO;
        let mut generated_frames = 0usize;
        for frame_index in 0..request.max_frames {
            if semantic == 2_150 {
                break;
            }
            let acoustic = self
                .predictor
                .generate_acoustic_codes_sampled_with_session(
                    &mut self.predictor_session,
                    &last_hidden,
                    semantic,
                    request.temperature,
                    request.top_k,
                    &mut seed,
                    &mut self.kernels,
                )?;
            let mut codes = Vec::with_capacity(self.config.num_code_groups);
            codes.push(semantic);
            codes.extend_from_slice(&acoustic);
            if request.chunk_frames == 0 {
                all_frames.push(codes);
            } else {
                pending.push(codes);
            }
            generated_frames += 1;
            if request.chunk_frames > 0 && pending.len() >= request.chunk_frames {
                codec_elapsed += decode_codec_chunk(
                    &self.codec,
                    &mut pending,
                    &mut transformed_context,
                    codec_session
                        .as_mut()
                        .expect("chunked decoding has a codec transformer session"),
                    request.chunk_context,
                    &mut samples,
                    &mut self.kernels,
                )?;
            }

            let semantic_embedding = self
                .frontend
                .codec_embeddings(&[semantic], &mut self.kernels)?;
            let acoustic_embedding = self.predictor.acoustic_embeddings_sum_with_session(
                &mut self.predictor_session,
                &mut self.kernels,
            )?;
            let text_embedding = if frame_index < inputs.trailing_tokens {
                let start = frame_index * self.config.hidden_size;
                &inputs.trailing_text[start..start + self.config.hidden_size]
            } else {
                &inputs.text_pad
            };
            let next_input = semantic_embedding
                .iter()
                .zip(acoustic_embedding)
                .zip(text_embedding)
                .map(|((semantic, acoustic), text)| semantic + acoustic + text)
                .collect::<Vec<_>>();
            last_hidden = self.talker.forward_session(
                &mut talker_session,
                &next_input,
                1,
                &self.config,
                &mut self.kernels,
            )?;
            semantic = self.frontend.semantic_speech_sample_with_session(
                &mut self.semantic_session,
                &last_hidden,
                &generated_semantics,
                request.temperature,
                request.top_k,
                1.05,
                &mut seed,
                &mut self.kernels,
            )?;
            generated_semantics.push(semantic);
        }
        let generation_elapsed = generation_started.elapsed().saturating_sub(codec_elapsed);
        ensure!(
            generated_frames > 0,
            "model emitted EOS before producing audio"
        );

        let decode_started = Instant::now();
        if request.chunk_frames == 0 {
            samples = self
                .codec
                .decode_frames_audio(&all_frames, &mut self.kernels)?;
            codec_elapsed = decode_started.elapsed();
        } else if !pending.is_empty() {
            codec_elapsed += decode_codec_chunk(
                &self.codec,
                &mut pending,
                &mut transformed_context,
                codec_session
                    .as_mut()
                    .expect("chunked decoding has a codec transformer session"),
                request.chunk_context,
                &mut samples,
                &mut self.kernels,
            )?;
        }
        ensure!(
            semantic == 2_150 || generated_frames < request.max_frames,
            "generation reached the {}-frame safety limit before EOS",
            request.max_frames
        );
        Ok(SynthesisOutput {
            samples,
            generated_frames,
            prompt_elapsed,
            generation_elapsed,
            codec_elapsed,
        })
    }
}

fn normalize_speaker_name(voice: &str) -> String {
    let normalized = voice.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "alloy" => "ryan".into(),
        "echo" => "aiden".into(),
        "fable" => "eric".into(),
        "onyx" => "uncle_fu".into(),
        "nova" => "serena".into(),
        "shimmer" => "vivian".into(),
        _ => normalized,
    }
}

fn decode_codec_chunk(
    codec: &CodecQuantizer,
    pending: &mut Vec<Vec<i32>>,
    context: &mut Vec<f32>,
    session: &mut CodecTransformerSession,
    context_frames: usize,
    samples: &mut Vec<f32>,
    kernels: &mut chew_kernel::GpuKernels,
) -> anyhow::Result<Duration> {
    if pending.is_empty() {
        return Ok(Duration::ZERO);
    }
    const SAMPLES_PER_FRAME: usize = 1920;
    const CHANNELS: usize = 1024;
    let prefix_frames = context.len() / CHANNELS;
    let new_frames = pending.len();
    let started = Instant::now();
    let transformed = codec.decode_frames_transformer_session(pending, session, kernels)?;
    pending.clear();

    let total_frames = prefix_frames + new_frames;
    let mut decode_frames = vec![0.0; CHANNELS * total_frames];
    for channel in 0..CHANNELS {
        let destination = channel * total_frames;
        if prefix_frames > 0 {
            let source = channel * prefix_frames;
            decode_frames[destination..destination + prefix_frames]
                .copy_from_slice(&context[source..source + prefix_frames]);
        }
        let source = channel * new_frames;
        decode_frames[destination + prefix_frames..destination + total_frames]
            .copy_from_slice(&transformed[source..source + new_frames]);
    }
    let decoded = codec.decode_transformed_audio(&decode_frames, total_frames, kernels)?;
    samples.extend_from_slice(&decoded[prefix_frames * SAMPLES_PER_FRAME..]);

    let keep = context_frames.min(total_frames);
    let first_kept = total_frames - keep;
    context.resize(CHANNELS * keep, 0.0);
    for channel in 0..CHANNELS {
        let source = channel * total_frames + first_kept;
        let destination = channel * keep;
        context[destination..destination + keep]
            .copy_from_slice(&decode_frames[source..source + keep]);
    }
    Ok(started.elapsed())
}

fn load_qwen_tokenizer(model_dir: &Path) -> anyhow::Result<Tokenizer> {
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let vocab = vocab.to_string_lossy();
    let merges = merges.to_string_lossy();
    let model = BPE::from_file(&vocab, &merges)
        .build()
        .map_err(|error| anyhow::anyhow!("could not build Qwen BPE: {error}"))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));

    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_dir.join("tokenizer_config.json"))
            .context("could not read tokenizer_config.json")?,
    )?;
    let decoder = config["added_tokens_decoder"]
        .as_object()
        .context("tokenizer_config.json has no added_tokens_decoder")?;
    let mut entries = decoder
        .iter()
        .map(|(id, value)| Ok((id.parse::<u32>().context("invalid added-token ID")?, value)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    entries.sort_by_key(|(id, _)| *id);
    let added = entries
        .into_iter()
        .map(|(_, value)| {
            AddedToken::from(
                value["content"].as_str().unwrap_or_default(),
                value["special"].as_bool().unwrap_or(false),
            )
            .single_word(value["single_word"].as_bool().unwrap_or(false))
            .lstrip(value["lstrip"].as_bool().unwrap_or(false))
            .rstrip(value["rstrip"].as_bool().unwrap_or(false))
            .normalized(value["normalized"].as_bool().unwrap_or(false))
        })
        .collect::<Vec<_>>();
    tokenizer.add_tokens(&added);
    Ok(tokenizer)
}
