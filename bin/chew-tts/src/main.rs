use anyhow::Context;
use chew_model_chatterbox::{
    ChatterboxConditioning, ChatterboxF0Predictor, ChatterboxFlowEstimator,
    ChatterboxFlowResnetBlock, ChatterboxFlowTimeEmbedding, ChatterboxFlowTransformerBlock,
    ChatterboxHiFT, ChatterboxS3ConformerLayer, ChatterboxS3Encoder, ChatterboxS3Flow,
    ChatterboxT3Frontend, ChatterboxT3Layer, ChatterboxT3Transformer, ChatterboxTokenizer,
    HIDDEN_SIZE as CHATTERBOX_HIDDEN_SIZE, INTERMEDIATE_SIZE as CHATTERBOX_INTERMEDIATE_SIZE,
    S3_HIDDEN_SIZE, inspect_model as inspect_chatterbox_model,
};
use chew_model_kokoro::{
    KokoroAdaInResBlock, KokoroAlbert, KokoroBiLstm, KokoroCheckpoint, KokoroDecoderFrontend,
    KokoroF0Noise, KokoroGenerator, KokoroProsodyFrontend, KokoroTextEncoder, KokoroVoice,
    inspect_model as inspect_kokoro_model,
};
use chew_model_qwen3_tts::{
    CodePredictorTransformer, CodecEncoder, CodecQuantizer, CodecTransformerSession,
    SpeakerEncoder, TalkerDecoderLayer, TalkerFrontend, TalkerLayerKvCache, TalkerTransformer,
    inspect_model, load_f16_tensor,
};
use clap::{Parser, Subcommand};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

mod audio_input;
mod chatterbox_engine;
mod kokoro_engine;
mod server;
mod voice_design;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a persistent OpenAI-compatible VoiceDesign HTTP server.
    Serve {
        /// Qwen3-TTS VoiceDesign model directory.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Listen port.
        #[arg(long, default_value_t = 18001)]
        port: u16,
        /// Per-request codec-frame safety limit and session capacity.
        #[arg(long, default_value_t = 2048)]
        max_frames: usize,
        /// Maximum number of waiting HTTP requests.
        #[arg(long, default_value_t = 16)]
        queue_capacity: usize,
    },
    /// Run Fleet's raw JSON-over-TCP protocol and return f32le/24 kHz audio.
    FleetServe {
        /// Qwen3-TTS model directory.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Listen port.
        #[arg(long, default_value_t = 18001)]
        port: u16,
        /// Per-request codec-frame safety limit and session capacity.
        #[arg(long, default_value_t = 2048)]
        max_frames: usize,
        /// Maximum number of waiting synthesis requests.
        #[arg(long, default_value_t = 16)]
        queue_capacity: usize,
    },
    /// Validate a Qwen3-TTS model and print its inference geometry.
    Inspect {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
    },
    /// Validate a Kokoro model and print its native checkpoint geometry.
    InspectKokoro {
        /// Directory containing config.json and kokoro-v1_0.pth.
        model_dir: PathBuf,
        /// Optional official Kokoro .pt voice pack to validate.
        #[arg(long)]
        voice: Option<PathBuf>,
        /// Optional already-phonemized input to validate and map to token IDs.
        #[arg(long)]
        phonemes: Option<String>,
    },
    /// Run Kokoro's native twelve-pass shared ALBERT and 512-wide projection.
    CudaKokoroAlbertSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        phonemes: String,
    },
    /// Validate Kokoro's native bidirectional LSTM CUDA cell.
    CudaKokoroLstmSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 8)]
        frames: usize,
    },
    /// Run Kokoro ALBERT, duration encoder, and duration projection.
    CudaKokoroProsodySmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        phonemes: String,
        #[arg(long)]
        wav: Option<PathBuf>,
    },
    /// Validate one native Kokoro AdaIN residual block.
    CudaKokoroAdaInSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 8)]
        frames: usize,
        #[arg(long)]
        upsample: bool,
    },
    /// Validate Chatterbox Multilingual V3 T3, S3Gen, and voice-encoder weights.
    InspectChatterbox {
        /// Directory containing t3_mtl23ls_v3, S3Gen, and ve weights.
        model_dir: PathBuf,
    },
    /// Tokenize multilingual text with Chatterbox's local grapheme BPE.
    TokenizeChatterbox {
        /// Directory containing grapheme_mtl_merged_expanded_v1.json.
        model_dir: PathBuf,
        /// ISO language code such as de or en.
        language: String,
        /// Text to encode.
        text: String,
    },
    /// Run one complete Chatterbox V3 T3 decoder layer on local CUDA.
    CudaChatterboxLayerSmoke {
        /// Directory containing t3_mtl23ls_v3.safetensors.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// T3 decoder layer to validate.
        #[arg(long, default_value_t = 0)]
        layer: usize,
        /// Run the complete 30-layer transformer and final norm.
        #[arg(long)]
        stack: bool,
        /// Number of prepared hidden-state tokens.
        #[arg(long, default_value_t = 1)]
        seq_len: usize,
        /// Prefill this many tokens, then append the rest one at a time.
        #[arg(long)]
        decode_split: Option<usize>,
        /// Compare a split cached decode against one-pass prefill.
        #[arg(long)]
        compare_cache: bool,
    },
    /// Run one native S3Gen relative-attention Conformer block on local CUDA.
    CudaChatterboxS3LayerSmoke {
        /// Directory containing s3gen_v3.safetensors.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Conformer layer within the selected six- or four-layer stage.
        #[arg(long, default_value_t = 0)]
        layer: usize,
        /// Select the four layers after the 2x upsampling operation.
        #[arg(long)]
        upsampled: bool,
        /// Number of deterministic test frames.
        #[arg(long, default_value_t = 8)]
        seq_len: usize,
    },
    /// Run the complete native S3Gen token-conditioning encoder.
    CudaChatterboxS3EncoderSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 8)]
        tokens: usize,
    },
    /// Run one transformer block from the S3Gen conditional-flow estimator.
    CudaChatterboxFlowBlockSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value = "flow.decoder.estimator.mid_blocks.0.1.0")]
        prefix: String,
        #[arg(long, default_value_t = 8)]
        seq_len: usize,
    },
    /// Run one causal ResNet block from the S3Gen flow estimator.
    CudaChatterboxFlowResnetSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value = "flow.decoder.estimator.mid_blocks.0.0")]
        prefix: String,
        #[arg(long, default_value_t = 8)]
        seq_len: usize,
        #[arg(long, default_value_t = 256)]
        input_channels: usize,
    },
    /// Run the S3Gen flow estimator's sinusoidal timestep MLP.
    CudaChatterboxFlowTimeSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 0.35)]
        timestep: f32,
    },
    /// Run one complete native S3Gen conditional-flow velocity evaluation.
    CudaChatterboxFlowEstimatorSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 8)]
        frames: usize,
        #[arg(long, default_value_t = 0.35)]
        timestep: f32,
    },
    /// Run native S3Gen conditioning and CFM Euler sampling to mel frames.
    CudaChatterboxFlowSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        generated_tokens: usize,
    },
    /// Run the native HiFT convolutional F0 predictor on deterministic mel.
    CudaChatterboxF0Smoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 16)]
        frames: usize,
    },
    /// Run the complete native HiFT vocoder on deterministic mel.
    CudaChatterboxHiFtSmoke {
        model_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        #[arg(long, default_value_t = 8)]
        frames: usize,
        #[arg(long)]
        wav: PathBuf,
    },
    /// Generate native Chatterbox speech tokens through the complete T3 path.
    CudaChatterboxGenerationSmoke {
        /// Directory containing Chatterbox V3 weights, tokenizer, and conds.pt.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// ISO language code such as de or en.
        #[arg(long, default_value = "de")]
        language: String,
        /// Text to synthesize into speech tokens.
        #[arg(long)]
        text: String,
        /// Maximum number of greedy speech tokens.
        #[arg(long, default_value_t = 8)]
        max_tokens: usize,
        /// Classifier-free guidance strength.
        #[arg(long, default_value_t = 0.5)]
        cfg_weight: f32,
        /// Optionally continue through S3Gen and HiFT into a 24-kHz WAV.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Conditional-flow Euler steps when --wav is set.
        #[arg(long, default_value_t = 10)]
        flow_steps: usize,
    },
    /// Tokenize text with the model's local Qwen2 BPE files.
    Tokenize {
        /// Qwen3-TTS model directory.
        model_dir: PathBuf,
        /// Text to encode.
        text: String,
    },
    /// Compile and load Chew's CUDA kernels for the selected GPU.
    CudaSmoke {
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
    },
    /// Validate one real Qwen linear layer against a CPU reference.
    CudaLinearSmoke {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Two-dimensional weight tensor to test.
        #[arg(long, default_value = "talker.model.layers.0.self_attn.q_proj.weight")]
        tensor: String,
        /// Use the native decode GEMV instead of cuBLAS GEMM.
        #[arg(long)]
        gemv: bool,
    },
    /// Run one complete native Qwen talker decoder layer on CUDA.
    CudaLayerSmoke {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Decoder layer to load.
        #[arg(long, default_value_t = 0)]
        layer: usize,
        /// Number of causal prefill tokens to validate.
        #[arg(long, default_value_t = 1)]
        seq_len: usize,
        /// Split after this many tokens and decode the remainder one-by-one.
        #[arg(long)]
        decode_split: Option<usize>,
        /// Optional raw little-endian f32 reference output.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Load and execute the complete GPU-resident Qwen talker stack.
    CudaTalkerSmoke {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Number of synthetic hidden-state tokens.
        #[arg(long, default_value_t = 1)]
        seq_len: usize,
        /// Prefill this many tokens, then append the remainder one-by-one.
        #[arg(long)]
        decode_split: Option<usize>,
        /// Optional raw little-endian f32 reference output.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Load and execute the complete Qwen code-predictor transformer.
    CudaPredictorSmoke {
        /// Directory containing config.json and Safetensors weights.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Number of prepared predictor tokens.
        #[arg(long, default_value_t = 1)]
        seq_len: usize,
        /// Generate all 15 acoustic codes instead of a prepared hidden pass.
        #[arg(long)]
        frame: bool,
        /// Semantic codec token used by --frame.
        #[arg(long, default_value_t = 42)]
        semantic_token: i32,
        /// Number of complete code frames generated by --frame.
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Optional raw little-endian f32 reference output.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Decode one 16-codebook frame into a codec latent.
    CudaCodecLatentSmoke {
        /// Directory containing speech_tokenizer/model.safetensors.
        tokenizer_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Sixteen comma-separated codec IDs.
        #[arg(
            long,
            default_value = "42,146,1921,714,1858,646,2036,1792,1614,912,581,708,1202,991,1341,259"
        )]
        codes: String,
        /// Also apply the codec's causal pre-convolution.
        #[arg(long)]
        preconv: bool,
        /// Also run the codec's eight-layer pre-transformer.
        #[arg(long)]
        transformer: bool,
        /// Also run both 2x ConvNeXt upsampling stages.
        #[arg(long)]
        upsample: bool,
        /// Decode the complete 1,920-sample waveform frame.
        #[arg(long)]
        audio: bool,
        /// Number of identical codec frames to decode jointly.
        #[arg(long, default_value_t = 1)]
        frames: usize,
        /// Validate persistent codec KV state by splitting after this frame.
        #[arg(long)]
        stream_split: Option<usize>,
        /// Number of times to run the selected codec stage.
        #[arg(long, default_value_t = 1)]
        repeats: usize,
        /// Optional PCM16 WAV output; requires --audio.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Optional raw little-endian f32 reference latent.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Generate acoustic codes with the predictor and decode them to WAV.
    CudaPredictorCodecSmoke {
        /// Qwen3-TTS model directory containing speech_tokenizer.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Semantic codec token supplied by the talker.
        #[arg(long, default_value_t = 42)]
        semantic_token: i32,
        /// Number of continuous codec frames to generate.
        #[arg(long, default_value_t = 3)]
        frames: usize,
        /// PCM16 WAV output path.
        #[arg(long)]
        wav: PathBuf,
    },
    /// Validate native talker embeddings, text projection, and codec head.
    CudaTalkerFrontendSmoke {
        /// Qwen3-TTS model directory.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Comma-separated text token IDs.
        #[arg(long, default_value = "1,42,1000")]
        text_ids: String,
        /// Tokenize this text instead of using --text-ids.
        #[arg(long)]
        text: Option<String>,
        /// Optional raw little-endian f32 projected-text reference.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Validate native WAV/mel preprocessing and the Base speaker encoder.
    CudaSpeakerSmoke {
        /// Qwen3-TTS Base model directory.
        model_dir: PathBuf,
        /// Reference WAV.
        #[arg(long)]
        wav: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Write the 2048-value little-endian f32 embedding.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Write the channel-last speaker mel as little-endian f32.
        #[arg(long)]
        mel_output: Option<PathBuf>,
        /// Compare against a raw little-endian f32 reference embedding.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Encode reference audio into native 16-codebook Qwen codec frames.
    CudaCodecEncoderSmoke {
        /// Qwen3-TTS model directory containing speech_tokenizer.
        model_dir: PathBuf,
        /// Reference WAV.
        #[arg(long)]
        wav: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Write frame-major little-endian i32 codec IDs.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Compare with frame-major little-endian i32 codec IDs.
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Generate VoiceDesign codec frames and decode them to a WAV file.
    CudaVoiceDesignSmoke {
        /// Qwen3-TTS VoiceDesign model directory.
        model_dir: PathBuf,
        /// Zero-based CUDA device index.
        #[arg(long, default_value_t = 0)]
        gpu: usize,
        /// Text to speak.
        #[arg(long)]
        text: String,
        /// Natural-language voice description.
        #[arg(long)]
        instruction: String,
        /// Language name from the model config.
        #[arg(long, default_value = "german")]
        language: String,
        /// Safety limit in 80-ms codec frames; generation normally stops at EOS.
        #[arg(long = "max-frames", alias = "frames", default_value_t = 2048)]
        max_frames: usize,
        /// Deterministic sampling seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Semantic and acoustic sampling temperature.
        #[arg(long, default_value_t = 0.9)]
        temperature: f32,
        /// Semantic and acoustic top-k.
        #[arg(long, default_value_t = 50)]
        top_k: usize,
        /// Decode after this many frames; zero keeps exact full-sequence decoding.
        #[arg(long, default_value_t = 32)]
        chunk_frames: usize,
        /// Previous frames retained as causal context for chunked decoding.
        #[arg(long, default_value_t = 4)]
        chunk_context: usize,
        /// PCM16 WAV output path.
        #[arg(long)]
        wav: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    match Cli::parse().command {
        Command::Serve {
            model_dir,
            gpu,
            host,
            port,
            max_frames,
            queue_capacity,
        } => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(server::run(
                model_dir,
                gpu,
                host,
                port,
                max_frames,
                queue_capacity,
            ))?,
        Command::FleetServe {
            model_dir,
            gpu,
            host,
            port,
            max_frames,
            queue_capacity,
        } => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(server::run_fleet(
                model_dir,
                gpu,
                host,
                port,
                max_frames,
                queue_capacity,
            ))?,
        Command::Inspect { model_dir } => {
            let inspection = inspect_model(&model_dir)
                .with_context(|| format!("could not inspect {}", model_dir.display()))?;
            let talker = &inspection.config.talker_config;
            let predictor = &talker.code_predictor_config;
            println!(
                "Qwen3-TTS {} {:?}",
                inspection.config.tts_model_size, inspection.config.tts_model_type
            );
            println!(
                "talker: {} layers, hidden {}, {} Q heads / {} KV heads",
                talker.num_hidden_layers,
                talker.hidden_size,
                talker.num_attention_heads,
                talker.num_key_value_heads
            );
            println!(
                "code predictor: {} layers, hidden {}, {} acoustic steps/frame",
                predictor.num_hidden_layers,
                predictor.hidden_size,
                talker.num_code_groups - 1
            );
            println!(
                "weights: {} tensors in {} file(s), {:.2} GiB",
                inspection.tensors.len(),
                inspection.weight_files.len(),
                inspection.total_weight_bytes as f64 / 1024.0_f64.powi(3)
            );
        }
        Command::InspectKokoro {
            model_dir,
            voice,
            phonemes,
        } => {
            let inspection = inspect_kokoro_model(&model_dir)
                .with_context(|| format!("could not inspect {}", model_dir.display()))?;
            println!(
                "Kokoro: {} tokens, context {}, hidden {}, style {}",
                inspection.config.n_token,
                inspection.config.plbert.max_position_embeddings,
                inspection.config.hidden_dim,
                inspection.config.style_dim,
            );
            println!(
                "Albert: {} shared layers, hidden {}, {} heads",
                inspection.config.plbert.num_hidden_layers,
                inspection.config.plbert.hidden_size,
                inspection.config.plbert.num_attention_heads,
            );
            println!(
                "iSTFTNet: upsample {:?}, kernels {:?}, n_fft {}, hop {}",
                inspection.config.istftnet.upsample_rates,
                inspection.config.istftnet.upsample_kernel_sizes,
                inspection.config.istftnet.gen_istft_n_fft,
                inspection.config.istftnet.gen_istft_hop_size,
            );
            println!(
                "weights: {} tensors, {:.2} MiB ({})",
                inspection.tensors.len(),
                inspection.total_weight_bytes as f64 / 1024.0_f64.powi(2),
                inspection.checkpoint.display(),
            );
            if let Some(phonemes) = phonemes {
                let tokens = inspection.config.tokenize_phonemes(&phonemes)?;
                println!(
                    "phonemes: {} mapped, {} skipped, {} tokens with boundaries",
                    tokens.phoneme_count,
                    tokens.skipped_phonemes,
                    tokens.ids.len(),
                );
                if let Some(path) = voice.as_deref() {
                    let voice = KokoroVoice::load(path, &inspection.config)?;
                    let style = voice.style_for_phoneme_count(tokens.phoneme_count)?;
                    println!(
                        "voice: {} entries x {}, selected style {} finite values ({})",
                        voice.entries(),
                        voice.style_width(),
                        style.len(),
                        path.display(),
                    );
                }
            } else if let Some(path) = voice.as_deref() {
                let voice = KokoroVoice::load(path, &inspection.config)?;
                println!(
                    "voice: {} entries x {} finite values ({})",
                    voice.entries(),
                    voice.style_width(),
                    path.display(),
                );
            }
        }
        Command::CudaKokoroAlbertSmoke {
            model_dir,
            gpu,
            phonemes,
        } => {
            let config = chew_model_kokoro::KokoroConfig::load(&model_dir)?;
            let tokens = config.tokenize_phonemes(&phonemes)?;
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 768 * 2_048, 2_048)?;
            let output =
                KokoroAlbert::load(&model_dir, &stream)?.encode(&tokens.ids, &mut kernels)?;
            println!(
                "Kokoro ALBERT CUDA: tokens={}, sum={:.9}, sum_sq={:.9}, first={:?}",
                tokens.ids.len(),
                output.iter().map(|x| f64::from(*x)).sum::<f64>(),
                output
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                &output[..8]
            );
        }
        Command::CudaKokoroLstmSmoke {
            model_dir,
            gpu,
            frames,
        } => {
            anyhow::ensure!(frames > 0, "frame count must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 768 * 2_048, 2_048)?;
            let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
            let input = (0..frames * 640)
                .map(|index| (index as f32 * 0.013).sin() * 0.2)
                .collect::<Vec<_>>();
            let output =
                KokoroBiLstm::load(&checkpoint, "predictor", "module.lstm", 640, 256, &stream)?
                    .forward(&input, frames, &mut kernels)?;
            println!(
                "Kokoro BiLSTM CUDA: frames={}, sum={:.9}, sum_sq={:.9}, first={:?}",
                frames,
                output.iter().map(|x| f64::from(*x)).sum::<f64>(),
                output
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                &output[..8]
            );
        }
        Command::CudaKokoroProsodySmoke {
            model_dir,
            gpu,
            speed,
            phonemes,
            wav,
        } => {
            let config = chew_model_kokoro::KokoroConfig::load(&model_dir)?;
            let tokens = config.tokenize_phonemes(&phonemes)?;
            let voice = KokoroVoice::load(&model_dir.join("voices/af_heart.pt"), &config)?;
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 768 * 2_048, 2_048)?;
            let prosody = KokoroProsodyFrontend::load(&model_dir, &stream)?.predict(
                &tokens.ids,
                &voice,
                speed,
                &mut kernels,
            )?;
            let asr = KokoroTextEncoder::load(&model_dir, &stream)?.encode_aligned(
                &tokens.ids,
                &prosody.durations,
                &mut kernels,
            )?;
            let (f0, noise) = KokoroF0Noise::load(&model_dir, &stream)?.predict(
                &prosody.aligned,
                prosody.acoustic_frames,
                &prosody.predictor_style,
                &mut kernels,
            )?;
            let decoder = KokoroDecoderFrontend::load(&model_dir, &stream)?.decode(
                &asr,
                &f0,
                &noise,
                prosody.acoustic_frames,
                &prosody.decoder_style,
                &mut kernels,
            )?;
            if let Some(path) = wav {
                let audio = KokoroGenerator::load(&model_dir, &stream)?.synthesize(
                    &decoder,
                    &f0,
                    f0.len(),
                    &prosody.decoder_style,
                    42,
                    &mut kernels,
                )?;
                write_pcm16_wav(&path, &audio, 24_000)?;
                println!(
                    "Kokoro end-to-end WAV: samples={}, seconds={:.3}, peak={:.6}, wav={}",
                    audio.len(),
                    audio.len() as f64 / 24_000.0,
                    audio.iter().map(|value| value.abs()).fold(0.0f32, f32::max),
                    path.display()
                );
            }
            println!(
                "Kokoro prosody CUDA: tokens={}, acoustic_frames={}, durations={:?}, aligned_sum={:.9}, aligned_sum_sq={:.9}, asr_sum={:.9}, asr_sum_sq={:.9}, f0_sum={:.9}, f0_sum_sq={:.9}, noise_sum={:.9}, noise_sum_sq={:.9}, decoder_sum={:.9}, decoder_sum_sq={:.9}",
                tokens.ids.len(),
                prosody.acoustic_frames,
                prosody.durations,
                prosody.aligned.iter().map(|x| f64::from(*x)).sum::<f64>(),
                prosody
                    .aligned
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                asr.iter().map(|x| f64::from(*x)).sum::<f64>(),
                asr.iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                f0.iter().map(|x| f64::from(*x)).sum::<f64>(),
                f0.iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                noise.iter().map(|x| f64::from(*x)).sum::<f64>(),
                noise
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                decoder.iter().map(|x| f64::from(*x)).sum::<f64>(),
                decoder
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
            );
        }
        Command::CudaKokoroAdaInSmoke {
            model_dir,
            gpu,
            frames,
            upsample,
        } => {
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 768 * 2_048, 2_048)?;
            let checkpoint = KokoroCheckpoint::open(model_dir.join("kokoro-v1_0.pth"))?;
            let (prefix, output) = if upsample {
                ("module.F0.1", 256)
            } else {
                ("module.F0.0", 512)
            };
            let input = (0..frames * 512)
                .map(|index| (index as f32 * 0.013).sin() * 0.2)
                .collect::<Vec<_>>();
            let style = (0..128)
                .map(|index| (index as f32 * 0.031).cos() * 0.1)
                .collect::<Vec<_>>();
            let result = KokoroAdaInResBlock::load(
                &checkpoint,
                "predictor",
                prefix,
                512,
                output,
                upsample,
                &stream,
            )?
            .forward(&input, frames, &style, &mut kernels)?;
            println!(
                "Kokoro AdaIN CUDA: frames={}, output_frames={}, channels={}, sum={:.9}, sum_sq={:.9}, first={:?}",
                frames,
                if upsample { frames * 2 } else { frames },
                output,
                result.iter().map(|x| f64::from(*x)).sum::<f64>(),
                result
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                &result[..8]
            );
        }
        Command::InspectChatterbox { model_dir } => {
            let inspection = inspect_chatterbox_model(&model_dir)
                .with_context(|| format!("could not inspect {}", model_dir.display()))?;
            println!("Chatterbox Multilingual V3: 30-layer Llama, hidden 1024, 16 heads");
            println!(
                "T3: {} tensors, text vocab 2454, speech vocab 8194 ({})",
                inspection.t3_tensors.len(),
                inspection.t3_path.display(),
            );
            println!(
                "S3Gen: {} tensors ({})",
                inspection.s3gen_tensor_count,
                inspection.s3gen_path.display(),
            );
            println!(
                "voice encoder: {} tensors ({})",
                inspection.voice_encoder_tensor_count,
                inspection.voice_encoder_path.display(),
            );
            println!(
                "weights: {:.2} GiB",
                inspection.total_weight_bytes as f64 / 1024.0_f64.powi(3),
            );
            let conditioning_path = model_dir.join("conds.pt");
            if conditioning_path.is_file() {
                let conditioning = ChatterboxConditioning::load(&conditioning_path)?;
                println!(
                    "conditioning: {} T3 prompt tokens, {} S3 prompt tokens, {} S3 feature frames",
                    conditioning.prompt_speech_tokens.len(),
                    conditioning.s3_prompt_tokens.len(),
                    conditioning.s3_prompt_feature_frames,
                );
            }
        }
        Command::TokenizeChatterbox {
            model_dir,
            language,
            text,
        } => {
            let tokenizer = ChatterboxTokenizer::load(&model_dir)?;
            println!("{:?}", tokenizer.encode(&text, &language)?);
        }
        Command::CudaChatterboxLayerSmoke {
            model_dir,
            gpu,
            layer,
            stack,
            seq_len,
            decode_split,
            compare_cache,
        } => {
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(
                gpu < allocator.gpu_count(),
                "GPU index {gpu} is out of range; detected {} device(s)",
                allocator.gpu_count()
            );
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(
                &stream,
                CHATTERBOX_HIDDEN_SIZE * CHATTERBOX_INTERMEDIATE_SIZE,
                CHATTERBOX_INTERMEDIATE_SIZE,
            )?;
            let hidden = (0..seq_len * CHATTERBOX_HIDDEN_SIZE)
                .map(|index| ((index as f32 * 0.013).sin() * 0.2) + 0.01)
                .collect::<Vec<_>>();
            let output = if stack {
                let transformer = ChatterboxT3Transformer::load(&model_dir, &stream)?;
                let max_batch = decode_split.unwrap_or(seq_len).max(1);
                let mut session = transformer.start_session(seq_len, max_batch, &stream)?;
                if let Some(split) = decode_split {
                    anyhow::ensure!(
                        split > 0 && split < seq_len,
                        "decode split must be inside 1..seq_len"
                    );
                    let mut output = transformer.forward_session(
                        &mut session,
                        &hidden[..split * CHATTERBOX_HIDDEN_SIZE],
                        split,
                        &mut kernels,
                    )?;
                    for token in split..seq_len {
                        output.extend(transformer.forward_session(
                            &mut session,
                            &hidden[token * CHATTERBOX_HIDDEN_SIZE
                                ..(token + 1) * CHATTERBOX_HIDDEN_SIZE],
                            1,
                            &mut kernels,
                        )?);
                    }
                    if compare_cache {
                        let mut full_session =
                            transformer.start_session(seq_len, seq_len, &stream)?;
                        let full = transformer.forward_session(
                            &mut full_session,
                            &hidden,
                            seq_len,
                            &mut kernels,
                        )?;
                        let mut max_delta = 0.0f32;
                        let mut sum_delta = 0.0f64;
                        for (cached, full) in output.iter().zip(&full) {
                            let delta = (cached - full).abs();
                            max_delta = max_delta.max(delta);
                            sum_delta += f64::from(delta);
                        }
                        println!(
                            "cache parity: mean_abs={:.9}, max_abs={max_delta:.9}",
                            sum_delta / output.len() as f64
                        );
                    }
                    output
                } else {
                    anyhow::ensure!(!compare_cache, "--compare-cache requires --decode-split");
                    transformer.forward_session(&mut session, &hidden, seq_len, &mut kernels)?
                }
            } else {
                anyhow::ensure!(
                    seq_len == 1 && decode_split.is_none() && !compare_cache,
                    "multi-token validation requires --stack"
                );
                ChatterboxT3Layer::load(&model_dir, layer, &stream)?
                    .forward_first_token(&hidden, &mut kernels)?
            };
            let sum = output.iter().map(|value| f64::from(*value)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>();
            println!(
                "Chatterbox T3 {} CUDA: sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                if stack { "stack" } else { "layer" },
                &output[..8]
            );
        }
        Command::CudaChatterboxS3LayerSmoke {
            model_dir,
            gpu,
            layer,
            upsampled,
            seq_len,
        } => {
            anyhow::ensure!(seq_len > 0, "sequence length must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(
                gpu < allocator.gpu_count(),
                "GPU index {gpu} is out of range; detected {} device(s)",
                allocator.gpu_count()
            );
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels =
                chew_kernel::GpuKernels::load(&stream, S3_HIDDEN_SIZE * 2_048, 2_048)?;
            let hidden = (0..seq_len * S3_HIDDEN_SIZE)
                .map(|index| ((index as f32 * 0.013).sin() * 0.2) + 0.01)
                .collect::<Vec<_>>();
            let output = ChatterboxS3ConformerLayer::load(&model_dir, upsampled, layer, &stream)?
                .forward(&hidden, seq_len, &mut kernels)?;
            let sum = output.iter().map(|value| f64::from(*value)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>();
            println!(
                "Chatterbox S3Gen layer CUDA: sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                &output[..8]
            );
        }
        Command::CudaChatterboxS3EncoderSmoke {
            model_dir,
            gpu,
            tokens,
        } => {
            anyhow::ensure!(tokens > 0, "token count must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels =
                chew_kernel::GpuKernels::load(&stream, S3_HIDDEN_SIZE * 2_048, 2_048)?;
            let model = ChatterboxS3Encoder::load(&model_dir, &stream)?;
            let ids = (0..tokens)
                .map(|index| ((index * 977 + 31) % 6_561) as i32)
                .collect::<Vec<_>>();
            let output = model.encode(&ids, &mut kernels)?;
            let sum = output.iter().map(|x| f64::from(*x)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>();
            println!(
                "Chatterbox S3Gen encoder CUDA: frames={}, sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                output.len() / 80,
                &output[..8]
            );
        }
        Command::CudaChatterboxFlowBlockSmoke {
            model_dir,
            gpu,
            prefix,
            seq_len,
        } => {
            anyhow::ensure!(seq_len > 0, "sequence length must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 256 * 1_024, 1_024)?;
            let input = (0..seq_len * 256)
                .map(|i| (i as f32 * 0.013).sin() * 0.2 + 0.01)
                .collect::<Vec<_>>();
            let output = ChatterboxFlowTransformerBlock::load(&model_dir, &prefix, &stream)?
                .forward(&input, seq_len, &mut kernels)?;
            let sum = output.iter().map(|x| f64::from(*x)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>();
            println!(
                "Chatterbox flow block CUDA: sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                &output[..8]
            );
        }
        Command::CudaChatterboxFlowResnetSmoke {
            model_dir,
            gpu,
            prefix,
            seq_len,
            input_channels,
        } => {
            anyhow::ensure!(seq_len > 0, "sequence length must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 512 * 1_024, 1_024)?;
            let input = (0..seq_len * input_channels)
                .map(|i| (i as f32 * 0.013).sin() * 0.2 + 0.01)
                .collect::<Vec<_>>();
            let time = (0..1_024)
                .map(|i| (i as f32 * 0.009).cos() * 0.15)
                .collect::<Vec<_>>();
            let output = ChatterboxFlowResnetBlock::load(&model_dir, &prefix, &stream)?.forward(
                &input,
                seq_len,
                &time,
                &mut kernels,
            )?;
            let sum = output.iter().map(|x| f64::from(*x)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>();
            println!(
                "Chatterbox flow ResNet CUDA: sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                &output[..8]
            );
        }
        Command::CudaChatterboxFlowTimeSmoke {
            model_dir,
            gpu,
            timestep,
        } => {
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 1_024 * 1_024, 1_024)?;
            let output = ChatterboxFlowTimeEmbedding::load(&model_dir, &stream)?
                .forward(timestep, &mut kernels)?;
            let sum = output.iter().map(|x| f64::from(*x)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>();
            println!(
                "Chatterbox flow time CUDA: sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                &output[..8]
            );
        }
        Command::CudaChatterboxFlowEstimatorSmoke {
            model_dir,
            gpu,
            frames,
            timestep,
        } => {
            anyhow::ensure!(frames > 0, "frame count must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(&stream, 1_024 * 1_024, 1_024)?;
            let input = (0..frames * 320)
                .map(|i| (i as f32 * 0.013).sin() * 0.2 + 0.01)
                .collect::<Vec<_>>();
            let output = ChatterboxFlowEstimator::load(&model_dir, &stream)?.forward(
                &input,
                frames,
                timestep,
                &mut kernels,
            )?;
            let sum = output.iter().map(|x| f64::from(*x)).sum::<f64>();
            let sum_sq = output
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>();
            println!(
                "Chatterbox flow estimator CUDA: sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                &output[..8]
            );
        }
        Command::CudaChatterboxFlowSmoke {
            model_dir,
            gpu,
            steps,
            generated_tokens,
        } => {
            anyhow::ensure!(
                generated_tokens > 0,
                "generated token count must be non-zero"
            );
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels =
                chew_kernel::GpuKernels::load(&stream, S3_HIDDEN_SIZE * 2_048, 2_048)?;
            let conditioning = ChatterboxConditioning::load(&model_dir.join("conds.pt"))?;
            let tokens = (0..generated_tokens)
                .map(|index| ((index * 977 + 31) % 6_561) as i32)
                .collect::<Vec<_>>();
            let mel = ChatterboxS3Flow::load(&model_dir, &stream)?.generate_mel(
                &tokens,
                &conditioning,
                steps,
                42,
                &mut kernels,
            )?;
            let sum = mel.iter().map(|x| f64::from(*x)).sum::<f64>();
            let sum_sq = mel
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>();
            println!(
                "Chatterbox CFM CUDA: frames={}, sum={sum:.9}, sum_sq={sum_sq:.9}, first={:?}",
                mel.len() / 80,
                &mel[..8]
            );
        }
        Command::CudaChatterboxF0Smoke {
            model_dir,
            gpu,
            frames,
        } => {
            anyhow::ensure!(frames > 0, "frame count must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels =
                chew_kernel::GpuKernels::load(&stream, S3_HIDDEN_SIZE * 2_048, 2_048)?;
            let mel = (0..frames * 80)
                .map(|index| (index as f32 * 0.017).sin() * 0.25)
                .collect::<Vec<_>>();
            let f0 = ChatterboxF0Predictor::load(&model_dir, &stream)?.predict(
                &mel,
                frames,
                &mut kernels,
            )?;
            println!(
                "Chatterbox HiFT F0 CUDA: frames={}, sum={:.9}, sum_sq={:.9}, first={:?}",
                frames,
                f0.iter().map(|x| f64::from(*x)).sum::<f64>(),
                f0.iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>(),
                &f0[..f0.len().min(8)]
            );
        }
        Command::CudaChatterboxHiFtSmoke {
            model_dir,
            gpu,
            frames,
            wav,
        } => {
            anyhow::ensure!(frames > 0, "frame count must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(gpu < allocator.gpu_count(), "GPU index out of range");
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels =
                chew_kernel::GpuKernels::load(&stream, S3_HIDDEN_SIZE * 2_048, 2_048)?;
            let mel = (0..frames * 80)
                .map(|index| (index as f32 * 0.017).sin() * 0.25)
                .collect::<Vec<_>>();
            let audio = ChatterboxHiFT::load(&model_dir, &stream)?.synthesize(
                &mel,
                frames,
                42,
                &mut kernels,
            )?;
            write_pcm16_wav(&wav, &audio, 24_000)?;
            println!(
                "Chatterbox HiFT CUDA: frames={}, samples={}, peak={:.6}, rms={:.6}, wav={}",
                frames,
                audio.len(),
                audio.iter().map(|value| value.abs()).fold(0.0f32, f32::max),
                (audio.iter().map(|value| value * value).sum::<f32>() / audio.len() as f32).sqrt(),
                wav.display()
            );
        }
        Command::CudaChatterboxGenerationSmoke {
            model_dir,
            gpu,
            language,
            text,
            max_tokens,
            cfg_weight,
            wav,
            flow_steps,
        } => {
            anyhow::ensure!(max_tokens > 0, "max tokens must be non-zero");
            let allocator = chew_vram::VramAllocator::init()?;
            anyhow::ensure!(
                gpu < allocator.gpu_count(),
                "GPU index {gpu} is out of range; detected {} device(s)",
                allocator.gpu_count()
            );
            let stream = std::sync::Arc::clone(allocator.stream(gpu));
            let mut kernels = chew_kernel::GpuKernels::load(
                &stream,
                CHATTERBOX_HIDDEN_SIZE * CHATTERBOX_INTERMEDIATE_SIZE,
                CHATTERBOX_INTERMEDIATE_SIZE,
            )?;
            let tokenizer = ChatterboxTokenizer::load(&model_dir)?;
            let text_tokens = tokenizer.encode(&text, &language)?;
            let conditioning = ChatterboxConditioning::load(&model_dir.join("conds.pt"))?;
            let frontend = ChatterboxT3Frontend::load(&model_dir, &stream)?;
            let prefix = frontend.build_prefix(&text_tokens, &conditioning, &mut kernels)?;
            let transformer = ChatterboxT3Transformer::load(&model_dir, &stream)?;
            let capacity = prefix.tokens + max_tokens;
            let mut conditional = transformer.start_session(capacity, prefix.tokens, &stream)?;
            let mut unconditional = transformer.start_session(capacity, prefix.tokens, &stream)?;
            let mut conditional_hidden = transformer.forward_session(
                &mut conditional,
                &prefix.conditional,
                prefix.tokens,
                &mut kernels,
            )?;
            let mut unconditional_hidden = transformer.forward_session(
                &mut unconditional,
                &prefix.unconditional,
                prefix.tokens,
                &mut kernels,
            )?;
            let mut generated = Vec::new();
            for position in 0..max_tokens {
                let conditional_logits = frontend.speech_logits(
                    &conditional_hidden[conditional_hidden.len() - CHATTERBOX_HIDDEN_SIZE..],
                    &mut kernels,
                )?;
                let unconditional_logits = frontend.speech_logits(
                    &unconditional_hidden[unconditional_hidden.len() - CHATTERBOX_HIDDEN_SIZE..],
                    &mut kernels,
                )?;
                let token = conditional_logits
                    .iter()
                    .zip(&unconditional_logits)
                    .enumerate()
                    .max_by(
                        |(_, (cond_left, uncond_left)), (_, (cond_right, uncond_right))| {
                            let left = cond_left.to_f32()
                                + cfg_weight * (cond_left.to_f32() - uncond_left.to_f32());
                            let right = cond_right.to_f32()
                                + cfg_weight * (cond_right.to_f32() - uncond_right.to_f32());
                            left.total_cmp(&right)
                        },
                    )
                    .map(|(token, _)| token as i32)
                    .context("Chatterbox speech head returned no logits")?;
                generated.push(token);
                if token == 6_562 {
                    break;
                }
                let embedding = frontend.speech_embedding(token, position + 1, &mut kernels)?;
                conditional_hidden =
                    transformer.forward_session(&mut conditional, &embedding, 1, &mut kernels)?;
                unconditional_hidden =
                    transformer.forward_session(&mut unconditional, &embedding, 1, &mut kernels)?;
            }
            println!(
                "Chatterbox T3 generated {} token(s) from {} text tokens and {} prefix tokens: {:?}",
                generated.len(),
                text_tokens.len(),
                prefix.tokens,
                generated,
            );
            if let Some(wav) = wav {
                let generated = generated
                    .into_iter()
                    .take_while(|token| *token != 6_562)
                    .collect::<Vec<_>>();
                anyhow::ensure!(!generated.is_empty(), "T3 produced no speech tokens");
                drop(conditional_hidden);
                drop(unconditional_hidden);
                drop(conditional);
                drop(unconditional);
                drop(transformer);
                drop(frontend);
                let started = std::time::Instant::now();
                let mel = ChatterboxS3Flow::load(&model_dir, &stream)?.generate_mel(
                    &generated,
                    &conditioning,
                    flow_steps,
                    42,
                    &mut kernels,
                )?;
                let mel_frames = mel.len() / 80;
                let audio = ChatterboxHiFT::load(&model_dir, &stream)?.synthesize(
                    &mel,
                    mel_frames,
                    43,
                    &mut kernels,
                )?;
                write_pcm16_wav(&wav, &audio, 24_000)?;
                println!(
                    "Chatterbox end-to-end: mel_frames={}, samples={}, audio_seconds={:.3}, post-T3_seconds={:.3}, wav={}",
                    mel_frames,
                    audio.len(),
                    audio.len() as f64 / 24_000.0,
                    started.elapsed().as_secs_f64(),
                    wav.display()
                );
            }
        }
        Command::Tokenize { model_dir, text } => {
            let tokenizer = load_qwen_tokenizer(&model_dir)?;
            let encoded = tokenizer
                .encode(text.as_str(), false)
                .map_err(|error| anyhow::anyhow!("could not encode text: {error}"))?;
            println!("{:?}", encoded.get_ids());
        }
        Command::CudaSmoke { gpu } => {
            let allocator = chew_vram::VramAllocator::init()?;
            if gpu >= allocator.gpu_count() {
                anyhow::bail!(
                    "GPU index {gpu} is out of range; detected {} device(s)",
                    allocator.gpu_count()
                );
            }
            let free_before = allocator.free_bytes(gpu)?;
            let stream = allocator.stream(gpu);
            let _kernels = chew_kernel::GpuKernels::load(stream, 1024 * 1024, 6 * 1024)
                .context("could not compile and load CUDA kernels")?;
            stream.synchronize()?;
            let free_after = allocator.free_bytes(gpu)?;
            println!(
                "CUDA device {gpu} ready: {:.1} MiB free, {:.1} MiB kernel/runtime allocation",
                free_after as f64 / 1024.0_f64.powi(2),
                free_before.saturating_sub(free_after) as f64 / 1024.0_f64.powi(2),
            );
        }
        Command::CudaLinearSmoke {
            model_dir,
            gpu,
            tensor,
            gemv,
        } => cuda_linear_smoke(&model_dir, gpu, &tensor, gemv)?,
        Command::CudaLayerSmoke {
            model_dir,
            gpu,
            layer,
            seq_len,
            decode_split,
            reference,
        } => cuda_layer_smoke(
            &model_dir,
            gpu,
            layer,
            seq_len,
            decode_split,
            reference.as_deref(),
        )?,
        Command::CudaTalkerSmoke {
            model_dir,
            gpu,
            seq_len,
            decode_split,
            reference,
        } => cuda_talker_smoke(&model_dir, gpu, seq_len, decode_split, reference.as_deref())?,
        Command::CudaPredictorSmoke {
            model_dir,
            gpu,
            seq_len,
            frame,
            semantic_token,
            repeats,
            reference,
        } => cuda_predictor_smoke(
            &model_dir,
            gpu,
            seq_len,
            frame,
            semantic_token,
            repeats,
            reference.as_deref(),
        )?,
        Command::CudaCodecLatentSmoke {
            tokenizer_dir,
            gpu,
            codes,
            preconv,
            transformer,
            upsample,
            audio,
            frames,
            stream_split,
            repeats,
            wav,
            reference,
        } => cuda_codec_latent_smoke(
            &tokenizer_dir,
            gpu,
            &codes,
            preconv,
            transformer,
            upsample,
            audio,
            frames,
            stream_split,
            repeats,
            wav.as_deref(),
            reference.as_deref(),
        )?,
        Command::CudaPredictorCodecSmoke {
            model_dir,
            gpu,
            semantic_token,
            frames,
            wav,
        } => cuda_predictor_codec_smoke(&model_dir, gpu, semantic_token, frames, &wav)?,
        Command::CudaTalkerFrontendSmoke {
            model_dir,
            gpu,
            text_ids,
            text,
            reference,
        } => cuda_talker_frontend_smoke(
            &model_dir,
            gpu,
            &text_ids,
            text.as_deref(),
            reference.as_deref(),
        )?,
        Command::CudaSpeakerSmoke {
            model_dir,
            wav,
            gpu,
            output,
            mel_output,
            reference,
        } => cuda_speaker_smoke(
            &model_dir,
            &wav,
            gpu,
            output.as_deref(),
            mel_output.as_deref(),
            reference.as_deref(),
        )?,
        Command::CudaCodecEncoderSmoke {
            model_dir,
            wav,
            gpu,
            output,
            reference,
        } => cuda_codec_encoder_smoke(
            &model_dir,
            &wav,
            gpu,
            output.as_deref(),
            reference.as_deref(),
        )?,
        Command::CudaVoiceDesignSmoke {
            model_dir,
            gpu,
            text,
            instruction,
            language,
            max_frames,
            seed,
            temperature,
            top_k,
            chunk_frames,
            chunk_context,
            wav,
        } => cuda_voice_design_smoke(
            &model_dir,
            gpu,
            &text,
            &instruction,
            &language,
            max_frames,
            seed,
            temperature,
            top_k,
            chunk_frames,
            chunk_context,
            &wav,
        )?,
    }
    Ok(())
}

fn cuda_codec_encoder_smoke(
    model_dir: &std::path::Path,
    wav: &std::path::Path,
    gpu: usize,
    output: Option<&std::path::Path>,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let bytes = std::fs::read(wav).with_context(|| format!("could not read {}", wav.display()))?;
    let (samples, sample_rate) = audio_input::decode_wav(&bytes)?;
    let samples = audio_input::resample(&samples, sample_rate, 24_000);
    let allocator = chew_vram::VramAllocator::init()?;
    anyhow::ensure!(
        gpu < allocator.gpu_count(),
        "GPU index {gpu} is out of range; detected {} device(s)",
        allocator.gpu_count()
    );
    let stream = allocator.stream(gpu);
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    let max_matrix = (config.intermediate_size * config.hidden_size)
        .max(config.text_hidden_size * config.text_hidden_size);
    let max_vector = config.intermediate_size.max(config.text_hidden_size);
    let mut kernels = chew_kernel::GpuKernels::load(stream, max_matrix, max_vector)?;
    let encoder = CodecEncoder::load(&model_dir.join("speech_tokenizer"), stream)?;
    let started = std::time::Instant::now();
    let frames = encoder.encode(&samples, &mut kernels)?;
    let elapsed = started.elapsed();
    let flat = frames.iter().flatten().copied().collect::<Vec<_>>();
    if let Some(output) = output {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for value in &flat {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(output, bytes)
            .with_context(|| format!("could not write {}", output.display()))?;
    }
    if let Some(reference) = reference {
        let bytes = std::fs::read(reference)
            .with_context(|| format!("could not read {}", reference.display()))?;
        anyhow::ensure!(
            bytes.len() % 4 == 0,
            "codec reference length is not divisible by four"
        );
        let expected = bytes
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            expected.len() == flat.len(),
            "codec reference has {} IDs, native encoder produced {}",
            expected.len(),
            flat.len()
        );
        let matches = flat
            .iter()
            .zip(&expected)
            .filter(|(left, right)| left == right)
            .count();
        println!(
            "codec parity: {matches}/{} IDs ({:.3}%)",
            flat.len(),
            matches as f64 * 100.0 / flat.len() as f64
        );
        anyhow::ensure!(matches == flat.len(), "codec ID parity failed");
    }
    println!(
        "codec encoder: {} frame(s), {:.3}s",
        frames.len(),
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn cuda_speaker_smoke(
    model_dir: &std::path::Path,
    wav: &std::path::Path,
    gpu: usize,
    output: Option<&std::path::Path>,
    mel_output: Option<&std::path::Path>,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let bytes = std::fs::read(wav).with_context(|| format!("could not read {}", wav.display()))?;
    let (samples, sample_rate) = audio_input::decode_wav(&bytes)?;
    let samples = audio_input::resample(&samples, sample_rate, 24_000);
    let (mel, frames) = audio_input::speaker_mel(&samples)?;
    if let Some(output) = mel_output {
        let mut bytes = Vec::with_capacity(mel.len() * 4);
        for value in &mel {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(output, bytes)
            .with_context(|| format!("could not write {}", output.display()))?;
    }
    let allocator = chew_vram::VramAllocator::init()?;
    anyhow::ensure!(
        gpu < allocator.gpu_count(),
        "GPU index {gpu} is out of range; detected {} device(s)",
        allocator.gpu_count()
    );
    let stream = allocator.stream(gpu);
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    let max_matrix = (config.intermediate_size * config.hidden_size)
        .max(config.text_hidden_size * config.text_hidden_size);
    let max_vector = config.intermediate_size.max(config.text_hidden_size);
    let mut kernels = chew_kernel::GpuKernels::load(stream, max_matrix, max_vector)?;
    let encoder = SpeakerEncoder::load(model_dir, stream)?;
    let started = std::time::Instant::now();
    let embedding = encoder.encode_mel(&mel, frames, &mut kernels)?;
    let elapsed = started.elapsed();
    if let Some(output) = output {
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for value in &embedding {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(output, bytes)
            .with_context(|| format!("could not write {}", output.display()))?;
    }
    compare_reference(&embedding, reference, 0.08)?;
    println!(
        "speaker embedding: {} mel frame(s), {} values, {:.3}s",
        frames,
        embedding.len(),
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn load_qwen_tokenizer(model_dir: &std::path::Path) -> anyhow::Result<Tokenizer> {
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

fn cuda_talker_frontend_smoke(
    model_dir: &std::path::Path,
    gpu: usize,
    text_ids: &str,
    text: Option<&str>,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    let text_ids = if let Some(text) = text {
        load_qwen_tokenizer(model_dir)?
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("could not encode text: {error}"))?
            .get_ids()
            .iter()
            .map(|id| *id as i32)
            .collect()
    } else {
        text_ids
            .split(',')
            .map(|value| value.trim().parse::<i32>())
            .collect::<Result<Vec<_>, _>>()
            .context("text IDs must be comma-separated integers")?
    };
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(
        stream,
        config.text_hidden_size * config.text_hidden_size,
        config.text_hidden_size,
    )?;
    let load_started = std::time::Instant::now();
    let frontend = TalkerFrontend::load(model_dir, config, stream)?;
    let free_loaded = allocator.free_bytes(gpu)?;
    let started = std::time::Instant::now();
    let projected = frontend.project_text_tokens(&text_ids, &mut kernels)?;
    let codec = frontend.codec_embeddings(&[0, 42, 2150], &mut kernels)?;
    let semantic = frontend.semantic_argmax(
        &projected[projected.len() - config.hidden_size..],
        &mut kernels,
    )?;
    println!(
        "talker frontend: {:.1} MiB VRAM, load {:.3}s, forward {:.3}ms",
        free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2),
        load_started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() * 1000.0,
    );
    println!(
        "text projected[0..8]={:?}, codec[0..8]={:?}, semantic argmax={semantic}",
        &projected[..8],
        &codec[..8],
    );
    println!("text IDs: {text_ids:?}");
    compare_reference(&projected, reference, 0.05)?;
    Ok(())
}

fn cuda_predictor_codec_smoke(
    model_dir: &std::path::Path,
    gpu: usize,
    semantic_token: i32,
    frames: usize,
    wav: &std::path::Path,
) -> anyhow::Result<()> {
    if frames == 0 {
        anyhow::bail!("frames must be non-zero");
    }
    if let Some(parent) = wav.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        if !parent.is_dir() {
            anyhow::bail!("WAV output directory does not exist: {}", parent.display());
        }
    }
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config.code_predictor_config;
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(
        stream,
        config.intermediate_size * config.hidden_size,
        config.intermediate_size,
    )?;
    let predictor = CodePredictorTransformer::load(model_dir, config, stream)?;
    let codec = CodecQuantizer::load(model_dir.join("speech_tokenizer"), stream)?;
    let free_loaded = allocator.free_bytes(gpu)?;

    let started = std::time::Instant::now();
    let mut codec_frames = Vec::with_capacity(frames);
    for frame in 0..frames {
        let frame_started = std::time::Instant::now();
        let talker_hidden = (0..inspection.config.talker_config.hidden_size)
            .map(|index| {
                let position = index + frame * inspection.config.talker_config.hidden_size;
                ((position as f32 + 1.0) * 0.013).sin() * 0.125
            })
            .collect::<Vec<_>>();
        let acoustic = predictor.generate_acoustic_codes_argmax(
            &talker_hidden,
            semantic_token,
            &mut kernels,
        )?;
        let mut codes = Vec::with_capacity(16);
        codes.push(semantic_token);
        codes.extend(acoustic);
        println!(
            "frame {frame}: {codes:?} ({:.3}ms)",
            frame_started.elapsed().as_secs_f64() * 1000.0
        );
        codec_frames.push(codes);
    }
    let predictor_elapsed = started.elapsed();
    let decode_started = std::time::Instant::now();
    let audio = codec.decode_frames_audio(&codec_frames, &mut kernels)?;
    let decode_elapsed = decode_started.elapsed();
    write_pcm16_wav(wav, &audio, 24_000)?;
    println!(
        "{} frame(s), {:.3}s audio: predictor {:.3}ms, codec {:.3}ms, {:.1} MiB VRAM, {}",
        frames,
        audio.len() as f64 / 24_000.0,
        predictor_elapsed.as_secs_f64() * 1000.0,
        decode_elapsed.as_secs_f64() * 1000.0,
        free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2),
        wav.display()
    );
    Ok(())
}

fn cuda_voice_design_smoke(
    model_dir: &std::path::Path,
    gpu: usize,
    text: &str,
    instruction: &str,
    language: &str,
    max_frames: usize,
    mut seed: u64,
    temperature: f32,
    top_k: usize,
    chunk_frames: usize,
    chunk_context: usize,
    wav: &std::path::Path,
) -> anyhow::Result<()> {
    if max_frames == 0 {
        anyhow::bail!("max-frames must be non-zero");
    }
    if let Some(parent) = wav.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        if !parent.is_dir() {
            anyhow::bail!("WAV output directory does not exist: {}", parent.display());
        }
    }
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    let language_key = language.to_ascii_lowercase();
    let language_codec_id = config
        .codec_language_id
        .get(&language_key)
        .copied()
        .with_context(|| {
            let mut supported = config.codec_language_id.keys().cloned().collect::<Vec<_>>();
            supported.sort();
            format!("unsupported language {language:?}; supported: {supported:?}")
        })? as i32;

    let tokenizer = load_qwen_tokenizer(model_dir)?;
    let encode = |value: &str| -> anyhow::Result<Vec<i32>> {
        Ok(tokenizer
            .encode(value, false)
            .map_err(|error| anyhow::anyhow!("could not encode text: {error}"))?
            .get_ids()
            .iter()
            .map(|id| *id as i32)
            .collect())
    };
    let text_ids = encode(text)?;
    let instruction_ids = encode(&format!("<|im_start|>user\n{instruction}<|im_end|>\n"))?;

    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let max_matrix = (config.intermediate_size * config.hidden_size)
        .max(config.text_hidden_size * config.text_hidden_size);
    let max_vector = config.intermediate_size.max(config.text_hidden_size);
    let mut kernels = chew_kernel::GpuKernels::load(stream, max_matrix, max_vector)?;

    let load_started = std::time::Instant::now();
    let talker = TalkerTransformer::load(model_dir, config, stream)?;
    let frontend = TalkerFrontend::load(model_dir, config, stream)?;
    let predictor =
        CodePredictorTransformer::load(model_dir, &config.code_predictor_config, stream)?;
    let mut predictor_session = predictor.start_generation_session(stream)?;
    let mut semantic_session = frontend.start_semantic_sampling_session(max_frames, stream)?;
    let codec = CodecQuantizer::load(model_dir.join("speech_tokenizer"), stream)?;
    let load_elapsed = load_started.elapsed();
    let free_loaded = allocator.free_bytes(gpu)?;

    let prompt_started = std::time::Instant::now();
    let inputs = frontend.build_voice_design_inputs(
        &text_ids,
        &instruction_ids,
        language_codec_id,
        &mut kernels,
    )?;
    let max_seq_len = inputs.prefill_tokens + max_frames;
    if max_seq_len > config.max_position_embeddings {
        anyhow::bail!(
            "prompt plus generation is {max_seq_len} tokens, model limit is {}",
            config.max_position_embeddings
        );
    }
    let mut session = talker.start_session(max_seq_len, inputs.prefill_tokens, config, stream)?;
    let normalized = talker.forward_session(
        &mut session,
        &inputs.prefill,
        inputs.prefill_tokens,
        config,
        &mut kernels,
    )?;
    let mut last_hidden = normalized[normalized.len() - config.hidden_size..].to_vec();
    let mut generated_semantics = Vec::with_capacity(max_frames);
    let mut semantic = frontend.semantic_speech_sample_with_session(
        &mut semantic_session,
        &last_hidden,
        &generated_semantics,
        temperature,
        top_k,
        1.05,
        &mut seed,
        &mut kernels,
    )?;
    generated_semantics.push(semantic);
    let prompt_elapsed = prompt_started.elapsed();

    let generation_started = std::time::Instant::now();
    let mut codec_frames = Vec::with_capacity(if chunk_frames == 0 {
        max_frames.min(1024)
    } else {
        chunk_frames
    });
    let mut codec_context = Vec::with_capacity(1024 * chunk_context);
    let mut codec_session = if chunk_frames > 0 {
        Some(codec.start_transformer_session(max_frames, stream)?)
    } else {
        None
    };
    let mut wav_writer = if chunk_frames > 0 {
        Some(Pcm16WavWriter::create(wav, 24_000)?)
    } else {
        None
    };
    let mut codec_elapsed = std::time::Duration::ZERO;
    let mut generated_frames = 0usize;
    for frame_index in 0..max_frames {
        if semantic == 2_150 {
            println!("codec EOS at frame {frame_index}");
            break;
        }
        let frame_started = std::time::Instant::now();
        let acoustic = predictor.generate_acoustic_codes_sampled_with_session(
            &mut predictor_session,
            &last_hidden,
            semantic,
            temperature,
            top_k,
            &mut seed,
            &mut kernels,
        )?;
        let mut codes = Vec::with_capacity(config.num_code_groups);
        codes.push(semantic);
        codes.extend_from_slice(&acoustic);
        codec_frames.push(codes);
        generated_frames += 1;
        println!(
            "frame {frame_index}: semantic {semantic}, acoustics {:?}, {:.3}ms",
            &acoustic[..3.min(acoustic.len())],
            frame_started.elapsed().as_secs_f64() * 1000.0
        );
        if chunk_frames > 0 && codec_frames.len() >= chunk_frames {
            codec_elapsed += decode_codec_chunk(
                &codec,
                &mut codec_frames,
                &mut codec_context,
                codec_session
                    .as_mut()
                    .expect("chunked decoding has a codec transformer session"),
                chunk_context,
                wav_writer
                    .as_mut()
                    .expect("chunked decoding has a WAV writer"),
                &mut kernels,
            )?;
        }

        let semantic_embedding = frontend.codec_embeddings(&[semantic], &mut kernels)?;
        let acoustic_embedding =
            predictor.acoustic_embeddings_sum_with_session(&mut predictor_session, &mut kernels)?;
        let text_embedding = if frame_index < inputs.trailing_tokens {
            let start = frame_index * config.hidden_size;
            &inputs.trailing_text[start..start + config.hidden_size]
        } else {
            &inputs.text_pad
        };
        let next_input = semantic_embedding
            .iter()
            .zip(acoustic_embedding)
            .zip(text_embedding)
            .map(|((semantic, acoustic), text)| semantic + acoustic + text)
            .collect::<Vec<_>>();
        last_hidden = talker.forward_session(&mut session, &next_input, 1, config, &mut kernels)?;
        semantic = frontend.semantic_speech_sample_with_session(
            &mut semantic_session,
            &last_hidden,
            &generated_semantics,
            temperature,
            top_k,
            1.05,
            &mut seed,
            &mut kernels,
        )?;
        generated_semantics.push(semantic);
    }
    let generation_elapsed = generation_started.elapsed().saturating_sub(codec_elapsed);
    if generated_frames == 0 {
        anyhow::bail!("model emitted EOS before producing audio");
    }
    let truncated = semantic != 2_150 && generated_frames == max_frames;

    let decode_started = std::time::Instant::now();
    let audio_samples = if chunk_frames == 0 {
        let audio = codec.decode_frames_audio(&codec_frames, &mut kernels)?;
        write_pcm16_wav(wav, &audio, 24_000)?;
        audio.len()
    } else {
        if !codec_frames.is_empty() {
            codec_elapsed += decode_codec_chunk(
                &codec,
                &mut codec_frames,
                &mut codec_context,
                codec_session
                    .as_mut()
                    .expect("chunked decoding has a codec transformer session"),
                chunk_context,
                wav_writer
                    .as_mut()
                    .expect("chunked decoding has a WAV writer"),
                &mut kernels,
            )?;
        }
        wav_writer
            .take()
            .expect("chunked decoding has a WAV writer")
            .finish()?
    };
    let decode_elapsed = if chunk_frames == 0 {
        decode_started.elapsed()
    } else {
        codec_elapsed
    };
    let audio_seconds = audio_samples as f64 / 24_000.0;
    let inference_seconds = prompt_elapsed.as_secs_f64()
        + generation_elapsed.as_secs_f64()
        + decode_elapsed.as_secs_f64();
    println!(
        "VoiceDesign: {} frame(s), {:.3}s audio, {:.3}s inference, RTF {:.3}",
        generated_frames,
        audio_seconds,
        inference_seconds,
        inference_seconds / audio_seconds,
    );
    println!(
        "load {:.3}s, prompt {:.3}s, generation {:.3}s, codec {:.3}s, {:.1} MiB VRAM, {}",
        load_elapsed.as_secs_f64(),
        prompt_elapsed.as_secs_f64(),
        generation_elapsed.as_secs_f64(),
        decode_elapsed.as_secs_f64(),
        free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2),
        wav.display(),
    );
    if truncated {
        anyhow::bail!(
            "generation reached the {max_frames}-frame safety limit before EOS; \
             {} contains truncated audio",
            wav.display()
        );
    }
    Ok(())
}

fn decode_codec_chunk(
    codec: &CodecQuantizer,
    pending: &mut Vec<Vec<i32>>,
    context: &mut Vec<f32>,
    session: &mut CodecTransformerSession,
    context_frames: usize,
    output: &mut Pcm16WavWriter,
    kernels: &mut chew_kernel::GpuKernels,
) -> anyhow::Result<std::time::Duration> {
    if pending.is_empty() {
        return Ok(std::time::Duration::ZERO);
    }
    const SAMPLES_PER_FRAME: usize = 1920;
    const CHANNELS: usize = 1024;
    let prefix_frames = context.len() / CHANNELS;
    let new_frames = pending.len();

    let started = std::time::Instant::now();
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
    let elapsed = started.elapsed();
    output.write_samples(&decoded[prefix_frames * SAMPLES_PER_FRAME..])?;

    let keep = context_frames.min(total_frames);
    let first_kept = total_frames - keep;
    context.resize(CHANNELS * keep, 0.0);
    for channel in 0..CHANNELS {
        let source = channel * total_frames + first_kept;
        let destination = channel * keep;
        context[destination..destination + keep]
            .copy_from_slice(&decode_frames[source..source + keep]);
    }
    Ok(elapsed)
}

fn cuda_codec_latent_smoke(
    tokenizer_dir: &PathBuf,
    gpu: usize,
    codes: &str,
    preconv: bool,
    transformer: bool,
    upsample: bool,
    audio: bool,
    frames: usize,
    stream_split: Option<usize>,
    repeats: usize,
    wav: Option<&std::path::Path>,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    if frames == 0 {
        anyhow::bail!("frames must be non-zero");
    }
    if repeats == 0 {
        anyhow::bail!("repeats must be non-zero");
    }
    if wav.is_some() && !audio {
        anyhow::bail!("--wav requires --audio");
    }
    let codes = codes
        .split(',')
        .map(|value| value.trim().parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .context("codec IDs must be comma-separated integers")?;
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(stream, 512 * 256, 512)?;
    let quantizer = CodecQuantizer::load(tokenizer_dir, stream)?;
    let free_loaded = allocator.free_bytes(gpu)?;
    let codec_frames = vec![codes.clone(); frames];
    if let Some(split) = stream_split {
        if split == 0 || split >= frames {
            anyhow::bail!("stream-split must be between 1 and frames - 1");
        }
        let full = quantizer.decode_frames_transformer(&codec_frames, &mut kernels)?;
        let mut session = quantizer.start_transformer_session(frames, stream)?;
        let first = quantizer.decode_frames_transformer_session(
            &codec_frames[..split],
            &mut session,
            &mut kernels,
        )?;
        let second = quantizer.decode_frames_transformer_session(
            &codec_frames[split..],
            &mut session,
            &mut kernels,
        )?;
        let mut streamed = vec![0.0f32; full.len()];
        for channel in 0..1024 {
            streamed[channel * frames..channel * frames + split]
                .copy_from_slice(&first[channel * split..(channel + 1) * split]);
            streamed[channel * frames + split..(channel + 1) * frames].copy_from_slice(
                &second[channel * (frames - split)..(channel + 1) * (frames - split)],
            );
        }
        let (max, sum) =
            full.iter()
                .zip(&streamed)
                .fold((0.0f32, 0.0f64), |(max, sum), (full, streamed)| {
                    let delta = (full - streamed).abs();
                    (max.max(delta), sum + f64::from(delta))
                });
        let mean = sum / full.len() as f64;
        if max > 0.1 {
            anyhow::bail!("streamed codec transformer delta {max:.7} exceeds 0.1 (mean {mean:.7})");
        }
        println!(
            "streamed codec transformer: split {split}/{frames}, \
             max delta {max:.7}, mean {mean:.7}"
        );
    }
    if repeats > 1 {
        if audio {
            if frames == 1 {
                quantizer.decode_frame_audio(&codes, &mut kernels)?;
            } else {
                quantizer.decode_frames_audio(&codec_frames, &mut kernels)?;
            }
        } else if upsample {
            if frames == 1 {
                quantizer.decode_frame_upsampled(&codes, &mut kernels)?;
            } else {
                quantizer.decode_frames_upsampled(&codec_frames, &mut kernels)?;
            }
        } else if transformer {
            if frames == 1 {
                quantizer.decode_frame_transformer(&codes, &mut kernels)?;
            } else {
                quantizer.decode_frames_transformer(&codec_frames, &mut kernels)?;
            }
        } else if preconv {
            quantizer.decode_frame_preconv(&codes, &mut kernels)?;
        } else {
            quantizer.decode_frame(&codes, &mut kernels)?;
        }
    }
    let started = std::time::Instant::now();
    let mut latent = Vec::new();
    for _ in 0..repeats {
        latent = if audio {
            if frames == 1 {
                quantizer.decode_frame_audio(&codes, &mut kernels)?
            } else {
                quantizer.decode_frames_audio(&codec_frames, &mut kernels)?
            }
        } else if upsample {
            if frames == 1 {
                quantizer.decode_frame_upsampled(&codes, &mut kernels)?
            } else {
                quantizer.decode_frames_upsampled(&codec_frames, &mut kernels)?
            }
        } else if transformer {
            if frames == 1 {
                quantizer.decode_frame_transformer(&codes, &mut kernels)?
            } else {
                quantizer.decode_frames_transformer(&codec_frames, &mut kernels)?
            }
        } else if preconv {
            quantizer.decode_frame_preconv(&codes, &mut kernels)?
        } else {
            quantizer.decode_frame(&codes, &mut kernels)?
        };
    }
    let stage = if audio {
        "audio"
    } else if upsample {
        "upsampled"
    } else if transformer {
        "transformer"
    } else if preconv {
        "pre-conv"
    } else {
        "quantizer"
    };
    println!(
        "codec {}: {:.1} MiB VRAM, {:.3}ms/run ({} run(s)), output[0..8]={:?}",
        stage,
        free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2),
        started.elapsed().as_secs_f64() * 1000.0 / repeats as f64,
        repeats,
        &latent[..8]
    );
    if let Some(path) = wav {
        write_pcm16_wav(path, &latent, 24_000)?;
        println!("wrote {} samples to {}", latent.len(), path.display());
    }
    compare_reference(
        &latent,
        reference,
        if audio || transformer || upsample {
            0.1
        } else {
            0.02
        },
    )?;
    Ok(())
}

fn write_pcm16_wav(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> anyhow::Result<()> {
    let mut writer = Pcm16WavWriter::create(path, sample_rate)?;
    writer.write_samples(samples)?;
    writer.finish().map(|_| ())
}

struct Pcm16WavWriter {
    file: std::fs::File,
    samples: usize,
}

impl Pcm16WavWriter {
    fn create(path: &std::path::Path, sample_rate: u32) -> anyhow::Result<Self> {
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("could not create WAV {}", path.display()))?;
        file.write_all(b"RIFF")?;
        file.write_all(&36u32.to_le_bytes())?;
        file.write_all(b"WAVEfmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&(sample_rate * 2).to_le_bytes())?;
        file.write_all(&2u16.to_le_bytes())?;
        file.write_all(&16u16.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&0u32.to_le_bytes())?;
        Ok(Self { file, samples: 0 })
    }

    fn write_samples(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        let new_samples = self
            .samples
            .checked_add(samples.len())
            .context("WAV sample count overflow")?;
        let data_bytes = new_samples
            .checked_mul(2)
            .context("WAV data is too large")?;
        u32::try_from(data_bytes).context("WAV data exceeds 4 GiB")?;

        let mut pcm = Vec::with_capacity(samples.len() * 2);
        for sample in samples {
            let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            pcm.extend_from_slice(&value.to_le_bytes());
        }
        self.file.write_all(&pcm)?;
        self.samples = new_samples;
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<usize> {
        let data_bytes = u32::try_from(self.samples * 2).context("WAV data exceeds 4 GiB")?;
        let riff_size = 36u32
            .checked_add(data_bytes)
            .context("WAV RIFF size overflow")?;
        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&riff_size.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&data_bytes.to_le_bytes())?;
        self.file.flush()?;
        Ok(self.samples)
    }
}

fn cuda_predictor_smoke(
    model_dir: &PathBuf,
    gpu: usize,
    seq_len: usize,
    generate_frame: bool,
    semantic_token: i32,
    repeats: usize,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config.code_predictor_config;
    if seq_len == 0 {
        anyhow::bail!("sequence length must be non-zero");
    }
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(
        stream,
        config.intermediate_size * config.hidden_size,
        config.intermediate_size,
    )?;
    let load_started = std::time::Instant::now();
    let predictor = CodePredictorTransformer::load(model_dir, config, stream)?;
    let mut predictor_session = predictor.start_generation_session(stream)?;
    let load_elapsed = load_started.elapsed();
    let free_loaded = allocator.free_bytes(gpu)?;

    if generate_frame {
        if repeats == 0 {
            anyhow::bail!("frame repeats must be non-zero");
        }
        let talker_hidden = (0..2048)
            .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.125)
            .collect::<Vec<_>>();
        println!(
            "{} predictor layers loaded in {:.3}s, {:.1} MiB VRAM",
            predictor.layer_count(),
            load_elapsed.as_secs_f64(),
            free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2)
        );
        for repeat in 0..repeats {
            let frame_started = std::time::Instant::now();
            let codes = predictor.generate_acoustic_codes_argmax_with_session(
                &mut predictor_session,
                &talker_hidden,
                semantic_token,
                &mut kernels,
            )?;
            println!(
                "frame {repeat}: semantic {semantic_token} -> acoustic {codes:?} in {:.3}ms",
                frame_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        return Ok(());
    }

    let hidden = (0..seq_len * config.hidden_size)
        .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.125)
        .collect::<Vec<_>>();
    let forward_started = std::time::Instant::now();
    let output = predictor.forward_hidden(&hidden, seq_len, seq_len, &mut kernels)?;
    let forward_elapsed = forward_started.elapsed();
    let checksum = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 + 1.0) * f64::from(*value))
        .sum::<f64>();
    println!(
        "{} predictor layers loaded in {:.3}s, {:.1} MiB VRAM",
        predictor.layer_count(),
        load_elapsed.as_secs_f64(),
        free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2)
    );
    println!(
        "{seq_len} token(s) forwarded in {:.3}ms, weighted checksum {checksum:.9}",
        forward_elapsed.as_secs_f64() * 1000.0
    );
    compare_reference(&output, reference, 0.02)?;
    Ok(())
}

fn cuda_talker_smoke(
    model_dir: &PathBuf,
    gpu: usize,
    seq_len: usize,
    decode_split: Option<usize>,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    if seq_len == 0 {
        anyhow::bail!("sequence length must be non-zero");
    }
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(
        stream,
        config.intermediate_size * config.hidden_size,
        config.intermediate_size,
    )?;
    let load_started = std::time::Instant::now();
    let talker = TalkerTransformer::load(model_dir, config, stream)?;
    let load_elapsed = load_started.elapsed();
    let free_loaded = allocator.free_bytes(gpu)?;

    let hidden = (0..seq_len * config.hidden_size)
        .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.125)
        .collect::<Vec<_>>();
    let forward_started = std::time::Instant::now();
    let mut cached_baseline = None;
    let output = if let Some(split) = decode_split {
        if split == 0 || split > seq_len {
            anyhow::bail!("decode split must be within 1..={seq_len}");
        }
        let mut session = talker.start_session(seq_len, split.max(1), config, stream)?;
        let mut output = talker.forward_session(
            &mut session,
            &hidden[..split * config.hidden_size],
            split,
            config,
            &mut kernels,
        )?;
        for token in split..seq_len {
            output.extend(talker.forward_session(
                &mut session,
                &hidden[token * config.hidden_size..(token + 1) * config.hidden_size],
                1,
                config,
                &mut kernels,
            )?);
        }
        cached_baseline =
            Some(talker.forward_hidden(&hidden, seq_len, seq_len, config, &mut kernels)?);
        output
    } else {
        talker.forward_hidden(&hidden, seq_len, seq_len, config, &mut kernels)?
    };
    let forward_elapsed = forward_started.elapsed();
    let checksum = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 + 1.0) * f64::from(*value))
        .sum::<f64>();
    println!(
        "{} talker layers loaded in {:.3}s, {:.1} MiB VRAM",
        talker.layer_count(),
        load_elapsed.as_secs_f64(),
        free_before.saturating_sub(free_loaded) as f64 / 1024.0_f64.powi(2)
    );
    println!(
        "{seq_len} token(s) forwarded in {:.3}ms, weighted checksum {checksum:.9}",
        forward_elapsed.as_secs_f64() * 1000.0
    );
    if let Some(baseline) = cached_baseline {
        let (max, mean) =
            output
                .iter()
                .zip(&baseline)
                .fold((0.0f32, 0.0f64), |(max, sum), (cached, full)| {
                    let delta = (cached - full).abs();
                    (max.max(delta), sum + f64::from(delta))
                });
        let mean = mean / output.len() as f64;
        if max > 0.1 {
            anyhow::bail!("cached talker delta {max:.7} exceeds 0.1");
        }
        let head = load_f16_tensor(model_dir, "talker.codec_head.weight")?;
        let cached_token = cpu_semantic_argmax(
            &output[output.len() - config.hidden_size..],
            &head.values,
            config.hidden_size,
            config.vocab_size,
        );
        let full_token = cpu_semantic_argmax(
            &baseline[baseline.len() - config.hidden_size..],
            &head.values,
            config.hidden_size,
            config.vocab_size,
        );
        if cached_token != full_token {
            anyhow::bail!(
                "cached semantic token {cached_token} differs from full token {full_token}"
            );
        }
        println!("cached/full delta: max {max:.7}, mean {mean:.7}, semantic token {cached_token}");
    }
    compare_reference(&output, reference, 0.05)?;
    Ok(())
}

fn cpu_semantic_argmax(
    hidden: &[f32],
    weight: &[half::f16],
    hidden_size: usize,
    vocab_size: usize,
) -> usize {
    let mut best_token = 0usize;
    let mut best_logit = f32::NEG_INFINITY;
    for token in 0..vocab_size {
        let row = &weight[token * hidden_size..(token + 1) * hidden_size];
        let logit = hidden
            .iter()
            .zip(row)
            .map(|(value, weight)| value * weight.to_f32())
            .sum::<f32>();
        if logit > best_logit {
            best_logit = logit;
            best_token = token;
        }
    }
    best_token
}

fn compare_reference(
    output: &[f32],
    reference: Option<&std::path::Path>,
    tolerance: f32,
) -> anyhow::Result<()> {
    let Some(reference) = reference else {
        return Ok(());
    };
    let bytes = std::fs::read(reference)?;
    if bytes.len() != output.len() * 4 {
        anyhow::bail!(
            "{} has {} bytes, expected {} raw f32 bytes",
            reference.display(),
            bytes.len(),
            output.len() * 4
        );
    }
    let expected = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    let mut max_abs_error = 0.0f32;
    let mut max_error_index = 0usize;
    let mut mean_abs_error = 0.0f64;
    for (index, (actual, expected)) in output.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        if error > max_abs_error {
            max_abs_error = error;
            max_error_index = index;
        }
        mean_abs_error += f64::from(error);
    }
    mean_abs_error /= output.len() as f64;
    if max_abs_error > tolerance {
        anyhow::bail!(
            "reference parity failed: max delta {max_abs_error:.7} exceeds {tolerance:.7}"
        );
    }
    println!(
        "reference delta: max {max_abs_error:.7} at {max_error_index} \
         (CUDA {:.7}, reference {:.7}), mean {mean_abs_error:.7}",
        output[max_error_index], expected[max_error_index]
    );
    Ok(())
}

fn cuda_linear_smoke(
    model_dir: &PathBuf,
    gpu: usize,
    tensor_name: &str,
    use_gemv: bool,
) -> anyhow::Result<()> {
    let tensor = load_f16_tensor(model_dir, tensor_name)
        .with_context(|| format!("could not load tensor {tensor_name}"))?;
    let [n, k]: [usize; 2] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|shape: Vec<usize>| anyhow::anyhow!("expected matrix, got shape {shape:?}"))?;

    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let stream = allocator.stream(gpu);
    let kernels = chew_kernel::GpuKernels::load(stream, n * k, k)?;

    let input = (0..k)
        .map(|index| half::f16::from_f32(((index as f32 + 1.0) * 0.013).sin() * 0.125))
        .collect::<Vec<_>>();
    let input_gpu = stream.clone_htod(&input)?;
    let weights_gpu = stream.clone_htod(&tensor.values)?;
    let mut output_gpu = stream.alloc_zeros::<half::f16>(n)?;
    if use_gemv {
        kernels.gemv.gemv_f16(
            &input_gpu,
            &weights_gpu,
            &mut output_gpu,
            n as u32,
            k as u32,
        )?;
    } else {
        kernels.gemm.matmul_f16(
            &input_gpu,
            &weights_gpu,
            &mut output_gpu,
            1,
            n as u32,
            k as u32,
        )?;
    }
    let mut output = vec![half::f16::ZERO; n];
    stream.memcpy_dtoh(&output_gpu, &mut output)?;

    let sample_rows = [0, n / 3, (2 * n) / 3, n - 1];
    let mut max_abs_error = 0.0f32;
    for row in sample_rows {
        let weights = &tensor.values[row * k..(row + 1) * k];
        let expected = weights
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight.to_f32() * value.to_f32())
            .sum::<f32>();
        let actual = output[row].to_f32();
        let error = (expected - actual).abs();
        max_abs_error = max_abs_error.max(error);
        println!("row {row}: GPU={actual:.6}, CPU={expected:.6}, abs_error={error:.6}");
    }
    if max_abs_error > 0.08 {
        anyhow::bail!("linear parity failed: maximum absolute error {max_abs_error:.6}");
    }
    println!(
        "{} parity passed for {tensor_name} [{n}, {k}], max abs error {max_abs_error:.6}",
        if use_gemv { "GEMV" } else { "GEMM" }
    );
    Ok(())
}

fn cuda_layer_smoke(
    model_dir: &PathBuf,
    gpu: usize,
    layer: usize,
    seq_len: usize,
    decode_split: Option<usize>,
    reference: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    let inspection = inspect_model(model_dir)?;
    let config = &inspection.config.talker_config;
    if layer >= config.num_hidden_layers {
        anyhow::bail!(
            "layer {layer} is out of range for {} talker layers",
            config.num_hidden_layers
        );
    }
    let allocator = chew_vram::VramAllocator::init()?;
    if gpu >= allocator.gpu_count() {
        anyhow::bail!(
            "GPU index {gpu} is out of range; detected {} device(s)",
            allocator.gpu_count()
        );
    }
    let free_before = allocator.free_bytes(gpu)?;
    let stream = allocator.stream(gpu);
    let mut kernels = chew_kernel::GpuKernels::load(
        stream,
        config.intermediate_size * config.hidden_size,
        config.intermediate_size,
    )?;
    let decoder = TalkerDecoderLayer::load(model_dir, layer, config, stream)?;
    if seq_len == 0 {
        anyhow::bail!("sequence length must be non-zero");
    }
    let hidden = (0..seq_len * config.hidden_size)
        .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.125)
        .collect::<Vec<_>>();
    let output = if let Some(split) = decode_split {
        if split == 0 || split >= seq_len {
            anyhow::bail!("decode split must be between 1 and seq_len - 1");
        }
        let mut cache = TalkerLayerKvCache::allocate(seq_len, config, stream)?;
        let mut output = decoder.forward_cached(
            &hidden[..split * config.hidden_size],
            split,
            config,
            &mut kernels,
            &mut cache,
        )?;
        for token in split..seq_len {
            output.extend(decoder.forward_cached(
                &hidden[token * config.hidden_size..(token + 1) * config.hidden_size],
                1,
                config,
                &mut kernels,
                &mut cache,
            )?);
        }
        output
    } else {
        decoder.forward_prefill(&hidden, seq_len, config, &mut kernels)?
    };
    let free_after = allocator.free_bytes(gpu)?;
    let checksum = output
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 + 1.0) * f64::from(*value))
        .sum::<f64>();
    let first = output.iter().take(8).copied().collect::<Vec<_>>();
    println!(
        "layer {layer}, {seq_len} token(s), decode split {decode_split:?}, output[0..8]: {first:?}"
    );
    println!("weighted checksum: {checksum:.9}");
    if let Some(reference) = reference {
        let bytes = std::fs::read(reference)?;
        if bytes.len() != output.len() * 4 {
            anyhow::bail!(
                "{} has {} bytes, expected {} raw f32 bytes",
                reference.display(),
                bytes.len(),
                output.len() * 4
            );
        }
        let expected = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let mut max_abs_error = 0.0f32;
        let mut max_error_index = 0usize;
        let mut mean_abs_error = 0.0f64;
        for (index, (actual, expected)) in output.iter().zip(&expected).enumerate() {
            let error = (actual - expected).abs();
            if error > max_abs_error {
                max_abs_error = error;
                max_error_index = index;
            }
            mean_abs_error += f64::from(error);
        }
        mean_abs_error /= output.len() as f64;
        if max_abs_error > 0.003 {
            anyhow::bail!(
                "layer parity failed: max abs error {max_abs_error:.7} at {max_error_index} \
                 (CUDA {:.7}, reference {:.7}), mean {mean_abs_error:.7}",
                output[max_error_index],
                expected[max_error_index]
            );
        }
        println!(
            "layer parity passed: max abs error {max_abs_error:.7} at {max_error_index} \
             (CUDA {:.7}, reference {:.7}), mean {mean_abs_error:.7}",
            output[max_error_index], expected[max_error_index]
        );
    }
    println!(
        "CUDA allocation: {:.1} MiB",
        free_before.saturating_sub(free_after) as f64 / 1024.0_f64.powi(2)
    );
    Ok(())
}
