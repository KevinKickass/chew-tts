use crate::voice_design::{SynthesisRequest, VoiceDesignEngine};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

const SAMPLE_RATE: u32 = 24_000;
const DEFAULT_INSTRUCTION: &str = "A natural, clear studio voice.";

struct Job {
    request: SynthesisRequest,
    response: oneshot::Sender<Result<GeneratedAudio, String>>,
}

struct GeneratedAudio {
    samples: Vec<f32>,
    frames: usize,
    inference_ms: f64,
}

struct AppState {
    jobs: mpsc::Sender<Job>,
    model: String,
    max_frames: usize,
    vram_bytes: u64,
}

#[derive(Deserialize)]
struct SpeechRequest {
    #[serde(default = "default_model")]
    model: String,
    input: String,
    #[serde(default = "default_voice")]
    voice: String,
    #[serde(default)]
    instruct: Option<String>,
    #[serde(default)]
    instruction: Option<String>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_response_format")]
    response_format: String,
    #[serde(default = "default_speed")]
    speed: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    max_frames: Option<usize>,
}

fn default_model() -> String {
    "tts-voice-design".into()
}
fn default_voice() -> String {
    "alloy".into()
}
fn default_language() -> String {
    "de".into()
}
fn default_response_format() -> String {
    "mp3".into()
}
fn default_speed() -> f32 {
    1.0
}
fn default_temperature() -> f32 {
    0.9
}
fn default_top_k() -> usize {
    50
}
fn default_seed() -> u64 {
    42
}

pub async fn run(
    model_dir: PathBuf,
    gpu: usize,
    host: String,
    port: u16,
    max_frames: usize,
    queue_capacity: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(queue_capacity > 0, "queue capacity must be non-zero");
    let (jobs, receiver) = mpsc::channel(queue_capacity);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("chew-tts-gpu".into())
        .spawn(move || {
            let engine = VoiceDesignEngine::load(&model_dir, gpu, max_frames);
            match engine {
                Ok(engine) => {
                    let metadata = (engine.load_elapsed, engine.vram_bytes);
                    if ready_tx.send(Ok(metadata)).is_ok() {
                        gpu_worker(engine, receiver);
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(format!("{error:#}")));
                }
            }
        })?;
    let (load_elapsed, vram_bytes) = ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("GPU worker exited during startup"))?
        .map_err(anyhow::Error::msg)?;

    let state = Arc::new(AppState {
        jobs,
        model: "tts-voice-design".into(),
        max_frames,
        vram_bytes,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/audio/speech", post(speech))
        .route("/internal/audio/raw", post(raw_speech))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    info!(
        host,
        port,
        load_seconds = load_elapsed.as_secs_f64(),
        vram_mib = vram_bytes as f64 / 1024.0_f64.powi(2),
        "Chew TTS ready"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        error!(%error, "could not install shutdown signal");
    }
}

fn gpu_worker(mut engine: VoiceDesignEngine, mut jobs: mpsc::Receiver<Job>) {
    while let Some(job) = jobs.blocking_recv() {
        let result = engine
            .synthesize(&job.request)
            .map(|output| GeneratedAudio {
                frames: output.generated_frames,
                inference_ms: output.inference_elapsed().as_secs_f64() * 1_000.0,
                samples: output.samples,
            })
            .map_err(|error| format!("{error:#}"));
        let _ = job.response.send(result);
    }
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "model": state.model,
        "max_frames": state.max_frames,
        "vram_bytes": state.vram_bytes,
    }))
}

async fn models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model,
            "object": "model",
            "owned_by": "simplellm"
        }]
    }))
}

async fn speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Response {
    let format = match AudioFormat::parse(&request.response_format) {
        Ok(format) => format,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };
    let speed = request.speed;
    match submit(&state, request).await {
        Ok(audio) => {
            let duration_ms = encoded_duration_header(&audio, speed);
            match encode_audio(audio.samples, format, speed).await {
                Ok(encoded) => Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, format.content_type())
                    .header("x-sllm-audio-duration-ms", duration_ms)
                    .header("x-sllm-generation-frames", audio.frames)
                    .header(
                        "server-timing",
                        format!("synth;dur={:.3}", audio.inference_ms),
                    )
                    .body(axum::body::Body::from(encoded))
                    .unwrap_or_else(|error| {
                        error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            &format!("could not build response: {error}"),
                        )
                    }),
                Err(message) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &message),
            }
        }
        Err(response) => response,
    }
}

async fn raw_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Response {
    match submit(&state, request).await {
        Ok(audio) => {
            let mut raw = Vec::with_capacity(audio.samples.len() * 4);
            for sample in audio.samples {
                raw.extend_from_slice(&sample.to_le_bytes());
            }
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "audio/x-f32le")
                .header("x-sllm-generation-frames", audio.frames)
                .header(
                    "server-timing",
                    format!("synth;dur={:.3}", audio.inference_ms),
                )
                .body(axum::body::Body::from(raw))
                .unwrap()
        }
        Err(response) => response,
    }
}

async fn submit(state: &Arc<AppState>, request: SpeechRequest) -> Result<GeneratedAudio, Response> {
    if request.input.trim().is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "input must not be empty",
        ));
    }
    if request.input.chars().count() > 4096 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "input exceeds 4096 characters",
        ));
    }
    if !request.speed.is_finite() || !(0.25..=4.0).contains(&request.speed) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "speed must be between 0.25 and 4.0",
        ));
    }
    if !matches!(
        request.model.as_str(),
        "tts-voice-design" | "qwen3-tts-voice-design" | "qwen3-tts"
    ) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            &format!("unsupported model {:?}", request.model),
        ));
    }
    let instruction = request
        .instruct
        .or(request.instruction)
        .or(request.instructions)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            if request.voice.eq_ignore_ascii_case("alloy") {
                DEFAULT_INSTRUCTION.into()
            } else {
                request.voice.clone()
            }
        });
    let synthesis = SynthesisRequest {
        text: request.input,
        instruction,
        language: normalize_language(&request.language),
        max_frames: request.max_frames.unwrap_or(state.max_frames),
        seed: request.seed,
        temperature: request.temperature,
        top_k: request.top_k,
        chunk_frames: 32,
        chunk_context: 4,
    };
    let (response_tx, response_rx) = oneshot::channel();
    state
        .jobs
        .try_send(Job {
            request: synthesis,
            response: response_tx,
        })
        .map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                error_response(StatusCode::TOO_MANY_REQUESTS, "TTS queue is full")
            }
            mpsc::error::TrySendError::Closed(_) => {
                error_response(StatusCode::SERVICE_UNAVAILABLE, "TTS worker is unavailable")
            }
        })?;
    let started = Instant::now();
    match response_rx.await {
        Ok(Ok(audio)) => {
            info!(
                frames = audio.frames,
                audio_seconds = audio.samples.len() as f64 / SAMPLE_RATE as f64,
                inference_ms = audio.inference_ms,
                wall_ms = started.elapsed().as_secs_f64() * 1_000.0,
                "speech generated"
            );
            Ok(audio)
        }
        Ok(Err(message)) => {
            error!(error = %message, "speech generation failed");
            Err(error_response(StatusCode::BAD_REQUEST, &message))
        }
        Err(_) => Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "TTS worker exited",
        )),
    }
}

fn normalize_language(language: &str) -> String {
    match language.trim().to_ascii_lowercase().as_str() {
        "de" | "deutsch" => "german".into(),
        "en" => "english".into(),
        "zh" => "chinese".into(),
        "ja" => "japanese".into(),
        "ko" => "korean".into(),
        "fr" => "french".into(),
        "ru" => "russian".into(),
        "pt" => "portuguese".into(),
        "es" => "spanish".into(),
        "it" => "italian".into(),
        other => other.into(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AudioFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl AudioFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "mp3" => Ok(Self::Mp3),
            "opus" | "ogg" => Ok(Self::Opus),
            "aac" => Ok(Self::Aac),
            "flac" => Ok(Self::Flac),
            "wav" => Ok(Self::Wav),
            "pcm" => Ok(Self::Pcm),
            other => Err(format!("unsupported response_format {other:?}")),
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

async fn encode_audio(
    samples: Vec<f32>,
    format: AudioFormat,
    speed: f32,
) -> Result<Vec<u8>, String> {
    if speed == 1.0 {
        let pcm = float32_to_pcm16(&samples);
        return match format {
            AudioFormat::Pcm => Ok(pcm),
            AudioFormat::Wav => Ok(wav_from_pcm16(&pcm)),
            _ => transcode(samples, format, speed).await,
        };
    }
    transcode(samples, format, speed).await
}

async fn transcode(samples: Vec<f32>, format: AudioFormat, speed: f32) -> Result<Vec<u8>, String> {
    let (codec, container) = match format {
        AudioFormat::Mp3 => ("libmp3lame", "mp3"),
        AudioFormat::Opus => ("libopus", "ogg"),
        AudioFormat::Aac => ("aac", "adts"),
        AudioFormat::Flac => ("flac", "flac"),
        AudioFormat::Wav => ("pcm_s16le", "wav"),
        AudioFormat::Pcm => ("pcm_s16le", "s16le"),
    };
    let mut raw = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        raw.extend_from_slice(&sample.to_le_bytes());
    }
    let mut command =
        Command::new(std::env::var("CHEW_TTS_FFMPEG").unwrap_or_else(|_| "ffmpeg".into()));
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-vn",
    ]);
    if speed != 1.0 {
        command.args(["-af", &atempo_filter(speed)]);
    }
    let mut child = command
        .args(["-c:a", codec, "-f", container, "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not start ffmpeg: {error}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "ffmpeg stdin is unavailable".to_string())?;
    let writer = tokio::spawn(async move {
        stdin.write_all(&raw).await?;
        stdin.shutdown().await
    });
    let output = child
        .wait_with_output()
        .await
        .map_err(|error| format!("ffmpeg failed: {error}"))?;
    writer
        .await
        .map_err(|error| format!("ffmpeg input task failed: {error}"))?
        .map_err(|error| format!("ffmpeg input failed: {error}"))?;
    if !output.status.success() {
        let mut stderr = String::new();
        use tokio::io::AsyncReadExt;
        BufReader::new(output.stderr.as_slice())
            .read_to_string(&mut stderr)
            .await
            .ok();
        return Err(format!("ffmpeg encoding failed: {}", stderr.trim()));
    }
    Ok(output.stdout)
}

fn atempo_filter(speed: f32) -> String {
    let mut remaining = speed;
    let mut filters = Vec::new();
    while remaining < 0.5 {
        filters.push("atempo=0.5".to_string());
        remaining /= 0.5;
    }
    while remaining > 2.0 {
        filters.push("atempo=2.0".to_string());
        remaining /= 2.0;
    }
    filters.push(format!("atempo={remaining:.6}"));
    filters.join(",")
}

fn float32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        pcm.extend_from_slice(&scaled.to_le_bytes());
    }
    pcm
}

fn wav_from_pcm16(pcm: &[u8]) -> Vec<u8> {
    let data_len = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32.saturating_add(data_len)).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

fn encoded_duration_header(audio: &GeneratedAudio, speed: f32) -> u64 {
    ((audio.samples.len() as f64 * 1_000.0 / SAMPLE_RATE as f64) / f64::from(speed)).round() as u64
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_openai_language_codes() {
        assert_eq!(normalize_language("de"), "german");
        assert_eq!(normalize_language("EN"), "english");
        assert_eq!(normalize_language("german"), "german");
    }

    #[test]
    fn atempo_stays_inside_ffmpeg_limits() {
        assert_eq!(atempo_filter(0.25), "atempo=0.5,atempo=0.500000");
        assert_eq!(atempo_filter(4.0), "atempo=2.0,atempo=2.000000");
        assert_eq!(atempo_filter(1.1), "atempo=1.100000");
    }

    #[test]
    fn wav_header_describes_pcm_payload() {
        let pcm = [1u8, 2, 3, 4];
        let wav = wav_from_pcm16(&pcm);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
        assert_eq!(&wav[44..], &pcm);
    }
}
