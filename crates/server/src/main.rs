use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use minijinja::Environment;
use chew_engine::{ChewEngine, sample::SampleParams};
use chew_gguf::GgufFile;
use chew_vram::VramAllocator;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

const UI_HTML: &str = include_str!("ui.html");

struct AppState {
    engine: Mutex<ChewEngine>,
    tokenizer: Tokenizer,
    model_name: String,
    model_arch: String,
    chat_template: Option<String>,
    bos_token_id: Option<u32>,
    bos_token: Option<String>,
    eos_token_id: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let model_path = std::env::args()
        .nth(1)
        .expect("usage: chew <model.gguf> [--port 8080] [--context N]");

    let args: Vec<String> = std::env::args().collect();

    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let max_context: Option<u32> = args
        .iter()
        .position(|a| a == "--context")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    info!("chew v0.1 — GPU inference engine");
    info!(model = %model_path, port, "starting");

    // Extract tokenizer from GGUF metadata
    let gguf = GgufFile::open(&model_path)?;
    let tokenizer = chew_gguf::extract_tokenizer(&gguf.header)
        .ok_or_else(|| anyhow::anyhow!("GGUF has no tokenizer metadata (tokenizer.ggml.tokens missing)"))?;

    let bos_token_id = gguf.header.bos_token_id();
    let bos_token = bos_token_id.and_then(|id| tokenizer.id_to_token(id));
    let eos_token_id = gguf
        .header
        .preferred_eos_token_id()
        .or_else(|| tokenizer.token_to_id("<turn|>"))
        .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
        .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .unwrap_or(2);

    info!(eos_token_id, "tokenizer loaded");

    let alloc = VramAllocator::init()?;
    let engine = ChewEngine::load(&model_path, &alloc, 0, max_context)?;

    let model_arch = engine.config().arch.clone();
    let model_name = gguf.header.model_name().unwrap_or(model_arch.as_str()).to_string();
    info!(arch = %model_name, "model loaded, starting server");
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        model_name,
        model_arch,
        chat_template: gguf.header.chat_template().map(str::to_string),
        bos_token_id,
        bos_token,
        eos_token_id,
    });

    let app = Router::new()
        .route("/", get(ui_handler))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!(port, "listening");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn ui_handler() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> u32 { 512 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Build a chat prompt based on model architecture.
fn build_prompt_fallback(messages: &[ChatMessage], arch: &str) -> String {
    match arch {
        "gemma4" => build_prompt_gemma4(messages),
        "gemma3" | "gemma2" => build_prompt_gemma_legacy(messages),
        _ => build_prompt_llama(messages),
    }
}

fn render_chat_template(
    template_src: &str,
    messages: &[ChatMessage],
    bos_token: Option<&str>,
) -> anyhow::Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template_src)?;
    let tpl = env.get_template("chat")?;
    let rendered = tpl.render(json!({
        "messages": messages,
        "add_generation_prompt": true,
        "enable_thinking": false,
        "tools": [],
        "bos_token": bos_token.unwrap_or(""),
    }))?;
    Ok(rendered)
}

/// Build a Gemma 4 chat prompt.
/// Uses <|turn> (ID 105) and <turn|> (ID 106) as special tokens.
fn build_prompt_gemma4(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => "user",
            r => r,
        };
        prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
    }
    prompt.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    prompt
}

fn strip_model_control_tokens(mut text: String) -> String {
    for marker in [
        "<|channel>final\n",
        "<|channel>final",
        "<|channel>analysis\n",
        "<|channel>analysis",
        "<|channel>commentary\n",
        "<|channel>commentary",
        "<|channel>thought\n",
        "<|channel>thought",
        "<channel|>",
        "<turn|>",
    ] {
        text = text.replace(marker, "");
    }
    text.trim().to_string()
}

const GEMMA4_SPECIAL_TOKENS: &[&str] = &[
    "<|tool_response>",
    "<tool_response|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|\"|>",
    "<bos>",
    "<eos>",
];

fn encode_prompt(
    tokenizer: &Tokenizer,
    model_arch: &str,
    prompt: &str,
) -> Result<Vec<u32>, String> {
    if model_arch != "gemma4" {
        return tokenizer
            .encode(prompt, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string());
    }

    let mut ids = Vec::new();
    let mut i = 0usize;
    while i < prompt.len() {
        let rest = &prompt[i..];
        let mut matched = None;
        for tok in GEMMA4_SPECIAL_TOKENS {
            if rest.starts_with(tok) {
                matched = Some(*tok);
                break;
            }
        }

        if let Some(tok) = matched {
            let id = tokenizer
                .token_to_id(tok)
                .ok_or_else(|| format!("missing special token id: {tok}"))?;
            ids.push(id);
            i += tok.len();
            continue;
        }

        let next_special = GEMMA4_SPECIAL_TOKENS
            .iter()
            .filter_map(|tok| rest.find(tok))
            .min()
            .unwrap_or(rest.len());
        let chunk = &rest[..next_special];
        if !chunk.is_empty() {
            let escaped = chunk.replace(' ', "▁");
            let enc = tokenizer.encode(escaped, false).map_err(|e| e.to_string())?;
            ids.extend_from_slice(enc.get_ids());
        }
        i += next_special;
    }

    Ok(ids)
}

fn extract_chat_response(text: &str) -> String {
    const CHANNEL_OPEN: &str = "<|channel>";
    const CHANNEL_CLOSE: &str = "<channel|>";
    const TURN_CLOSE: &str = "<turn|>";

    let mut final_text = String::new();
    let mut visible_text = String::new();
    let mut active_channel: Option<String> = None;
    let mut body_start = 0usize;
    let mut i = 0usize;

    while let Some(rel_open) = text[i..].find(CHANNEL_OPEN) {
        let open = i + rel_open;
        if active_channel.is_some() {
            let chunk = &text[body_start..open];
            if active_channel.as_deref() == Some("final") {
                final_text.push_str(chunk);
            } else if active_channel.as_deref() != Some("thought") {
                visible_text.push_str(chunk);
            }
        } else {
            visible_text.push_str(&text[i..open]);
        }

        let name_start = open + CHANNEL_OPEN.len();
        let Some(rel_close) = text[name_start..].find(CHANNEL_CLOSE) else {
            i = open;
            break;
        };
        let close = name_start + rel_close;
        active_channel = Some(text[name_start..close].trim().to_string());
        body_start = close + CHANNEL_CLOSE.len();
        i = body_start;
    }

    let tail = &text[i..];
    if let Some(channel) = active_channel.as_deref() {
        let tail = tail.split(TURN_CLOSE).next().unwrap_or("");
        if channel == "final" {
            final_text.push_str(tail);
        } else if channel != "thought" {
            visible_text.push_str(tail);
        }
    } else {
        visible_text.push_str(tail);
    }

    let chosen = if final_text.trim().is_empty() {
        visible_text
    } else {
        final_text
    };
    strip_model_control_tokens(chosen)
}

/// Build a Gemma 2/3 chat prompt (legacy <start_of_turn> format).
fn build_prompt_gemma_legacy(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => "user",
            r => r,
        };
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Build a Llama 3.1 Instruct chat prompt.
fn build_prompt_llama(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let prompt = match state.chat_template.as_deref() {
        Some(template) => render_chat_template(
            template,
            &req.messages,
            state.bos_token.as_deref(),
        )
            .unwrap_or_else(|_| build_prompt_fallback(&req.messages, &state.model_arch)),
        None => build_prompt_fallback(&req.messages, &state.model_arch),
    };

    let mut input_tokens = encode_prompt(&state.tokenizer, &state.model_arch, prompt.as_str())
        .map_err(|e| AppError(format!("tokenize: {e}")))?;
    if let Some(bos) = state.bos_token_id {
        // Match llama.cpp behavior for Gemma4: the model can auto-add BOS even if
        // the rendered prompt already starts with an explicit <bos> token.
        if state.model_arch == "gemma4" {
            input_tokens.insert(0, bos);
        } else if input_tokens.first().copied() != Some(bos) {
            input_tokens.insert(0, bos);
        }
    }
    let prompt_len = input_tokens.len() as u32;

    info!(prompt = %prompt, prompt_tokens = prompt_len, max_tokens = req.max_tokens, "generating");

    let params = SampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        ..Default::default()
    };

    let generated = {
        let mut engine = state.engine.lock().await;
        engine
            .generate(&input_tokens, req.max_tokens, &params, state.eos_token_id)
            .map_err(|e| AppError(format!("generate: {e}")))?
    };

    let completion_len = generated.len() as u32;

    let raw_response_text = state
        .tokenizer
        .decode(&generated, false)
        .map_err(|e| AppError(format!("detokenize: {e}")))?;
    let raw_response_text = if state.model_arch == "gemma4" {
        raw_response_text.replace('▁', " ")
    } else {
        raw_response_text
    };
    info!(raw_response_text = %raw_response_text, "decoded raw response");
    let response_text = extract_chat_response(&raw_response_text);

    info!(completion_tokens = completion_len, ?generated, "done");

    let model = if req.model.is_empty() {
        state.model_name.clone()
    } else {
        req.model
    };

    Ok(Json(ChatResponse {
        id: format!("chew-{}", uuid_simple()),
        object: "chat.completion".into(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: response_text,
            },
            finish_reason: if generated.last() == Some(&state.eos_token_id) {
                "stop".into()
            } else {
                "length".into()
            },
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    }))
}

struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!(error = %self.0, "request failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": self.0,
                    "type": "server_error",
                }
            })),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    owned_by: String,
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelEntry {
            id: state.model_name.clone(),
            object: "model".into(),
            owned_by: "chew".into(),
        }],
    })
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{t:x}")
}
