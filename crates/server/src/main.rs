use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use chew_engine::{ChewEngine, sample::SampleParams};
use chew_vram::VramAllocator;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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
        .expect("usage: chew <model.gguf> [--tokenizer path/to/tokenizer.json] [--port 8080]");

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

    // Look for tokenizer: --tokenizer flag, or tokenizer.json next to model
    let tokenizer_path = args
        .iter()
        .position(|a| a == "--tokenizer")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let model = PathBuf::from(&model_path);
            model.parent().unwrap_or(".".as_ref()).join("tokenizer.json")
        });

    info!("chew v0.1 — GPU inference engine");
    info!(model = %model_path, port, tokenizer = %tokenizer_path.display(), "starting");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {}: {e}", tokenizer_path.display()))?;

    let eos_token_id = tokenizer
        .token_to_id("<end_of_turn>")  // Gemma 4
        .or_else(|| tokenizer.token_to_id("<|eot_id|>"))  // Llama
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .unwrap_or(2);

    info!(eos_token_id, "tokenizer loaded");

    let alloc = VramAllocator::init()?;
    let engine = ChewEngine::load(&model_path, &alloc, 0, max_context)?;

    let model_name = engine.config().arch.clone();
    info!(arch = %model_name, "model loaded, starting server");

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        model_name,
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
fn build_prompt(messages: &[ChatMessage], arch: &str) -> String {
    match arch {
        "gemma4" | "gemma3" | "gemma2" => build_prompt_gemma(messages),
        _ => build_prompt_llama(messages),
    }
}

/// Build a Gemma-style chat prompt.
fn build_prompt_gemma(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => "user", // Gemma doesn't have system role, treat as user
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
    let prompt = build_prompt(&req.messages, &state.model_name);

    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| AppError(format!("tokenize: {e}")))?;

    let input_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = input_tokens.len() as u32;

    info!(prompt_tokens = prompt_len, max_tokens = req.max_tokens, "generating");

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

    let response_text = state
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| AppError(format!("detokenize: {e}")))?;

    info!(completion_tokens = completion_len, "done");

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
