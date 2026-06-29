use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use chew_engine::{ChewEngine, sample::SampleParams};
use chew_gguf::GgufFile;
use chew_vram::VramAllocator;
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

const UI_HTML: &str = include_str!("ui.html");
const JELLY_MONSTER_ASCII: &str = r#"
                            %@%:
                            +@:------:-=@
                           @:-------------:@
                          @:-:--------------=+
                         @::------------------==@
                         :-:---------------------=#@                 +@@@@@@@@=
                        #-:---------------------------::------:-----------------:+@#
                       @--:---------------------------------------------------------=@
                       =--:-----------------------------::----:::--------------------=*
                      #---------------------------------------------------------------=+
                      *----------------------------------------------------------------=@
                     :-----------------------------------------------------------------==
                   #:---------------------:%%------------------------------------------==#
                @--------------------@%:-----------------------------------------------=-:
             @:-------------:---------------------------------------------------------=-+
            @:--------=@@@@@@-----------------=======-----------------------------------@
           @---------=        -#-------------:+.     @:--------------------------------*
           @--:------@   #@-    @----------==          @----------------------------===@
           @--:------@   @@@    @----------*  @@@       @--------------------------=--@
            +---------:         @----------:            @------------------------====@
            @--------+:@@      @-=-:------#--          -:-----------------------====@
            @---------:%--::::--@--:-------:#::-@@%*@%--------------------------=-=@
            @--------------+@%=---:------------*@@@@---------------------------===@
           *---------------------:---------------------------------:@----------==@
          @:----------------------------------------------------------:@-------==@
         %:----------------------------------=---------*@@   *****@@-----------==@
       @:--:--------------------------=====-------=@@%****   *******@:---=:-----=+@
      @--::-------------:----------------------#@%   ******%%********@:::-@------=@%
     @:-:--------------:--::%@%**   +****   %*******@****************%:---@------==+@
    @:-:-----------------@********=  *****% %***************+********@:-:-%:::--:-==-@#
   @---------------------**************+******************+**********@---=::::------=--@
   %---------------------=******************************************@---=@-----------===#@
   @----:------------------#%********************************@@%***@----%--------------==-@
    @--------------------------@#*********##********@#********   @:----=:--------------====@
     %@--------=-------------------=@%*****   *******  %******@@-------:----------------===%*
                =@*--------------------:::*@@@@%%#***%*@@@@@*---------------------------===+@
                    %------------------------------------------------------------------====@:
                     @---------------------------------------------------------------:=====@
                      :-------------------------------------------------------------======@
                      @----------------------------------------------------------======-@%
                       *----------------------------------====--====================--@@
                       @=-----------------------------========-=========---------=%@@
                        @==-----------------------=====-=-=@@@            *@@@@.
                         @==-------------------========@@
                          @======--------====-=====-@@
                           @#-===================%@
                             @@==============-+@
                                @@========-@@
"#;

struct AppState {
    engine: Mutex<ChewEngine>,
    tokenizer: Tokenizer,
    model_name: String,
    model_arch: String,
    chat_template: Option<String>,
    tokenizer_add_bos: bool,
    bos_token_id: Option<u32>,
    bos_token: Option<String>,
    eos_token_id: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
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

    println!("{JELLY_MONSTER_ASCII}");
    info!("chew v0.1 — GPU inference engine");
    info!(model = %model_path, port, max_context, "starting");

    // Extract tokenizer from GGUF metadata
    let gguf = GgufFile::open(&model_path)?;
    let tokenizer = chew_gguf::extract_tokenizer(&gguf.header).ok_or_else(|| {
        anyhow::anyhow!("GGUF has no tokenizer metadata (tokenizer.ggml.tokens missing)")
    })?;

    let bos_token_id = gguf.header.bos_token_id();
    let bos_token = bos_token_id.and_then(|id| tokenizer.id_to_token(id));
    let tokenizer_add_bos = gguf.header.add_bos_token().unwrap_or(false);
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
    let model_name = gguf
        .header
        .model_name()
        .unwrap_or(model_arch.as_str())
        .to_string();
    info!(arch = %model_name, "model loaded, starting server");
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        model_name,
        model_arch,
        chat_template: gguf.header.chat_template().map(str::to_string),
        tokenizer_add_bos,
        bos_token_id,
        bos_token,
        eos_token_id,
    });

    let app = Router::new()
        .route("/", get(ui_handler))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
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
    #[serde(default = "default_top_k")]
    top_k: u32,
    #[serde(default = "default_repeat_penalty")]
    repeat_penalty: f32,
    #[serde(default = "default_repeat_window")]
    repeat_window: usize,
}

fn default_max_tokens() -> u32 {
    2048
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_top_k() -> u32 {
    40
}
fn default_repeat_penalty() -> f32 {
    // Default fast path: allows pure GPU top-k sampling.
    1.0
}
fn default_repeat_window() -> usize {
    64
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingsRequest {
    #[serde(default)]
    model: String,
    input: EmbeddingInput,
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct EmbeddingsUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    object: String,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: EmbeddingsUsage,
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
        "llama" => build_prompt_llama(messages),
        "mamba" => build_prompt_mamba_base(messages),
        _ => build_prompt_generic_chat(messages),
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
    prompt.push_str("<|turn>model\n<|channel>final\n<channel|>");
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

fn strip_trailing_eos_markers(mut text: String) -> String {
    const EOS_MARKERS: &[&str] = &["<|endoftext|>", "<|eot_id|>", "</s>", "<eos>"];
    text = text.trim().to_string();
    loop {
        let mut stripped = false;
        for marker in EOS_MARKERS {
            if text.ends_with(marker) {
                let keep = text.len().saturating_sub(marker.len());
                text.truncate(keep);
                text = text.trim_end().to_string();
                stripped = true;
            }
        }
        if !stripped {
            break;
        }
    }
    text
}

fn decode_hex_byte_markers(text: &str) -> String {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0usize;

    while i < text.len() {
        if let Some((byte, next_i)) = parse_hex_byte_marker(text, i) {
            out.push(byte);
            i = next_i;
            continue;
        }

        let ch = text[i..].chars().next().expect("valid utf-8 boundary");
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        i += ch.len_utf8();
    }

    String::from_utf8_lossy(&out).into_owned()
}

fn parse_hex_byte_marker(text: &str, start: usize) -> Option<(u8, usize)> {
    let rest = text.get(start..)?;
    let bytes = rest.as_bytes();
    if bytes.len() < 6
        || bytes[0] != b'<'
        || bytes[1] != b'0'
        || bytes[2] != b'x'
        || bytes[5] != b'>'
    {
        return None;
    }

    let hi = (bytes[3] as char).to_digit(16)?;
    let lo = (bytes[4] as char).to_digit(16)?;
    Some((((hi << 4) | lo) as u8, start + 6))
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
    if model_arch == "bert" {
        return tokenizer
            .encode(prompt, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string());
    }

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
            let enc = tokenizer.encode(chunk, false).map_err(|e| e.to_string())?;
            ids.extend_from_slice(enc.get_ids());
        }
        i += next_special;
    }

    Ok(ids)
}

fn extract_chat_response(text: &str, starts_in_thought: bool) -> String {
    const CHANNEL_OPEN: &str = "<|channel>";
    const CHANNEL_CLOSE: &str = "<channel|>";
    const TURN_CLOSE: &str = "<turn|>";

    let text = text.split(TURN_CLOSE).next().unwrap_or(text);

    let mut final_text = String::new();
    let mut visible_text = String::new();
    let mut active_channel: Option<String> = if starts_in_thought {
        Some("thought".to_string())
    } else {
        None
    };
    let mut i = 0usize;

    while i < text.len() {
        let rest = &text[i..];
        let next_open = rest.find(CHANNEL_OPEN);
        let next_close = rest.find(CHANNEL_CLOSE);

        match (next_open, next_close) {
            // <|channel>NAME<channel|> — switch to named channel
            (Some(op), Some(cp)) if op <= cp => {
                collect_channel_chunk(
                    &mut final_text,
                    &mut visible_text,
                    &active_channel,
                    &rest[..op],
                );
                let name_start = op + CHANNEL_OPEN.len();
                active_channel = Some(rest[name_start..cp].trim().to_string());
                i += cp + CHANNEL_CLOSE.len();
            }
            // Standalone <channel|> — close current channel
            (_, Some(cp)) => {
                collect_channel_chunk(
                    &mut final_text,
                    &mut visible_text,
                    &active_channel,
                    &rest[..cp],
                );
                active_channel = None;
                i += cp + CHANNEL_CLOSE.len();
            }
            // Orphan <|channel> without matching close
            (Some(op), None) => {
                collect_channel_chunk(
                    &mut final_text,
                    &mut visible_text,
                    &active_channel,
                    &rest[..op],
                );
                break;
            }
            (None, None) => break,
        }
    }

    if i < text.len() {
        collect_channel_chunk(
            &mut final_text,
            &mut visible_text,
            &active_channel,
            &text[i..],
        );
    }

    let chosen = if final_text.trim().is_empty() {
        visible_text
    } else {
        final_text
    };
    strip_model_control_tokens(chosen)
}

fn collect_channel_chunk(
    final_text: &mut String,
    visible_text: &mut String,
    channel: &Option<String>,
    chunk: &str,
) {
    match channel.as_deref() {
        Some("final") => final_text.push_str(chunk),
        Some("thought") => {}
        _ => visible_text.push_str(chunk),
    }
}

/// Build a Gemma 2/3 chat prompt (legacy <start_of_turn> format).
fn build_prompt_gemma_legacy(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => "user",
            r => r,
        };
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
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

/// Build a generic role-tagged chat prompt for architectures without a dedicated template.
fn build_prompt_generic_chat(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => "System",
            "assistant" => "Assistant",
            _ => "User",
        };
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(msg.content.trim());
        prompt.push('\n');
    }
    prompt.push_str("Assistant: ");
    prompt
}

/// Build a plain continuation prompt for base Mamba checkpoints (no chat template).
/// This avoids role wrappers that can bias GPT2-style Mamba models into immediate EOT.
fn build_prompt_mamba_base(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        if !msg.content.is_empty() {
            prompt.push_str(msg.content.trim());
            prompt.push('\n');
        }
    }
    prompt
}

fn should_prepend_bos(model_arch: &str, tokenizer_add_bos: bool) -> bool {
    // Align with llama.cpp: metadata drives BOS policy. Gemma4 is a known
    // metadata edge-case where some GGUFs incorrectly set add_bos=false.
    if model_arch == "gemma4" && !tokenizer_add_bos {
        return true;
    }
    tokenizer_add_bos
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let prompt = match state.chat_template.as_deref() {
        Some(template) => render_chat_template(template, &req.messages, state.bos_token.as_deref())
            .unwrap_or_else(|_| build_prompt_fallback(&req.messages, &state.model_arch)),
        None => build_prompt_fallback(&req.messages, &state.model_arch),
    };

    let mut input_tokens = encode_prompt(&state.tokenizer, &state.model_arch, prompt.as_str())
        .map_err(|e| AppError(format!("tokenize: {e}")))?;
    if let Some(bos) = state.bos_token_id {
        if should_prepend_bos(&state.model_arch, state.tokenizer_add_bos)
            && input_tokens.first().copied() != Some(bos)
        {
            input_tokens.insert(0, bos);
        }
    }
    let prompt_len = input_tokens.len() as u32;

    info!(
        prompt_tokens = prompt_len,
        max_tokens = req.max_tokens,
        "generating"
    );

    let params = SampleParams {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        repeat_penalty: req.repeat_penalty,
        repeat_window: req.repeat_window,
        ..Default::default()
    };

    let gen_t0 = Instant::now();
    let generated = {
        let mut engine = state.engine.lock().await;
        engine
            .generate(&input_tokens, req.max_tokens, &params, state.eos_token_id)
            .map_err(|e| AppError(format!("generate: {e}")))?
    };
    let gen_elapsed = gen_t0.elapsed();

    let completion_len = generated.len() as u32;
    let elapsed_s = gen_elapsed.as_secs_f64().max(1e-9);
    let completion_tps = completion_len as f64 / elapsed_s;
    let total_tps = (prompt_len + completion_len) as f64 / elapsed_s;
    info!(
        prompt_tokens = prompt_len,
        completion_tokens = completion_len,
        elapsed_ms = format!("{:.2}", gen_elapsed.as_secs_f64() * 1000.0),
        completion_tok_s = format!("{:.2}", completion_tps),
        total_tok_s = format!("{:.2}", total_tps),
        "generation perf"
    );

    let raw_response_text = state
        .tokenizer
        .decode(&generated, false)
        .map_err(|e| AppError(format!("detokenize: {e}")))?;
    let raw_response_text = if state.model_arch == "gemma4" {
        decode_hex_byte_markers(&raw_response_text.replace('▁', " "))
    } else {
        raw_response_text
    };
    info!(raw_response_text = %raw_response_text, "decoded raw response");
    let starts_in_thought = prompt.contains("<|channel>thought");
    let response_text =
        strip_trailing_eos_markers(extract_chat_response(&raw_response_text, starts_in_thought));

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

fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

fn mean_pool_last_hidden(hidden: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    if seq_len == 0 {
        return out;
    }
    for t in 0..seq_len {
        let row = &hidden[t * dim..(t + 1) * dim];
        for (o, &x) in out.iter_mut().zip(row.iter()) {
            *o += x;
        }
    }
    let inv = 1.0 / seq_len as f32;
    for x in &mut out {
        *x *= inv;
    }
    l2_normalize(&mut out);
    out
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, AppError> {
    let texts: Vec<String> = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Batch(v) => v,
    };

    let model = if req.model.is_empty() {
        format!("{}-embeddings", state.model_name)
    } else {
        req.model
    };

    let mut data = Vec::with_capacity(texts.len());
    let mut total_tokens = 0u32;

    for (index, text) in texts.iter().enumerate() {
        let mut tokens = encode_prompt(&state.tokenizer, &state.model_arch, text)
            .map_err(|e| AppError(format!("tokenize: {e}")))?;
        if should_prepend_bos(&state.model_arch, state.tokenizer_add_bos) {
            if let Some(bos) = state.bos_token_id {
                if tokens.first().copied() != Some(bos) {
                    tokens.insert(0, bos);
                }
            }
        }
        total_tokens += tokens.len() as u32;

        let hidden = {
            let mut engine = state.engine.lock().await;
            engine
                .encode_hidden(&tokens)
                .map_err(|e| AppError(format!("embed: {e}")))?
        };
        let dim = hidden.len() / tokens.len().max(1);
        let embedding = mean_pool_last_hidden(&hidden, tokens.len(), dim);
        data.push(EmbeddingObject {
            object: "embedding".into(),
            embedding,
            index,
        });
    }

    Ok(Json(EmbeddingsResponse {
        object: "list".into(),
        data,
        model,
        usage: EmbeddingsUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
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

#[cfg(test)]
mod tests {
    use super::{
        ChatMessage, build_prompt_fallback, decode_hex_byte_markers, extract_chat_response,
        should_prepend_bos, strip_trailing_eos_markers,
    };

    #[test]
    fn mamba_fallback_uses_generic_prompt() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let prompt = build_prompt_fallback(&messages, "mamba");
        assert_eq!(prompt, "hi\n");
        assert!(!prompt.contains("<|begin_of_text|>"));
    }

    #[test]
    fn unknown_arch_fallback_uses_generic_prompt() {
        let messages = vec![ChatMessage {
            role: "system".into(),
            content: "rules".into(),
        }];
        let prompt = build_prompt_fallback(&messages, "unknown-arch");
        assert!(prompt.starts_with("System: rules\n"));
        assert!(prompt.ends_with("Assistant: "));
        assert!(!prompt.contains("<|start_header_id|>"));
    }

    #[test]
    fn llama_fallback_keeps_llama_template() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let prompt = build_prompt_fallback(&messages, "llama");
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn gemma4_fallback_starts_in_final_channel() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let prompt = build_prompt_fallback(&messages, "gemma4");
        assert!(prompt.ends_with("<|turn>model\n<|channel>final\n<channel|>"));
    }

    #[test]
    fn bos_prepend_policy_respects_metadata_with_gemma4_workaround() {
        assert!(!should_prepend_bos("mamba", false));
        assert!(should_prepend_bos("mamba", true));
        assert!(!should_prepend_bos("bert", false));
        assert!(should_prepend_bos("bert", true));
        assert!(!should_prepend_bos("llama", false));
        assert!(should_prepend_bos("llama", true));
        assert!(should_prepend_bos("gemma4", false));
        assert!(should_prepend_bos("gemma4", true));
    }

    #[test]
    fn strips_trailing_eos_markers() {
        assert_eq!(
            strip_trailing_eos_markers("Paris<|endoftext|>".into()),
            "Paris"
        );
        assert_eq!(strip_trailing_eos_markers("hello </s> ".into()), "hello");
        assert_eq!(
            strip_trailing_eos_markers("plain text".into()),
            "plain text"
        );
    }

    #[test]
    fn extract_chat_response_handles_close_only_after_thought() {
        let raw = "Plan:\n1. Think\n<channel|>Visible answer<turn|>";
        assert_eq!(extract_chat_response(raw, true), "Visible answer");
    }

    #[test]
    fn decodes_hex_byte_markers() {
        assert_eq!(
            decode_hex_byte_markers("Hello<0x20>world<0x0A>"),
            "Hello world\n"
        );
    }
}
