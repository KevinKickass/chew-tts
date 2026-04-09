//! Extract a HuggingFace `tokenizers::Tokenizer` from GGUF metadata.
//!
//! GGUF files embed the full tokenizer under `tokenizer.ggml.*` keys:
//! - `tokenizer.ggml.model`: "gpt2" (BPE) or "llama" (SPM/BPE)
//! - `tokenizer.ggml.tokens`: string array of vocab
//! - `tokenizer.ggml.scores`: f32 array (SPM scores, unused for pure BPE)
//! - `tokenizer.ggml.token_type`: i32 array (1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte)
//! - `tokenizer.ggml.merges`: string array of BPE merge rules ("a b")
//! - `tokenizer.ggml.bos_token_id`, `eos_token_id`, etc.

use crate::types::{GgufHeader, MetadataValue};
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use tracing::info;

/// Token type codes from GGUF spec.
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;

const TOKEN_TYPE_BYTE: i32 = 6;

/// Try to build a `Tokenizer` from GGUF metadata.
/// Returns `None` if the required metadata keys are missing.
pub fn extract_tokenizer(header: &GgufHeader) -> Option<Tokenizer> {
    let tokens = get_string_array(&header.metadata, "tokenizer.ggml.tokens")?;
    let merges = get_string_array(&header.metadata, "tokenizer.ggml.merges");
    let token_types = get_i32_array(&header.metadata, "tokenizer.ggml.token_type");

    let model_type = header
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(|v| v.as_str())
        .unwrap_or("bpe");

    let bos_id = header.metadata.get("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u32());
    let eos_id = header.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32());
    let unk_id = header.metadata.get("tokenizer.ggml.unknown_token_id").and_then(|v| v.as_u32());
    let pad_id = header.metadata.get("tokenizer.ggml.padding_token_id").and_then(|v| v.as_u32());

    info!(
        vocab_size = tokens.len(),
        merges = merges.as_ref().map(|m| m.len()).unwrap_or(0),
        model_type,
        "extracting tokenizer from GGUF"
    );

    // Build the tokenizer.json structure that HuggingFace tokenizers expects
    let tokenizer_json = build_tokenizer_json(
        &tokens,
        merges.as_deref(),
        token_types.as_deref(),
        model_type,
        bos_id,
        eos_id,
        unk_id,
        pad_id,
    );

    let json_str = serde_json::to_string(&tokenizer_json).ok()?;
    match Tokenizer::from_bytes(json_str.as_bytes()) {
        Ok(tokenizer) => {
            info!("tokenizer extracted from GGUF metadata");
            Some(tokenizer)
        }
        Err(e) => {
            tracing::error!(%e, model_type, "failed to build tokenizer from GGUF metadata");
            None
        }
    }
}

fn build_tokenizer_json(
    tokens: &[String],
    merges: Option<&[String]>,
    token_types: Option<&[i32]>,
    model_type: &str,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    pad_id: Option<u32>,
) -> Value {
    // Build vocab map: token -> id
    let vocab: Value = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), Value::Number(i.into())))
        .collect::<serde_json::Map<String, Value>>()
        .into();

    // Collect added_tokens (control tokens, byte tokens, special tokens)
    let mut added_tokens = Vec::new();
    let types = token_types.unwrap_or(&[]);

    for (id, token) in tokens.iter().enumerate() {
        let tt = types.get(id).copied().unwrap_or(TOKEN_TYPE_NORMAL);
        let is_special = matches!(tt, TOKEN_TYPE_CONTROL | TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_BYTE)
            || Some(id as u32) == bos_id
            || Some(id as u32) == eos_id
            || Some(id as u32) == unk_id
            || Some(id as u32) == pad_id;

        if is_special {
            added_tokens.push(json!({
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }));
        }
    }

    // Build merges array
    let merges_json: Vec<Value> = merges
        .unwrap_or(&[])
        .iter()
        .map(|m| Value::String(m.clone()))
        .collect();

    // Determine the HF model type
    // GGUF "llama" and "gpt2" both use BPE in the HF tokenizer
    let hf_type = "BPE";

    // unk_token
    let unk_token = unk_id
        .and_then(|id| tokens.get(id as usize))
        .cloned()
        .unwrap_or_else(|| "<unk>".to_string());

    // For SPM-style (sentencepiece) models, byte_fallback is typical
    let byte_fallback = matches!(model_type, "llama" | "gemma" | "gemma4");

    let mut model = json!({
        "type": hf_type,
        "vocab": vocab,
        "merges": merges_json,
        "unk_token": unk_token,
        "byte_fallback": byte_fallback,
    });

    // Some models need fuse_unk
    if model_type == "llama" || model_type.starts_with("gemma") {
        model["fuse_unk"] = json!(true);
    }

    let mut result = json!({
        "version": "1.0",
        "model": model,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
    });

    // Add pre_tokenizer and decoder based on model type
    match model_type {
        "gpt2" => {
            result["pre_tokenizer"] = json!({
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            });
            result["decoder"] = json!({
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            });
        }
        "llama" | "gemma" | "gemma2" | "gemma3" | "gemma4" => {
            // SentencePiece-style: ▁ = space marker in vocab
            // Pre-tokenizer: prepend ▁ then replace all spaces with ▁
            // so the BPE model can match tokens like "▁Hello"
            result["pre_tokenizer"] = json!({
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false}
                ]
            });
            result["decoder"] = json!({
                "type": "Sequence",
                "decoders": [
                    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false},
                    {"type": "ByteFallback"}
                ]
            });
        }
        _ => {}
    }

    result
}

fn get_string_array(
    metadata: &std::collections::HashMap<String, MetadataValue>,
    key: &str,
) -> Option<Vec<String>> {
    match metadata.get(key)? {
        MetadataValue::Array(arr) => {
            let strings: Vec<String> = arr
                .iter()
                .filter_map(|v| match v {
                    MetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if strings.is_empty() {
                None
            } else {
                Some(strings)
            }
        }
        _ => None,
    }
}

fn get_i32_array(
    metadata: &std::collections::HashMap<String, MetadataValue>,
    key: &str,
) -> Option<Vec<i32>> {
    match metadata.get(key)? {
        MetadataValue::Array(arr) => {
            let vals: Vec<i32> = arr
                .iter()
                .filter_map(|v| match v {
                    MetadataValue::Int32(n) => Some(*n),
                    MetadataValue::Uint32(n) => Some(*n as i32),
                    MetadataValue::Int8(n) => Some(*n as i32),
                    MetadataValue::Uint8(n) => Some(*n as i32),
                    MetadataValue::Int16(n) => Some(*n as i32),
                    MetadataValue::Uint16(n) => Some(*n as i32),
                    _ => None,
                })
                .collect();
            if vals.is_empty() { None } else { Some(vals) }
        }
        _ => None,
    }
}
