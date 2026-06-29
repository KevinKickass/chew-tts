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
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use tracing::info;

/// Token type codes from GGUF spec.
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

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

    let bos_id = header
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.as_u32());
    let eos_id = header
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.as_u32());
    let unk_id = header
        .metadata
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(|v| v.as_u32());
    let pad_id = header
        .metadata
        .get("tokenizer.ggml.padding_token_id")
        .and_then(|v| v.as_u32());

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

    // Collect added_tokens (control/user-defined/special tokens).
    // Byte tokens are part of the normal Gemma4/GPT-style vocabulary and must
    // stay in the model vocab rather than being elevated to special tokens.
    let mut added_tokens = Vec::new();
    let types = token_types.unwrap_or(&[]);

    for (id, token) in tokens.iter().enumerate() {
        let tt = types.get(id).copied().unwrap_or(TOKEN_TYPE_NORMAL);
        let is_special = matches!(
            tt,
            TOKEN_TYPE_CONTROL | TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_USER_DEFINED
        ) || Some(id as u32) == bos_id
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
        "gemma4" => {
            // Gemma 4 uses a GPT-2 style regex splitter plus a byte-level decoder.
            // Treating it like a SentencePiece tokenizer breaks both whitespace and
            // control-token boundaries.
            result["pre_tokenizer"] = json!({
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {
                            "Regex": "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
                        },
                        "behavior": "Isolated",
                        "invert": false
                    }
                ]
            });
            result["decoder"] = json!({
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false
            });
        }
        "llama" | "gemma" | "gemma2" | "gemma3" => {
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

#[cfg(test)]
mod tests {
    use super::{TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_UNKNOWN, build_tokenizer_json};
    use tokenizers::Tokenizer;

    #[test]
    fn gemma4_tokenizer_preserves_spaces_and_control_tokens() {
        let tokens = vec![
            "<pad>".to_string(),
            "<eos>".to_string(),
            "<bos>".to_string(),
            "<unk>".to_string(),
            "<|channel>".to_string(),
            "<channel|>".to_string(),
            "<|turn>".to_string(),
            "<turn|>".to_string(),
            "Hello".to_string(),
            " world".to_string(),
            "!".to_string(),
            "final".to_string(),
        ];
        let token_types = vec![
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_UNKNOWN,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
        ];
        let json = build_tokenizer_json(
            &tokens,
            Some(&[]),
            Some(&token_types),
            "gemma4",
            Some(2),
            Some(1),
            Some(3),
            Some(0),
        );
        let tokenizer =
            Tokenizer::from_bytes(serde_json::to_vec(&json).expect("serialize tokenizer json"))
                .expect("build tokenizer");

        assert_eq!(tokenizer.token_to_id("<|channel>"), Some(4));
        assert_eq!(tokenizer.token_to_id("<channel|>"), Some(5));
        assert_eq!(tokenizer.token_to_id("<|turn>"), Some(6));
        assert_eq!(tokenizer.token_to_id("<turn|>"), Some(7));
        assert_eq!(
            tokenizer
                .decode(&[8, 9, 10], false)
                .expect("decode plain text"),
            "Hello world!"
        );

        let with_controls = tokenizer
            .decode(&[4, 11, 5, 8, 9, 10, 7], false)
            .expect("decode control tokens");
        assert_eq!(
            with_controls,
            "<|channel>final<channel|>Hello world!<turn|>"
        );
    }
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
