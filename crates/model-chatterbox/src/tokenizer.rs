use crate::{MAX_TEXT_TOKENS, START_TEXT_TOKEN, STOP_TEXT_TOKEN};
use anyhow::ensure;
use std::path::Path;
use tokenizers::Tokenizer;
use unicode_normalization::UnicodeNormalization;

const SUPPORTED_LANGUAGES: [&str; 23] = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
    "pl", "pt", "ru", "sv", "sw", "tr", "zh",
];

pub struct ChatterboxTokenizer {
    tokenizer: Tokenizer,
}

impl ChatterboxTokenizer {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let path = model_dir.join("grapheme_mtl_merged_expanded_v1.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("could not load {}: {error}", path.display()))?;
        for (token, expected) in [
            ("[STOP]", STOP_TEXT_TOKEN as u32),
            ("[START]", START_TEXT_TOKEN as u32),
            ("[SPACE]", 2),
        ] {
            ensure!(
                tokenizer.token_to_id(token) == Some(expected),
                "Chatterbox tokenizer {token} has an unexpected ID"
            );
        }
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str, language: &str) -> anyhow::Result<Vec<i32>> {
        let language = language.trim().to_ascii_lowercase();
        ensure!(
            SUPPORTED_LANGUAGES.contains(&language.as_str()),
            "unsupported Chatterbox language {language:?}"
        );
        let text = normalize_multilingual_text(text)?;
        let text = text.to_lowercase().nfkd().collect::<String>();
        let input = format!("[{language}]{}", text.replace(' ', "[SPACE]"));
        let encoded = self
            .tokenizer
            .encode(input, false)
            .map_err(|error| anyhow::anyhow!("could not tokenize Chatterbox text: {error}"))?;
        let mut ids = Vec::with_capacity(encoded.len() + 2);
        ids.push(START_TEXT_TOKEN as i32);
        ids.extend(encoded.get_ids().iter().map(|id| *id as i32));
        ids.push(STOP_TEXT_TOKEN as i32);
        ensure!(
            ids.len() <= MAX_TEXT_TOKENS + 2,
            "Chatterbox text requires {} tokens, model limit is {}",
            ids.len(),
            MAX_TEXT_TOKENS + 2
        );
        Ok(ids)
    }
}

pub fn normalize_multilingual_text(text: &str) -> anyhow::Result<String> {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    ensure!(!text.is_empty(), "Chatterbox text must not be empty");
    let mut text = text;
    for (old, new) in [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ] {
        text = text.replace(old, new);
    }
    text = text.trim_end().to_owned();
    if ![".", "!", "?", "-", ","]
        .iter()
        .any(|ending| text.ends_with(ending))
    {
        text.push('.');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_multilingual_punctuation() {
        assert_eq!(
            normalize_multilingual_text("  Hallo   Welt…  ").unwrap(),
            "Hallo Welt,"
        );
        assert_eq!(
            normalize_multilingual_text("Guten Abend").unwrap(),
            "Guten Abend."
        );
    }

    #[test]
    fn rejects_empty_text() {
        assert!(normalize_multilingual_text(" \n ").is_err());
    }
}
