use crate::VibeVoiceConfig;
use anyhow::{Context, ensure};
use chew_safetensors::MappedSafetensors;
use half::bf16;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VibeVoicePromptKv {
    pub key: Vec<bf16>,
    pub value: Vec<bf16>,
    pub tokens: usize,
}

#[derive(Debug, Clone)]
pub struct VibeVoicePromptBranch {
    pub last_hidden_state: Vec<bf16>,
    pub tokens: usize,
    pub layers: Vec<VibeVoicePromptKv>,
}

#[derive(Debug, Clone)]
pub struct VibeVoicePrompt {
    pub lm: VibeVoicePromptBranch,
    pub tts_lm: VibeVoicePromptBranch,
    pub negative_lm: VibeVoicePromptBranch,
    pub negative_tts_lm: VibeVoicePromptBranch,
}

impl VibeVoicePrompt {
    /// Load Chew's safe, mmap-friendly prompt format. Official `.pt` prompt
    /// files are converted once during model installation; inference does not
    /// load pickle or require Python/PyTorch.
    pub fn load(path: &Path, config: &VibeVoiceConfig) -> anyhow::Result<Self> {
        let weights = MappedSafetensors::open(path)
            .with_context(|| format!("could not open VibeVoice prompt {}", path.display()))?;
        Ok(Self {
            lm: load_branch(&weights, "lm", config.text_layers(), config)?,
            tts_lm: load_branch(
                &weights,
                "tts_lm",
                config.tts_backbone_num_hidden_layers,
                config,
            )?,
            negative_lm: load_branch(&weights, "neg_lm", config.text_layers(), config)?,
            negative_tts_lm: load_branch(
                &weights,
                "neg_tts_lm",
                config.tts_backbone_num_hidden_layers,
                config,
            )?,
        })
    }

    pub fn total_tokens(&self) -> usize {
        self.lm.tokens + self.tts_lm.tokens + self.negative_lm.tokens + self.negative_tts_lm.tokens
    }
}

fn load_branch(
    weights: &MappedSafetensors,
    prefix: &str,
    layer_count: usize,
    config: &VibeVoiceConfig,
) -> anyhow::Result<VibeVoicePromptBranch> {
    let hidden_name = format!("{prefix}.last_hidden_state");
    let (hidden_shape, last_hidden_state) = weights
        .tensor_bf16(&hidden_name)
        .with_context(|| format!("could not load VibeVoice prompt tensor {hidden_name}"))?;
    ensure!(
        hidden_shape.len() == 3
            && hidden_shape[0] == 1
            && hidden_shape[2] == config.decoder_config.hidden_size,
        "{hidden_name} has shape {hidden_shape:?}, expected [1, tokens, {}]",
        config.decoder_config.hidden_size
    );
    let tokens = hidden_shape[1];
    ensure!(tokens > 0, "{prefix} prompt is empty");

    let heads = config.decoder_config.num_key_value_heads;
    let head_dim = config.head_dim();
    let mut layers = Vec::with_capacity(layer_count);
    for layer in 0..layer_count {
        let key_name = format!("{prefix}.key.{layer}");
        let value_name = format!("{prefix}.value.{layer}");
        let (key_shape, key) = weights
            .tensor_bf16(&key_name)
            .with_context(|| format!("could not load VibeVoice prompt tensor {key_name}"))?;
        let (value_shape, value) = weights
            .tensor_bf16(&value_name)
            .with_context(|| format!("could not load VibeVoice prompt tensor {value_name}"))?;
        let expected = [1, heads, tokens, head_dim];
        ensure!(
            key_shape == expected,
            "{key_name} has shape {key_shape:?}, expected {expected:?}"
        );
        ensure!(
            value_shape == expected,
            "{value_name} has shape {value_shape:?}, expected {expected:?}"
        );
        layers.push(VibeVoicePromptKv { key, value, tokens });
    }
    Ok(VibeVoicePromptBranch {
        last_hidden_state,
        tokens,
        layers,
    })
}
