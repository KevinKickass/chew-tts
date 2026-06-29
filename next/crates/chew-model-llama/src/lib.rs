//! Llama-style causal model semantics.
//! This crate should stay close to llama.cpp behavior, but expressed in Rust.

use chew_runtime::{AttentionMode, KvCacheLayout, KvSharing, LayerKvSpec, RuntimeLimits};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaModelConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

#[derive(Debug)]
pub struct LlamaModel {
    config: LlamaModelConfig,
}

impl LlamaModel {
    pub fn new(config: LlamaModelConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> LlamaModelConfig {
        self.config
    }

    pub fn validate_limits(&self, limits: RuntimeLimits) -> bool {
        limits.n_ctx >= limits.n_batch
            && limits.n_batch >= limits.n_ubatch
            && limits.n_batch >= limits.n_seq_max
    }

    pub fn kv_cache_layout(&self) -> KvCacheLayout {
        KvCacheLayout {
            layers: (0..self.config.n_layers)
                .map(|layer_idx| LayerKvSpec {
                    layer_idx,
                    kv_heads: self.config.n_kv_heads,
                    head_dim: self.config.head_dim,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                })
                .collect(),
        }
    }

    pub fn n_embd_k_gqa(&self) -> u32 {
        self.config.n_kv_heads * self.config.head_dim
    }

    pub fn n_embd_v_gqa(&self) -> u32 {
        self.config.n_kv_heads * self.config.head_dim
    }
}

#[cfg(test)]
mod tests {
    use chew_runtime::RuntimeLimits;

    use super::*;

    #[test]
    fn llama_layout_is_full_attention_and_dedicated_kv() {
        let model = LlamaModel::new(LlamaModelConfig {
            n_layers: 4,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
        });

        let layout = model.kv_cache_layout();
        assert_eq!(layout.layer_count(), 4);
        assert_eq!(layout.full_attention_layer_count(), 4);
        assert_eq!(layout.sliding_window_layer_count(), 0);
        assert_eq!(layout.shared_group_count(), 0);
        assert_eq!(layout.dedicated_layer_count(), 4);
        assert_eq!(layout.layers[0].kv_heads, 8);
        assert_eq!(model.n_embd_k_gqa(), 8 * 128);
        assert_eq!(model.n_embd_v_gqa(), 8 * 128);
    }

    #[test]
    fn llama_limits_follow_runtime_contract() {
        let model = LlamaModel::new(LlamaModelConfig {
            n_layers: 4,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
        });

        assert!(model.validate_limits(RuntimeLimits {
            n_ctx: 8192,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 4,
        }));
        assert!(!model.validate_limits(RuntimeLimits {
            n_ctx: 1024,
            n_batch: 512,
            n_ubatch: 256,
            n_seq_max: 1024,
        }));
    }
}
