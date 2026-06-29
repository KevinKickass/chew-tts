//! Gemma 4 model semantics.
//! SWA, KV sharing, per-layer head dimensions and other Gemma-specific behavior live here.

use chew_runtime::{AttentionMode, KvCacheLayout, KvSharing, LayerKvSpec, RuntimeLimits};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma4ModelConfig {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub sliding_window: u32,
    pub full_attention_stride: u32,
    pub shared_kv_layers: u32,
}

#[derive(Debug)]
pub struct Gemma4Model {
    config: Gemma4ModelConfig,
}

impl Gemma4Model {
    pub fn new(config: Gemma4ModelConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> Gemma4ModelConfig {
        self.config
    }

    pub fn validate_limits(&self, limits: RuntimeLimits) -> bool {
        limits.n_ctx >= limits.n_batch
            && limits.n_batch >= limits.n_ubatch
            && limits.n_batch >= limits.n_seq_max
    }

    pub fn kv_cache_layout(&self) -> KvCacheLayout {
        let full_stride = self.config.full_attention_stride.max(1);
        let shared_prefix = self.config.shared_kv_layers.min(self.config.n_layers);

        KvCacheLayout {
            layers: (0..self.config.n_layers)
                .map(|layer_idx| {
                    let attention = if layer_idx % full_stride == 0 {
                        AttentionMode::FullCausal
                    } else {
                        AttentionMode::SlidingWindow {
                            window: self.config.sliding_window,
                        }
                    };
                    let sharing = if layer_idx < shared_prefix {
                        // Exact Gemma 4 grouping will come from model metadata later.
                        KvSharing::Shared { group: 0 }
                    } else {
                        KvSharing::Dedicated
                    };
                    LayerKvSpec {
                        layer_idx,
                        kv_heads: self.config.n_kv_heads,
                        head_dim: self.config.head_dim,
                        attention,
                        sharing,
                    }
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
    use chew_runtime::{AttentionMode, KvSharing, RuntimeLimits};

    use super::*;

    #[test]
    fn gemma4_layout_captures_sliding_window_and_shared_kv_prefix() {
        let model = Gemma4Model::new(Gemma4ModelConfig {
            n_layers: 6,
            n_kv_heads: 4,
            head_dim: 256,
            sliding_window: 1024,
            full_attention_stride: 3,
            shared_kv_layers: 2,
        });

        let layout = model.kv_cache_layout();
        assert_eq!(layout.layer_count(), 6);
        assert_eq!(layout.full_attention_layer_count(), 2);
        assert_eq!(layout.sliding_window_layer_count(), 4);
        assert_eq!(layout.shared_group_count(), 1);
        assert_eq!(layout.dedicated_layer_count(), 4);
        assert_eq!(layout.layers[0].attention, AttentionMode::FullCausal);
        assert_eq!(
            layout.layers[1].attention,
            AttentionMode::SlidingWindow { window: 1024 }
        );
        assert_eq!(layout.layers[0].sharing, KvSharing::Shared { group: 0 });
        assert_eq!(layout.layers[2].sharing, KvSharing::Dedicated);
        assert_eq!(model.n_embd_k_gqa(), 4 * 256);
        assert_eq!(model.n_embd_v_gqa(), 4 * 256);
    }

    #[test]
    fn gemma4_limits_follow_runtime_contract() {
        let model = Gemma4Model::new(Gemma4ModelConfig {
            n_layers: 6,
            n_kv_heads: 4,
            head_dim: 256,
            sliding_window: 1024,
            full_attention_stride: 3,
            shared_kv_layers: 2,
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
