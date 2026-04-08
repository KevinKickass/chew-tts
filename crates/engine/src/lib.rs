// chew-engine: Transformer inference engine
//
// Planned:
// - Model loading: GGUF parse → VRAM alloc → upload weights
// - Forward pass: embed → N * (attention + MLP) → logits
// - KV cache management (in our VRAM pool)
// - Sampling: top-k, top-p, temperature, repetition penalty
// - Tokenizer integration (HuggingFace tokenizers crate)

pub struct ChewEngine;
