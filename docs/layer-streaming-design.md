# Layer Streaming: GPU/CPU Mixed Inference

Run models that don't fit in VRAM by streaming layer weights from pinned RAM.

## When to use

| Model | Weights | Fits 10GB VRAM? | Mode |
|-------|---------|-----------------|------|
| Llama 3.1 8B Q4_K | 4.5G | ✓ | normal |
| Gemma 4 E4B Q4_K | 4.6G | ✓ | normal |
| Magistral Small 24B Q4_K | 13.3G | ❌ | **streaming** |
| OLMo 3 32B Q4_K | 18.1G | ❌ | **streaming** |

## Architecture: Layer Ping-Pong

```
Compute stream: [exec L0 from A] → [exec L1 from B] → [exec L2 from A] → ...
DMA stream:        [copy L1 → B]  → [copy L2 → A]  → [copy L3 → B]  → ...
                   └─ event sync ─┘  └─ event sync ─┘
```

- 2 GPU slots (A, B), each holds 1 layer's quantized weights
- While GPU computes layer N from slot A, DMA copies layer N+1 into slot B
- Swap after each layer
- KV cache + token embeddings + output projection stay resident in VRAM

## VRAM Budget (Magistral Small 24B)

| Component | Size |
|-----------|------|
| 2 layer slots (2 × 161MB) | 322 MB |
| Token embeddings (Q4_K) | 378 MB |
| Output projection (Q4_K) | 378 MB |
| Output norm (f16) | 10 KB |
| All layer norms (f16, resident) | 2 MB |
| KV cache (4k context) | 960 MB |
| Scratch + cuBLAS | 165 MB |
| Driver headroom | 256 MB |
| **Total** | **~2.5 GB** |

Fits easily in 10GB VRAM. Could do 16k context (~3.8GB KV).

## Performance

| Metric | Value |
|--------|-------|
| Per-layer weight size | ~161 MB |
| PCIe 4.0 x16 bandwidth | ~25 GB/s |
| Transfer time per layer | **6.4 ms** |
| GPU compute time per layer | ~0.2 ms |
| Bottleneck | **PCIe** (30:1 ratio) |
| 48 layers × 6.4ms | ~307 ms/token |
| **Decode speed** | **~3.3 tok/s** |

The compute (0.2ms) cannot hide the transfer (6.4ms). Overlap helps minimally.

For **prefill** (seq_len ≥ 32): compute scales with batch → transfer IS hidden → near full speed.

## Implementation

### New structs

```
StreamingWeights {
    // Resident on GPU
    token_embd: QuantWeight,      // kept quantized, dequant on lookup
    output_norm: CudaSlice<f16>,
    output: QuantWeight,
    attn_norms: Vec<CudaSlice<f16>>,  // all layers, tiny
    ffn_norms: Vec<CudaSlice<f16>>,

    // Pinned host memory
    host_layers: PinnedBuffer<u8>,    // all layer weights
    layer_byte_offsets: Vec<(usize, usize)>,

    // Double-buffer GPU slots
    slot: [CudaSlice<u8>; 2],
    dma_stream: CudaStream,
    events: [CudaEvent; 2],
}
```

### Forward pass changes

```rust
fn forward_streaming(hidden, weights, config, kernels, kv_cache, scratch, seq_len) {
    // Layer 0 pre-loaded in slot 0
    for layer_idx in 0..n_layers {
        let cur = layer_idx % 2;
        let next = 1 - cur;

        // Wait for current slot DMA to complete
        compute_stream.wait(events[cur]);

        // Start DMA for layer+1 into other slot
        if layer_idx + 1 < n_layers {
            dma_stream.wait(compute_done_event);
            dma_stream.memcpy_htod(host[layer+1], slot[next]);
            dma_stream.record(events[next]);
        }

        // Build LayerWeights view from slot[cur]
        let layer = view_from_slot(slot[cur], layer_idx);

        // Run layer (same kernels as normal forward)
        execute_layer(hidden, layer, norms[layer_idx], ...);

        compute_stream.record(compute_done_event);
    }
    // Final norm + logits with resident weights
}
```

### Files to modify

| File | Change |
|------|--------|
| `vram_plan.rs` | Add `compute_streaming()` budget |
| `weights.rs` | Add `StreamingWeights`, pinned host loading |
| `forward.rs` | Add `forward_streaming()` with ping-pong |
| `lib.rs` | Auto-detect: try normal load, fallback to streaming |

### Incompatibilities

- **CUDA Graph**: disabled in streaming mode (weight pointers change per layer)
- **Fused norm+Q8**: works unchanged (norm weights stay resident)
- **Dual GEMV (K+V, Gate+Up)**: works unchanged (reads from current slot)

## OLMo2 Architecture

OLMo2 differs from Llama in **norm placement**: post-norm instead of pre-norm.

```
Llama:  x = x + attn(norm(x))     // norm BEFORE sublayer
OLMo2:  x = x + norm(attn(x))     // norm AFTER sublayer
```

Needs: new `forward_olmo2()` or parameterized norm placement in `forward()`.
New kernel: `rmsnorm_add_f32_f16` (norm the delta, add to residual).
Weight loading: same tensor names, just different application order.
