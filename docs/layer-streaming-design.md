# Layer Streaming: GPU/CPU Mixed Inference

Run models that don't fit in VRAM by using an adaptive weight cache.

## When to use

| Model | Weights | Fits 10GB VRAM? | Mode |
|-------|---------|-----------------|------|
| Llama 3.1 8B Q4_K | 4.5G | ✓ | normal |
| Gemma 4 E4B Q4_K | 4.6G | ✓ | normal |
| Magistral Small 24B Q4_K | 13.3G | ❌ | **streaming** |
| OLMo 3 32B Q4_K | 18.1G | ❌ | **streaming** |

## Architecture: Adaptive Weight Cache

NOT a fixed 2-slot ping-pong. Instead: **fill VRAM with as many layers as possible**, stream only the rest.

```
VRAM:
┌──────────────────────────────────┐
│ Fixed: KV Cache + Scratch + Emb  │  ~2.5 GB (always resident)
├──────────────────────────────────┤
│ Weight Cache: max layers         │  ~5-7 GB (depends on model/ctx)
│ [L0][L1][L2]...[LN-1]           │  e.g. 44 of 48 layers for 24B
├──────────────────────────────────┤
│ DMA Buffer: 2 streaming slots    │  ~0.3 GB (for non-resident layers)
└──────────────────────────────────┘
```

### Key insight

For Magistral 24B on 10GB VRAM:
- Fixed overhead: ~2.5GB
- Available for weights: ~7.2GB
- Per layer: ~161MB
- **44 of 48 layers fit permanently** in VRAM
- Only 4 layers need streaming
- 4 × 6.4ms PCIe transfer = 25.6ms extra per token
- During the 44 resident layers (~8.8ms compute), we can preload ~1.4 streamed layers
- **Effective: ~35-40 tok/s** (vs 3.3 tok/s with naive 2-slot design)

### Predictive preloading

```
Resident layers:  [L0] [L1] ... [L43]     ← GPU compute, no DMA needed
                                     ↓ during this compute time, DMA streams ahead:
DMA stream:       ............[L44→slot A] [L45→slot B]
                  
Streamed layers:  [L44 from A] [L45 from B] [L46→A preloaded] [L47→B preloaded]
```

While GPU executes the resident layers, the DMA stream prefetches the first streamed layers. By the time GPU reaches L44, it's already in VRAM.

### Algorithm

```
1. At load time:
   - Calculate: n_resident = (vram_free - fixed - 2*layer_size) / layer_size
   - Load layers 0..n_resident directly to VRAM (permanent)
   - Allocate 2 DMA slots for remaining layers
   - Copy all layer weights to pinned host RAM

2. At decode time:
   for layer in 0..n_layers:
     if layer < n_resident:
       // Fast path: weights already in VRAM
       execute_layer(resident_weights[layer])
       // Predictive: if close to end of resident zone, start DMA
       if layer == n_resident - 2:
         dma_stream.copy(host[n_resident] → slot_a)
     else:
       // Streaming path: use DMA slot
       slot = (layer - n_resident) % 2
       compute_stream.wait(event[slot])  // wait for DMA
       execute_layer(slot_weights[slot])
       // Prefetch next streamed layer
       next_stream = n_resident + ((layer - n_resident) + 2)
       if next_stream < n_layers:
         dma_stream.copy(host[next_stream] → slot[other])
         dma_stream.record(event[other])
```

## VRAM Budget (Magistral Small 24B, 4k context)

| Component | Size |
|-----------|------|
| Token embeddings (Q4_K) | 378 MB |
| Output projection (Q4_K) | 378 MB |
| Norms (all layers, f16) | 2 MB |
| KV cache (4k context) | 960 MB |
| Scratch + cuBLAS | 165 MB |
| Driver headroom | 256 MB |
| **Fixed total** | **~2,139 MB** |
| Available for weights | **~7,731 MB** |
| Per-layer weight size | 161 MB |
| 2 DMA slots | 322 MB |
| **Resident layers** | **(7731 - 322) / 161 = 46** |
| Streamed layers | **2 of 48** |

With only 2 layers streaming: 2 × 6.4ms = 12.8ms extra, fully hidden by prefetch during resident compute. **Effectively zero overhead** — near full GPU speed (~45-50 tok/s for 24B).

## Performance Estimates

| Model | Layers | Resident | Streamed | Extra latency | Est. tok/s |
|-------|--------|----------|----------|---------------|-----------|
| Magistral 24B Q4_K (4k ctx) | 48 | 46 | 2 | ~0ms (hidden) | **~45** |
| Magistral 24B Q4_K (8k ctx) | 48 | 40 | 8 | ~25ms | **~30** |
| OLMo 32B Q4_K (4k ctx) | 64 | 30 | 34 | ~180ms | **~5** |
| OLMo 32B Q4_K (2k ctx) | 64 | 38 | 26 | ~130ms | **~7** |

## Implementation

### Files to modify

| File | Change |
|------|--------|
| `vram_plan.rs` | `compute_streaming()`: calculate n_resident, slot sizes |
| `weights.rs` | `StreamingWeights`: resident + host + slots |
| `forward.rs` | `forward_streaming()`: fast path + DMA path |
| `lib.rs` | Auto-fallback: try normal → try streaming |

### CUDA Graph compatibility

- **Resident layers**: CUDA Graph works (same pointers every time)
- **Streamed layers**: no Graph (pointers change)
- Hybrid: capture Graph for resident portion, fall through to non-graph for streamed layers

## OLMo2 Architecture

OLMo2 differs from Llama in **norm placement**: post-norm instead of pre-norm.

```
Llama:  x = x + attn(norm(x))     // norm BEFORE sublayer
OLMo2:  x = x + norm(attn(x))     // norm AFTER sublayer
```

Needs: new `forward_olmo2()` or parameterized norm placement in `forward()`.
New kernel: `rmsnorm_add_f32_f16` (norm the delta, add to residual).
Weight loading: same tensor names, just different application order.
