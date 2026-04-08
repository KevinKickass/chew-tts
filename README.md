# Chew

GPU inference engine for GGUF models. Full VRAM control, quantized weights stay quantized on GPU, on-the-fly dequant per GEMM.

Built in Rust with raw CUDA (cudarc + NVRTC), no llama.cpp dependency.

## Features

- **Exact VRAM budgeting** — knows before loading whether the model fits. No OOM surprises.
- **Quantized weights on GPU** — Q4_K_M, Q6_K, IQ2_S, etc. stay quantized in VRAM. Only dequantized on-the-fly per matrix multiply into a reusable 128 MB scratch buffer.
- **Chunked output GEMM** — large matrices (e.g. output [128k, 4096]) are processed in chunks, keeping the dequant scratch small.
- **Auto context fitting** — tries max context, halves until it fits with 256 MB headroom.
- **OpenAI-compatible API** — `/v1/chat/completions` endpoint.
- **Built-in Web UI** — dark theme chat interface at `/`.
- **NVRTC runtime compilation** — CUDA kernels compiled at startup, auto-detects GPU arch and patches PTX versions.

## Architecture

```
crates/
  gguf/      GGUF parser (tensors, metadata, quantization types)
  vram/      VRAM allocator (multi-GPU ready)
  kernel/    CUDA kernels (dequant, ops, GEMM) via NVRTC
  engine/    Inference engine (weights, forward pass, KV cache, sampling, VRAM plan)
  server/    HTTP server (Axum, OpenAI API, Web UI)
```

### VRAM Budget

Before allocating anything, `VramPlan` computes exact requirements:

| Component | Description |
|-----------|-------------|
| Weights | Quantized tensors at disk size + norms/embeddings as f16 |
| Dequant scratch | 128 MB cap — largest weight chunk dequanted to f16 for cuBLAS |
| KV cache | 2 * layers * context * kv_heads * head_dim * 2 bytes |
| Forward scratch | All intermediate buffers (norm, QKV, FFN, logits) |
| cuBLAS workspace | 32 MB |
| Loading peak | Temporary overhead during upload_and_dequant |

The plan reports **steady-state** and **peak** (during loading) VRAM, and a clear **FITS** or **DOES NOT FIT** verdict.

### Weight Storage

Large weight matrices stay quantized on GPU (`CudaSlice<u8>` + `GgmlType`). Per GEMM call:

1. Dequant chunk of weight matrix into 128 MB f16 scratch buffer (GPU kernel)
2. cuBLAS hgemm on the f16 chunk
3. Repeat for remaining chunks (only needed for matrices > 64M elements like output projection)

Small tensors (norms, embeddings) are dequanted once to f16 at load time.

### Forward Pass

Standard Llama transformer: RMSNorm -> QKV projection -> RoPE -> KV cache -> Fused MHA (with GQA) -> Output projection -> SiLU FFN -> Residual connections -> Final norm -> Logits.

All matmuls use on-the-fly dequant from quantized weights.

## Usage

### Run server

```bash
cargo run --release -- /path/to/model.gguf \
  --tokenizer /path/to/tokenizer.json \
  --port 9090 \
  --context 4096
```

Open `http://localhost:9090` for the Web UI.

### VRAM budget check (without loading)

```bash
cargo run --example vram_plan -- /path/to/model.gguf
```

Output:

```
Context     Weights  Dequant       KV  Scratch   cuBLAS    Total     Peak
--------------------------------------------------------------------------
512          5404 MB    128 MB     64 MB    193 MB     32 MB   5822 MB   6104 MB
1024         5404 MB    128 MB    128 MB    386 MB     32 MB   6079 MB   6361 MB
2048         5404 MB    128 MB    256 MB    773 MB     32 MB   6593 MB   6875 MB
...

>>> FITS - GO <<<
```

### API

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Health check

```bash
curl http://localhost:9090/health
```

## Requirements

- NVIDIA GPU with compute capability >= 7.0
- CUDA toolkit (for NVRTC runtime compilation)
- Rust 2024 edition

## Supported Quantization Types

Q4_0, Q4_K, Q6_K, Q8_0, Q2_K, Q3_K, BF16, F16, F32, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS

## Workspace Crates

| Crate | Description |
|-------|-------------|
| `chew-gguf` | GGUF file parser — tensors, metadata, CPU dequantization |
| `chew-vram` | GPU memory allocator with device enumeration |
| `chew-kernel` | CUDA kernels compiled at runtime via NVRTC |
| `chew-engine` | Core inference engine — weights, forward pass, KV cache, sampling |
| `chew-server` | HTTP server with OpenAI-compatible API and Web UI |
