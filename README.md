# Chew TTS

Chew TTS is a Rust and CUDA speech-synthesis engine built for predictable VRAM
usage, high throughput, and a single deployable binary. CUDA kernels are
compiled at startup with NVRTC for the GPU that is actually present.

The project is a TTS-focused fork of
[Chew](https://github.com/KevinKickass/chew). It intentionally has its own
repository and release lifecycle: Fleet can deploy a small speech engine
without carrying an LLM API or llama.cpp dependency.

## Status

Qwen3-TTS support is under active development.

Implemented:

- model-independent TTS request, audio, and capability types;
- memory-mapped native Safetensors access;
- Qwen3-TTS Base, CustomVoice, and VoiceDesign configuration parsing;
- validation of the talker and code-predictor geometry;
- inspection of real Hugging Face model directories;
- native F16 CUDA execution of a complete Qwen talker decoder layer;
- causal multi-token prefill and incremental decoding with a native KV cache;
- GPU-resident execution of all 28 talker layers without host round-trips;
- a native dense F16 decode GEMV path;
- GPU-resident execution of the five-layer Qwen code predictor;
- complete 15-codebook acoustic generation with GPU embeddings and argmax;
- native decoding of all 16 codec codebooks into the 512-channel latent;
- the codec's causal pre-convolution and eight-layer transformer on CUDA;
- PyTorch parity checks for real Qwen weights, RoPE, GQA, and cached decoding.

Next:

- one CUDA graph for a complete 16-codebook audio frame;
- the convolutional upsampling stages of the 12 Hz speech-tokenizer decoder;
- speaker and reference-audio encoders for voice cloning;
- Kokoro as the second model family.

## Why a separate repository?

TTS and LLM serving share CUDA primitives, but not their product surface or
hot path. Qwen3-TTS performs one large talker step followed by 15 small
code-predictor steps for every 12.5 Hz audio frame. It also needs convolutional
audio decoders, resampling, phonemization, and audio encoders that an LLM
server does not need.

The fork currently retains Chew's proven LLM crates as implementation
references while the CUDA runtime is separated from GGUF-specific types. They
will leave the final TTS binary and eventually the workspace after the Qwen
path is verified.

## Workspace

```text
bin/
  chew-tts/             TTS CLI and, later, HTTP server
crates/
  tts-core/             common request, audio, and capability types
  safetensors/          mmap Safetensors access
  model-qwen3-tts/      Qwen configuration and inference
  kernel/               inherited CUDA/NVRTC kernels
  vram/                 inherited VRAM ownership and budgeting
```

The inherited `gguf`, `engine`, and `server` crates are temporary references
and are not dependencies of the `chew-tts` binary.

## Inspect a model

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  inspect /models/Qwen3-TTS-12Hz-1.7B-Base
```

Example:

```text
Qwen3-TTS 1b7 Base
talker: 28 layers, hidden 2048, 16 Q heads / 8 KV heads
code predictor: 5 layers, hidden 1024, 15 acoustic steps/frame
weights: 480 tensors in 1 file(s), 3.59 GiB
```

## CUDA validation

Compile the inherited kernels with NVRTC for the installed GPU:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- cuda-smoke --gpu 0
```

Run one real Qwen talker decoder layer:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  cuda-layer-smoke /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --gpu 0 --layer 0 --seq-len 4 --decode-split 2
```

An optional raw little-endian F32 output can be supplied with `--reference`.
The command then fails if CUDA and the reference differ beyond the configured
correctness tolerance. `--decode-split` fills the KV cache with a prompt prefix
and processes the remaining tokens individually.

Run the complete talker stack from a prepared hidden state:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  cuda-talker-smoke /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --gpu 0
```

On an RTX 3080, the initial dense F16 implementation loads the 28-layer talker
into approximately 2.7 GiB of VRAM and executes one synthetic decode token in
approximately 7 ms. The five-layer code predictor occupies another 172 MiB and
executes in approximately 0.7 ms per codebook step. A warm, complete
15-codebook predictor frame including projection, embeddings, heads, and GPU
argmax takes approximately 10 ms. Talker plus predictor therefore take roughly
17 ms per 80-ms audio frame before the speech codec.

Decode one complete codec frame through its quantizers, causal pre-convolution,
and eight-layer transformer:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  cuda-codec-latent-smoke /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/speech_tokenizer \
  --gpu 0 --transformer
```

This codec front end uses approximately 108 MiB of VRAM and takes approximately
0.8 ms per frame on an RTX 3080. Its output is checked against an independent
F32 PyTorch implementation; the measured maximum absolute delta is below
0.00005 for the documented validation frame.

## Requirements

- NVIDIA GPU with compute capability 7.0 or newer
- CUDA toolkit with NVRTC
- Rust 2024 edition

F16 is the common correctness baseline for V100 and A6000. BF16 and
architecture-specific fast paths are enabled only after output parity.

## References and licensing

Chew TTS is MIT licensed. Qwen3-TTS porting behavior is checked against the
official implementation and the pinned MIT Rust reference documented in
[docs/PORTING_QWEN3_TTS.md](docs/PORTING_QWEN3_TTS.md).
