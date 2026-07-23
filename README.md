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
- inspection of real Hugging Face model directories.

Next:

- F16 talker prefill and decode on Chew CUDA kernels;
- code-predictor decode and GPU-side sampling;
- one CUDA graph for a complete 16-codebook audio frame;
- the 12 Hz speech-tokenizer decoder;
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
