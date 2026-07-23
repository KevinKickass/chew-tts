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
- complete 15-codebook acoustic generation with GPU embeddings, deterministic
  argmax, and temperature/top-k sampling;
- reusable predictor KV/scratch sessions and exact GPU-resident Top-K sampling
  for the 2,048-token acoustic vocabulary;
- native decoding of all 16 codec codebooks into the 512-channel latent;
- the codec's causal pre-convolution and eight-layer transformer on CUDA;
- both 2x causal ConvNeXt codec upsampling stages;
- the complete BigVGAN waveform decoder, including SnakeBeta activations,
  dilated residual units, and all four upsampling blocks;
- joint multi-frame decoding with causal transformer attention and convolution
  history preserved across frame boundaries;
- mono 24-kHz PCM16 WAV output from the native codec path;
- direct code-predictor to continuous-codec integration;
- local Qwen2 byte-level BPE tokenization from `vocab.json` and `merges.txt`;
- native talker text/codec embeddings, SwiLU text projection, and semantic head;
- exact VoiceDesign ChatML/control-token prefill, persistent talker KV cache,
  autoregressive semantic/acoustic generation, and end-to-end WAV output;
- PyTorch parity checks for real Qwen weights, RoPE, GQA, and cached decoding.

Next:

- one CUDA graph for a complete 16-codebook audio frame;
- optimized one-pass model loading and GPU-resident sampling;
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

Add `--upsample --repeats 10` to validate and benchmark both causal ConvNeXt
upsampling stages. After one cuBLAS warm-up pass, the complete path through the
four 1024-channel output steps takes approximately 1.1 ms per frame on the same
GPU. Against the independent F32 reference, the mean absolute delta is below
0.002.

Add `--audio --repeats 10` to run through the complete BigVGAN decoder and
produce one 1,920-sample frame at 24 kHz. The warm single-frame path takes
approximately 24 ms on an RTX 3080, including the codec front end. Its mean
absolute delta against the independent F32 waveform reference is below 0.0002
and its maximum delta below 0.0009.

Decode several frames jointly and write a WAV file:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  cuda-codec-latent-smoke /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/speech_tokenizer \
  --gpu 0 --audio --frames 3 --repeats 10 --wav /tmp/codec.wav
```

The multi-frame path preserves the causal transformer and convolution context
instead of joining independently decoded 80-ms chunks. Three frames take
approximately 69 ms warm for 240 ms of output on an RTX 3080 (codec RTF 0.29).
All 5,760 samples match the independent F32 path with a mean absolute delta
below 0.00018 and a maximum delta below 0.00085.

Run the predictor-to-codec integration without manually supplying acoustic
codebooks:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  cuda-predictor-codec-smoke /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --gpu 0 --semantic-token 42 --frames 3 --wav /tmp/predictor-codec.wav
```

This currently uses deterministic prepared talker hidden states to exercise
the boundary. On an RTX 3080, predictor plus codec occupy approximately
589 MiB and generate three distinct acoustic code frames plus 240 ms of valid
PCM audio end to end.

The local tokenizer matches Hugging Face token IDs without requiring Python or
network access:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  tokenize /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign "Hallo Welt!"
```

`cuda-talker-frontend-smoke` validates token embeddings, the two-layer SwiLU
text projection, codec control-token embeddings, and the semantic codec head.
The frontend occupies approximately 684 MiB on an RTX 3080. Against the
independent F32 PyTorch path, projected text has a mean absolute delta below
0.000008 and both paths select the same semantic argmax token.

Run the complete native VoiceDesign path:

```bash
CARGO_TARGET_DIR=/tmp/chew-tts-target \
  cargo run --release -p chew-tts -- \
  cuda-voice-design-smoke /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --gpu 0 \
  --language german \
  --text "Guten Abend, schön dass du da bist." \
  --instruction "A quiet, breathy German female whisper." \
  --max-frames 2048 \
  --seed 42 \
  --wav /tmp/voice-design.wav
```

The command uses the model's production sampling defaults: temperature 0.9,
top-k 50, and semantic repetition penalty 1.05. Generation normally ends at
codec EOS; `--max-frames` is only a safety limit and reports an error instead
of silently presenting truncated audio. On an RTX 3080, the first
correctness-oriented implementation occupies approximately 3.9 GiB of VRAM
and generates 0.96 seconds of sampled audio in approximately 0.77 seconds
(RTF 0.80). The optimized binary maps and uploads the complete model in
approximately 4.2 seconds from a warm filesystem cache. Debug builds are not
representative because their element-wise F16 conversion is intentionally
unoptimized.

A 48-frame stability run produces 3.84 seconds of continuous audio in
approximately 2.14 seconds (RTF 0.56). Convolution kernels place the audio
timeline on CUDA's large X grid dimension, so output is not limited by the
65,535-block Y dimension once a waveform exceeds roughly 2.7 seconds.

By default, generated codes are drained through a bounded 32-frame codec
buffer with 64 previous frames of causal context. This caps the codec working
set at 96 frames and enables early audio delivery. PCM chunks are written
immediately through a streaming WAV sink; generated audio is not accumulated
in RAM, and the header is finalized at EOS. For the 6.88-second
VoiceDesign validation sample, chunked output differs from full-sequence
decoding by only 0.72 PCM16 levels on average (26 maximum) and remains faster
than real time at RTF 0.83. Use `--chunk-frames 0` for exact full-sequence
decoding.

The current optimized acoustic path sustains roughly 95% GPU utilization on an
RTX 3080. Moving exact Top-K 50 sampling onto the GPU reduced a 2.48-second
VoiceDesign validation run to approximately 1.35 seconds total inference
(RTF 0.54), while repeated runs with the same seed produce byte-identical WAV
files.

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
