# Qwen3-TTS porting reference

Chew TTS ports Qwen3-TTS inference onto Chew's CUDA/NVRTC runtime. It does not
embed Candle, PyTorch, libtorch, or ONNX Runtime in the production path.

## Upstream references

- Official implementation: <https://github.com/QwenLM/Qwen3-TTS>
- Rust behavioral reference: <https://github.com/TrevorS/qwen3-tts-rs>
- Pinned Rust reference commit: `711ceee07cad92673f86de8997bdf54c30caa49f`

Both Chew and the Rust behavioral reference are MIT licensed. Code copied or
adapted from the reference must retain attribution in the affected source file.
The official implementation remains the source of truth when behavior differs.

## Runtime graph

For every 12.5 Hz audio frame:

1. The talker performs one autoregressive decode step and samples semantic
   codebook 0.
2. The five-layer code predictor performs 15 autoregressive steps for acoustic
   codebooks 1 through 15.
3. The 16 codes are embedded and summed to form the next talker input.
4. After generation, the speech tokenizer decoder converts all codebooks to a
   24 kHz mono waveform.

The hot path therefore has many small launches. The intended Chew
implementation captures one complete frame, including sampling and the 15
predictor steps, as a CUDA graph. Token IDs and positions live in stable device
buffers so generation does not require a host synchronization per codebook.

## Port order

1. Parse and validate native Hugging Face Safetensors.
2. Produce golden text embeddings and projected prompt embeddings.
3. Match talker prefill and one decode step against the official Python model.
4. Match all 15 code-predictor tokens for one frame.
5. Capture the frame hot path as a CUDA graph.
6. Port the 12 Hz speech-tokenizer decoder.
7. Add speaker encoder and reference-audio encoder for voice cloning.

F16 is the first correctness target because it works on both V100 and A6000.
BF16 may be enabled on Ampere after parity. Quantization follows only after the
F16 path produces matching codes and audio.
