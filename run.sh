#!/bin/bash
pkill -9 -f "target/release/chew" 2>/dev/null; sleep 2
cd /run/media/kevin/KioxiaNVMe/KI-kram/chew
cargo build --release 2>&1 | tail -2
RUST_LOG=${RUST_LOG:-info} exec ./target/release/chew \
  "/run/media/kevin/KioxiaNVMe/NVMeR0/AI/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  --tokenizer "/run/media/kevin/KioxiaNVMe/KI-kram/models/llama3.1-8b-exl2-4bpw/tokenizer.json"
