#!/bin/bash
# Wrapper: run nvcc with patched CUDA headers to fix glibc 2.42 conflict
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec /usr/local/cuda-13.1/bin/nvcc \
  -I"$SCRIPT_DIR/cuda_patched" \
  "$@"
