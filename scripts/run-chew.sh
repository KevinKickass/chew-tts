#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.gguf> [--port N] [--context N] [--tokenizer PATH] [extra args...]" >&2
  exit 1
fi

MODEL="$1"
shift

PORT="${PORT:-8080}"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--port" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    PORT="${ARGS[$((i + 1))]}"
    break
  fi
done

"$ROOT/scripts/kill-chew-port.sh" "$PORT"

cd "$ROOT"
cargo run --quiet -p chew-server -- "$MODEL" "${ARGS[@]}"
