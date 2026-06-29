#!/usr/bin/env bash
set -euo pipefail

ROOT="/run/media/kevin/KioxiaNVMe/KI-kram/chew"
MODEL="${MODEL:-/run/media/kevin/KioxiaNVMe/KI-kram/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-/run/media/kevin/KioxiaNVMe/KI-kram/models/gemma4-tokenizer.json}"
PORT="${PORT:-9093}"
CONTEXT="${CONTEXT:-512}"
LOG_FILE="${LOG_FILE:-/tmp/chew-gemma-profile.log}"
PAYLOAD="${PAYLOAD:-{\"messages\":[{\"role\":\"user\",\"content\":\"Say only: hi\"}],\"max_tokens\":8,\"temperature\":0}}"

cd "$ROOT"

"$ROOT/scripts/kill-chew-port.sh" "$PORT"

echo "Starting chew profile run"
echo "  model:    $MODEL"
echo "  tokenizer:$TOKENIZER"
echo "  port:     $PORT"
echo "  context:  $CONTEXT"
echo "  log:      $LOG_FILE"

echo "Building release binary..."
cargo build --release -p chew-server >/dev/null

echo "Launching server..."
RUST_LOG="${RUST_LOG:-info}" CHEW_PROFILE=1 \
  ./target/release/chew \
  "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --port "$PORT" \
  --context "$CONTEXT" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"

echo "Waiting for server to come up..."
for _ in $(seq 1 60); do
  if grep -q "listening" "$LOG_FILE" 2>/dev/null; then
    break
  fi
  sleep 1
done

if ! grep -q "listening" "$LOG_FILE" 2>/dev/null; then
  echo "Server did not start. Last log lines:"
  tail -80 "$LOG_FILE" || true
  exit 1
fi

echo "Server ready on port $PORT"
echo "Warmup request..."
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD"
echo

echo "Recent profiling lines:"
grep -E "PROFILE decode step|decode timing|streaming scheduler stats" "$LOG_FILE" | tail -20 || true
