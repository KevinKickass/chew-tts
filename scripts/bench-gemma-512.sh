#!/usr/bin/env bash
set -euo pipefail

ROOT="/run/media/kevin/KioxiaNVMe/KI-kram/chew"
PORT="${PORT:-9093}"
RUNS="${RUNS:-3}"
LOG_FILE="${LOG_FILE:-/tmp/chew-gemma-profile.log}"
PAYLOAD="${PAYLOAD:-{\"messages\":[{\"role\":\"user\",\"content\":\"Say only: hi\"}],\"max_tokens\":8,\"temperature\":0}}"
OUT_PREFIX="${OUT_PREFIX:-/tmp/chew-bench}"

cd "$ROOT"

echo "Benchmarking running chew server"
echo "  port:    $PORT"
echo "  runs:    $RUNS"
echo "  log:     $LOG_FILE"
echo "  out:     ${OUT_PREFIX}-N.json"
echo

for i in $(seq 1 "$RUNS"); do
  out_file="${OUT_PREFIX}-${i}.json"
  echo "=== RUN $i ==="
  /usr/bin/time -f 'wall_s=%e rss_kb=%M' \
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$PAYLOAD" \
      -o "$out_file"

  python - <<PY
import json
path = ${out_file@Q}
with open(path) as f:
    obj = json.load(f)
print("content=", obj["choices"][0]["message"]["content"])
print("usage=", obj["usage"])
PY
  echo
  sleep 1
done

echo "Recent decode timing lines:"
grep -E "decode timing|PROFILE decode step|streaming scheduler stats" "$LOG_FILE" | tail -40 || true
