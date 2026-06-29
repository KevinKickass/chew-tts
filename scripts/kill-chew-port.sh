#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <port>" >&2
  exit 1
fi

PORT="$1"

declare -a PIDS=()

while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  if [[ "$pid" -eq "$$" ]]; then
    continue
  fi
  cmdline_file="/proc/$pid/cmdline"
  [[ -r "$cmdline_file" ]] || continue
  cmdline="$(tr '\0' ' ' < "$cmdline_file")"
  if [[ "$cmdline" == *"/chew "* && "$cmdline" == *"--port $PORT"* ]]; then
    PIDS+=("$pid")
  fi
done < <(pgrep -f -- "--port $PORT" || true)

if [[ ${#PIDS[@]} -eq 0 ]]; then
  exit 0
fi

echo "Stopping stale chew processes on port $PORT: ${PIDS[*]}"
kill "${PIDS[@]}" 2>/dev/null || true

for _ in $(seq 1 20); do
  declare -a ALIVE=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      ALIVE+=("$pid")
    fi
  done
  if [[ ${#ALIVE[@]} -eq 0 ]]; then
    exit 0
  fi
  PIDS=("${ALIVE[@]}")
  sleep 0.2
done

echo "Force-killing chew processes on port $PORT: ${PIDS[*]}"
kill -9 "${PIDS[@]}" 2>/dev/null || true
