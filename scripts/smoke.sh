#!/usr/bin/env bash
# Smoke test a packaged artifact: start the server on the bundled tiny model
# and complete one non-streaming and one streaming chat completion.
# Usage: scripts/smoke.sh <extracted-artifact-dir>
set -euo pipefail

DIR="${1:?usage: smoke.sh <extracted-artifact-dir>}"
DIR="$(cd "$DIR" && pwd)"
BIN="$DIR/nimbus-edge"
MODEL_DIR="$DIR/models/tiny-gpt2"

[[ -x "$BIN" ]] || { echo "FAIL: $BIN not found (extract the artifact first)"; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "FAIL: bundled model missing"; exit 1; }

LOG="$(mktemp)"
"$BIN" serve --model "$MODEL_DIR" --port 0 > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  URL="$(grep -m1 '^listening on ' "$LOG" | awk '{print $3}')"
  [[ -n "${URL:-}" ]] && break
  sleep 0.2
done
[[ -n "${URL:-}" ]] || { echo "FAIL: server did not report its address"; cat "$LOG"; exit 1; }
echo "server up at $URL (pid $SERVER_PID)"

curl -fsS "$URL/healthz" > /dev/null || { echo "FAIL: healthz"; exit 1; }

BODY="$(curl -fsS "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"tiny-gpt2","messages":[{"role":"user","content":"hello"}],"max_tokens":8}')"
echo "$BODY" | grep -q '"object":"chat.completion"' || { echo "FAIL: chat body: $BODY"; exit 1; }
echo "non-streaming chat completion: OK"

STREAM="$(curl -fsS -N "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"tiny-gpt2","messages":[{"role":"user","content":"hi"}],"max_tokens":6,"stream":true}')"
echo "$STREAM" | grep -q 'data: \[DONE\]' || { echo "FAIL: stream end: $STREAM"; exit 1; }
echo "streaming chat completion:    OK"

kill "$SERVER_PID" 2>/dev/null || true
echo "SMOKE: PASS ($URL, model tiny-gpt2)"
