#!/usr/bin/env bash
# Package the Linux release artifact: binary + docs + bundled tiny model,
# then produce the SHA256 checksum file. Run from the nimbus-edge root after
# `cargo build --release`. Requires ../nimbus-infer-rs (the model fixture).
set -euo pipefail

VERSION="$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["packages"][0]["version"])')"
OUT="dist/nimbus-edge-v${VERSION}-x86_64-unknown-linux-gnu"
STAGE="$(mktemp -d)/nimbus-edge-v${VERSION}-linux-x64"

mkdir -p "$STAGE/models" dist
cp target/release/nimbus-edge "$STAGE/"
cp README.md LICENSE "$STAGE/"

# Bundle the tiny fixture model (random weights, ~3.4 MB) so the artifact
# works offline. A sidecar edge-meta.json enables chat completions.
cp -r ../nimbus-infer-rs/tests/fixtures/tiny-gpt2 "$STAGE/models/tiny-gpt2"
cat > "$STAGE/models/tiny-gpt2/edge-meta.json" <<META
{
  "format": "nimbus-checkpoint-v1",
  "name": "tiny-gpt2",
  "created_at": "1970-01-01T00:00:00Z",
  "source": "bundled fixture (random weights, for smoke testing only)",
  "chat_template": "chatml",
  "arch": "gpt2",
  "quant_hint": "fp32",
  "files": {}
}
META

tar -czf "$OUT.tar.gz" -C "$(dirname "$STAGE")" "$(basename "$STAGE")"
sha256sum "$OUT.tar.gz" > "$OUT.tar.gz.sha256"
echo "packaged: $OUT.tar.gz"
sha256sum "$OUT.tar.gz"
