#!/usr/bin/env bash
# One-command local gate: everything CI runs, runnable before pushing.
set -euo pipefail
cd "$(dirname "$0")/.."

PKGS=(-p nimbus-edge-core -p nimbus-edge-server -p nimbus-edge-cli -p nimbus-edge-tests)
FEATURES=(--features nimbus-edge-core/test-support)

echo "=== fmt ==="
cargo fmt "${PKGS[@]}" -- --check

echo "=== clippy (-D warnings) ==="
cargo clippy --workspace --all-targets "${FEATURES[@]}" -- -D warnings

echo "=== tests (unit + integration + black-box) ==="
cargo test --workspace "${FEATURES[@]}"

echo "=== release build ==="
cargo build --release

echo
echo "CHECK: PASS"
