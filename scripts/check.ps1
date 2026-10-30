# One-command local gate: everything CI runs, runnable before pushing.
# Usage: powershell -File scripts\check.ps1   (from the nimbus-edge root)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot/..

$packages = @("nimbus-edge-core", "nimbus-edge-server", "nimbus-edge-cli", "nimbus-edge-tests")
$features = @("--features", "nimbus-edge-core/test-support")

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Step "fmt"
$fmtArgs = @(); foreach ($p in $packages) { $fmtArgs += @("-p", $p) }
cargo fmt @fmtArgs -- --check
if ($LASTEXITCODE -ne 0) { exit 1 }

Step "clippy (-D warnings)"
cargo clippy --workspace --all-targets @features -- -D warnings
if ($LASTEXITCODE -ne 0) { exit 1 }

Step "tests (unit + integration + black-box)"
cargo test --workspace @features
if ($LASTEXITCODE -ne 0) { exit 1 }

Step "release build"
cargo build --release
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "`nCHECK: PASS" -ForegroundColor Green
