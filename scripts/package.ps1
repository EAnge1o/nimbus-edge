# Package the Windows release artifact: binary + docs + bundled tiny model,
# then produce the SHA256 checksum file. Run from the nimbus-edge root after
# `cargo build --release`. Requires ..\nimbus-infer-rs (the model fixture).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot/..

$version = (cargo metadata --no-deps --format-version 1 | ConvertFrom-Json).packages[0].version
$out = "dist/nimbus-edge-v$version-x86_64-pc-windows-msvc"
$stage = "$env:TEMP/nimbus-edge-v$version-windows-x64"

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path "$stage/models", dist | Out-Null

Copy-Item target/release/nimbus-edge.exe $stage/
Copy-Item README.md, LICENSE $stage/

# Bundle the tiny fixture model (random weights, ~3.4 MB) so the artifact
# works offline. A sidecar edge-meta.json enables chat completions.
Copy-Item -Recurse ../nimbus-infer-rs/tests/fixtures/tiny-gpt2 "$stage/models/tiny-gpt2"
@'
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
'@ | Set-Content -Encoding ASCII "$stage/models/tiny-gpt2/edge-meta.json"

Compress-Archive -Path "$stage" -DestinationPath "${out}.zip" -Force
$hash = (Get-FileHash "${out}.zip" -Algorithm SHA256).Hash.ToLower()
"$hash  $(Split-Path "${out}.zip" -Leaf)" | Set-Content -Encoding ASCII "${out}.zip.sha256"
Write-Host "packaged: ${out}.zip"
Write-Host "$hash  $(Split-Path "${out}.zip" -Leaf)"
