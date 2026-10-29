# Smoke test a packaged artifact: start the server on the bundled tiny model
# and complete one non-streaming and one streaming chat completion.
# Usage: pwsh -File scripts/smoke.ps1 -ArtifactDir <extracted-artifact-dir>
param(
    [Parameter(Mandatory = $true)][string]$ArtifactDir
)
$ErrorActionPreference = "Stop"

$dir = (Resolve-Path $ArtifactDir).Path
$bin = Join-Path $dir "nimbus-edge.exe"
$model = Join-Path $dir "models/tiny-gpt2"
if (-not (Test-Path $bin)) { Write-Host "FAIL: $bin not found"; exit 1 }
if (-not (Test-Path "$model/config.json")) { Write-Host "FAIL: bundled model missing"; exit 1 }

$log = "$env:TEMP/nimbus-edge-smoke-$PID.log"
$proc = Start-Process -FilePath $bin -ArgumentList @("serve", "--model", $model, "--port", "0") `
    -RedirectStandardOutput $log -RedirectStandardError "$env:TEMP/nimbus-edge-smoke-$PID.err.log" `
    -PassThru -NoNewWindow
try {
    $url = $null
    foreach ($i in 1..100) {
        $line = (Get-Content $log -ErrorAction SilentlyContinue | Where-Object { $_ -like "listening on *" } | Select-Object -First 1)
        if ($line) { $url = ($line -split " ")[2]; break }
        Start-Sleep -Milliseconds 200
    }
    if (-not $url) { Write-Host "FAIL: server did not report its address"; Get-Content $log; exit 1 }
    Write-Host "server up at $url (pid $($proc.Id))"

    $health = Invoke-RestMethod -Uri "$url/healthz" -TimeoutSec 5
    if ($health.status -ne "ok") { Write-Host "FAIL: healthz"; exit 1 }

    $body = @{
        model    = "tiny-gpt2"
        messages = @(@{ role = "user"; content = "hello" })
        max_tokens = 8
    } | ConvertTo-Json -Depth 5
    $resp = Invoke-RestMethod -Uri "$url/v1/chat/completions" -Method Post -Body $body `
        -ContentType "application/json" -TimeoutSec 30
    if ($resp.object -ne "chat.completion") { Write-Host "FAIL: chat body"; exit 1 }
    Write-Host "non-streaming chat completion: OK"

    # Streaming: read the raw SSE body and require [DONE].
    $streamBody = @{
        model    = "tiny-gpt2"
        messages = @(@{ role = "user"; content = "hi" })
        max_tokens = 6
        stream   = $true
    } | ConvertTo-Json -Depth 5
    $req = [System.Net.HttpWebRequest]::Create("$url/v1/chat/completions")
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Timeout = 30000
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($streamBody)
    $reqStream = $req.GetRequestStream()
    $reqStream.Write($bytes, 0, $bytes.Length)
    $reqStream.Close()
    $reader = New-Object System.IO.StreamReader($req.GetResponse().GetResponseStream())
    $text = $reader.ReadToEnd()
    $reader.Close()
    if ($text -notmatch "data: \[DONE\]") { Write-Host "FAIL: stream end: $text"; exit 1 }
    Write-Host "streaming chat completion:    OK"

    Write-Host "SMOKE: PASS ($url, model tiny-gpt2)"
}
finally {
    try { $proc.Kill() } catch {}
    Remove-Item $log, "$env:TEMP/nimbus-edge-smoke-$PID.err.log" -ErrorAction SilentlyContinue
}
