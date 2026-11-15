# Starts Ollama check -> FastAPI backend -> Cloudflare quick tunnel, then prints the public URL.
# Usage (PowerShell, project root):  powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path outputs | Out-Null

# 1. API key in .env (created once)
if (-not (Test-Path .env)) {
    $key = -join ((48..57) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
    "CAR_API_KEY=$key" | Out-File -Encoding ascii .env
    Write-Host "Created .env with a new CAR_API_KEY"
}
$env:CAR_API_KEY = ((Get-Content .env | Where-Object { $_ -like "CAR_API_KEY=*" }) -split "=", 2)[1]

# 2. Model check (fall back to the base model if the fine-tuned one is missing)
$models = ollama list | Out-String
if ($models -match "car-assistant") { $env:LLM_MODEL = "car-assistant" }
elseif ($models -match "qwen2.5:3b-instruct") {
    $env:LLM_MODEL = "qwen2.5:3b-instruct"
    Write-Warning "car-assistant not found in Ollama: using base qwen2.5:3b-instruct"
} else { throw "No model found. Run: ollama pull qwen2.5:3b-instruct" }
Write-Host "Model: $env:LLM_MODEL"

# 3. Backend
Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
$api = Start-Process -PassThru -WindowStyle Hidden -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8000" `
    -RedirectStandardOutput outputs\api.log -RedirectStandardError outputs\api.err.log
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-RestMethod http://127.0.0.1:8000/health).status -eq "ok") { break } } catch {}
    Start-Sleep -Seconds 1
}
Write-Host "Backend running (PID $($api.Id))"

# 4. Tunnel
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared not installed. Run: winget install --id Cloudflare.cloudflared"
}
Remove-Item outputs\tunnel.log -ErrorAction SilentlyContinue
$tun = Start-Process -PassThru -WindowStyle Hidden -FilePath "cloudflared" `
    -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000" -RedirectStandardError outputs\tunnel.log
$url = $null
for ($i = 0; $i -lt 60 -and -not $url; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path outputs\tunnel.log) {
        $m = Select-String -Path outputs\tunnel.log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value }
    }
}
if (-not $url) { throw "Tunnel URL not found; see outputs\tunnel.log" }
for ($i = 0; $i -lt 30; $i++) {
    try { if ((Invoke-RestMethod "$url/health").status -eq "ok") { break } } catch {}
    Start-Sleep -Seconds 2
}
$url | Out-File -Encoding ascii outputs\backend_url.txt
Write-Host ""
Write-Host "PUBLIC BACKEND URL: $url" -ForegroundColor Green
Write-Host "Paste into Streamlit secrets as BACKEND_URL, with CAR_API_KEY = $env:CAR_API_KEY"
Write-Host "Stop with: Stop-Process -Id $($api.Id),$($tun.Id)"
