# Voiceprint System - Windows Setup Script
# Usage: powershell -ExecutionPolicy ByPass -File scripts/setup.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Voiceprint System - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Install uv
Write-Host "[1/5] Checking uv..." -ForegroundColor Yellow
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  Installing uv..." -ForegroundColor Gray
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "  [FAIL] uv installation failed. Install manually: https://docs.astral.sh/uv/" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [OK] uv found: $(uv --version)" -ForegroundColor Green
Write-Host ""

# 2. Install ffmpeg
Write-Host "[2/5] Checking ffmpeg..." -ForegroundColor Yellow
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "  ffmpeg not found. Attempting install via winget..." -ForegroundColor Gray
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Host "  [WARN] ffmpeg installed but not in PATH. Restart your terminal." -ForegroundColor DarkYellow
        } else {
            Write-Host "  [OK] ffmpeg installed" -ForegroundColor Green
        }
    } else {
        Write-Host "  [WARN] winget not found. Please install ffmpeg manually:" -ForegroundColor DarkYellow
        Write-Host "    Download: https://ffmpeg.org/download.html" -ForegroundColor Gray
    }
} else {
    Write-Host "  [OK] ffmpeg found" -ForegroundColor Green
}
Write-Host ""

# 3. Install Python via uv
Write-Host "[3/5] Ensuring Python 3.11..." -ForegroundColor Yellow
uv python install 3.11
Write-Host "  [OK] Python ready" -ForegroundColor Green
Write-Host ""

# 4. Sync dependencies
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
uv sync
Write-Host "  [OK] Dependencies installed" -ForegroundColor Green
Write-Host ""

# 5. Run doctor
Write-Host "[5/5] Running environment check..." -ForegroundColor Yellow
uv run python scripts/doctor.py
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Cyan
Write-Host "  Start: uv run python web_server.py" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
