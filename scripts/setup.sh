#!/usr/bin/env bash
# Voiceprint System - Linux Setup Script
# Usage: bash scripts/setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo "  Voiceprint System - Linux Setup"
echo "========================================"
echo ""

# 1. Install uv
echo "[1/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "  [FAIL] uv installation failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi
echo "  [OK] uv found: $(uv --version)"
echo ""

# 2. Install system dependencies
echo "[2/5] Checking system dependencies..."

install_sys_deps() {
    if command -v apt-get &>/dev/null; then
        echo "  Detected: Debian/Ubuntu"
        sudo apt-get update -qq
        sudo apt-get install -y -qq ffmpeg libsndfile1 portaudio19-dev
    elif command -v dnf &>/dev/null; then
        echo "  Detected: Fedora/RHEL"
        sudo dnf install -y ffmpeg libsndfile portaudio-devel
    elif command -v pacman &>/dev/null; then
        echo "  Detected: Arch Linux"
        sudo pacman -S --needed --noconfirm ffmpeg libsndfile portaudio
    else
        echo "  [WARN] Unknown distro. Install manually: ffmpeg libsndfile portaudio"
        return
    fi
    echo "  [OK] System dependencies installed"
}

if command -v ffmpeg &>/dev/null; then
    echo "  [OK] ffmpeg found"
else
    echo "  ffmpeg not found. Installing..."
    install_sys_deps
fi
echo ""

# 3. Install Python via uv
echo "[3/5] Ensuring Python 3.11..."
uv python install 3.11
echo "  [OK] Python ready"
echo ""

# 4. Sync dependencies
echo "[4/5] Installing dependencies..."
uv sync
echo "  [OK] Dependencies installed"
echo ""

# 5. Run doctor
echo "[5/5] Running environment check..."
uv run python scripts/doctor.py
echo ""

echo "========================================"
echo "  Setup complete!"
echo "  Start: uv run python web_server.py"
echo "========================================"
