# Voiceprint Recognition System

ECAPA-TDNN based voiceprint recognition system for power cabinet operator authentication.

## Features

- User management (add / delete)
- Voiceprint registration (upload audio / browser recording / guided multi-segment fusion)
- Speaker identification (1:N) and verification (1:1)
- Verification logs
- System reset (admin authentication)
- **GPU auto-detection**: uses NVIDIA GPU when available, falls back to CPU automatically

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (installed automatically by setup scripts)
- [ffmpeg](https://ffmpeg.org/download.html) (installed automatically by setup scripts)

### Windows

```powershell
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system
.\scripts\setup.ps1
uv run python web_server.py
```

### Linux

```bash
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system
bash scripts/setup.sh
uv run python web_server.py
```

Open http://localhost:5000 in your browser.

## GPU / CPU Auto-Detection

The setup scripts automatically detect your hardware:

| Scenario | Behavior |
|---|---|
| NVIDIA GPU + driver | Installs CUDA PyTorch, runs on GPU |
| No NVIDIA GPU | Installs CPU PyTorch, runs on CPU |
| GPU install fails | Falls back to CPU automatically |
| GPU runtime fails | Falls back to CPU automatically |

### Force a specific mode

Windows:

```powershell
.\scripts\setup.ps1 -Mode gpu   # Force GPU (errors if unavailable)
.\scripts\setup.ps1 -Mode cpu   # Force CPU
.\scripts\setup.ps1 -Mode auto  # Auto-detect (default)
```

Linux:

```bash
bash scripts/setup.sh --gpu
bash scripts/setup.sh --cpu
bash scripts/setup.sh --auto
```

### Verify environment

```bash
uv run python scripts/doctor.py
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `VOICEPRINT_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOICEPRINT_ALLOW_CPU_FALLBACK` | `true` | Fall back to CPU on GPU failure |
| `VOICEPRINT_CUDA_DEVICE_INDEX` | `0` | GPU index for multi-GPU systems |
| `VOICEPRINT_DATA_DIR` | `data` | Data storage directory |
| `VOICEPRINT_DB_PATH` | `data/voiceprints.db` | Database file path |
| `VOICEPRINT_THRESHOLD` | `0.75` | Similarity threshold (0-1) |
| `VOICEPRINT_SAMPLE_RATE` | `16000` | Audio sample rate |
| `VOICEPRINT_HOST` | `0.0.0.0` | Server listen address |
| `VOICEPRINT_PORT` | `5000` | Server listen port |

## Common Commands

| Action | Command |
|---|---|
| Install dependencies | `uv sync` |
| Install GPU/CPU torch | `uv run python scripts/bootstrap.py --auto` |
| Start server | `uv run python web_server.py` |
| Check environment | `uv run python scripts/doctor.py` |
| Add a package | `uv add <package>` |

## Project Structure

```
voiceprint-system/
├── web_server.py       # Flask Web server
├── config.py           # Configuration (env var support)
├── database.py         # Database management
├── voiceprint.py       # Voiceprint processing core
├── pyproject.toml      # Project metadata & dependencies
├── uv.lock             # Locked dependencies
├── .python-version     # Python version pin
├── scripts/
│   ├── setup.ps1       # Windows setup (GPU/CPU auto)
│   ├── setup.sh        # Linux setup (GPU/CPU auto)
│   ├── bootstrap.py    # GPU detection & torch installer
│   └── doctor.py       # Environment verification
├── templates/
│   └── index.html      # Frontend page
├── static/
│   ├── css/style.css   # Styles
│   └── js/app.js       # Frontend logic
└── data/               # Runtime data (not committed)
    ├── audio_files/
    ├── models/
    └── voiceprints.db
```

## Tech Stack

- Voiceprint model: SpeechBrain ECAPA-TDNN
- Backend: Flask
- Frontend: HTML + CSS + JavaScript
- Database: SQLite
- Audio: soundfile + scipy + ffmpeg
- Deep Learning: PyTorch (CUDA or CPU, auto-detected)
- Environment: uv

## License

MIT License
