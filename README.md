# Voiceprint Recognition System

ECAPA-TDNN based voiceprint recognition system for power cabinet operator authentication.
This version requires NVIDIA CUDA GPU acceleration. CPU-only runtime is unsupported.

## Features

- User management (add / delete)
- Voiceprint registration (upload audio / browser recording / guided multi-segment fusion)
- Speaker identification (1:N) and verification (1:1)
- Verification logs
- System reset (admin authentication)
- **GPU-only runtime**: requires NVIDIA CUDA and fails fast when GPU acceleration is unavailable
- Detailed web status panel showing model, CUDA, PyTorch, GPU, driver, memory, and data statistics

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (installed automatically by setup scripts)
- [ffmpeg](https://ffmpeg.org/download.html) (installed automatically by setup scripts)

### Windows

```powershell
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system
.\scripts\setup.ps1
.\.venv\Scripts\python.exe web_server.py
```

### Linux

```bash
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system
bash scripts/setup.sh
.venv/bin/python web_server.py
```

Open http://localhost:5000 in your browser.

## GPU Requirement

The setup scripts detect NVIDIA GPU and install a CUDA PyTorch wheel. If no CUDA-capable GPU is available, setup and runtime fail with an explicit error.

| Scenario | Behavior |
|---|---|
| NVIDIA GPU + driver | Installs CUDA PyTorch, runs on GPU |
| No NVIDIA GPU | Setup fails |
| CPU-only PyTorch | Doctor/runtime fail |
| GPU runtime fails | Runtime fails; no CPU fallback |

### Setup mode

Windows:

```powershell
.\scripts\setup.ps1 -Mode gpu
```

Linux:

```bash
bash scripts/setup.sh --gpu
```

### Verify environment

```bash
uv run --no-sync python scripts/doctor.py
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
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
| Install CUDA torch | `uv run python scripts/bootstrap.py --mode gpu` |
| Start server | `uv run --no-sync python web_server.py` |
| Check environment | `uv run --no-sync python scripts/doctor.py` |
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
│   ├── setup.ps1       # Windows setup (GPU required)
│   ├── setup.sh        # Linux setup (GPU required)
│   ├── bootstrap.py    # GPU detection & CUDA torch installer
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
- Deep Learning: PyTorch CUDA (GPU required)
- Environment: uv

## License

MIT License
