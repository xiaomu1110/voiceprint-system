# Voiceprint Recognition System

ECAPA-TDNN based voiceprint recognition system for power cabinet operator authentication.

## Features

- User management (add / delete)
- Voiceprint registration (upload audio / browser recording / guided multi-segment fusion)
- Speaker identification (1:N) and verification (1:1)
- Verification logs
- System reset (admin authentication)

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

### Verify Environment

```bash
uv run python scripts/doctor.py
```

Open http://localhost:5000 in your browser.

## Configuration

Edit `config.py` or set environment variables:

| Variable | Default | Description |
|---|---|---|
| `VOICEPRINT_DATA_DIR` | `data` | Data storage directory |
| `VOICEPRINT_DB_PATH` | `data/voiceprints.db` | Database file path |
| `VOICEPRINT_THRESHOLD` | `0.75` | Similarity threshold (0-1) |
| `VOICEPRINT_SAMPLE_RATE` | `16000` | Audio sample rate |
| `VOICEPRINT_HOST` | `0.0.0.0` | Server listen address |
| `VOICEPRINT_PORT` | `5000` | Server listen port |
| `VOICEPRINT_DEBUG` | `false` | Enable Flask debug mode |

## Common Commands

| Action | Command |
|---|---|
| Install dependencies | `uv sync` |
| Start server | `uv run python web_server.py` |
| Check environment | `uv run python scripts/doctor.py` |
| Add a package | `uv add <package>` |

## Project Structure

```
voiceprint-system/
├── web_server.py       # Flask Web server
├── config.py           # Configuration
├── database.py         # Database management
├── voiceprint.py       # Voiceprint processing core
├── pyproject.toml      # Project metadata & dependencies
├── uv.lock             # Locked dependencies
├── .python-version     # Python version
├── scripts/
│   ├── setup.ps1       # Windows setup
│   ├── setup.sh        # Linux setup
│   └── doctor.py       # Environment check
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
- Environment: uv

## License

MIT License
