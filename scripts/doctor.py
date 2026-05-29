"""
Voiceprint System - Environment Doctor

Checks whether the current environment is ready to run the system.
Run: uv run voiceprint-doctor
"""
import importlib
import os
import shutil
import socket
import subprocess
import sys


def _check(label: str) -> str:
    return f"  [OK]   {label}"


def _warn(label: str) -> str:
    return f"  [WARN] {label}"


def _fail(label: str) -> str:
    return f"  [FAIL] {label}"


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def check_python() -> list[str]:
    v = sys.version.split()[0]
    major, minor = int(v.split(".")[0]), int(v.split(".")[1])
    if major == 3 and 11 <= minor <= 12:
        return [_check(f"Python {v}")]
    return [_fail(f"Python {v} (need >=3.11,<3.13)")]


def check_ffmpeg() -> list[str]:
    if _cmd_exists("ffmpeg"):
        try:
            r = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ver = r.stdout.splitlines()[0] if r.stdout else "unknown"
            return [_check(f"ffmpeg found: {ver}")]
        except Exception:
            return [_check("ffmpeg found (version check failed)")]
    return [
        _fail("ffmpeg not found"),
        "",
        "  Install ffmpeg:",
        "    Windows : winget install Gyan.FFmpeg",
        "    Ubuntu  : sudo apt-get install -y ffmpeg",
        "    Fedora  : sudo dnf install -y ffmpeg",
        "    Arch    : sudo pacman -S ffmpeg",
    ]


def check_portable_imports() -> list[str]:
    """Check libraries that may depend on system libs (soundfile, sounddevice)."""
    lines: list[str] = []
    for mod in ("soundfile", "sounddevice"):
        try:
            importlib.import_module(mod)
            lines.append(_check(f"import {mod}"))
        except Exception as e:
            lines.append(_fail(f"import {mod}: {e}"))
    return lines


def check_imports() -> list[str]:
    mods = ["flask", "flask_cors", "numpy", "scipy", "torch", "torchaudio", "speechbrain"]
    lines: list[str] = []
    for mod in mods:
        try:
            importlib.import_module(mod)
            lines.append(_check(f"import {mod}"))
        except Exception as e:
            lines.append(_fail(f"import {mod}: {e}"))
    return lines


def check_cuda() -> list[str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return [_check(f"CUDA available: {name}")]
        return [_warn("CUDA not available (will use CPU)")]
    except Exception:
        return [_warn("CUDA check skipped (torch not available)")]


def check_data_dirs() -> list[str]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines: list[str] = []
    for rel in ("data", "data/audio_files", "data/models"):
        p = os.path.join(base, rel)
        if os.path.isdir(p):
            lines.append(_check(f"directory {rel}/ exists"))
        else:
            try:
                os.makedirs(p, exist_ok=True)
                lines.append(_check(f"directory {rel}/ created"))
            except Exception as e:
                lines.append(_fail(f"cannot create {rel}/: {e}"))
    return lines


def check_db_writable() -> list[str]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base, "data", "voiceprints.db")
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return [_check(f"database writable: data/voiceprints.db")]
    except Exception as e:
        return [_fail(f"database not writable: {e}")]


def check_port(port: int = 5000) -> list[str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return [_check(f"port {port} available")]
        except OSError:
            return [_warn(f"port {port} already in use")]


def main() -> int:
    print("=" * 50)
    print("  Voiceprint System - Environment Doctor")
    print("=" * 50)
    print()

    sections = [
        ("Python", check_python),
        ("System Dependencies", check_ffmpeg),
        ("Portable Libraries", check_portable_imports),
        ("Python Packages", check_imports),
        ("GPU", check_cuda),
        ("Data Directories", check_data_dirs),
        ("Database", check_db_writable),
        ("Network", lambda: check_port()),
    ]

    has_fail = False
    for title, fn in sections:
        print(f"[{title}]")
        for line in fn():
            print(line)
            if "[FAIL]" in line:
                has_fail = True
        print()

    print("=" * 50)
    if has_fail:
        print("  Environment has issues. Fix [FAIL] items above.")
        print("=" * 50)
        return 1
    print("  Environment ready!")
    print("  Run: uv run voiceprint-system")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
