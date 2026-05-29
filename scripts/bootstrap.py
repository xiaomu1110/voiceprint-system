"""
Voiceprint System - Bootstrap

Detects GPU, selects the correct PyTorch wheel, installs dependencies.
Run: uv run python scripts/bootstrap.py [--auto|--gpu|--cpu]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, **kwargs)


def _nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def detect_gpu() -> dict:
    """Detect NVIDIA GPU via nvidia-smi. Returns dict with gpu info or empty dict."""
    smi = _nvidia_smi_path()
    if not smi:
        return {}

    try:
        r = _run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if r.returncode != 0 or not r.stdout.strip():
            return {}

        line = r.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        gpu_name = parts[0] if parts else "Unknown"
        driver_ver = parts[1] if len(parts) > 1 else "Unknown"

        # Parse CUDA version from nvidia-smi header
        r2 = _run([smi])
        cuda_ver = None
        for line2 in r2.stdout.splitlines():
            m = re.search(r"CUDA Version:\s*([\d.]+)", line2)
            if m:
                cuda_ver = m.group(1)
                break

        return {
            "available": True,
            "name": gpu_name,
            "driver_version": driver_ver,
            "cuda_version": cuda_ver,
        }
    except Exception:
        return {}


def select_wheel(cuda_version: str | None) -> str:
    """Select PyTorch CUDA wheel tag based on driver CUDA support."""
    if not cuda_version:
        return "cu126"

    major, minor = cuda_version.split(".")[:2]
    cuda_num = float(f"{major}.{minor}")

    # Map driver CUDA support to PyTorch wheel tag
    # Higher CUDA drivers support lower CUDA wheels
    if cuda_num >= 12.6:
        return "cu126"
    elif cuda_num >= 12.4:
        return "cu124"
    elif cuda_num >= 12.1:
        return "cu121"
    elif cuda_num >= 11.8:
        return "cu118"
    else:
        return "cpu"


def _uninstall_torch(uv: str) -> None:
    """Uninstall existing torch and torchaudio to allow clean reinstall."""
    print("  Uninstalling existing torch/torchaudio...")
    _run([uv, "pip", "uninstall", "torch", "torchaudio", "-y"])


def install_torch(mode: str, cuda_version: str | None = None) -> bool:
    """Install torch + torchaudio. Returns True if GPU install succeeded."""
    uv = shutil.which("uv")
    if not uv:
        print("  [FAIL] uv not found")
        return False

    if mode == "gpu":
        wheel = select_wheel(cuda_version)
        index_url = f"https://download.pytorch.org/whl/{wheel}"
        print(f"  Installing GPU torch ({wheel})...")
        _uninstall_torch(uv)
        r = _run([uv, "pip", "install", "torch", "torchaudio", "--index-url", index_url])
        if r.returncode != 0:
            print(f"  [WARN] GPU install failed: {r.stderr[:200]}")
            print(f"  Falling back to CPU...")
            return install_torch("cpu")
        return True
    else:
        print("  Installing CPU torch...")
        _uninstall_torch(uv)
        r = _run([uv, "pip", "install", "torch", "torchaudio", "--index-url",
                   "https://download.pytorch.org/whl/cpu"])
        if r.returncode != 0:
            print(f"  [FAIL] CPU install failed: {r.stderr[:200]}")
            return False
        return True


def verify_install() -> dict:
    """Verify torch installation by importing and checking CUDA."""
    code = (
        "import torch; "
        "print(f'VER:{torch.__version__}'); "
        "print(f'CUDA:{torch.version.cuda}'); "
        "print(f'AVAIL:{torch.cuda.is_available()}'); "
        "print(f'COUNT:{torch.cuda.device_count()}'); "
        "print(f'NAME:{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'NAME:N/A')"
    )
    try:
        r = _run([sys.executable, "-c", code])
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr[:200]}

        info = {}
        for line in r.stdout.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()

        return {
            "ok": True,
            "version": info.get("VER", ""),
            "cuda_tag": info.get("CUDA", ""),
            "cuda_available": info.get("AVAIL", "") == "True",
            "device_count": int(info.get("COUNT", "0")),
            "device_name": info.get("NAME", ""),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}



def main() -> int:
    parser = argparse.ArgumentParser(description="Voiceprint System Bootstrap")
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"], default="auto",
                        help="Installation mode (default: auto)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Voiceprint System - Bootstrap")
    print("=" * 50)
    print()

    # Step 1: Detect GPU
    print("[1/3] Detecting GPU...")
    gpu = detect_gpu()
    if gpu.get("available"):
        print(f"  [OK] GPU: {gpu['name']}")
        print(f"  [OK] Driver: {gpu['driver_version']}")
        print(f"  [OK] CUDA: {gpu['cuda_version']}")
    else:
        print("  [INFO] No NVIDIA GPU detected")
    print()

    # Step 2: Decide mode
    print("[2/3] Selecting mode...")
    mode = args.mode
    if mode == "auto":
        if gpu.get("available"):
            mode = "gpu"
            print(f"  [OK] Auto-selected: GPU")
        else:
            mode = "cpu"
            print(f"  [OK] Auto-selected: CPU")
    else:
        print(f"  [OK] Forced mode: {mode.upper()}")
    print()

    # Step 3: Install torch
    print(f"[3/3] Installing PyTorch ({mode.upper()})...")
    gpu_ok = install_torch(mode, gpu.get("cuda_version"))

    # Verify
    print()
    print("Verifying installation...")
    result = verify_install()

    if result.get("ok"):
        print(f"  torch: {result['version']}")
        print(f"  cuda_tag: {result['cuda_tag'] or 'N/A'}")
        print(f"  cuda_available: {result['cuda_available']}")
        if result["cuda_available"]:
            print(f"  device: {result['device_name']}")
            print()
            print("=" * 50)
            print("  GPU mode ready!")
            print("  Start: uv run python web_server.py")
            print("=" * 50)
            return 0
        else:
            print()
            print("=" * 50)
            print("  CPU mode ready (GPU not available)")
            print("  Start: uv run python web_server.py")
            print("=" * 50)
            return 0
    else:
        print(f"  [FAIL] {result.get('error', 'unknown error')}")
        print()
        print("=" * 50)
        print("  Installation failed!")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
