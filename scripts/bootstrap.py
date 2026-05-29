"""
Voiceprint System - Bootstrap

Detects GPU, selects the correct CUDA PyTorch wheel, installs dependencies.
Run: uv run python scripts/bootstrap.py [--mode gpu]
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
        raise RuntimeError("Cannot determine driver CUDA support from nvidia-smi")

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
        raise RuntimeError(f"Driver CUDA support {cuda_version} is too old; need >= 11.8")


def _uninstall_torch(uv: str) -> None:
    """Uninstall existing torch and torchaudio to allow clean reinstall."""
    print("  Uninstalling existing torch/torchaudio...")
    _run([uv, "pip", "uninstall", "torch", "torchaudio", "-y"])


def install_torch(cuda_version: str | None = None) -> bool:
    """Install CUDA torch + torchaudio. Returns True if CUDA install succeeded."""
    uv = shutil.which("uv")
    if not uv:
        print("  [FAIL] uv not found")
        return False

    try:
        wheel = select_wheel(cuda_version)
    except RuntimeError as e:
        print(f"  [FAIL] {e}")
        return False

    index_url = f"https://download.pytorch.org/whl/{wheel}"
    print(f"  Installing CUDA torch ({wheel})...")
    _uninstall_torch(uv)
    r = _run([uv, "pip", "install", "torch", "torchaudio", "--index-url", index_url])
    if r.returncode != 0:
        print(f"  [FAIL] CUDA torch install failed: {r.stderr[:500]}")
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
    parser.add_argument("--mode", choices=["auto", "gpu"], default="gpu",
                        help="Installation mode. Both modes require a CUDA-capable NVIDIA GPU.")
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
        print("  [FAIL] No NVIDIA GPU detected. This system requires CUDA GPU acceleration.")
        print()
        print("=" * 50)
        print("  Bootstrap failed!")
        print("=" * 50)
        return 1
    print()

    # Step 2: Decide mode
    print("[2/3] Selecting mode...")
    print(f"  [OK] GPU required ({args.mode})")
    print()

    # Step 3: Install torch
    print("[3/3] Installing PyTorch (CUDA)...")
    gpu_ok = install_torch(gpu.get("cuda_version"))
    if not gpu_ok:
        print()
        print("=" * 50)
        print("  CUDA PyTorch installation failed!")
        print("=" * 50)
        return 1

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
            print("  Start: uv run --no-sync python web_server.py")
            print("=" * 50)
            return 0
        else:
            print()
            print("=" * 50)
            print("  [FAIL] CUDA is not available after installing PyTorch.")
            print("  The system will not run in CPU mode.")
            print("=" * 50)
            return 1
    else:
        print(f"  [FAIL] {result.get('error', 'unknown error')}")
        print()
        print("=" * 50)
        print("  Installation failed!")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
