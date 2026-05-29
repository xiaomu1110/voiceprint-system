"""
Voiceprint System - Configuration

All values can be overridden via environment variables.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.environ.get("VOICEPRINT_DATA_DIR", os.path.join(BASE_DIR, "data"))
AUDIO_DIR = os.path.join(DATA_DIR, "audio_files")
MODEL_DIR = os.path.join(DATA_DIR, "models")
DB_PATH = os.environ.get("VOICEPRINT_DB_PATH", os.path.join(DATA_DIR, "voiceprints.db"))

# Voiceprint
SIMILARITY_THRESHOLD = float(os.environ.get("VOICEPRINT_THRESHOLD", "0.75"))
SAMPLE_RATE = int(os.environ.get("VOICEPRINT_SAMPLE_RATE", "16000"))
EMBEDDING_DIM = int(os.environ.get("VOICEPRINT_EMBEDDING_DIM", "192"))
MODEL_NAME = os.environ.get("VOICEPRINT_MODEL_NAME", "speechbrain/spkrec-ecapa-voxceleb")

# Device
DEVICE_PREFERENCE = os.environ.get("VOICEPRINT_DEVICE", "auto").lower()
ALLOW_CPU_FALLBACK = os.environ.get("VOICEPRINT_ALLOW_CPU_FALLBACK", "true").lower() in ("1", "true", "yes")
CUDA_DEVICE_INDEX = int(os.environ.get("VOICEPRINT_CUDA_DEVICE_INDEX", "0"))

# Server
HOST = os.environ.get("VOICEPRINT_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOICEPRINT_PORT", "5000"))
DEBUG = os.environ.get("VOICEPRINT_DEBUG", "false").lower() in ("1", "true", "yes")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
