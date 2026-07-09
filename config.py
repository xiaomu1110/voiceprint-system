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
# 1:1 verification: threshold on raw cosine similarity.
VERIFY_THRESHOLD = float(os.environ.get("VOICEPRINT_VERIFY_THRESHOLD", str(SIMILARITY_THRESHOLD)))
# 1:N identification: absolute raw-cosine floor; defends against low-score false accept when N is small
# (without this, softmax confidence is a relative rank measure and N=1 always matches).
IDENTIFY_ABS_FLOOR = float(os.environ.get("VOICEPRINT_IDENTIFY_ABS_FLOOR", "0.55"))
# 1:N identification: minimum top-1 over top-2 raw-cosine margin, used only when N >= 2.
IDENTIFY_MARGIN = float(os.environ.get("VOICEPRINT_IDENTIFY_MARGIN", "0.08"))
# Enrollment consistency: minimum cosine between a new segment and the user's existing voiceprint centroid.
CONSISTENCY_THRESHOLD = float(os.environ.get("VOICEPRINT_CONSISTENCY_THRESHOLD", "0.45"))
# Enrollment fusion: drop the bottom fraction of segments by cosine-to-centroid before fusing (0 disables).
FUSION_TRIM_RATIO = float(os.environ.get("VOICEPRINT_FUSION_TRIM_RATIO", "0.1"))
# Adaptive symmetric score normalization (S-norm/AS-norm). Off by default — enable only after
# re-calibrating thresholds on the target domain, since it changes the score scale.
USE_ASNORM = os.environ.get("VOICEPRINT_USE_ASNORM", "false").lower() in ("1", "true", "yes")
AS_NORM_K = int(os.environ.get("VOICEPRINT_ASNORM_K", "200"))
# Use Silero VAD for effective-speech-duration estimation in quality gating (falls back to energy).
USE_SILERO_VAD = os.environ.get("VOICEPRINT_USE_SILERO_VAD", "true").lower() in ("1", "true", "yes")
# Default speaker encoder backend: 'erres2netv2' (Chinese-strong, default) or 'ecapa' (fallback).
ENCODER_NAME = os.environ.get("VOICEPRINT_ENCODER", "erres2netv2")
ERES2NETV2_MODEL_ID = os.environ.get(
    "VOICEPRINT_ERES2NETV2_MODEL", "iic/speech_eres2netv2_sv_zh-cn_16k-common"
)
SAMPLE_RATE = int(os.environ.get("VOICEPRINT_SAMPLE_RATE", "16000"))
EMBEDDING_DIM = int(os.environ.get("VOICEPRINT_EMBEDDING_DIM", "192"))
MODEL_NAME = os.environ.get("VOICEPRINT_MODEL_NAME", "speechbrain/spkrec-ecapa-voxceleb")
CONFIDENCE_SCALE = float(os.environ.get("VOICEPRINT_CONFIDENCE_SCALE", "8.0"))

# Device: this system requires NVIDIA CUDA. CPU execution is intentionally unsupported.
CUDA_DEVICE_INDEX = int(os.environ.get("VOICEPRINT_CUDA_DEVICE_INDEX", "0"))

# Server
HOST = os.environ.get("VOICEPRINT_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOICEPRINT_PORT", "5000"))
DEBUG = os.environ.get("VOICEPRINT_DEBUG", "false").lower() in ("1", "true", "yes")
ADMIN_USER = os.environ.get("VOICEPRINT_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("VOICEPRINT_ADMIN_PASS", "123456")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
