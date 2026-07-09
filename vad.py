"""
Voice activity detection with a Silero VAD front-end and an energy fallback.

Silero VAD gives far more accurate speech/silence segmentation than an energy
threshold (especially under industrial noise), which makes enrollment quality
gating and effective-duration estimation more reliable. Silero is optional:
if the model/dependency is unavailable, we degrade gracefully to the
energy-based estimator so the system keeps working.
"""
from __future__ import annotations

import numpy as np

_silero_model = None
_silero_attempted = False


def _load_silero():
    """Lazily load the Silero VAD model. Returns the model or None."""
    global _silero_model, _silero_attempted
    if _silero_attempted:
        return _silero_model
    _silero_attempted = True
    try:
        try:
            from silero_vad import get_speech_ts  # type: ignore
            return None  # package API differs; fall through to torch.hub path
        except Exception:
            pass
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            force_reload=False,
        )
        _silero_model = model
        return _silero_model
    except Exception as e:
        print(f"[vad] Silero VAD unavailable, using energy fallback: {e}")
        return None


def is_silero_available() -> bool:
    return _load_silero() is not None


def _energy_speech_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    """Energy-based speech mask (fallback). Returns boolean array of speech samples."""
    if len(audio) == 0:
        return np.zeros(0, dtype=bool)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    threshold = max(0.01, rms * 0.35)
    return np.abs(audio) > threshold


def speech_timestamps(audio: np.ndarray, sr: int, use_silero: bool = True):
    """Return a list of {'start': sample, 'end': sample} speech segments."""
    if use_silero:
        model = _load_silero()
        if model is not None:
            try:
                import torch
                from scipy.signal import resample as scipy_resample
                wav = audio.astype(np.float32)
                if sr != 16000:
                    wav = scipy_resample(wav, int(len(wav) * 16000 / sr))
                    sr = 16000
                tensor = torch.from_numpy(wav).unsqueeze(0)
                ts = model.get_speech_timestamps(tensor, sampling_rate=sr)
                # remap sample indices if we resampled
                scale = len(audio) / max(len(wav), 1)
                return [
                    {"start": int(t["start"] * scale), "end": int(t["end"] * scale)}
                    for t in ts
                ]
            except Exception as e:
                print(f"[vad] Silero failed at runtime, falling back: {e}")
    # energy fallback
    mask = _energy_speech_mask(audio, sr)
    ts = []
    in_speech = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_speech:
            start = i
            in_speech = True
        elif not m and in_speech:
            ts.append({"start": start, "end": i})
            in_speech = False
    if in_speech:
        ts.append({"start": start, "end": len(mask)})
    return ts


def effective_speech_duration(audio: np.ndarray, sr: int, use_silero: bool = True) -> float:
    """Seconds of detected speech."""
    if len(audio) == 0 or sr == 0:
        return 0.0
    ts = speech_timestamps(audio, sr, use_silero=use_silero)
    samples = sum(t["end"] - t["start"] for t in ts)
    return samples / sr
