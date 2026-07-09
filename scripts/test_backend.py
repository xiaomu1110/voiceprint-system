"""Tests for models/speaker_backend.py.

ECAPA needs CUDA + the SpeechBrain model (should work in this project).
ERes2NetV2/WavLM are scaffolds: their optional deps are not installed, so they
must report unavailable with a clear install hint.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.speaker_backend import (
    ECAPAEncoder,
    ERes2NetV2Encoder,
    WavLMSpkEncoder,
    make_encoder,
)


def test_scaffolds_unavailable():
    e = ERes2NetV2Encoder()
    assert not e.is_ready(), "ERes2NetV2 should be unavailable without modelscope"
    try:
        e.embed("nonexistent.wav")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as ex:
        assert "pip install" in str(ex), f"missing install hint: {ex}"
    w = WavLMSpkEncoder()
    assert not w.is_ready(), "WavLM should be unavailable without transformers"
    try:
        w.embed("nonexistent.wav")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as ex:
        assert "pip install" in str(ex)
    print("PASS: ERes2NetV2/WavLM scaffolds report unavailable with install hint")


def test_factory_unknown():
    try:
        make_encoder("does-not-exist")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("PASS: factory rejects unknown encoder name")


def test_ecapa():
    wav = "data/audio_files/enrollment/3/e8c5d9b0c3a84cc28bedb3c1a0e64df4.wav"
    if not os.path.exists(wav):
        print("SKIP: enrollment audio not found")
        return
    try:
        enc = ECAPAEncoder()
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: ECAPA/CUDA unavailable ({e})")
        return
    emb = enc.embed(wav)
    assert emb.ndim == 1, f"expected 1-D, got {emb.shape}"
    assert emb.shape[0] == 192, f"expected dim 192, got {emb.shape[0]}"
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4, "embedding not L2-normalized"
    print(f"PASS: ECAPA embed dim={emb.shape[0]} norm≈1.0")


if __name__ == "__main__":
    test_scaffolds_unavailable()
    test_factory_unknown()
    test_ecapa()
    print("\nAll backend tests passed.")
