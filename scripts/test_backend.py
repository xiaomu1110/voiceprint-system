"""Tests for models/speaker_backend.py.

ECAPA needs CUDA + the SpeechBrain model. ERes2NetV2 is available when modelscope
is installed (embeds -> 192-dim); otherwise it reports unavailable with a hint.
WavLM is a scaffold until `transformers` is installed.
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

WAV = "data/audio_files/enrollment/3/e8c5d9b0c3a84cc28bedb3c1a0e64df4.wav"


def test_erres2netv2():
    e = ERes2NetV2Encoder()
    if not e.is_ready():
        try:
            e.embed("nonexistent.wav")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as ex:
            assert "pip install" in str(ex), f"missing install hint: {ex}"
        print("PASS: ERes2NetV2 unavailable with install hint")
        return
    if not os.path.exists(WAV):
        print("SKIP: ERes2NetV2 ready but no audio to embed")
        return
    emb = e.embed(WAV)
    assert emb.ndim == 1 and emb.shape[0] == 192, f"expected (192,), got {emb.shape}"
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4, "not L2-normalized"
    print(f"PASS: ERes2NetV2 embed dim={emb.shape[0]} norm≈1.0")


def test_wavlm_unavailable():
    w = WavLMSpkEncoder()
    if w.is_ready():
        print("SKIP: WavLM available (transformers installed)")
        return
    try:
        w.embed("nonexistent.wav")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as ex:
        assert "pip install" in str(ex)
    print("PASS: WavLM unavailable with install hint")


def test_factory_unknown():
    try:
        make_encoder("does-not-exist")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("PASS: factory rejects unknown encoder name")


def test_ecapa():
    if not os.path.exists(WAV):
        print("SKIP: enrollment audio not found")
        return
    try:
        enc = ECAPAEncoder()
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: ECAPA/CUDA unavailable ({e})")
        return
    emb = enc.embed(WAV)
    assert emb.ndim == 1, f"expected 1-D, got {emb.shape}"
    assert emb.shape[0] == 192, f"expected dim 192, got {emb.shape[0]}"
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4, "embedding not L2-normalized"
    print(f"PASS: ECAPA embed dim={emb.shape[0]} norm≈1.0")


if __name__ == "__main__":
    test_erres2netv2()
    test_wavlm_unavailable()
    test_factory_unknown()
    test_ecapa()
    print("\nAll backend tests passed.")
