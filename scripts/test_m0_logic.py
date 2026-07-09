"""M0 logic tests — no model/GPU required.

Validates the security fix (N=1 / low-score false accept) and the normalized-space
fusion, without instantiating the GPU model.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IDENTIFY_ABS_FLOOR, IDENTIFY_MARGIN, FUSION_TRIM_RATIO
from voiceprint import normalize_confidence_scores, VoiceprintExtractor


def _decision(best_score, best_confidence, second_score, threshold=0.75, n=1):
    """Replica of the new 1:N joint decision in VoiceprintDatabase.verify_speaker."""
    top1_margin = (best_score - second_score) if n >= 2 else None
    return (
        best_confidence >= threshold
        and best_score >= IDENTIFY_ABS_FLOOR
        and (top1_margin is None or top1_margin >= IDENTIFY_MARGIN)
    )


class FakeExtractor:
    """Stand-in exposing the static/instance helpers used by _fuse_embeddings."""
    _normalize = staticmethod(VoiceprintExtractor._normalize)
    _fuse_embeddings = VoiceprintExtractor._fuse_embeddings

    def compute_similarity(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return float(np.dot(a, b))


def test_softmax_single_element_is_one():
    # documents the root cause: softmax over one element is always 1.0
    out = normalize_confidence_scores([("u1", 0.05)])
    assert out[0]["normalized_confidence"] == 1.0
    print("PASS: normalize_confidence_scores([single]) == 1.0  (root cause confirmed)")


def test_n1_low_score_now_rejected():
    # N=1, raw cosine 0.05, confidence 1.0 -> OLD code matched; NEW code must reject.
    matched = _decision(best_score=0.05, best_confidence=1.0, second_score=0.0, n=1)
    assert matched is False, "N=1 low-score must NOT match (security fix)"
    print("PASS: N=1 low-score (0.05) rejected despite confidence=1.0")


def test_n1_high_score_accepted():
    matched = _decision(best_score=0.80, best_confidence=1.0, second_score=0.0, n=1)
    assert matched is True
    print("PASS: N=1 high-score (0.80) accepted")


def test_low_score_relative_winner_rejected():
    # [0.30, 0.20, 0.20, 0.20] -> top confidence ~0.995 but raw 0.30 < floor
    out = normalize_confidence_scores([("a", 0.30), ("b", 0.20), ("c", 0.20), ("d", 0.20)])
    conf = out[0]["normalized_confidence"]
    matched = _decision(best_score=0.30, best_confidence=conf, second_score=0.20, n=4)
    assert matched is False, "low-score relative winner must NOT match"
    print(f"PASS: low-score relative winner (0.30) rejected (confidence was {conf:.3f})")


def test_small_margin_rejected():
    # high enough score but top-1/top-2 too close -> ambiguous -> reject
    matched = _decision(best_score=0.70, best_confidence=0.99, second_score=0.68, n=2)
    assert matched is False, "tiny top-1 margin must reject"
    print("PASS: small top-1 margin (0.02 < 0.08) rejected")


def test_fusion_is_normalized():
    fe = FakeExtractor()
    rng = np.random.default_rng(0)
    embs = [rng.standard_normal(192) for _ in range(5)]
    fused = fe._fuse_embeddings(embs)
    assert abs(float(np.linalg.norm(fused)) - 1.0) < 1e-5
    print("PASS: fused embedding is L2-normalized")


def test_fusion_in_normalized_space_not_norm_weighted():
    # A high-norm junk vector must not dominate the fused centroid.
    fe = FakeExtractor()
    base = np.zeros(192); base[0] = 1.0  # clean unit vector
    junk = np.zeros(192); junk[1] = 1000.0  # huge norm, orthogonal direction
    fused = fe._fuse_embeddings([base, junk])
    # if norm-weighted (bug), fused would point ~junk direction (component[1] dominant).
    # in normalized space, both contribute equally -> component[0] and [1] comparable.
    assert abs(fused[0]) > 0.3 and abs(fused[1]) > 0.3, "fusion must operate in normalized space"
    print(f"PASS: fusion not norm-weighted (fused[0]={fused[0]:.3f}, fused[1]={fused[1]:.3f})")


if __name__ == "__main__":
    test_softmax_single_element_is_one()
    test_n1_low_score_now_rejected()
    test_n1_high_score_accepted()
    test_low_score_relative_winner_rejected()
    test_small_margin_rejected()
    test_fusion_is_normalized()
    test_fusion_in_normalized_space_not_norm_weighted()
    print(f"\nAll M0 logic tests passed. (abs_floor={IDENTIFY_ABS_FLOOR}, margin={IDENTIFY_MARGIN}, trim={FUSION_TRIM_RATIO})")
