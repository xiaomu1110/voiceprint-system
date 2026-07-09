"""
Voiceprint System - eval/calibrate logic self-test (GPU-free, no real audio content)

Validates the pure-logic pieces of scripts/calibrate.py:
  - build_trials produces correct target/nontarget pairs from a synthetic dir tree
    (only directory structure + file extensions matter; audio content is irrelevant).
  - compute_eer: ~0 for perfectly separable scores, ~0.5 for random scores.
  - compute_min_dcf: ~0 for perfect separation, finite and >= 0 for random.

No CUDA, no SpeechBrain, no real audio decoding required.
Run:
    .venv/Scripts/python.exe scripts/test_eval_logic.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calibrate import build_trials, compute_eer, compute_min_dcf  # noqa: E402


def _make_dataset(root, domain="mandarin", users=("1", "2"), segs_per_user=3):
    """Create <root>/<domain>/<user>/segN.wav dummy files (content irrelevant)."""
    for uid in users:
        udir = os.path.join(root, domain, uid)
        os.makedirs(udir, exist_ok=True)
        for k in range(segs_per_user):
            with open(os.path.join(udir, f"seg{k}.wav"), "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")  # minimal dummy bytes


def test_build_trials_structure():
    tmp = tempfile.mkdtemp(prefix="vp_eval_")
    try:
        _make_dataset(tmp)
        trials = build_trials(tmp, "mandarin", max_pairs_per_user=50)
        targets = [t for t in trials if t[4]]
        nontargets = [t for t in trials if not t[4]]

        # 2 users x C(3,2)=3 target pairs each = 6 target trials.
        assert len(targets) == 6, f"expected 6 target pairs, got {len(targets)}"
        # all target pairs share user_a == user_b
        assert all(t[2] == t[3] for t in targets), "target pairs must share user"
        # each target pair must be two different audio paths
        assert all(t[0] != t[1] for t in targets), "target pair must use different audio"

        # nontarget pairs cross users.
        assert len(nontargets) > 0, "should produce nontarget pairs"
        assert all(t[2] != t[3] for t in nontargets), "nontarget pairs must differ in user"

        print(f"PASS: build_trials -> {len(targets)} targets, {len(nontargets)} nontargets")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_trials_cap():
    tmp = tempfile.mkdtemp(prefix="vp_cap_")
    try:
        # one user with many segments; target pairs must be capped.
        _make_dataset(tmp, users=("9",), segs_per_user=20)
        trials = build_trials(tmp, "mandarin", max_pairs_per_user=5)
        targets = [t for t in trials if t[4]]
        assert len(targets) <= 5, f"target cap not respected: {len(targets)}"
        print(f"PASS: build_trials respects per-user cap ({len(targets)} <= 5)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_trials_missing():
    tmp = tempfile.mkdtemp(prefix="vp_missing_")
    try:
        assert build_trials(tmp, "nope", 50) == []
        print("PASS: build_trials on missing domain -> []")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_eer_perfect():
    # targets scored 1.0, nontargets 0.0 -> perfectly separable -> EER ~ 0.
    scores = [1.0] * 50 + [0.0] * 50
    labels = [1] * 50 + [0] * 50
    eer, thr, fpr, tpr = compute_eer(scores, labels)
    assert eer < 1e-9, f"perfect EER should be ~0, got {eer}"
    assert fpr.size == tpr.size and fpr.size > 0
    assert abs(fpr[0] - 0.0) < 1e-9 and abs(tpr[0] - 0.0) < 1e-9  # reject-all point
    print(f"PASS: compute_eer perfect -> EER={eer:.4f}, thr={thr:.4f}")


def test_eer_random():
    rng = np.random.default_rng(123)
    n = 500
    scores = rng.random(n)
    labels = rng.integers(0, 2, n)
    eer, _thr, _fpr, _tpr = compute_eer(scores, labels)
    assert 0.3 < eer < 0.7, f"random EER should be ~0.5, got {eer}"
    print(f"PASS: compute_eer random -> EER={eer:.4f} (expect ~0.5)")


def test_eer_empty():
    eer, thr, fpr, tpr = compute_eer([], [])
    assert np.isnan(eer) and np.isnan(thr) and fpr.size == 0 and tpr.size == 0
    print("PASS: compute_eer empty -> NaN + empty arrays")


def test_eer_single_class():
    # only targets (no nontargets) -> cannot compute EER -> NaN
    eer, _thr, _fpr, _tpr = compute_eer([0.9, 0.8, 0.7], [1, 1, 1])
    assert np.isnan(eer)
    print("PASS: compute_eer single-class -> NaN")


def test_min_dcf_perfect():
    scores = [1.0] * 50 + [0.0] * 50
    labels = [1] * 50 + [0] * 50
    dcf, thr = compute_min_dcf(scores, labels, p_target=0.01)
    assert dcf < 1e-9, f"perfect minDCF should be ~0, got {dcf}"
    print(f"PASS: compute_min_dcf perfect -> {dcf:.6f} @ thr={thr:.4f}")


def test_min_dcf_random_finite():
    rng = np.random.default_rng(7)
    n = 300
    scores = rng.random(n)
    labels = rng.integers(0, 2, n)
    dcf, thr = compute_min_dcf(scores, labels)
    assert np.isfinite(dcf) and dcf >= 0.0, f"minDCF must be finite & >= 0, got {dcf}"
    assert np.isfinite(thr)
    print(f"PASS: compute_min_dcf random -> {dcf:.6f} @ thr={thr:.4f}")


def test_min_dcf_parameters():
    # Higher P_target should weight misses more; result still finite and >= 0.
    scores = [0.8, 0.6, 0.4, 0.2]
    labels = [1, 1, 0, 0]
    d1, _ = compute_min_dcf(scores, labels, p_target=0.01, c_miss=1, c_fa=1)
    d2, _ = compute_min_dcf(scores, labels, p_target=0.5, c_miss=10, c_fa=1)
    assert np.isfinite(d1) and np.isfinite(d2)
    print(f"PASS: compute_min_dcf parameterized -> Ptarget=0.01:{d1:.6f}, Ptarget=0.5,Cmiss=10:{d2:.6f}")


def main():
    test_build_trials_structure()
    test_build_trials_cap()
    test_build_trials_missing()
    test_eer_perfect()
    test_eer_random()
    test_eer_empty()
    test_eer_single_class()
    test_min_dcf_perfect()
    test_min_dcf_random_finite()
    test_min_dcf_parameters()
    print("\nAll eval-logic self-tests passed.")


if __name__ == "__main__":
    raise SystemExit(0 if main() is None else 0)
