"""Unit tests for scoring.py (AS-norm, cohort, fusion) — no model/GPU needed."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import CohortManager, score_fusion, _normalize


def _emb(direction, dim=128, scale=1.0):
    v = np.zeros(dim)
    v[direction % dim] = scale
    return _normalize(v)


def test_fusion_equal_weights():
    s = score_fusion([0.8, 0.6])
    assert abs(s - 0.7) < 1e-9
    print("PASS: score_fusion equal weights -> 0.7")


def test_fusion_weighted():
    s = score_fusion([0.8, 0.6], weights=[3.0, 1.0])
    assert abs(s - (3 * 0.8 + 0.6) / 4) < 1e-9
    print("PASS: score_fusion weighted -> 0.75")


def test_fusion_empty():
    assert score_fusion([]) == 0.0
    print("PASS: score_fusion empty -> 0.0")


def test_cohort_not_ready_below_min():
    cm = CohortManager(min_cohort=5)
    cm.load_from_pairs([(i, _emb(i)) for i in range(3)])
    assert not cm.is_ready()
    assert cm.as_norm(_emb(0), _emb(1), 0.5) == 0.5  # falls back to raw
    print("PASS: AS-norm falls back to raw when cohort too small")


def test_as_norm_shifts_score():
    # cohort of many impostors all orthogonal-ish to test/ref.
    rng = np.random.default_rng(7)
    cohort_pairs = [(i, rng.standard_normal(128)) for i in range(60)]
    cm = CohortManager(k=20, min_cohort=20)
    cm.load_from_pairs(cohort_pairs)

    test = rng.standard_normal(128)
    ref = test * 0.9 + rng.standard_normal(128) * 0.1  # similar to test
    raw = float(np.dot(_normalize(test), _normalize(ref)))
    z = cm.as_norm(test, ref, raw)

    # a genuine pair should get a positive z (above impostor mean)
    assert z > 0, f"genuine pair z should be > 0, got {z}"
    assert z != raw  # normalization actually happened
    print(f"PASS: AS-norm shifts genuine-pair score (raw={raw:.3f} -> z={z:.3f})")


def test_as_norm_excludes_target_user():
    rng = np.random.default_rng(1)
    test = rng.standard_normal(128)
    ref = rng.standard_normal(128)  # independent of test -> moderate raw score
    raw = float(np.dot(_normalize(test), _normalize(ref)))
    # cohort: 40 random vectors, except user 5 is a near-duplicate of test
    # (so it lands in test's top-K and excluding it changes the impostor stats).
    cohort_pairs = []
    for i in range(40):
        if i == 5:
            cohort_pairs.append((5, test + rng.standard_normal(128) * 0.02))
        else:
            cohort_pairs.append((i, rng.standard_normal(128)))
    cm = CohortManager(k=20, min_cohort=20)
    cm.load_from_pairs(cohort_pairs)
    z_all = cm.as_norm(test, ref, raw, exclude_user_id=None)
    z_excl = cm.as_norm(test, ref, raw, exclude_user_id=5)
    assert z_all != z_excl, "excluding a top-K impostor should change the normalized score"
    print(f"PASS: AS-norm respects exclude_user_id (z_all={z_all:.3f}, z_excl={z_excl:.3f})")


def test_load_from_db_mock():
    class FakeDB:
        def get_all_voiceprints(self):
            rng = np.random.default_rng(3)
            return [{"user_id": i, "embedding": rng.standard_normal(64)} for i in range(30)]
    cm = CohortManager(min_cohort=10)
    cm.load_from_db(FakeDB())
    assert cm.is_ready()
    assert cm.size == 30
    print("PASS: CohortManager.load_from_db loads embeddings")


if __name__ == "__main__":
    test_fusion_equal_weights()
    test_fusion_weighted()
    test_fusion_empty()
    test_cohort_not_ready_below_min()
    test_as_norm_shifts_score()
    test_as_norm_excludes_target_user()
    test_load_from_db_mock()
    print("\nAll scoring tests passed.")
