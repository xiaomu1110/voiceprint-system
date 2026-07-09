"""
Score normalization and fusion for speaker verification.

Provides:
  - CohortManager: builds/refreshes an impostor embedding pool for adaptive norm.
  - as_norm: adaptive symmetric score normalization (S-norm / AS-norm).
  - score_fusion: weighted fusion of scores from multiple model backends.

All routines operate on L2-normalized embeddings and raw cosine scores.
"""
from __future__ import annotations

import threading
import numpy as np


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-9:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _cos_batch(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine of one vector a against each row of B (B assumed normalized)."""
    a = _normalize(a)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-9, None)
    return Bn @ a


class CohortManager:
    """Impostor embedding pool for adaptive score normalization.

    The cohort is a matrix of L2-normalized embeddings drawn from registered
    users (or a dedicated impostor set). For a target user we exclude that
    user's own embeddings so they do not bias the impostor distribution.
    """

    def __init__(self, k: int = 200, min_cohort: int = 20):
        self._lock = threading.Lock()
        self._cohort: np.ndarray | None = None  # (N, D)
        self._user_index: list[tuple[int, int]] = []  # (user_id, row_idx)
        self.k = k
        self.min_cohort = min_cohort

    @property
    def size(self) -> int:
        return 0 if self._cohort is None else self._cohort.shape[0]

    def is_ready(self) -> bool:
        return self._cohort is not None and self.size >= self.min_cohort

    def load_from_pairs(self, pairs: list[tuple[int, np.ndarray]]) -> None:
        """pairs: iterable of (user_id, embedding). Replaces the current cohort."""
        rows = []
        index = []
        for uid, emb in pairs:
            rows.append(_normalize(emb))
            index.append((uid, len(rows) - 1))
        with self._lock:
            self._cohort = np.asarray(rows, dtype=np.float32) if rows else None
            self._user_index = index

    def load_from_db(self, db_manager, extractor=None) -> None:
        vps = db_manager.get_all_voiceprints()
        pairs = []
        for vp in vps:
            emb = vp["embedding"]
            if extractor is not None:
                emb = extractor._normalize(emb)
            pairs.append((vp["user_id"], emb))
        self.load_from_pairs(pairs)

    def _cohort_for(self, exclude_user_id: int | None) -> np.ndarray | None:
        if self._cohort is None:
            return None
        if exclude_user_id is None:
            return self._cohort
        keep = [i for (uid, i) in self._user_index if uid != exclude_user_id]
        if not keep:
            return None
        return self._cohort[keep]

    def as_norm(self, test_emb: np.ndarray, ref_emb: np.ndarray,
                raw_score: float, exclude_user_id: int | None = None) -> float:
        """Adaptive symmetric score normalization.

        Returns a normalized score; falls back to the raw score if the cohort
        is too small. Higher is still "more similar".
        """
        cohort = self._cohort_for(exclude_user_id)
        if cohort is None or cohort.shape[0] < self.min_cohort:
            return float(raw_score)
        k = min(self.k, cohort.shape[0])

        t_sims = _cos_batch(test_emb, cohort)
        r_sims = _cos_batch(ref_emb, cohort)
        # top-K most similar impostors (the ones that would most confusable)
        t_top = np.sort(t_sims)[-k:]
        r_top = np.sort(r_sims)[-k:]
        mean_t, std_t = float(t_top.mean()), float(t_top.std())
        mean_r, std_r = float(r_top.mean()), float(r_top.std())
        mean_z = (mean_t + mean_r) / 2.0
        std_z = (std_t + std_r) / 2.0
        if std_z < 1e-9:
            return float(raw_score)
        z = (float(raw_score) - mean_z) / std_z
        # keep it in a sane range; cosine-like interpretation
        return float(np.clip(z, -10.0, 10.0))


def score_fusion(scores: list[float], weights: list[float] | None = None) -> float:
    """Weighted average of scores from multiple backends."""
    if not scores:
        return 0.0
    if weights is None:
        w = np.ones(len(scores)) / len(scores)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(scores):
            raise ValueError("weights length must match scores length")
        total = float(w.sum())
        if total <= 0:
            w = np.ones(len(scores)) / len(scores)
        else:
            w = w / total
    return float(np.dot(np.asarray(scores, dtype=np.float64), w))
