"""
Voiceprint System - Threshold Calibration

Builds target/nontarget trial pairs from an evaluation dataset, scores them with
the ECAPA voiceprint extractor (cosine similarity), and reports EER / minDCF /
optimal operating-point thresholds per domain.

Pure-numpy EER/minDCF implementation (no sklearn dependency).
The GPU model (VoiceprintExtractor) is instantiated only when embeddings must be
extracted; the trial-construction and metric functions are GPU-free.

Dataset layout: <eval_dir>/<domain>/<user_id>/<audio.ext>
Example: data/eval/mandarin/3/xxx.wav

Run:
    .venv/Scripts/python.exe scripts/calibrate.py --eval-dir data/eval
"""
import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Audio extensions recognised by VoiceprintExtractor.SUPPORTED_FORMATS.
AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aac', '.wma', '.opus', '.webm'}


def build_trials(eval_dir, domain, max_pairs_per_user=50, seed=42):
    """Construct target/nontarget trial pairs for one domain.

    Directory layout: <eval_dir>/<domain>/<user_id>/<audio.ext>
    Returns a list of (wav_a, wav_b, user_a, user_b, is_target) tuples where
    is_target is True for same-user pairs (target) and False for cross-user
    pairs (nontarget). Per-user pair counts are capped to avoid combinatorial
    explosion; sampling is deterministic (seeded).
    """
    domain_dir = Path(eval_dir) / domain
    if not domain_dir.is_dir():
        return []

    user_audios = {}
    for user_dir in sorted(domain_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        audios = sorted(p for p in user_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
        if audios:
            user_audios[user_dir.name] = audios

    if not user_audios:
        return []

    rng = random.Random(seed)
    users = sorted(user_audios.keys())
    trials = []

    # Target pairs: same user, two different audio segments.
    for uid in users:
        audios = user_audios[uid]
        if len(audios) < 2:
            continue
        pairs = list(itertools.combinations(range(len(audios)), 2))
        if len(pairs) > max_pairs_per_user:
            pairs = rng.sample(pairs, max_pairs_per_user)
        for i, j in pairs:
            trials.append((str(audios[i]), str(audios[j]), uid, uid, True))

    # Nontarget pairs: audio from different users.
    for uid in users:
        audios = user_audios[uid]
        others = [u for u in users if u != uid]
        if not others or not audios:
            continue
        seen = set()
        made = 0
        attempts = 0
        cap = max_pairs_per_user
        while made < cap and attempts < cap * 5:
            attempts += 1
            other = rng.choice(others)
            a = rng.choice(audios)
            b = rng.choice(user_audios[other])
            key = (str(a), str(b))
            if key in seen:
                continue
            seen.add(key)
            trials.append((str(a), str(b), uid, other, False))
            made += 1

    return trials


def compute_eer(scores, labels):
    """Equal Error Rate (pure numpy).

    scores: higher => more likely target. labels: 1=target, 0=nontarget.
    Convention: accept when score >= threshold.
    Returns (eer, threshold, fpr_array, tpr_array).
    EER is the operating point where FPR == FNR; the reported value is the
    average of FPR and FNR at the crossing point.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    empty = np.array([], dtype=np.float64)
    nan = float('nan')

    n_target = int(np.sum(labels == 1))
    n_nontarget = int(np.sum(labels == 0))
    if n_target == 0 or n_nontarget == 0 or scores.size == 0:
        return nan, nan, empty, empty

    order = np.argsort(-scores, kind='mergesort')
    s_sorted = scores[order]
    y_sorted = labels[order]

    tp_cum = np.cumsum(y_sorted == 1).astype(np.float64)
    fp_cum = np.cumsum(y_sorted == 0).astype(np.float64)

    fpr = fp_cum / n_nontarget
    fnr = (n_target - tp_cum) / n_target
    tpr = tp_cum / n_target

    # Prepend the "reject all" operating point (threshold above the max score).
    fpr = np.concatenate([[0.0], fpr])
    fnr = np.concatenate([[1.0], fnr])
    tpr = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[s_sorted[0] + 1.0], s_sorted])

    diff = fpr - fnr
    idx = int(np.argmin(np.abs(diff)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = float(thresholds[idx])
    return float(eer), threshold, fpr, tpr


def compute_min_dcf(scores, labels, p_target=0.01, c_miss=1, c_fa=1):
    """Minimum Detection Cost (pure numpy).

    DCF(t) = C_miss * P_target * FNR(t) + C_fa * (1 - P_target) * FPR(t)
    Returns (min_dcf, threshold) where min_dcf is the raw (non-normalized) DCF
    value at its minimum over all thresholds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    nan = float('nan')

    n_target = int(np.sum(labels == 1))
    n_nontarget = int(np.sum(labels == 0))
    if n_target == 0 or n_nontarget == 0 or scores.size == 0:
        return nan, nan

    order = np.argsort(-scores, kind='mergesort')
    s_sorted = scores[order]
    y_sorted = labels[order]

    tp_cum = np.cumsum(y_sorted == 1).astype(np.float64)
    fp_cum = np.cumsum(y_sorted == 0).astype(np.float64)
    fpr = fp_cum / n_nontarget
    fnr = (n_target - tp_cum) / n_target

    fpr = np.concatenate([[0.0], fpr])
    fnr = np.concatenate([[1.0], fnr])
    thresholds = np.concatenate([[s_sorted[0] + 1.0], s_sorted])

    dcf = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    idx = int(np.argmin(dcf))
    return float(dcf[idx]), float(thresholds[idx])


def _scores_summary(scores, labels):
    """Summary statistics for the score distribution split by label."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def _stats(a):
        if a.size == 0:
            return None
        return {
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()) if a.size > 1 else 0.0,
        }

    tgt = scores[labels == 1]
    non = scores[labels == 0]
    return {
        "all": _stats(scores),
        "target": _stats(tgt),
        "nontarget": _stats(non),
        "target_count": int(tgt.size),
        "nontarget_count": int(non.size),
    }


def score_trials(trials, extractor):
    """Score each trial with cosine similarity, caching embeddings per audio file.

    Uses extractor._normalize + dot product (embeddings from extract_embedding are
    already normalized, but the explicit _normalize is applied per the spec).
    Returns a list of (score, is_target) tuples.
    """
    cache = {}

    def _emb(path):
        if path not in cache:
            cache[path] = extractor.extract_embedding(path)
        return cache[path]

    out = []
    for wav_a, wav_b, _user_a, _user_b, is_target in trials:
        ea = extractor._normalize(_emb(wav_a))
        eb = extractor._normalize(_emb(wav_b))
        out.append((float(np.dot(ea, eb)), bool(is_target)))
    return out


def load_extractor(model):
    """Instantiate VoiceprintExtractor lazily; the GPU model is loaded only here."""
    try:
        from voiceprint import VoiceprintExtractor
    except Exception as e:
        print(f"[ERROR] 无法导入 VoiceprintExtractor: {e}", file=sys.stderr)
        raise
    if model and model != "ecapa":
        print(f"[WARN] --model={model} 暂未实现，回退到默认 ECAPA 模型", file=sys.stderr)
    try:
        return VoiceprintExtractor()
    except Exception as e:
        print(f"[ERROR] VoiceprintExtractor 实例化失败（需要 NVIDIA CUDA GPU）: {e}", file=sys.stderr)
        raise


def _calibrate_from_scores(scored, domain):
    """Compute the full calibration report dict from scored trials."""
    scores = [s for s, _ in scored]
    labels = [1 if t else 0 for _, t in scored]
    eer, thr_eer, _fpr, _tpr = compute_eer(scores, labels)
    min_dcf, thr_dcf = compute_min_dcf(scores, labels)
    num_targets = int(sum(labels))
    return {
        "domain": domain,
        "num_trials": len(scored),
        "num_targets": num_targets,
        "num_nontargets": len(scored) - num_targets,
        "eer": eer,
        "minDCF": min_dcf,
        "threshold_eer": thr_eer,
        "threshold_mindcf": thr_dcf,
        "scores_summary": _scores_summary(scores, labels),
    }


def _format_cell(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_table(rows, title, cols):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    str_rows = [[_format_cell(r.get(c, "")) for c in cols] for r in rows]
    all_lines = [cols] + str_rows
    widths = [max(len(line[i]) for line in all_lines) for i in range(len(cols))]
    template = "  ".join(f"{{:<{w}}}" for w in widths)
    print(template.format(*cols))
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    for row in str_rows:
        print(template.format(*row))


def main():
    parser = argparse.ArgumentParser(
        description="Voiceprint threshold calibration (EER / minDCF / optimal thresholds)"
    )
    parser.add_argument("--eval-dir", required=True, help="evaluation dataset root")
    parser.add_argument("--domain", default=None, help="single domain; default: all subdirs of eval-dir")
    parser.add_argument("--model", default="ecapa", help="model name (ecapa; erres2net/wavlm reserved)")
    parser.add_argument("--max-pairs-per-user", type=int, default=50,
                        help="cap target/nontarget pairs per user to avoid explosion")
    parser.add_argument("--out", default="data/eval/report.json", help="output JSON path")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"[ERROR] 评测目录不存在: {eval_dir}", file=sys.stderr)
        return 1

    if args.domain:
        domains = [args.domain]
    else:
        domains = sorted(d.name for d in eval_dir.iterdir() if d.is_dir())

    if not domains:
        print(f"[ERROR] 未在 {eval_dir} 下找到任何域目录", file=sys.stderr)
        return 1

    # Build trials first (pure logic, no GPU needed).
    domain_trials = []
    for domain in domains:
        domain_dir = eval_dir / domain
        if not domain_dir.is_dir():
            print(f"[WARN] 域目录不存在，跳过: {domain}", file=sys.stderr)
            continue
        trials = build_trials(str(eval_dir), domain, args.max_pairs_per_user)
        if not trials:
            print(f"[WARN] 域 {domain} 无可用 trial（音频为空或用户不足），跳过", file=sys.stderr)
            continue
        domain_trials.append((domain, trials))

    if not domain_trials:
        print("[ERROR] 所有域均无可评测数据。", file=sys.stderr)
        return 1

    # Instantiate the GPU model only now, when there is real work to score.
    try:
        extractor = load_extractor(args.model)
    except Exception:
        print("[ERROR] 无法加载模型，标定中止。请确认 CUDA 可用。", file=sys.stderr)
        return 1

    results = []
    for domain, trials in domain_trials:
        scored = score_trials(trials, extractor)
        if not scored:
            continue
        results.append(_calibrate_from_scores(scored, domain))

    if not results:
        print("[ERROR] 未能生成任何标定结果。", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    _print_table(
        results,
        "Calibration Report",
        ["domain", "num_trials", "num_targets", "eer", "minDCF", "threshold_eer", "threshold_mindcf"],
    )
    print(f"\nJSON report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
