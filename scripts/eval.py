"""
Voiceprint System - Regression Evaluation

Runs all trials for a domain at a fixed threshold and reports acceptance /
rejection metrics: correct-accept rate, correct-reject rate, FPR, FNIR,
accuracy, EER, and the target/nontarget x accept/reject confusion matrix.

Reuses the trial-construction, EER, and minDCF helpers from calibrate.py.

Dataset layout: <eval_dir>/<domain>/<user_id>/<audio.ext>
Example: data/eval/mandarin/3/xxx.wav

Run:
    .venv/Scripts/python.exe scripts/eval.py --eval-dir data/eval --domain mandarin
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from calibrate import (  # noqa: E402
    build_trials,
    compute_eer,
    compute_min_dcf,
    score_trials,
    load_extractor,
    _scores_summary,
)


def evaluate_domain(trials, domain, threshold, extractor):
    """Score prebuilt trials for one domain and compute threshold-based metrics.

    trials: list of (wav_a, wav_b, user_a, user_b, is_target) from build_trials.
    Returns a report dict, or None when trials is empty.
    """
    if not trials:
        return None

    scored = score_trials(trials, extractor)
    scores = np.array([s for s, _ in scored], dtype=np.float64)
    labels = np.array([1 if t else 0 for _, t in scored], dtype=np.int64)

    # Decision at the fixed threshold: accept when score >= threshold.
    accepted = scores >= threshold
    ta = int(np.sum((labels == 1) & accepted))   # target  -> accept (true accept)
    fr = int(np.sum((labels == 1) & ~accepted))  # target  -> reject (false reject)
    fa = int(np.sum((labels == 0) & accepted))   # nontarget -> accept (false accept)
    tr = int(np.sum((labels == 0) & ~accepted))  # nontarget -> reject (true reject)

    n_target = ta + fr
    n_nontarget = fa + tr
    total = n_target + n_nontarget

    correct_accept_rate = ta / n_target if n_target else 0.0       # = TPR = 1 - FNIR
    correct_reject_rate = tr / n_nontarget if n_nontarget else 0.0  # = TNR = 1 - FPR
    fpr = fa / n_nontarget if n_nontarget else 0.0                 # false accept rate
    fnir = fr / n_target if n_target else 0.0                      # false reject rate
    accuracy = (ta + tr) / total if total else 0.0

    eer, _eer_thr, _fpr, _tpr = compute_eer(scores, labels)
    min_dcf, _ = compute_min_dcf(scores, labels)

    return {
        "domain": domain,
        "threshold": float(threshold),
        "num_trials": total,
        "num_targets": n_target,
        "num_nontargets": n_nontarget,
        "correct_accept_rate": correct_accept_rate,
        "correct_reject_rate": correct_reject_rate,
        "fpr": fpr,
        "fnir": fnir,
        "accuracy": accuracy,
        "eer": eer,
        "minDCF": min_dcf,
        "confusion_matrix": {
            "target_accept": ta,
            "target_reject": fr,
            "nontarget_accept": fa,
            "nontarget_reject": tr,
        },
        "scores_summary": _scores_summary(scores, labels),
    }


def _print_table(rows):
    print()
    print("=" * 96)
    print("Evaluation Report")
    print("=" * 96)
    cols = ["domain", "thr", "trials", "CAR", "CRR", "FPR", "FNIR", "ACC", "EER",
            "TA", "FR", "FA", "TR"]
    str_rows = []
    for r in rows:
        c = r["confusion_matrix"]
        str_rows.append([
            r["domain"],
            f"{r['threshold']:.4f}",
            str(r["num_trials"]),
            f"{r['correct_accept_rate']:.4f}",
            f"{r['correct_reject_rate']:.4f}",
            f"{r['fpr']:.4f}",
            f"{r['fnir']:.4f}",
            f"{r['accuracy']:.4f}",
            f"{r['eer']:.4f}",
            str(c["target_accept"]),
            str(c["target_reject"]),
            str(c["nontarget_accept"]),
            str(c["nontarget_reject"]),
        ])
    all_lines = [cols] + str_rows
    widths = [max(len(line[i]) for line in all_lines) for i in range(len(cols))]
    template = "  ".join(f"{{:<{w}}}" for w in widths)
    print(template.format(*cols))
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    for row in str_rows:
        print(template.format(*row))
    print()
    print("Legend: CAR=正确接受率 CRR=正确拒绝率 FPR=错误接受率 FNIR=错误拒绝率 "
          "ACC=准确率 EER=等错误率")
    print("        TA=target-accept FR=target-reject FA=nontarget-accept TR=nontarget-reject")


def main():
    parser = argparse.ArgumentParser(description="Voiceprint regression evaluation at a fixed threshold")
    parser.add_argument("--eval-dir", required=True, help="evaluation dataset root")
    parser.add_argument("--domain", default=None, help="single domain; default: all subdirs of eval-dir")
    parser.add_argument("--threshold", type=float, default=None,
                        help="decision threshold (default: config.VERIFY_THRESHOLD)")
    parser.add_argument("--max-pairs-per-user", type=int, default=50,
                        help="cap target/nontarget pairs per user")
    parser.add_argument("--out", default="data/eval/eval_report.json", help="output JSON path")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"[ERROR] 评测目录不存在: {eval_dir}", file=sys.stderr)
        return 1

    threshold = args.threshold if args.threshold is not None else config.VERIFY_THRESHOLD

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
        extractor = load_extractor("ecapa")
    except Exception:
        print("[ERROR] 无法加载模型，评测中止。请确认 CUDA 可用。", file=sys.stderr)
        return 1

    results = []
    for domain, trials in domain_trials:
        res = evaluate_domain(trials, domain, threshold, extractor)
        if res is None:
            continue
        results.append(res)

    if not results:
        print("[ERROR] 未能生成任何评测结果。", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(threshold), "results": results}, f, ensure_ascii=False, indent=2)

    _print_table(results)
    print(f"\nJSON report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
