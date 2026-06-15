"""Compute and compare classification metrics for Doppelgangers predictions.

Reads one or more prediction JSONL files produced by ``infer_dg.py`` and reports
accuracy, precision, recall, F1, ROC-AUC and PR-AUC (average precision), plus a
ranking metric Precision@K for K in {3, 5, 10}.

Note on "AUC@K": standard ROC-AUC is a single number with no K. We interpret the
requested "AUC@K (3,5,10)" as **Precision@K** -- the precision among the K
highest-confidence positive predictions -- which is the natural top-K ranking
metric here. Adjust ``--ks`` or this function if you meant something else; metrics
are computed from the saved predictions so nothing needs to be re-run.

Example:
    python eval_metrics.py \
        --preds checkpoint-1000=eval/preds/checkpoint-1000.jsonl \
                checkpoint-10000=eval/preds/checkpoint-10000.jsonl \
        --out-dir eval/results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def parse_args():
    p = argparse.ArgumentParser(description="Metrics + comparison for DG predictions.")
    p.add_argument("--preds", nargs="+", required=True,
                   help="One or more NAME=path.jsonl entries (or bare paths).")
    p.add_argument("--out-dir", default="eval/results", help="Where to write tables/plots.")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10],
                   help="K values for Precision@K (default: 3 5 10).")
    return p.parse_args()


def load_preds(spec):
    """Accept 'name=path' or 'path'; return (name, records)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = os.path.splitext(os.path.basename(path))[0]
    records = [json.loads(line) for line in open(path) if line.strip()]
    return name, records


def precision_at_k(labels, scores, k):
    """Precision among the k highest-scoring (most confident 'yes') predictions."""
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    top = order[:k]
    return float(np.mean(np.asarray(labels)[top] == 1))


def compute_metrics(records, ks):
    labels = np.array([r["label"] for r in records])
    preds = np.array([r["pred"] for r in records])
    scores = np.array([r["score"] for r in records], dtype=float)

    out = {
        "n": int(len(labels)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    # AUC metrics need both classes present.
    if len(np.unique(labels)) == 2:
        out["roc_auc"] = roc_auc_score(labels, scores)
        out["pr_auc"] = average_precision_score(labels, scores)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    for k in ks:
        out[f"P@{k}"] = precision_at_k(labels, scores, k)
    return out, labels, scores


def format_table(results, ks):
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] + [f"P@{k}" for k in ks]
    name_w = max(len("checkpoint"), *(len(n) for n in results))
    header = "checkpoint".ljust(name_w) + "  n     " + "  ".join(c.rjust(8) for c in cols)
    lines = [header, "-" * len(header)]
    for name, (m, _, _) in results.items():
        row = name.ljust(name_w) + f"  {m['n']:<5d} "
        row += "  ".join(f"{m[c]:8.4f}" for c in cols)
        lines.append(row)
    return "\n".join(lines)


def plot_roc(results, out_path):
    plt.figure(figsize=(6, 6))
    for name, (m, labels, scores) in results.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bars(results, ks, out_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] + [f"P@{k}" for k in ks]
    names = list(results.keys())
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(names))
    plt.figure(figsize=(1.6 * len(metrics), 5))
    for i, name in enumerate(names):
        m = results[name][0]
        vals = [m[k] for k in metrics]
        plt.bar(x + i * w, vals, width=w, label=name)
    plt.xticks(x + w * (len(names) - 1) / 2, metrics, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Metric comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for spec in args.preds:
        name, records = load_preds(spec)
        results[name] = compute_metrics(records, args.ks)

    table = format_table(results, args.ks)
    print("\n" + table + "\n")

    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({n: m for n, (m, _, _) in results.items()}, f, indent=2)

    plot_roc(results, os.path.join(args.out_dir, "roc.png"))
    plot_bars(results, args.ks, os.path.join(args.out_dir, "metrics_bar.png"))
    print(f"Wrote table + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
