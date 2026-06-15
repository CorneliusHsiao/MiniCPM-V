"""Join accuracy metrics with efficiency stats across quantization methods.

Reuses ``eval/eval_metrics.py`` for the classification metrics (accuracy, F1,
ROC-AUC, PR-AUC, P@K) computed from each predictions JSONL, then merges the
``<preds>.meta.json`` sidecars (disk size, peak VRAM, latency) written by
``run_infer.py`` into one comparison table.

Usage (from ``finetune/``):
    python quantize/quant_compare.py --pred-dir quantize/preds --out-dir quantize/results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="Accuracy + efficiency comparison.")
    p.add_argument("--pred-dir", default="quantize/preds",
                   help="Directory of *.jsonl predictions (with optional *.meta.json).")
    p.add_argument("--preds", nargs="*", default=None,
                   help="Explicit NAME=path.jsonl specs (overrides --pred-dir scan).")
    p.add_argument("--out-dir", default="quantize/results")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
    return p.parse_args()


def discover(pred_dir):
    specs = []
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        specs.append(f"{name}={path}")
    return specs


def main():
    args = parse_args()
    from eval.eval_metrics import compute_metrics, load_preds

    specs = args.preds or discover(args.pred_dir)
    if not specs:
        raise SystemExit(f"No predictions found in {args.pred_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = {}
    for spec in specs:
        name, records = load_preds(spec)
        metrics, _, _ = compute_metrics(records, args.ks)
        path = spec.split("=", 1)[1] if "=" in spec else spec
        meta_path = path + ".meta.json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        rows[name] = {**metrics, **{
            "disk_gb": (meta.get("disk_bytes") or 0) / 1e9,
            "vram_gb": (meta.get("peak_vram_bytes") or 0) / 1e9,
            "ms_per_sample": meta.get("latency_ms_per_sample"),
        }}

    cols = ["n", "accuracy", "f1", "roc_auc", "pr_auc", "disk_gb", "vram_gb", "ms_per_sample"]
    name_w = max(len("method"), *(len(n) for n in rows))
    header = "method".ljust(name_w) + "  " + "  ".join(c.rjust(12) for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in rows.items():
        cells = []
        for c in cols:
            v = m.get(c)
            if v is None:
                cells.append("n/a".rjust(12))
            elif c == "n":
                cells.append(f"{int(v):12d}")
            else:
                cells.append(f"{v:12.4f}")
        lines.append(name.ljust(name_w) + "  " + "  ".join(cells))
    table = "\n".join(lines)
    print("\n" + table + "\n")

    with open(os.path.join(args.out_dir, "compare.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out_dir, "compare.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote comparison to {args.out_dir}/compare.txt and compare.json")


if __name__ == "__main__":
    main()
