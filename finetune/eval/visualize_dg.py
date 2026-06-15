"""Visualize Doppelgangers test cases: the two input images + model decision.

Reads a predictions JSONL (from ``infer_dg.py``) and renders N cases as a grid,
each row showing image_0, image_1, and a text panel with the ground-truth label,
the model prediction, and the confidence score. Correct predictions are framed in
green, incorrect ones in red.

Example:
    python visualize_dg.py \
        --preds eval/preds/checkpoint-10000.jsonl \
        --num 10 --out eval/results/cases_checkpoint-10000.png
"""

import argparse
import json
import os
import random
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

LABEL_NAME = {1: "yes (same)", 0: "no (different)"}


def parse_args():
    p = argparse.ArgumentParser(description="Visualize DG predictions.")
    p.add_argument("--preds", required=True, help="Predictions JSONL from infer_dg.py.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--num", type=int, default=10, help="Number of cases to show.")
    p.add_argument("--select", default="random",
                   choices=["random", "first", "errors", "correct"],
                   help="Which cases to show (default: random).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick(records, select, num, seed):
    if select == "errors":
        pool = [r for r in records if r["pred"] != r["label"]]
    elif select == "correct":
        pool = [r for r in records if r["pred"] == r["label"]]
    else:
        pool = list(records)
    if select == "random":
        random.Random(seed).shuffle(pool)
    return pool[:num]


def main():
    args = parse_args()
    records = [json.loads(l) for l in open(args.preds) if l.strip()]
    cases = pick(records, args.select, args.num, args.seed)
    if not cases:
        raise SystemExit(f"No cases to show for select={args.select}.")

    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.4 * n),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, r in enumerate(cases):
        correct = r["pred"] == r["label"]
        edge = "#2ca02c" if correct else "#d62728"
        for j, key in enumerate(["image_0", "image_1"]):
            ax = axes[i][j]
            try:
                ax.imshow(Image.open(r[key]).convert("RGB"))
            except Exception as e:
                ax.text(0.5, 0.5, f"<image load error>\n{e}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(edge); s.set_linewidth(3)
            ax.set_title(f"image_{j}", fontsize=10)

        ax = axes[i][2]
        ax.axis("off")
        q = textwrap.fill(r.get("question", ""), 38)
        txt = (
            f"Q: {q}\n\n"
            f"Ground truth: {LABEL_NAME.get(r['label'], r['label'])}\n"
            f"Prediction:   {LABEL_NAME.get(r['pred'], r['pred'])}\n"
            f"Model text:   '{r.get('pred_text', '')}'\n"
            f"p(yes) score: {r['score']:.3f}\n\n"
            f"{'CORRECT' if correct else 'WRONG'}"
        )
        ax.text(0.0, 0.5, txt, fontsize=11, va="center", family="monospace",
                color=edge if not correct else "black")

    plt.suptitle(f"Doppelgangers cases ({args.select}) - {os.path.basename(args.preds)}",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out} ({n} cases)")


if __name__ == "__main__":
    main()
