#!/bin/bash
# Convenience runner: inference on two checkpoints -> metrics + visualizations.
#
# Usage:
#   bash eval/run_eval.sh [NUM_SAMPLES]
# NUM_SAMPLES defaults to 1000 (a quick balanced subset). Use -1 for the full
# eval set (~11k pairs, much slower).
set -e

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NUM_SAMPLES=${1:-1000}
DATA=/wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_eval.json
OUT_ROOT=output/output_minicpmv26
PRED_DIR=eval/preds
RES_DIR=eval/results

cd "$(dirname "$0")/.."   # run from finetune/

for CKPT in checkpoint-1000 checkpoint-10000; do
    echo "=== Inference: ${CKPT} (num_samples=${NUM_SAMPLES}) ==="
    # --skip-existing: if predictions already exist (eval done), skip straight to
    # visualization/metrics. Delete the .jsonl to force a re-run.
    python eval/infer_dg.py \
        --checkpoint ${OUT_ROOT}/${CKPT} \
        --data ${DATA} \
        --num-samples ${NUM_SAMPLES} \
        --output ${PRED_DIR}/${CKPT}.jsonl \
        --skip-existing
done

echo "=== Visualizing 10 cases for checkpoint-10000 ==="
python eval/visualize_dg.py \
    --preds ${PRED_DIR}/checkpoint-10000.jsonl \
    --num 10 --select random \
    --out ${RES_DIR}/cases_checkpoint-10000.png

echo "=== Metrics + comparison ==="
python eval/eval_metrics.py \
    --preds checkpoint-1000=${PRED_DIR}/checkpoint-1000.jsonl \
            checkpoint-10000=${PRED_DIR}/checkpoint-10000.jsonl \
    --out-dir ${RES_DIR}

echo "Done. See ${RES_DIR}/ for metrics.txt, roc.png, metrics_bar.png and case visualizations."
