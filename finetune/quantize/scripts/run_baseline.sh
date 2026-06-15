#!/bin/bash
# Baseline - un-quantized checkpoint inference (reference point for comparison).
#
#   bash quantize/scripts/run_baseline.sh
#   CUDA_DEVICES=1 NUM_SAMPLES=-1 bash quantize/scripts/run_baseline.sh
export CUDA_VISIBLE_DEVICES=6,7

set -e
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== Baseline: un-quantized $CHECKPOINT ==="
infer baseline baseline
echo "Baseline done -> $PRED_DIR/baseline.jsonl"
