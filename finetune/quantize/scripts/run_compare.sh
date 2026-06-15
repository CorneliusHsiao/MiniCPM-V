#!/bin/bash
# Compare - join accuracy (eval_metrics) + efficiency (disk/VRAM/latency) over all
# predictions in PRED_DIR.
#
#   bash quantize/scripts/run_compare.sh
set -e
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== Compare: $PRED_DIR -> $RES_DIR ==="
python quantize/quant_compare.py --pred-dir "$PRED_DIR" --out-dir "$RES_DIR"
echo "Compare done -> $RES_DIR/compare.txt"
