#!/bin/bash
# A1 - bitsandbytes NF4 / int8 (load-time). Then universal inference.
#
#   bash quantize/scripts/run_a1.sh                 # 4-bit nf4
#   BITS=8 bash quantize/scripts/run_a1.sh          # int8
#   CUDA_DEVICES=1 NUM_SAMPLES=-1 bash quantize/scripts/run_a1.sh
# Extra flags are forwarded to the method, e.g.:
#   bash quantize/scripts/run_a1.sh --quant-type fp4

export CUDA_VISIBLE_DEVICES=4,5

set -e
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== A1: bitsandbytes (bits=${BITS:-4}, type=${QUANT_TYPE:-nf4}) ==="
python quantize/methods/a1_bnb.py $COMMON_ARGS \
    --bits "${BITS:-4}" --quant-type "${QUANT_TYPE:-nf4}" "$@"

infer a1_bnb "$ARTIFACT_DIR/a1_bnb"
echo "A1 done -> $PRED_DIR/a1_bnb.jsonl"
