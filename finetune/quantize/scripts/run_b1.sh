#!/bin/bash
# B1 - GPTQ int4 on the Qwen2 backbone (calibrated). Then universal inference.
#
#   bash quantize/scripts/run_b1.sh
#   BITS=4 GROUP_SIZE=64 NUM_CALIB_SAMPLES=512 bash quantize/scripts/run_b1.sh
# Extra flags forwarded to the method, e.g.:
#   bash quantize/scripts/run_b1.sh --no-desc-act

export CUDA_VISIBLE_DEVICES=2,3

set -e
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== B1: GPTQ (bits=${BITS:-4}, group_size=${GROUP_SIZE:-128}, calib=${NUM_CALIB_SAMPLES}) ==="
python quantize/methods/b1_gptq.py $COMMON_ARGS \
    --bits "${BITS:-4}" --group-size "${GROUP_SIZE:-128}" "$@"

infer b1_gptq "$ARTIFACT_DIR/b1_gptq"
echo "B1 done -> $PRED_DIR/b1_gptq.jsonl"
