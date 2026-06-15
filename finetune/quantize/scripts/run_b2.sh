#!/bin/bash
# B2 - AWQ int4 on the Qwen2 backbone (activation-aware). Then universal inference.
#
#   bash quantize/scripts/run_b2.sh
#   W_BIT=4 Q_GROUP_SIZE=64 NUM_CALIB_SAMPLES=512 bash quantize/scripts/run_b2.sh
# Extra flags forwarded to the method, e.g.:
#   bash quantize/scripts/run_b2.sh --version GEMV
export CUDA_VISIBLE_DEVICES=4,5

set -e
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== B2: AWQ (w_bit=${W_BIT:-4}, group_size=${Q_GROUP_SIZE:-128}, calib=${NUM_CALIB_SAMPLES}) ==="
python quantize/methods/b2_awq.py $COMMON_ARGS \
    --w-bit "${W_BIT:-4}" --q-group-size "${Q_GROUP_SIZE:-128}" "$@"

infer b2_awq "$ARTIFACT_DIR/b2_awq"
echo "B2 done -> $PRED_DIR/b2_awq.jsonl"
