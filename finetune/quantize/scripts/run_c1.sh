#!/bin/bash
# C1 - QLoRA: 4-bit NF4 frozen base + trainable LoRA adapters. Then universal inference.
# NOTE: QLoRA is single-process (not torchrun) and incompatible with DeepSpeed ZeRO-3.
#
#   bash quantize/scripts/run_c1.sh
#   QLORA_MAX_STEPS=4000 LORA_R=128 LR=1e-4 bash quantize/scripts/run_c1.sh
# Extra flags forwarded to the method, e.g.:
#   bash quantize/scripts/run_c1.sh --tune-vision
export CUDA_VISIBLE_DEVICES=4,5

set -e
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== C1: QLoRA (steps=${QLORA_MAX_STEPS:-2000}, r=${LORA_R:-64}, lr=${LR:-1e-4}) ==="
python quantize/methods/c1_qlora.py $COMMON_ARGS \
    --max-steps "${QLORA_MAX_STEPS:-2000}" --lora-r "${LORA_R:-64}" \
    --learning-rate "${LR:-1e-4}" "$@"

infer c1_qlora "$ARTIFACT_DIR/c1_qlora"
echo "C1 done -> $PRED_DIR/c1_qlora.jsonl"
