# Shared setup for the per-module quantization scripts. Sourced, not executed.
#
# Resolves paths, loads configs/default.env, allows inline env overrides, and
# defines COMMON_ARGS (shared CLI flags) + the infer() helper that runs the
# universal evaluator. Every module script sources this first.

QZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> quantize/
cd "$QZ_ROOT/.."                                             # -> finetune/
source "$QZ_ROOT/configs/default.env"

# Inline overrides take precedence over default.env.
CUDA_DEVICES=${CUDA_DEVICES:-0}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
MAX_SLICE_NUMS=${MAX_SLICE_NUMS:-1}
NUM_CALIB_SAMPLES=${NUM_CALIB_SAMPLES:-256}
CALIB_MAX_LENGTH=${CALIB_MAX_LENGTH:-1024}

export USE_TF=0 TRANSFORMERS_NO_TF=1

# B1/B2 export the ~14GB fp16 Qwen2 backbone to a temp dir before quantizing.
# Default it to a /wekafs scratch path so it never lands on a full /home or a
# small /tmp. The MiniCPM-o remote code is cached, so run offline by default.
export TMPDIR=${TMPDIR:-$QZ_ROOT/.tmp}
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

COMMON_ARGS="--base-model $BASE_MODEL --checkpoint $CHECKPOINT --dataset $TRAIN_DATA \
  --eval-data $EVAL_DATA --output-dir $ARTIFACT_DIR --cuda-devices $CUDA_DEVICES \
  --max-slice-nums $MAX_SLICE_NUMS --num-calib-samples $NUM_CALIB_SAMPLES \
  --calib-max-length $CALIB_MAX_LENGTH"

# infer <name> <manifest-dir-or-'baseline'>
infer () {
    python quantize/run_infer.py --manifest "$2" --data "$EVAL_DATA" \
        --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" \
        --num-samples "$NUM_SAMPLES" --max-slice-nums "$MAX_SLICE_NUMS" \
        --cuda-devices "$CUDA_DEVICES" --output "$PRED_DIR/$1.jsonl"
}
