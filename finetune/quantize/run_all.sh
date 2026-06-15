#!/bin/bash
# Orchestrator: run the per-module scripts in quantize/scripts/ then compare.
# Each method has its own standalone script; this just chains them.
#
#   bash quantize/run_all.sh                          # baseline + a1 + b1 + b2 + c1 + compare
#   METHODS="baseline a1" bash quantize/run_all.sh    # subset
#   CUDA_DEVICES=1 NUM_SAMPLES=-1 bash quantize/run_all.sh
#
# Per-method knobs are forwarded via env (BITS, GROUP_SIZE, QLORA_MAX_STEPS, ...);
# see the individual scripts in quantize/scripts/.
set -e

SCRIPTS="$(cd "$(dirname "$0")" && pwd)/scripts"
METHODS=${METHODS:-"baseline a1 b1 b2 c1"}

for m in $METHODS; do
    case "$m" in
        baseline) bash "$SCRIPTS/run_baseline.sh" ;;
        a1)       bash "$SCRIPTS/run_a1.sh" ;;
        b1)       bash "$SCRIPTS/run_b1.sh" ;;
        b2)       bash "$SCRIPTS/run_b2.sh" ;;
        c1)       bash "$SCRIPTS/run_c1.sh" ;;
        *)        echo "unknown method: $m" ;;
    esac
done

bash "$SCRIPTS/run_compare.sh"
echo "All done."
