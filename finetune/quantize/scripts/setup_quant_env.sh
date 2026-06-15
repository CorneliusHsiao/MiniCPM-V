#!/bin/bash
# Build the conda environment that runs B1 (GPTQ) and B2 (AWQ).
#
# Why a separate env: gptqmodel + autoawq need a newer torch than the finetune
# env (torch 2.2 / transformers 4.51). Their maintained wheels also drag in
# transformers 5, which BREAKS the MiniCPM-o `trust_remote_code` model. This
# recipe pins the one combination that satisfies all three constraints at once:
#
#   torch 2.6.0+cu124   - works on this box's CUDA 12.8 driver (the default
#                         cu13 wheel reports CUDA unavailable)
#   transformers 4.51.2 - the version the remote code targets; also autoawq's
#                         last-tested version
#   tokenizers 0.21.x / huggingface_hub <1.0 - what transformers 4.51.2 wants
#   autoawq 0.2.9       - B2 (imports cleanly on transformers 4.51.2)
#   gptqmodel 2.2.0     - B1, built FROM SOURCE. The only prebuilt wheel (7.1.0)
#                         hard-requires transformers>=5.4 (uses
#                         transformers.masking_utils); 2.2.0 does not.
#
# Everything (env + caches + tmp) lives on /wekafs because /home is full.
#
# Usage:
#   bash quantize/scripts/setup_quant_env.sh
#   ENV_PREFIX=/path/to/env bash quantize/scripts/setup_quant_env.sh
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/wekafs/ict/hanyuanx/envs/mcpm-quant}
SCRATCH=${SCRATCH:-/wekafs/ict/hanyuanx}
CUDA_HOME_DIR=${CUDA_HOME_DIR:-/usr/local/cuda-12.8}

export CONDA_PKGS_DIRS="$SCRATCH/.conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH/.pip_cache"
export TMPDIR="$SCRATCH/.tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

conda create -y -p "$ENV_PREFIX" python=3.10
PIP="$ENV_PREFIX/bin/pip"

# 1) torch matched to the GPU driver (CUDA 12.8 -> cu124 wheels).
$PIP install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# 2) AWQ (B2). It pulls transformers 5; pin everything back to the remote-code
#    versions afterwards.
$PIP install autoawq
$PIP install "transformers==4.51.2" "tokenizers==0.21.1" "huggingface_hub<1.0"

# 3) Runtime deps the MiniCPM-o modeling file imports at load time (the dynamic
#    import check fails without them even with init_audio/init_tts=False).
$PIP install soundfile librosa vector_quantize_pytorch vocos einops sentencepiece

# 4) GPTQ (B1) built from source (nvcc from CUDA 12.8) so it keeps transformers 4.51.
CUDA_HOME="$CUDA_HOME_DIR" PATH="$CUDA_HOME_DIR/bin:$PATH" \
    $PIP install --no-build-isolation "gptqmodel==2.2.0"

# 5) torchao (pulled transitively) targets torch>=2.8 and crashes the
#    transformers import on torch 2.6 -> remove it.
$PIP uninstall -y torchao || true

echo
echo "Environment ready: $ENV_PREFIX"
echo "Verify with:"
echo "  conda activate $ENV_PREFIX"
echo "  python -c \"import torch,transformers,gptqmodel; from awq import AutoAWQForCausalLM; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, gptqmodel.__version__)\""
