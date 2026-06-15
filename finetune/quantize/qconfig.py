"""Shared configuration for every quantization method.

The same ``QuantConfig`` drives A1/B1/B2/C1 and the universal evaluator, so that
the checkpoint, base repo (for remote code + image processor), calibration /
training data, output location and the CUDA devices are controlled from one
place. CUDA device selection MUST be applied via :func:`apply_runtime_env`
*before* torch is imported, so entry points parse args first and only then pull
in the heavy libraries.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

DEFAULT_BASE_MODEL = "openbmb/MiniCPM-o-2_6"
DEFAULT_CHECKPOINT = "output/output_minicpmv26/checkpoint-10000"
DEFAULT_TRAIN_DATA = "/wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_train.json"
DEFAULT_EVAL_DATA = "/wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_eval.json"

# Sub-modules that must stay in high precision for an MLLM: the vision tower,
# the resampler, the (optional) audio/tts towers, the token embeddings and the
# LM head. Only the Qwen2 transformer blocks under ``llm.model.layers`` are
# worth quantizing.
KEEP_HIGH_PRECISION = [
    "vpm",
    "resampler",
    "apm",
    "tts",
    "audio",
    "embed_tokens",
    "lm_head",
]


@dataclass
class QuantConfig:
    # --- I/O -----------------------------------------------------------------
    base_model: str = DEFAULT_BASE_MODEL
    checkpoint: str = DEFAULT_CHECKPOINT
    dataset: str = DEFAULT_TRAIN_DATA          # calibration / QLoRA training data
    eval_data: str = DEFAULT_EVAL_DATA
    output_dir: str = "quantize/artifacts"
    run_name: Optional[str] = None             # sub-dir under output_dir; defaults per method

    # --- hardware ------------------------------------------------------------
    cuda_devices: str = "0"                    # maps to CUDA_VISIBLE_DEVICES
    dtype: str = "bf16"                         # bf16 | fp16 | fp32 (compute / non-quant params)

    # --- multimodal ----------------------------------------------------------
    max_slice_nums: int = 1                     # must match training (finetune used 1)

    # --- calibration ---------------------------------------------------------
    num_calib_samples: int = 128
    calib_max_length: int = 1024
    seed: int = 0

    def resolved_run_name(self, method: str) -> str:
        return self.run_name or method

    def artifact_dir(self, method: str) -> str:
        return os.path.join(self.output_dir, self.resolved_run_name(method))

    def to_dict(self) -> dict:
        return asdict(self)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("shared")
    g.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="Repo/dir with remote code + image processor (default: %(default)s).")
    g.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Fine-tuned checkpoint to quantize.")
    g.add_argument("--dataset", default=DEFAULT_TRAIN_DATA,
                   help="Calibration (B1/B2) or training (C1) data JSON.")
    g.add_argument("--eval-data", default=DEFAULT_EVAL_DATA, help="Eval JSON for run_infer.")
    g.add_argument("--output-dir", default="quantize/artifacts",
                   help="Root directory for produced artifacts + manifests.")
    g.add_argument("--run-name", default=None, help="Artifact sub-dir name (default: method id).")
    g.add_argument("--cuda-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                   help="CUDA_VISIBLE_DEVICES to pin (default: env or 0).")
    g.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Compute / non-quantized parameter dtype.")
    g.add_argument("--max-slice-nums", type=int, default=1)
    g.add_argument("--num-calib-samples", type=int, default=128)
    g.add_argument("--calib-max-length", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0)


def config_from_args(args: argparse.Namespace) -> QuantConfig:
    return QuantConfig(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        run_name=args.run_name,
        cuda_devices=args.cuda_devices,
        dtype=args.dtype,
        max_slice_nums=args.max_slice_nums,
        num_calib_samples=args.num_calib_samples,
        calib_max_length=args.calib_max_length,
        seed=args.seed,
    )


def apply_runtime_env(cuda_devices: Optional[str]) -> None:
    """Pin CUDA devices and force torch-only transformers. Call before importing torch."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_devices is not None and cuda_devices != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
