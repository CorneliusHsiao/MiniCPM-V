"""A1 - bitsandbytes weight-only quantization (NF4 4-bit or int8).

This is *load-time* quantization: there is no calibration and no separate weight
file to produce, so the manifest simply records the BitsAndBytesConfig kwargs and
points the evaluator back at the original checkpoint. The vision tower, resampler,
embeddings and LM head are kept in high precision; only the Qwen2 blocks are
quantized.

Usage (from the ``finetune/`` directory):
    python quantize/methods/a1_bnb.py --bits 4 --quant-type nf4 --cuda-devices 0
    python quantize/methods/a1_bnb.py --bits 8 --cuda-devices 0
"""

from __future__ import annotations

import argparse
import os
import sys

FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)

from quantize.qconfig import (  # noqa: E402
    KEEP_HIGH_PRECISION,
    QuantConfig,
    add_common_args,
    apply_runtime_env,
    config_from_args,
)


def bnb_kwargs(bits: int, quant_type: str, double_quant: bool, compute_dtype: str) -> dict:
    """Plain-dict spec; reconstructed into a BitsAndBytesConfig at load time."""
    if bits == 4:
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": quant_type,
            "bnb_4bit_use_double_quant": double_quant,
            "bnb_4bit_compute_dtype": compute_dtype,
            "llm_int8_skip_modules": KEEP_HIGH_PRECISION,
        }
    if bits == 8:
        return {
            "load_in_8bit": True,
            "llm_int8_skip_modules": KEEP_HIGH_PRECISION,
        }
    raise ValueError("bits must be 4 or 8")


def make_bnb_config(spec: dict):
    """Reconstruct a BitsAndBytesConfig from the manifest's string-only spec."""
    from transformers import BitsAndBytesConfig

    from quantize.common import torch_dtype

    kwargs = dict(spec)
    if "bnb_4bit_compute_dtype" in kwargs:
        kwargs["bnb_4bit_compute_dtype"] = torch_dtype(kwargs["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**kwargs)


def build(cfg: QuantConfig, bits: int = 4, quant_type: str = "nf4",
          double_quant: bool = True, verify: bool = True) -> dict:
    from quantize.common import dir_size_bytes, load_minicpm, write_manifest

    spec = bnb_kwargs(bits, quant_type, double_quant, cfg.dtype)
    artifact_dir = cfg.artifact_dir("a1_bnb")

    if verify:
        # Materialize once to confirm the config loads on this checkpoint/hardware.
        model = load_minicpm(cfg.checkpoint, cfg.dtype, quantization_config=make_bnb_config(spec))
        print(f"[a1] loaded {cfg.checkpoint} in {bits}-bit bnb OK "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        del model

    manifest = {
        "method": f"a1-bnb-{'nf4' if bits == 4 else 'int8'}",
        "bits": bits,
        "loader": "bnb",
        "base_model": cfg.base_model,
        "checkpoint": cfg.checkpoint,
        "model_path": cfg.checkpoint,         # load-time quant: weights come from here
        "dtype": cfg.dtype,
        "bnb_kwargs": spec,                    # compute_dtype stored as string name
        "disk_bytes": dir_size_bytes(cfg.checkpoint),
        "notes": "Load-time quantization; no separate weight artifact is produced.",
    }
    write_manifest(artifact_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 bitsandbytes quantization.")
    add_common_args(parser)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--no-double-quant", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the load check (faster, no GPU needed).")
    args = parser.parse_args()

    apply_runtime_env(args.cuda_devices)
    cfg = config_from_args(args)
    build(cfg, bits=args.bits, quant_type=args.quant_type,
          double_quant=not args.no_double_quant, verify=not args.no_verify)


if __name__ == "__main__":
    main()
