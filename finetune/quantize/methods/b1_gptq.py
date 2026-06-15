"""B1 - GPTQ int4 weight quantization of the Qwen2 language backbone.

GPTQ needs a standard CausalLM, so we extract ``model.llm`` (a Qwen2ForCausalLM)
from the multimodal wrapper, quantize it with calibration prompts drawn from the
task data, and save it. The universal evaluator reloads the quantized Qwen2 and
swaps it back into the wrapper (vision tower + resampler stay in bf16/fp16).

Backend: ``gptqmodel`` (the maintained successor to ``auto-gptq``).

Usage (from ``finetune/``):
    python quantize/methods/b1_gptq.py --bits 4 --group-size 128 \
        --num-calib-samples 256 --cuda-devices 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)

from quantize.qconfig import QuantConfig, add_common_args, apply_runtime_env, config_from_args  # noqa: E402


def build(cfg: QuantConfig, bits: int = 4, group_size: int = 128,
          desc_act: bool = True, sym: bool = True) -> dict:
    from quantize.common import (
        calibration_texts,
        dir_size_bytes,
        export_llm,
        load_minicpm,
        load_tokenizer,
        write_manifest,
    )

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as e:
        raise ImportError(
            "B1 needs `gptqmodel` (pip install gptqmodel). "
            "If you must use the legacy backend, adapt this module to auto_gptq."
        ) from e

    artifact_dir = cfg.artifact_dir("b1_gptq")
    llm_out = os.path.join(artifact_dir, "llm")
    os.makedirs(artifact_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg.checkpoint)
    print("[b1] loading base model to extract the LLM backbone ...")
    base = load_minicpm(cfg.checkpoint, cfg.dtype, device="cpu")

    tmp_llm = tempfile.mkdtemp(prefix="minicpm_llm_")
    try:
        export_llm(base, tmp_llm, tokenizer=tokenizer)
        del base

        print(f"[b1] building {cfg.num_calib_samples} calibration prompts ...")
        calib = calibration_texts(cfg.dataset, tokenizer, cfg.num_calib_samples,
                                  cfg.calib_max_length, seed=cfg.seed)

        quant_config = QuantizeConfig(bits=bits, group_size=group_size,
                                      desc_act=desc_act, sym=sym)
        print(f"[b1] quantizing Qwen2 backbone to int{bits} (group_size={group_size}) ...")
        gptq = GPTQModel.load(tmp_llm, quant_config, trust_remote_code=True)
        gptq.quantize(calib, batch_size=1)
        gptq.save(llm_out)
        tokenizer.save_pretrained(llm_out)
    finally:
        shutil.rmtree(tmp_llm, ignore_errors=True)

    manifest = {
        "method": f"b1-gptq-int{bits}",
        "bits": bits,
        "loader": "gptq_llm_swap",
        "backend": "gptqmodel",
        "base_model": cfg.base_model,
        "checkpoint": cfg.checkpoint,
        "model_path": cfg.checkpoint,          # vision/resampler/head from here (bf16/fp16)
        "llm_quant_dir": llm_out,              # quantized Qwen2 swapped in at eval
        "dtype": "fp16",                        # GPTQ kernels run fp16 compute
        "gptq": {"bits": bits, "group_size": group_size, "desc_act": desc_act, "sym": sym},
        "disk_bytes": dir_size_bytes(llm_out),
        "calib": {"n": cfg.num_calib_samples, "source": cfg.dataset, "kind": "text-prompt"},
    }
    write_manifest(artifact_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 GPTQ quantization (LLM backbone).")
    add_common_args(parser)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-desc-act", action="store_true",
                        help="Disable activation-order (faster quant, slightly lower quality).")
    parser.add_argument("--asym", action="store_true", help="Use asymmetric quantization.")
    args = parser.parse_args()

    apply_runtime_env(args.cuda_devices)
    cfg = config_from_args(args)
    build(cfg, bits=args.bits, group_size=args.group_size,
          desc_act=not args.no_desc_act, sym=not args.asym)


if __name__ == "__main__":
    main()
