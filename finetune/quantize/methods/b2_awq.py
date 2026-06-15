"""B2 - AWQ int4 (activation-aware) quantization of the Qwen2 backbone.

Mirrors B1's extract -> quantize -> swap-in design, but uses AutoAWQ. AWQ scales
weights by activation statistics, which typically preserves accuracy better than
plain round-to-nearest at int4 and is well supported by downstream serving
engines (vLLM/SGLang).

Backend: ``autoawq`` (``pip install autoawq``). For the OpenBMB multimodal fork
see ``tc-mb/AutoAWQ``; this module quantizes the standalone Qwen2 LLM, which the
upstream AutoAWQ already supports.

Usage (from ``finetune/``):
    python quantize/methods/b2_awq.py --w-bit 4 --q-group-size 128 \
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


def build(cfg: QuantConfig, w_bit: int = 4, q_group_size: int = 128,
          zero_point: bool = True, version: str = "GEMM") -> dict:
    from quantize.common import (
        calibration_texts,
        dir_size_bytes,
        export_llm,
        load_minicpm,
        load_tokenizer,
        write_manifest,
    )

    try:
        from awq import AutoAWQForCausalLM
    except ImportError as e:
        raise ImportError("B2 needs `autoawq` (pip install autoawq).") from e

    artifact_dir = cfg.artifact_dir("b2_awq")
    llm_out = os.path.join(artifact_dir, "llm")
    os.makedirs(artifact_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg.checkpoint)
    print("[b2] loading base model to extract the LLM backbone ...")
    base = load_minicpm(cfg.checkpoint, cfg.dtype, device="cpu")

    tmp_llm = tempfile.mkdtemp(prefix="minicpm_llm_")
    try:
        export_llm(base, tmp_llm, tokenizer=tokenizer)
        del base

        print(f"[b2] building {cfg.num_calib_samples} calibration prompts ...")
        calib = calibration_texts(cfg.dataset, tokenizer, cfg.num_calib_samples,
                                  cfg.calib_max_length, seed=cfg.seed)

        quant_config = {
            "zero_point": zero_point,
            "q_group_size": q_group_size,
            "w_bit": w_bit,
            "version": version,
        }
        print(f"[b2] AWQ quantizing Qwen2 backbone to int{w_bit} "
              f"(group_size={q_group_size}, {version}) ...")
        awq = AutoAWQForCausalLM.from_pretrained(tmp_llm, low_cpu_mem_usage=True)
        awq.quantize(tokenizer, quant_config=quant_config, calib_data=calib)
        awq.save_quantized(llm_out)
        tokenizer.save_pretrained(llm_out)
    finally:
        shutil.rmtree(tmp_llm, ignore_errors=True)

    manifest = {
        "method": f"b2-awq-int{w_bit}",
        "bits": w_bit,
        "loader": "awq_llm_swap",
        "backend": "autoawq",
        "base_model": cfg.base_model,
        "checkpoint": cfg.checkpoint,
        "model_path": cfg.checkpoint,
        "llm_quant_dir": llm_out,
        "dtype": "fp16",
        "awq": quant_config,
        "disk_bytes": dir_size_bytes(llm_out),
        "calib": {"n": cfg.num_calib_samples, "source": cfg.dataset, "kind": "text-prompt"},
    }
    write_manifest(artifact_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 AWQ quantization (LLM backbone).")
    add_common_args(parser)
    parser.add_argument("--w-bit", type=int, default=4, choices=[4])
    parser.add_argument("--q-group-size", type=int, default=128)
    parser.add_argument("--no-zero-point", action="store_true")
    parser.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV"])
    args = parser.parse_args()

    apply_runtime_env(args.cuda_devices)
    cfg = config_from_args(args)
    build(cfg, w_bit=args.w_bit, q_group_size=args.q_group_size,
          zero_point=not args.no_zero_point, version=args.version)


if __name__ == "__main__":
    main()
