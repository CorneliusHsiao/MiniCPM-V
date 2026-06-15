"""Universal evaluator for any quantized artifact.

Reads a ``quant_manifest.json`` (produced by A1/B1/B2/C1), loads the model with
the matching strategy, and runs the Doppelgangers pair inference, writing
predictions in the EXACT JSONL schema that ``infer_dg.py`` produces so that
``eval/eval_metrics.py`` and ``quantize/quant_compare.py`` work unchanged. A
sidecar ``<output>.meta.json`` records efficiency stats (disk size, peak VRAM,
latency) for comparison.

Pass ``--manifest baseline`` to evaluate the un-quantized checkpoint.

Usage (from ``finetune/``):
    python quantize/run_infer.py --manifest quantize/artifacts/a1_bnb \
        --num-samples 1000 --output quantize/preds/a1_bnb.jsonl --cuda-devices 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="Universal inference for quantized MiniCPM artifacts.")
    p.add_argument("--manifest", required=True,
                   help="Artifact dir / manifest path, or 'baseline' for the raw checkpoint.")
    p.add_argument("--data", default=None, help="Eval JSON (default: manifest eval_data or repo default).")
    p.add_argument("--output", required=True, help="Predictions JSONL path.")
    p.add_argument("--checkpoint", default="output/output_minicpmv26/checkpoint-10000",
                   help="Used only when --manifest baseline.")
    p.add_argument("--base-model", default="openbmb/MiniCPM-o-2_6")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--max-slice-nums", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--dtype", default=None, choices=[None, "bf16", "fp16", "fp32"],
                   help="Override compute dtype (default: manifest's recommended dtype).")
    p.add_argument("--cuda-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def baseline_manifest(args) -> dict:
    return {
        "method": "baseline-bf16",
        "loader": "plain",
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "model_path": args.checkpoint,
        "dtype": "bf16",
        "disk_bytes": None,
    }


def load_model(manifest: dict, dtype: str):
    """Dispatch on manifest['loader']; returns an object exposing MiniCPMO.generate."""
    from quantize.common import load_minicpm
    from quantize.common import swap_llm

    loader = manifest["loader"]
    model_path = manifest.get("model_path", manifest["checkpoint"])

    if loader == "plain":
        return load_minicpm(model_path, dtype)

    if loader == "bnb":
        from quantize.methods.a1_bnb import make_bnb_config
        qc = make_bnb_config(manifest["bnb_kwargs"])
        return load_minicpm(model_path, dtype, quantization_config=qc)

    if loader == "gptq_llm_swap":
        from gptqmodel import GPTQModel
        base = load_minicpm(model_path, dtype)
        quant = GPTQModel.load(manifest["llm_quant_dir"], trust_remote_code=True)
        # GPTQ saves the inner config's torch_dtype (bf16); align the non-quantized
        # submodules (embeddings/norms/lm_head) with the fp16 vision tower it swaps into.
        llm = quant.model.to(base.device).to(base.dtype)
        swap_llm(base, llm)
        return base

    if loader == "awq_llm_swap":
        from awq import AutoAWQForCausalLM
        base = load_minicpm(model_path, dtype)
        quant = AutoAWQForCausalLM.from_quantized(manifest["llm_quant_dir"], fuse_layers=False)
        swap_llm(base, quant.model.to(base.device))
        return base

    if loader == "qlora":
        from peft import PeftModel
        from quantize.methods.a1_bnb import make_bnb_config
        qc = make_bnb_config(manifest["bnb_kwargs"])
        base = load_minicpm(model_path, dtype, quantization_config=qc)
        peft_model = PeftModel.from_pretrained(base, manifest["adapter_path"]).eval()
        return peft_model.get_base_model()

    raise ValueError(f"Unknown loader: {loader}")


def main():
    args = parse_args()

    from quantize.qconfig import apply_runtime_env
    apply_runtime_env(args.cuda_devices)

    import torch
    from PIL import Image

    from eval.infer_dg import (
        first_token_ids,
        gt_label,
        image_paths,
        predict,
        reconstruct_question,
    )
    from quantize.common import load_processor, load_tokenizer, read_manifest

    manifest = baseline_manifest(args) if args.manifest == "baseline" else read_manifest(args.manifest)
    dtype = args.dtype or manifest.get("dtype", "bf16")
    data_path = args.data or manifest.get("eval_data") or \
        "/wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_eval.json"

    print(f"[infer] method={manifest['method']} loader={manifest['loader']} dtype={dtype}")
    model = load_model(manifest, dtype)
    tokenizer = load_tokenizer(manifest.get("checkpoint", args.checkpoint))
    processor = load_processor(manifest["base_model"])

    yes_ids = first_token_ids(tokenizer, ["yes", "Yes", "YES", " yes", " Yes"])
    no_ids = first_token_ids(tokenizer, ["no", "No", "NO", " no", " No"])

    data = json.load(open(data_path))
    if args.num_samples is not None and args.num_samples >= 0:
        data = data[: args.num_samples]
    print(f"[infer] evaluating {len(data)} samples -> {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    n_done = n_err = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for idx, sample in enumerate(data):
            try:
                paths = image_paths(sample)
                images = [Image.open(p).convert("RGB") for p in paths]
                question = reconstruct_question(sample)
                label = gt_label(sample)
                pred, score, text = predict(
                    model, tokenizer, processor, images, question, yes_ids, no_ids, args
                )
                fout.write(json.dumps({
                    "id": sample.get("id", idx),
                    "image_0": paths[0], "image_1": paths[1],
                    "question": question, "label": label,
                    "pred": pred, "score": score, "pred_text": text,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"[warn] sample {sample.get('id', idx)} failed: {e}", flush=True)
            if (idx + 1) % args.log_every == 0:
                rate = (idx + 1) / (time.time() - t0)
                print(f"  {idx + 1}/{len(data)} ({rate:.2f} it/s, errors={n_err})", flush=True)

    dt = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    meta = {
        "method": manifest["method"],
        "loader": manifest["loader"],
        "dtype": dtype,
        "n": n_done,
        "errors": n_err,
        "seconds": dt,
        "latency_ms_per_sample": (dt / n_done * 1000) if n_done else None,
        "peak_vram_bytes": int(peak_vram),
        "disk_bytes": manifest.get("disk_bytes"),
    }
    with open(args.output + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[infer] done: {n_done} ok, {n_err} err in {dt:.1f}s; "
          f"peak VRAM {peak_vram / 1e9:.2f} GB -> {args.output}")


if __name__ == "__main__":
    main()
