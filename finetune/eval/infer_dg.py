"""Run a fine-tuned MiniCPM-o checkpoint on the Doppelgangers pair eval set.

For every (image_0, image_1) pair the model is asked the binary question
"do these two images depict the same place/object? yes/no". We record:
  - the hard decision (yes -> 1, no -> 0)
  - a continuous confidence score p(yes) derived from the first generated
    token's logits over the yes / no vocabulary (needed for ROC-AUC etc.)

The predictions are written to a JSONL file so that metrics and
visualizations can be computed offline (cheaply, without re-running the model).

Example:
    USE_TF=0 CUDA_VISIBLE_DEVICES=0 python infer_dg.py \
        --checkpoint output/output_minicpmv26/checkpoint-10000 \
        --data /wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_eval.json \
        --num-samples 1000 \
        --output eval/preds/checkpoint-10000.jsonl
"""

import os

# Keep transformers in torch-only mode (this env has TF/Keras 3 that break the import).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
import json
import re
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Default base repo: holds the remote modeling code + image processor config that
# the training checkpoints do NOT save (they only store weights + tokenizer).
BASE_MODEL = "openbmb/MiniCPM-o-2_6"
IMAGE_PATTERN = re.compile(r"<image_\d+>\s*")


def parse_args():
    p = argparse.ArgumentParser(description="Inference for the Doppelgangers pair task.")
    p.add_argument("--checkpoint", required=True, help="Path to the fine-tuned checkpoint dir.")
    p.add_argument("--data", required=True, help="Path to the eval/test JSON (list of samples).")
    p.add_argument("--output", required=True, help="Where to write predictions (.jsonl).")
    p.add_argument("--base-model", default=BASE_MODEL,
                   help="Repo/dir providing the processor + remote code (default: %(default)s).")
    p.add_argument("--num-samples", type=int, default=-1,
                   help="Evaluate on the first N samples (-1 = all). Default: all.")
    p.add_argument("--max-slice-nums", type=int, default=1,
                   help="Image slicing; must match training (finetune used 1).")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip inference if the output file already has the expected number of lines.")
    return p.parse_args()


def first_token_ids(tokenizer, words):
    """First sub-token id for each spelling variant of a class word."""
    ids = set()
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def reconstruct_question(sample):
    """Strip the <image_xx> placeholders from the training prompt to get the raw question."""
    user_content = sample["conversations"][0]["content"]
    return IMAGE_PATTERN.sub("", user_content).strip()


def gt_label(sample):
    """Ground-truth: 1 if 'yes' (same place / true pair), 0 if 'no'."""
    raw = sample.get("final_decision")
    if raw is None and len(sample.get("conversations", [])) > 1:
        raw = sample["conversations"][1]["content"]
    raw = str(raw).strip().lower()
    return 1 if raw.startswith("y") else 0


def image_paths(sample):
    """Return the two image paths in a stable order (<image_00>, <image_01>)."""
    img = sample["image"]
    if isinstance(img, dict):
        return [img[k] for k in sorted(img.keys())]
    raise ValueError(f"Unexpected image field for sample {sample.get('id')}")


def build_inputs(model, processor, images, question, max_slice_nums):
    """Replicate MiniCPM-o `chat` input building for a single two-image turn."""
    placeholders = "\n".join(["(<image>./</image>)"] * len(images))
    msgs = [{"role": "user", "content": f"{placeholders}\n{question}"}]
    prompt = processor.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        [prompt],
        [images],
        max_slice_nums=max_slice_nums,
        use_image_id=True,
        return_tensors="pt",
        max_length=8192,
    ).to(model.device)
    inputs.pop("image_sizes", None)
    return inputs


@torch.inference_mode()
def predict(model, tokenizer, processor, images, question, yes_ids, no_ids, args):
    inputs = build_inputs(model, processor, images, question, args.max_slice_nums)
    result, outputs = model.generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        decode_text=True,
        do_sample=False,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )
    text = (result[0] if result else "").strip()

    # Confidence from the first generated token's distribution over yes/no tokens.
    first_logits = outputs.scores[0][0].float()
    probs = torch.softmax(first_logits, dim=-1)
    p_yes = probs[yes_ids].sum().item()
    p_no = probs[no_ids].sum().item()
    denom = p_yes + p_no
    score = p_yes / denom if denom > 0 else 0.5

    low = text.lower()
    if low.startswith("yes"):
        pred = 1
    elif low.startswith("no"):
        pred = 0
    else:
        pred = 1 if p_yes >= p_no else 0
    return pred, score, text


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Determine how many predictions a complete run would produce.
    data = json.load(open(args.data))
    n_total = len(data)
    expected = n_total if (args.num_samples is None or args.num_samples < 0) else min(args.num_samples, n_total)

    if args.skip_existing and count_lines(args.output) >= expected:
        print(f"Skip: {args.output} already has >= {expected} predictions. "
              f"Delete it to force re-evaluation.", flush=True)
        return

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=dtype,
        init_vision=True,
        init_audio=False,
        init_tts=False,
        attn_implementation="sdpa",
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    # Checkpoints don't save the image processor; load it from the base repo.
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    yes_ids = first_token_ids(tokenizer, ["yes", "Yes", "YES", " yes", " Yes"])
    no_ids = first_token_ids(tokenizer, ["no", "No", "NO", " no", " No"])
    print(f"yes token ids: {yes_ids} | no token ids: {no_ids}", flush=True)

    if args.num_samples is not None and args.num_samples >= 0:
        data = data[: args.num_samples]
    print(f"Evaluating {len(data)} samples.", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

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
                rec = {
                    "id": sample.get("id", idx),
                    "image_0": paths[0],
                    "image_1": paths[1],
                    "question": question,
                    "label": label,
                    "pred": pred,
                    "score": score,
                    "pred_text": text,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            except Exception as e:  # keep going on bad samples
                n_err += 1
                print(f"[warn] sample {sample.get('id', idx)} failed: {e}", flush=True)

            if (idx + 1) % args.log_every == 0:
                rate = (idx + 1) / (time.time() - t0)
                print(f"  {idx + 1}/{len(data)}  ({rate:.2f} it/s, errors={n_err})", flush=True)

    dt = time.time() - t0
    print(f"Done: {n_done} ok, {n_err} errors in {dt:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
