"""Shared runtime helpers: model loading, calibration data, manifests, sizing.

Everything here imports torch/transformers lazily inside functions so that
``apply_runtime_env`` (CUDA device pinning) can run first in the entry points.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

IMAGE_PATTERN = re.compile(r"<image_\d+>\s*")
MANIFEST_NAME = "quant_manifest.json"


# --------------------------------------------------------------------------- #
# dtype / model loading
# --------------------------------------------------------------------------- #
def torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_minicpm(checkpoint: str, dtype: str, quantization_config=None,
                 attn_implementation: str = "sdpa", device: Optional[str] = "cuda"):
    """Load the MiniCPM-o/V multimodal wrapper exactly as the eval harness does.

    Only vision + LLM are instantiated (init_audio/init_tts disabled) to match
    the fine-tuning recipe.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch_dtype(dtype),
        quantization_config=quantization_config,
        init_vision=True,
        init_audio=False,
        init_tts=False,
        attn_implementation=attn_implementation,
    )
    model = model.eval()
    if device and quantization_config is None:
        model = model.to(device)
    return model


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def load_processor(base_model: str):
    """Checkpoints don't ship the image processor; load it from the base repo."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(base_model, trust_remote_code=True)


# --------------------------------------------------------------------------- #
# calibration data
# --------------------------------------------------------------------------- #
def reconstruct_question(sample: dict) -> str:
    return IMAGE_PATTERN.sub("", sample["conversations"][0]["content"]).strip()


def calibration_texts(dataset_path: str, tokenizer, n: int, max_length: int,
                       seed: int = 0) -> List[str]:
    """Text-prompt calibration set for weight-only PTQ (B1/B2).

    Builds the Qwen2 chat-formatted prompt for each sample's question (image
    placeholders stripped). Text calibration of the LLM backbone is the standard,
    robust default; for vision-conditioned calibration see
    :func:`capture_llm_calibration` and the README "tricks" section.
    """
    import random

    data = json.load(open(dataset_path))
    rng = random.Random(seed)
    rng.shuffle(data)

    texts: List[str] = []
    for sample in data:
        try:
            q = reconstruct_question(sample)
            if not q:
                continue
            msgs = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            continue
        # length guard against pathological prompts
        if len(tokenizer.encode(prompt)) > max_length:
            continue
        texts.append(prompt)
        if len(texts) >= n:
            break
    if not texts:
        raise RuntimeError(f"No calibration prompts built from {dataset_path}")
    return texts


# --------------------------------------------------------------------------- #
# LLM swap-in (for B1/B2: quantized Qwen2 dropped back into the MM wrapper)
# --------------------------------------------------------------------------- #
def export_llm(model, dst: str, tokenizer=None) -> str:
    """Save the inner Qwen2 ``model.llm`` as a standalone HF CausalLM directory.

    The wrapper shares its (``minicpmo``) config object with the inner LLM, which
    confuses GPTQ/AWQ loaders. We write a clean ``Qwen2Config`` derived from the
    LLM-relevant fields so the artifact loads as a plain Qwen2ForCausalLM.
    """
    from transformers import Qwen2Config

    os.makedirs(dst, exist_ok=True)
    src = model.llm.config
    qcfg = Qwen2Config(
        vocab_size=src.vocab_size,
        hidden_size=src.hidden_size,
        intermediate_size=src.intermediate_size,
        num_hidden_layers=src.num_hidden_layers,
        num_attention_heads=src.num_attention_heads,
        num_key_value_heads=getattr(src, "num_key_value_heads", src.num_attention_heads),
        hidden_act=getattr(src, "hidden_act", "silu"),
        max_position_embeddings=getattr(src, "max_position_embeddings", 32768),
        rms_norm_eps=getattr(src, "rms_norm_eps", 1e-6),
        rope_theta=getattr(src, "rope_theta", 1000000.0),
        tie_word_embeddings=getattr(src, "tie_word_embeddings", False),
        bos_token_id=getattr(src, "bos_token_id", 151643),
        eos_token_id=getattr(src, "eos_token_id", 151645),
        sliding_window=getattr(src, "sliding_window", 131072),
        use_sliding_window=getattr(src, "use_sliding_window", False),
        max_window_layers=getattr(src, "max_window_layers", src.num_hidden_layers),
        attention_dropout=getattr(src, "attention_dropout", 0.0),
        torch_dtype=getattr(src, "torch_dtype", None),
    )
    orig = model.llm.config
    model.llm.config = qcfg
    try:
        model.llm.save_pretrained(dst)
    finally:
        model.llm.config = orig
    if tokenizer is not None:
        tokenizer.save_pretrained(dst)
    return dst


def swap_llm(model, quant_llm) -> None:
    """Replace the wrapper's language model with a quantized Qwen2 in place.

    The wrapper builds ``inputs_embeds`` (image features merged into the text
    embedding stream) and then calls ``self.llm(...)`` / ``self.llm.generate(...)``,
    so a drop-in Qwen2ForCausalLM-compatible module is all that's required.
    """
    model.llm = quant_llm


# --------------------------------------------------------------------------- #
# manifests + sizing
# --------------------------------------------------------------------------- #
def dir_size_bytes(path: str) -> int:
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def write_manifest(artifact_dir: str, manifest: dict) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    manifest.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(artifact_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {path}")
    return path


def read_manifest(path: str) -> dict:
    """Accept either a manifest file or the artifact directory containing it."""
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST_NAME)
    with open(path) as f:
        return json.load(f)
