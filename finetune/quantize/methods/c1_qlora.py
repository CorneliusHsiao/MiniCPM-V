"""C1 - QLoRA: 4-bit (NF4) frozen base + trainable LoRA adapters.

Unlike ``finetune/finetune_lora.sh`` (plain LoRA on a *full-precision* base), C1
loads the fine-tuned checkpoint in 4-bit NF4 and trains LoRA adapters on top, so
the deployed footprint is ~4-bit. It reuses the repo's ``SupervisedDataset`` and
``CPMTrainer`` (which knows to call the custom ``MiniCPMO.forward(data=...)``),
and saves only the adapter. The universal evaluator then loads the 4-bit base +
adapter.

QLoRA is incompatible with DeepSpeed ZeRO-3; run this as a single process
(``python``), not ``torchrun``. The vision tower is frozen by default.

Usage (from ``finetune/``):
    python quantize/methods/c1_qlora.py --max-steps 2000 --lora-r 64 --cuda-devices 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from types import MethodType

FINETUNE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if FINETUNE_DIR not in sys.path:
    sys.path.insert(0, FINETUNE_DIR)

from quantize.qconfig import QuantConfig, add_common_args, apply_runtime_env, config_from_args  # noqa: E402


def build(cfg: QuantConfig, *, lora_r: int = 64, lora_alpha: int = 64,
          lora_dropout: float = 0.05, max_steps: int = 2000, learning_rate: float = 1e-4,
          tune_vision: bool = False,
          target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
          ) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel

    # Repo internals (CPMTrainer passes inputs as `data=` to the MM wrapper).
    from dataset import SupervisedDataset, data_collator
    from finetune import TrainingArguments, build_transform
    from quantize.methods.a1_bnb import make_bnb_config
    from quantize.qconfig import KEEP_HIGH_PRECISION
    from quantize.common import dir_size_bytes, load_tokenizer, torch_dtype, write_manifest

    artifact_dir = cfg.artifact_dir("c1_qlora")
    adapter_dir = os.path.join(artifact_dir, "adapter")
    os.makedirs(artifact_dir, exist_ok=True)

    compute_dtype = torch_dtype(cfg.dtype)
    # Quantize only the Qwen2 backbone; keep vpm/resampler/embeddings/lm_head in
    # fp16 so the LoRA modules_to_save targets remain trainable (4-bit params
    # cannot require gradients).
    bnb_spec = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": cfg.dtype,
        "llm_int8_skip_modules": KEEP_HIGH_PRECISION,
    }
    qc = make_bnb_config(bnb_spec)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[c1] loading {cfg.checkpoint} in 4-bit NF4 ...")
    model = AutoModel.from_pretrained(
        cfg.checkpoint,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=qc,
        device_map={"": local_rank},
        init_vision=True,
        init_audio=False,
        init_tts=False,
        attn_implementation="sdpa",
    )
    tokenizer = load_tokenizer(cfg.checkpoint)

    if not tune_vision:
        model.vpm.requires_grad_(False)
    for _, p in model.llm.named_parameters():
        p.requires_grad = False

    modules_to_save = ["embed_tokens", "resampler"]
    if tune_vision:
        modules_to_save.append("vpm")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=modules_to_save,
    )

    if not hasattr(model, "get_input_embeddings"):
        def get_input_embeddings(self):
            return self.llm.get_input_embeddings()
        model.get_input_embeddings = MethodType(get_input_embeddings, model)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # --- data (reuse repo dataset + slice config) ----------------------------
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = cfg.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = cfg.max_slice_nums
        slice_config = model.config.to_dict()
    batch_vision = getattr(model.config, "batch_vision_input", False)

    transform = build_transform()
    train_json = json.load(open(cfg.dataset))
    train_dataset = SupervisedDataset(
        train_json, transform, tokenizer, slice_config=slice_config, llm_type="qwen",
        patch_size=model.config.patch_size, query_nums=model.config.query_num,
        batch_vision=batch_vision, max_length=cfg.calib_max_length,
    )

    training_args = TrainingArguments(
        output_dir=adapter_dir,
        use_lora=True,
        llm_type="qwen",
        tune_vision=tune_vision,
        tune_llm=False,
        model_max_length=cfg.calib_max_length,
        max_slice_nums=cfg.max_slice_nums,
        bf16=(cfg.dtype == "bf16"),
        fp16=(cfg.dtype == "fp16"),
        do_train=True,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.1,
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    from trainer import CPMTrainer

    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=partial(data_collator, max_length=cfg.calib_max_length),
    )
    trainer.train()
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "method": "c1-qlora-nf4",
        "bits": 4,
        "loader": "qlora",
        "base_model": cfg.base_model,
        "checkpoint": cfg.checkpoint,
        "model_path": cfg.checkpoint,          # 4-bit base
        "adapter_path": adapter_dir,
        "dtype": cfg.dtype,
        "bnb_kwargs": bnb_spec,
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
                 "target_modules": target_modules, "tune_vision": tune_vision},
        "train": {"max_steps": max_steps, "lr": learning_rate, "data": cfg.dataset},
        "disk_bytes": dir_size_bytes(adapter_dir),
    }
    write_manifest(artifact_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="C1 QLoRA (4-bit base + LoRA adapters).")
    add_common_args(parser)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tune-vision", action="store_true")
    args = parser.parse_args()

    apply_runtime_env(args.cuda_devices)
    cfg = config_from_args(args)
    build(cfg, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
          max_steps=args.max_steps, learning_rate=args.learning_rate, tune_vision=args.tune_vision)


if __name__ == "__main__":
    main()
