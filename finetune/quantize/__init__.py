"""Quantization toolkit for fine-tuned MiniCPM-o / MiniCPM-V checkpoints.

Implements four interchangeable methods (A1 bitsandbytes, B1 GPTQ, B2 AWQ,
C1 QLoRA) behind a shared config + manifest, plus a universal inference entry
point that emits predictions in the exact JSONL schema consumed by
``eval/eval_metrics.py`` so every method can be compared on the same footing.
"""
