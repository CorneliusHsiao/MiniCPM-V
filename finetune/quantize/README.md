# MiniCPM-o/V Quantization Toolkit

Module-wise quantization of a fine-tuned checkpoint (e.g.
`output/output_minicpmv26/checkpoint-10000`) with a shared config and a
**universal prediction format** so every method is evaluated and compared on the
exact same footing as the existing `eval/` harness.

All commands are run from the `finetune/` directory.

## Layout

```
quantize/
  qconfig.py            # shared config: CUDA devices, checkpoint/base/dataset, dtype, calib
  common.py             # model loading, calibration data, LLM swap, manifests, sizing
  methods/
    a1_bnb.py           # A1  bitsandbytes NF4 / int8        (load-time, no calibration)
    b1_gptq.py          # B1  GPTQ int4   (Qwen2 backbone, calibrated)
    b2_awq.py           # B2  AWQ  int4   (Qwen2 backbone, activation-aware)
    c1_qlora.py         # C1  QLoRA: 4-bit frozen base + trainable LoRA adapters
  run_infer.py          # universal: manifest -> predictions JSONL (infer_dg schema)
  quant_compare.py      # accuracy (eval_metrics) + efficiency (disk/VRAM/latency)
  scripts/
    _common.sh          # shared setup (paths, env overrides, COMMON_ARGS, infer())
    setup_quant_env.sh  # build the B1/B2 (GPTQ+AWQ) conda env on /wekafs
    run_a1.sh           # A1: build + infer
    run_b1.sh           # B1: build + infer
    run_b2.sh           # B2: build + infer
    run_c1.sh           # C1: train + infer
    run_baseline.sh     # un-quantized reference inference
    run_compare.sh      # comparison table
  run_all.sh            # orchestrator: chains the per-module scripts + compare
  configs/default.env   # CUDA devices + paths + sizes
  requirements-quant.txt
```

Only the **Qwen2 language backbone** (`llm.model.layers.*`) is quantized. The
SigLIP vision tower (`vpm`), `resampler`, token embeddings and LM head stay in
bf16/fp16 — this is the standard recipe for multimodal models and matches the
official OpenBMB int4 builds.

## Universal manifest & output format

Each method writes `quant_manifest.json` into its artifact dir describing how to
reload it (`loader`, `model_path`, optional `llm_quant_dir`/`adapter_path`,
quant kwargs, on-disk size). `run_infer.py` consumes the manifest and emits
predictions in the **identical JSONL schema** as `eval/infer_dg.py`
(`id, image_0, image_1, question, label, pred, score, pred_text`), plus a
`<output>.meta.json` with peak VRAM / latency / disk size. This means:

- `eval/eval_metrics.py` works on the outputs unchanged, and
- `quantize/quant_compare.py` merges accuracy + efficiency into one table.

## Quick start

```bash
# Safe: installs only bitsandbytes (A1 + C1), pinned for torch 2.2.0.
pip install --no-cache-dir -r quantize/requirements-quant.txt

# Everything: baseline + A1/B1/B2/C1 + compare.
CUDA_DEVICES=0 bash quantize/run_all.sh
METHODS="baseline a1 c1" bash quantize/run_all.sh     # subset (in-env methods)

# Per-module scripts (each = build + infer, fully standalone):
bash quantize/scripts/run_baseline.sh
bash quantize/scripts/run_a1.sh                       # BITS=8 ... for int8
GROUP_SIZE=64 NUM_CALIB_SAMPLES=512 bash quantize/scripts/run_b1.sh
bash quantize/scripts/run_b2.sh
QLORA_MAX_STEPS=4000 LORA_R=128 bash quantize/scripts/run_c1.sh
bash quantize/scripts/run_compare.sh
```

The per-module scripts read `configs/default.env` and accept inline env
overrides (`CUDA_DEVICES`, `NUM_SAMPLES`, `BITS`, `GROUP_SIZE`,
`NUM_CALIB_SAMPLES`, `QLORA_MAX_STEPS`, `LORA_R`, `LR`, ...); see the header of
each script. Any extra `--flag`s are forwarded straight to the underlying Python
method. The lower-level Python entry points (`methods/*.py`, `run_infer.py`)
remain usable directly with the shared CLI flags (`--cuda-devices`,
`--checkpoint`, `--base-model`, `--dataset`, `--eval-data`, `--dtype`, ...).

## Is C1 the same as `finetune/finetune_lora.sh`?

**No.** They are different techniques:

| | `finetune_lora.sh` (plain LoRA) | C1 (`c1_qlora.py`, QLoRA) |
|---|---|---|
| Base weights | full precision (fp16/bf16) | **4-bit NF4 (bitsandbytes)** |
| Trains | LoRA adapters (+ vpm/resampler/embed) | LoRA adapters on a frozen 4-bit base |
| Goal | cheap *fine-tuning*, full-precision deploy | **quantized** deploy (~4-bit footprint) |
| `prepare_model_for_kbit_training` | not used (no quantized base) | required |
| DeepSpeed | ZeRO-2 via `torchrun` | incompatible with ZeRO-3; single process |

Also note: the repo's `finetune.py` has a `q_lora` flag and calls
`prepare_model_for_kbit_training`, **but it never passes a `BitsAndBytesConfig`
to `from_pretrained`** — so that path prepares for k-bit training without
actually loading a 4-bit base. C1 closes that gap by loading the checkpoint in
NF4 first, then attaching LoRA. So `finetune_lora.sh` ≈ LoRA, C1 = QLoRA.

## Environment compatibility (important)

The finetune env is pinned to **torch 2.2.0 / transformers 4.51.2 / numpy 1.26.4**
(required by the MiniCPM-o remote code and the prebuilt `flash-attn 2.5.8`).

- **A1 + C1** run in this env as-is — they only need `bitsandbytes==0.43.1`
  (in `requirements-quant.txt`). Verified: the checkpoint loads in 4-bit NF4.
- **B1 (GPTQ) + B2 (AWQ)** need a **separate env** (`gptqmodel` / `autoawq`
  require newer torch, and their maintained wheels drag in `transformers>=5`
  which breaks the MiniCPM-o remote code). Do **not** install them into the
  finetune env.

### B1/B2 env (verified working)

Build it with the bundled script — it creates the env on `/wekafs` (since
`/home` is full) and pins the one combination that satisfies all constraints:

```bash
bash quantize/scripts/setup_quant_env.sh        # -> /wekafs/ict/hanyuanx/envs/mcpm-quant
conda activate /wekafs/ict/hanyuanx/envs/mcpm-quant
bash quantize/scripts/run_b2.sh                 # AWQ  int4  (build + infer)
bash quantize/scripts/run_b1.sh                 # GPTQ int4  (build + infer)
```

The pinned set (see the script header for the rationale):

| package        | version       | why                                            |
| -------------- | ------------- | ---------------------------------------------- |
| torch          | 2.6.0+cu124   | works on the CUDA 12.8 driver (cu13 → no CUDA) |
| transformers   | 4.51.2        | MiniCPM-o remote code; autoawq's tested ver.   |
| tokenizers     | 0.21.x        | what transformers 4.51.2 requires              |
| huggingface_hub| <1.0          | what transformers 4.51.2 requires              |
| autoawq        | 0.2.9         | B2 — imports cleanly on transformers 4.51.2    |
| gptqmodel      | 2.2.0 (source)| B1 — prebuilt 7.x needs transformers≥5         |

Verified end-to-end on this checkpoint:
- **B2/AWQ**: LLM 16.2 GB → ~5.5 GB; swap-in inference produces predictions.
- **B1/GPTQ**: LLM 14.18 GB → 5.19 GB (−63%); swap-in inference produces predictions.

Notes:
- `gptqmodel 2.2.0` is built from source (nvcc 12.8); the only prebuilt wheel
  (7.1.0) imports `transformers.masking_utils` and hard-requires `transformers>=5.4`.
- `autoawq` pulls `torchao`, which targets `torch>=2.8` and crashes the
  `transformers` import on torch 2.6 — the setup script removes it.
- The exported Qwen2 backbone carries the MiniCPM-o tokenizer (remote `auto_map`),
  so `run_infer.py` loads the GPTQ artifact with `trust_remote_code=True`.

If the finetune env ever gets upgraded by accident, restore it with:
```bash
pip install --no-cache-dir "numpy==1.26.4" "torch==2.2.0" "transformers==4.51.2" \
  "tokenizers==0.21.1" "huggingface_hub==0.30.2" "accelerate==0.30.1" \
  "protobuf==5.29.6" "pillow==10.1.0" "bitsandbytes==0.43.1"
# if torch then can't find libnccl.so.2:
pip install --no-cache-dir --force-reinstall --no-deps "nvidia-nccl-cu12==2.19.3"
```

## Notes / caveats

- **B1/B2 swap design.** The quantized Qwen2 is saved standalone and, at eval
  time, swapped back into the multimodal wrapper (`model.llm = quant_llm`). The
  wrapper merges image features into `inputs_embeds` and then calls
  `self.llm(...)`, so a Qwen2ForCausalLM-compatible module is sufficient. Run
  B1/B2 inference in `fp16` (the kernels' compute dtype) — `run_infer.py` picks
  this up from the manifest automatically.
- **Calibration** for B1/B2 uses the task prompts (text). For vision-conditioned
  calibration see the trick below.
- Package availability: `bitsandbytes`, `gptqmodel`, `autoawq` are CUDA-only and
  not in the base `requirements.txt`; install from `requirements-quant.txt`.

## Tricks for better quantization (CUDA / C++ low-level)

1. **Vision-conditioned calibration (accuracy).** Instead of text-only prompts,
   run the full multimodal forward on calibration images and capture the real
   `inputs_embeds` entering `model.llm` (register a forward pre-hook on
   `model.llm.model`). Quantizing on the true activation distribution — image
   tokens included — tightens GPTQ/AWQ scales for this task. Hook point: the
   wrapper builds `vllm_embedding` then calls the LLM; capture there.
2. **Group size & act-order.** Smaller `group_size` (64 vs 128) and `desc_act`
   (activation ordering) reduce int4 error at a modest speed cost; tune per the
   ROC-AUC sensitivity of the yes/no readout.
3. **Fused int4 kernels.** Use AWQ-GEMM / Marlin / ExLlamaV2 CUDA kernels for
   W4A16 (e.g. gptqmodel's Marlin backend, AWQ `version="GEMM"`). Marlin's
   tiling + `dp4a`/tensor-core paths give large decode speedups over naive
   dequantize-then-matmul.
4. **FP8 on Hopper.** On H100, W8A8 FP8 (torchao / native) uses the `__nv_fp8`
   tensor-core path — near-lossless, ~2x memory, faster than int4 dequant.
5. **KV-cache quantization.** Quantize the KV cache to int8/fp8 to cut memory
   and bandwidth during generation (independent of weight quantization).
6. **CUDA graphs + paged attention.** Capture the decode step in a CUDA graph to
   remove per-token launch overhead; pair with FlashAttention-2 (already a dep)
   or paged attention for the prefill of the long image-token prefix.
7. **Custom dequant fusion (C++/CUDA).** Fuse int4 weight dequantization into the
   GEMM epilogue (CUTLASS mixed-input GEMM) so weights are never materialized in
   fp16 in global memory — this is what Marlin/Machete do and is the single
   biggest low-level win for W4A16 throughput.
8. **Serving engines.** For production throughput, export B2 (AWQ) and serve via
   vLLM/SGLang, which ship tuned int4 + paged-attention CUDA kernels; reimplement
   the first-token yes/no confidence using the engine's logprobs API.
