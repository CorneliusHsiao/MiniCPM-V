#!/bin/bash

# Keep transformers in torch-only mode to avoid optional TF import/protobuf conflicts.
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0,1
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
 
MODEL="openbmb/MiniCPM-o-2_6"
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2_6
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_train.json"
EVAL_DATA="/wekafs/ict/hanyuanx/MiniCPM-V/SCRIPTS/dg_pairs_minicpm_eval.json"

# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
LLM_TYPE="qwen" 
MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --output_dir output/output_minicpmv26 \
    --logging_dir output/output_minicpmv26 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero3.json \
    --report_to "tensorboard" 
