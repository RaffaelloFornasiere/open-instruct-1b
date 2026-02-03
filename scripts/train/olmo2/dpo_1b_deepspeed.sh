#!/bin/bash
# DPO training for OLMo-2 1B using DeepSpeed ZeRO Stage 2
#
# Effective batch size: per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# Current config: 1 * 128 * 1 = 128
#
# Usage:
#   Single GPU:  ./scripts/train/olmo2/dpo_1b_deepspeed.sh
#   Multi GPU:   ./scripts/train/olmo2/dpo_1b_deepspeed.sh 4

NUM_GPUS=${1:-1}
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training with DeepSpeed ZeRO-2: $NUM_GPUS GPUs, batch size $BATCH_SIZE_PER_GPU per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Effective batch size: $TOTAL_BATCH_SIZE"

export WANDB_MODE=disabled

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/OLMo-2-0425-1B-SFT \
    --tokenizer_name allenai/OLMo-2-0425-1B-SFT \
    --use_flash_attn \
    --gradient_checkpointing \
    --mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/olmo2_1b_dpo_deepspeed/ \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --chat_template_name olmo \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
    # --with_tracking
