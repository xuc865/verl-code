#!/usr/bin/env bash
# Multi-turn SFT on APPS self-repair trajectories (verl fsdp_sft_trainer).
#
# Data: data/sft/apps_mt8_mix/{train,val}.parquet  (messages column)
# Ref hyperparams loosely follow Think-Anywhere coding SFT
#   (lr=1e-5, warmup=0.1, epochs=2, max_length=8192, grad ckpt),
# but we keep multiturn.enable=true (their public yaml is single-turn prompt/response).
#
# Usage (training machine with GPUs):
#   cd /mnt/z4/solariewang/verl-swe
#   bash scripts/build_apps_mt8_sft_dataset.sh   # once
#   bash examples/sft/apps_mt8/run_apps_mt8_sft.sh
#
# Default: physical GPUs 4,5,6,7 (4x H20 90GB) — plenty for 7B full SFT @ 8k.
#
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES  MODEL_PATH  TRAIN_FILES  VAL_FILES  SAVE_DIR  NPROC_PER_NODE
#   TRAIN_BATCH_SIZE  MICRO_BSZ  MAX_LENGTH  TOTAL_EPOCHS  LR  LORA_RANK
#   SP_SIZE  LOGGER
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
cd "$REPO_ROOT"

# Only use cards 4–7 (remap to local cuda:0..3 for torchrun).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

MODEL_PATH=${MODEL_PATH:-/mnt/z4/solariewang/models/Qwen2.5-Coder-7B-Instruct}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/sft/apps_mt8_mix_think}
TRAIN_FILES=${TRAIN_FILES:-$DATA_DIR/train.parquet}
VAL_FILES=${VAL_FILES:-$DATA_DIR/val.parquet}
SAVE_DIR=${SAVE_DIR:-$REPO_ROOT/checkpoints/apps_mt8_sft_qwen25_coder7b_think}

# Match visible GPU count (default 4).
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BSZ=${MICRO_BSZ:-1}          # 90GB H20 can try MICRO_BSZ=2 if OOM-free
MAX_LENGTH=${MAX_LENGTH:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
LR=${LR:-1e-5}
LORA_RANK=${LORA_RANK:-0}          # Think-Anywhere used 16; 0 = full FT (better for later GRPO merge)
LORA_ALPHA=${LORA_ALPHA:-32}
SP_SIZE=${SP_SIZE:-1}
LOGGER=${LOGGER:-"[console]"}      # e.g. "['console','wandb']"

if [ ! -f "$TRAIN_FILES" ] || [ ! -f "$VAL_FILES" ]; then
  echo "ERROR: missing parquet. Run: bash scripts/build_apps_mt8_sft_dataset.sh" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR" logs
export PYTHONUNBUFFERED=1

echo "========== APPS mt8 multi-turn SFT =========="
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  model=$MODEL_PATH"
echo "  train=$TRAIN_FILES"
echo "  val=$VAL_FILES"
echo "  save=$SAVE_DIR"
echo "  nproc=$NPROC_PER_NODE batch=$TRAIN_BATCH_SIZE micro=$MICRO_BSZ max_len=$MAX_LENGTH"
echo "  epochs=$TOTAL_EPOCHS lr=$LR lora_rank=$LORA_RANK sp=$SP_SIZE"
echo "  grad_accum ≈ $((TRAIN_BATCH_SIZE / (NPROC_PER_NODE * MICRO_BSZ)))"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BSZ" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=right \
  data.balance_dp_token=False \
  model.partial_pretrain="$MODEL_PATH" \
  model.enable_gradient_checkpointing=True \
  model.trust_remote_code=True \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  model.target_modules=all-linear \
  model.fsdp_config.cpu_offload=False \
  model.fsdp_config.offload_params=False \
  optim.lr="$LR" \
  optim.betas='[0.9,0.95]' \
  optim.weight_decay=0.01 \
  optim.warmup_steps_ratio=0.1 \
  optim.clip_grad=1.0 \
  ulysses_sequence_parallel_size="$SP_SIZE" \
  use_remove_padding=False \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.default_hdfs_dir=null \
  trainer.project_name=apps-mt8-sft \
  trainer.experiment_name=qwen25-coder7b-mt8 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.logger="$LOGGER" \
  trainer.seed=1 \
  "$@"

echo "SFT done. Checkpoint under: $SAVE_DIR"
