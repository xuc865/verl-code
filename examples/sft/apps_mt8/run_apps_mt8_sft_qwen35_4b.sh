#!/usr/bin/env bash
# Multi-turn SFT on APPS self-repair trajectories — Qwen3.5-4B.
# Mirrors run_apps_mt8_sft.sh (Qwen2.5-Coder-7B), adapted for Qwen3.5:
#   - MODEL_PATH = Qwen3.5-4B (multimodal HF; fsdp_sft_trainer loads via Vision2Seq)
#   - trust_remote_code=True (required)
#   - 6×GPU 2–7 (same cards as GRPO resume; free after RL finishes)
#   - MICRO_BSZ=1 by default (vision tower still loaded; raise only if OOM-free)
#
# Data: data/sft/apps_mt8_mix_think/{train,val}.parquet  (same as 7B think SFT)
#
# Usage (training machine):
#   cd /mnt/z4/solariewang/verl-swe
#   bash examples/sft/apps_mt8/run_apps_mt8_sft_qwen35_4b.sh
#
# Or auto-chain after GRPO:
#   bash scripts/launch_sft_qwen35_4b_after_grpo.sh
#
# Overrides:
#   CUDA_VISIBLE_DEVICES  MODEL_PATH  TRAIN_FILES  VAL_FILES  SAVE_DIR  NPROC_PER_NODE
#   TRAIN_BATCH_SIZE  MICRO_BSZ  MAX_LENGTH  TOTAL_EPOCHS  LR  LORA_RANK  LOGGER
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
cd "$REPO_ROOT"

# Physical GPUs 2–7 (remap to local cuda:0..5 for torchrun).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}

# Use SFT_MODEL_PATH only (ignore bare MODEL_PATH — GRPO shells often leave that set to a 7B ckpt).
MODEL_PATH=${SFT_MODEL_PATH:-/mnt/z4/solariewang/models/Qwen3.5-4B}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/sft/apps_mt8_mix_think}
TRAIN_FILES=${TRAIN_FILES:-$DATA_DIR/train.parquet}
VAL_FILES=${VAL_FILES:-$DATA_DIR/val.parquet}
SAVE_DIR=${SAVE_DIR:-$REPO_ROOT/checkpoints/apps_mt8_sft_qwen35_4b_think}

IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#_GPU_ARR[@]}}
# Global batch must be divisible by nproc. 7B recipe used 64/4=16 per GPU;
# keep 16/GPU → 96 on 6 GPUs. Default micro=1 (Qwen3.5 still carries vision tower;
# raise MICRO_BSZ=2 only after a smoke step is OOM-free).
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((16 * NPROC_PER_NODE))}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_LENGTH=${MAX_LENGTH:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
# Optional mid-epoch step saves; 0 = only save at each epoch end (default for SFT).
SAVE_FREQ=${SAVE_FREQ:-0}
# Keep at most this many local global_step_* ckpts; prune oldest when exceeded.
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-3}
LR=${LR:-1e-5}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}
SP_SIZE=${SP_SIZE:-1}
LOGGER=${LOGGER:-"[console]"}

if [ ! -f "$TRAIN_FILES" ] || [ ! -f "$VAL_FILES" ]; then
  echo "ERROR: missing parquet. Run: bash scripts/build_apps_mt8_sft_dataset.sh" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  exit 1
fi
if ! grep -qE 'Qwen3_5|qwen3_5|Qwen3\.5' "$MODEL_PATH/config.json"; then
  echo "ERROR: expected Qwen3.5 base/ckpt, got: $MODEL_PATH" >&2
  grep -E 'architectures|model_type' "$MODEL_PATH/config.json" >&2 || true
  echo "  tip: unset MODEL_PATH; use SFT_MODEL_PATH=/mnt/z4/solariewang/models/Qwen3.5-4B" >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be divisible by NPROC_PER_NODE ($NPROC_PER_NODE)" >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % (NPROC_PER_NODE * MICRO_BSZ) != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be divisible by nproc*micro ($NPROC_PER_NODE*$MICRO_BSZ)" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR" logs
export PYTHONUNBUFFERED=1
# Qwen3.5 / HF often need a reachable hub endpoint even for local files (tokenizer extras).
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com/}
export HF_HOME=${HF_HOME:-/mnt/z4/solariewang/datasets/hf_cache}

echo "========== APPS mt8 multi-turn SFT (Qwen3.5-4B) =========="
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  model=$MODEL_PATH"
echo "  train=$TRAIN_FILES"
echo "  val=$VAL_FILES"
echo "  save=$SAVE_DIR"
echo "  nproc=$NPROC_PER_NODE batch=$TRAIN_BATCH_SIZE micro=$MICRO_BSZ max_len=$MAX_LENGTH"
echo "  epochs=$TOTAL_EPOCHS save_freq=$SAVE_FREQ max_ckpt_to_keep=$MAX_CKPT_TO_KEEP lr=$LR lora_rank=$LORA_RANK sp=$SP_SIZE"
echo "  grad_accum ≈ $((TRAIN_BATCH_SIZE / (NPROC_PER_NODE * MICRO_BSZ)))"

# Prefer verl-agent (same env as GRPO / prior SFT).
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-verl-agent}"
fi

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
  trainer.experiment_name=qwen35-4b-mt8-think \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_ckpt_to_keep="$MAX_CKPT_TO_KEEP" \
  trainer.logger="$LOGGER" \
  trainer.seed=1 \
  "$@"

echo "SFT done. Checkpoint under: $SAVE_DIR"
