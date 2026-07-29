#!/bin/bash
# One-line launcher for the three small self-repair benchmarks on the TRAINING machine.
# Everything is hard-wired for /mnt/z4/solariewang and Qwen3.5-9B.
# Usage:
#   bash /mnt/z4/solariewang/verl-swe/examples/didpo_trainer/launch_small_benchmarks.sh
# Optional:
#   SERIAL=1 bash /mnt/z4/solariewang/verl-swe/examples/didpo_trainer/launch_small_benchmarks.sh
set -euo pipefail

# ---------------- fixed paths / environment ---------------- #
ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=$ROOT/verl-swe
CONDA_ENV=${CONDA_ENV:-verl-swe-qwen35}
MODEL=$ROOT/models/Qwen3.5-9B
export SWEBENCH_DATA_ROOT=$ROOT/datasets

# Keep data/model loading local. Remove these two lines only if you intentionally want HF network fallback.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# W&B is enabled by run_swebench.sh. Keep this local to the training machine shell/process.
export WANDB_API_KEY='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'
export WANDB_PROJECT=didpo_swebench
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=300

# Single-node H20: avoid NCCL probing IB/RDMA and make Ray logs less opaque.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---------------- auto activate conda env ---------------- #
set +x
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$ROOT/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$ROOT/miniforge3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found. Create/activate $CONDA_ENV first, or install conda." >&2
    exit 1
fi
conda activate "$CONDA_ENV"
python -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
set -x

cd "$REPO"
mkdir -p logs

# benchmark | visible GPUs. Each run sees two H20s and uses TP=2 / n_gpus_per_node=2.
runs=(
  "humaneval|0,1"
  "mbpp|2,3"
  "apps|4,5"
)

COMMON_OVERRIDES=(
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.trust_remote_code=True
  data.trust_remote_code=True
  trainer.project_name=didpo_swebench
  trainer.logger="['console','wandb']"
  trainer.n_gpus_per_node=2
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
  env.swebench.data_root="$SWEBENCH_DATA_ROOT"
)

for r in "${runs[@]}"; do
  bench="${r%%|*}"
  gpus="${r#*|}"
  log="logs/run_${bench}.log"
  echo "==> launch $bench on GPU $gpus, log=$REPO/$log"
  CMD=(bash examples/didpo_trainer/run_swebench.sh vllm
      env.swebench.benchmark="$bench"
      trainer.experiment_name="didpo_qwen35_9b_${bench}"
      "${COMMON_OVERRIDES[@]}")
  if [ "${SERIAL:-0}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpus" "${CMD[@]}" 2>&1 | tee "$log"
  else
    CUDA_VISIBLE_DEVICES="$gpus" setsid nohup "${CMD[@]}" > "$log" 2>&1 < /dev/null &
    echo "    pid=$!"
    sleep 5
  fi
done

echo "all launched. logs in $REPO/logs/"
echo "tail examples: tail -f $REPO/logs/run_humaneval.log"
