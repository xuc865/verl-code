#!/usr/bin/env bash
# Resume CodeRL+ GRPO from global_step_50 (same recipe as launch_grpo_coderl_sft_mt8.sh).
# Continues SwanLab under the previous run id (wt76kt18) when possible.
#
# Usage (train host):
#   bash /mnt/z4/solariewang/verl-swe/scripts/launch_grpo_coderl_sft_mt8_resume50.sh
#   FOREGROUND=1 bash .../launch_grpo_coderl_sft_mt8_resume50.sh
#
# Before starting: kill heavy unrelated jobs on this node (esp. transform_ui_fps),
# or Ray host-RAM OOM will kill workers again.

set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=${REPO:-$ROOT/verl-swe}
CONDA_ENV=${CONDA_ENV:-verl-agent}

# SFT path is still needed for tokenizer/config; GRPO weights come from resume ckpt.
MODEL_PATH=${MODEL_PATH:-$REPO/checkpoints/apps_mt8_sft_qwen25_coder7b_think/global_step_242}
EXP_NAME=${EXP_NAME:-grpo_coderl_qwen25_7b_sft_mt8}
PROJECT_NAME=${PROJECT_NAME:-grpo_coderl}
CKPT_DIR=${CKPT_DIR:-$REPO/checkpoints/${PROJECT_NAME}/${EXP_NAME}}
RESUME_STEP=${RESUME_STEP:-50}
RESUME_CKPT=${RESUME_CKPT:-$CKPT_DIR/global_step_${RESUME_STEP}}

# Append to previous SwanLab run (cloud: https://swanlab.cn/@swan114514/grpo_coderl/runs/wt76kt18)
SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-wt76kt18}
SWANLAB_RESUME=${SWANLAB_RESUME:-allow}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}
IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS=${N_GPUS:-${#_GPU_ARR[@]}}
TP_SIZE=${TP_SIZE:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$N_GPUS}
GROUP_SIZE=${GROUP_SIZE:-32}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-$TRAIN_BATCH_SIZE}
TOTAL_STEPS=${TOTAL_STEPS:-100}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.70}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-12288}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-False}
PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-4}
PPO_MINI_BSZ=${PPO_MINI_BSZ:-$((8 * N_GPUS))}

export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=$REPO/swanlog
export SWANLAB_RUN_ID SWANLAB_RESUME
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com/}
export HF_HOME=${HF_HOME:-$ROOT/datasets/hf_cache}
export SWEBENCH_DATA_ROOT=${SWEBENCH_DATA_ROOT:-$ROOT/datasets}
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
_np="${NO_PROXY:-}"
for _h in api.swanlab.cn swanlab.cn .swanlab.cn localhost 127.0.0.1; do
  case ",${_np}," in *",${_h},"*) ;; *) _np="${_np:+$_np,}${_h}" ;; esac
done
export NO_PROXY="$_np" no_proxy="$_np"
SWANLAB_API_KEY_FILE=${SWANLAB_API_KEY_FILE:-$ROOT/.swanlab_api_key}
if [[ -z "${SWANLAB_API_KEY:-}" && -f "$SWANLAB_API_KEY_FILE" ]]; then
  SWANLAB_API_KEY=$(head -1 "$SWANLAB_API_KEY_FILE" | tr -d '\r\n')
fi
if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
  echo "  swanlab key loaded (mode=local, resume id=$SWANLAB_RUN_ID)"
else
  echo "WARN: no SWANLAB_API_KEY / $SWANLAB_API_KEY_FILE (local logs still OK)" >&2
fi

export TRAIN_BATCH_SIZE GROUP_SIZE VAL_DATA_SIZE
export CUDA_VISIBLE_DEVICES
export SWEBENCH_TIMING_LOG=$REPO/logs/swebench_timing_${EXP_NAME}_resume${RESUME_STEP}.jsonl

cd "$REPO"
mkdir -p logs "$SWANLAB_LOG_DIR" "$HF_HOME"

# ---- preflight: ckpt + host RAM killers ----
if [[ ! -d "$RESUME_CKPT/actor" ]]; then
  echo "ERROR: missing resume actor ckpt: $RESUME_CKPT/actor" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: missing SFT config/tokenizer path: $MODEL_PATH" >&2
  exit 1
fi
# Keep auto-resume pointer on step 50 (in case a partial step wrote a bad pointer).
mkdir -p "$CKPT_DIR"
echo "$RESUME_STEP" >"$CKPT_DIR/latest_checkpointed_iteration.txt"
echo "  resume pointer -> $CKPT_DIR/latest_checkpointed_iteration.txt = $RESUME_STEP"

if pgrep -af 'transform_ui_fps' >/dev/null 2>&1; then
  echo "ERROR: transform_ui_fps is still running (previously ~1.8TB RAM; killed this job)." >&2
  echo "  Kill it first, e.g.: pkill -f transform_ui_fps" >&2
  pgrep -af 'transform_ui_fps' || true
  exit 1
fi

if (( ${#_GPU_ARR[@]} != N_GPUS )); then
  echo "ERROR: CUDA_VISIBLE_DEVICES has ${#_GPU_ARR[@]} ids (${CUDA_VISIBLE_DEVICES}) but N_GPUS=$N_GPUS" >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be divisible by N_GPUS ($N_GPUS)" >&2
  exit 1
fi
if (( N_GPUS % TP_SIZE != 0 )); then
  echo "ERROR: N_GPUS ($N_GPUS) must be divisible by TP_SIZE ($TP_SIZE)" >&2
  exit 1
fi
_norm_mini=$((PPO_MINI_BSZ / N_GPUS))
if (( PPO_MINI_BSZ % N_GPUS != 0 )) || (( _norm_mini % PPO_MICRO_BSZ != 0 )); then
  echo "ERROR: after norm, ppo_mini=$PPO_MINI_BSZ // n_gpus=$N_GPUS -> $_norm_mini must be divisible by micro=$PPO_MICRO_BSZ" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES N_GPUS TRAIN_BATCH_SIZE GROUP_SIZE VAL_DATA_SIZE PPO_MINI_BSZ PPO_MICRO_BSZ

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "==> preflight (fail fast on import / preset issues)"
REPO="$REPO" SWEBENCH_DATA_ROOT="${SWEBENCH_DATA_ROOT:-$ROOT/datasets}" \
  python3 "$REPO/scripts/preflight_grpo_coderl_sft.py"

LOG=$REPO/logs/${EXP_NAME}_resume${RESUME_STEP}.nohup.log
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  nohup env FOREGROUND=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    N_GPUS="$N_GPUS" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    GROUP_SIZE="$GROUP_SIZE" \
    VAL_DATA_SIZE="$VAL_DATA_SIZE" \
    GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
    PPO_MINI_BSZ="$PPO_MINI_BSZ" \
    PPO_MICRO_BSZ="$PPO_MICRO_BSZ" \
    SWANLAB_RUN_ID="$SWANLAB_RUN_ID" \
    SWANLAB_RESUME="$SWANLAB_RESUME" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    bash "$SCRIPT_PATH" >"$LOG" 2>&1 &
  echo "PID=$!  log=$LOG"
  echo "  resume=$RESUME_CKPT  swanlab_id=$SWANLAB_RUN_ID"
  echo "  devices=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS batch=$TRAIN_BATCH_SIZE mini=$PPO_MINI_BSZ micro=$PPO_MICRO_BSZ"
  echo "tail -f $LOG"
  exit 0
fi

ray stop --force || true
sleep 3

echo "RESUME GRPO from step $RESUME_STEP -> total $TOTAL_STEPS"
echo "  ckpt=$RESUME_CKPT"
echo "  model(config)=$MODEL_PATH | bench=apps_train_coderl"
echo "  batch=$TRAIN_BATCH_SIZE group=$GROUP_SIZE rollouts=$((TRAIN_BATCH_SIZE * GROUP_SIZE))"
echo "  gpus=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS tp=$TP_SIZE util=$GPU_MEMORY_UTIL"
echo "  swanlab=$SWANLAB_MODE id=$SWANLAB_RUN_ID resume=$SWANLAB_RESUME logdir=$SWANLAB_LOG_DIR"

bash examples/grpo_trainer/run_swebench.sh vllm \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.max_num_gen_batches=20 \
  env.swebench.benchmark=apps_train_coderl \
  env.swebench.test_feedback_mode=interactive \
  env.swebench.auto_test_after_edit=True \
  env.swebench.step_reward_coef=0.15 \
  env.swebench.io_max_cases=64 \
  env.swebench.io_memory_limit_mb=2048 \
  env.max_steps=8 \
  env.swebench.max_turns=8 \
  env.swebench.min_turns_before_finish=3 \
  env.history_length=5 \
  data.max_prompt_length=8192 \
  data.max_response_length=4096 \
  data.truncation=left \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BSZ" \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BSZ" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BSZ" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BSZ" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTIL" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
  actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE_ENGINE" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.total_epochs=9999 \
  trainer.val_before_train=False \
  trainer.save_freq=20 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.test_freq=-1 \
  trainer.skip_val_envs=True \
  trainer.resume_mode=auto \
  trainer.experiment_name="$EXP_NAME" \
  trainer.project_name="$PROJECT_NAME" \
  "trainer.logger=['console','swanlab']" \
  env.swebench.data_root="$ROOT/datasets" \
  env.resources_per_worker.num_cpus=0.5
