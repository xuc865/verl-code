#!/usr/bin/env bash
# CodeRL+ GRPO from Qwen3.5-4B APPS mt8 SFT ckpt (same recipe as
# launch_grpo_coderl_sft_mt8.sh / 7B), on GPUs 2–7.
#
# Usage (train host):
#   bash /mnt/z4/solariewang/verl-swe/scripts/launch_grpo_coderl_qwen35_4b_sft_mt8.sh
#   FOREGROUND=1 bash .../launch_grpo_coderl_qwen35_4b_sft_mt8.sh
#
# Or auto-chain after SFT:
#   bash scripts/launch_grpo_qwen35_4b_after_sft.sh
#
# Env overrides: MODEL_PATH EXP_NAME TOTAL_STEPS CUDA_VISIBLE_DEVICES ...
set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=${REPO:-$ROOT/verl-swe}
CONDA_ENV=${CONDA_ENV:-verl-agent}

SFT_CKPT_ROOT=${SFT_CKPT_ROOT:-$REPO/checkpoints/apps_mt8_sft_qwen35_4b_think}
# Default: latest global_step_* under SFT_CKPT_ROOT (set MODEL_PATH to pin a step).
# Sort by trailing step number — full paths have many '_' so sort -t_ -k3 is wrong
# (e.g. global_step_81 can sort "after" global_step_162 lexicographically).
if [[ -z "${MODEL_PATH:-}" ]]; then
  _latest="" _best_n=-1
  shopt -s nullglob
  for _d in "$SFT_CKPT_ROOT"/global_step_*; do
    _n="${_d##*_}"
    [[ "$_n" =~ ^[0-9]+$ ]] || continue
    if (( _n > _best_n )); then
      _best_n=$_n
      _latest=$_d
    fi
  done
  shopt -u nullglob
  MODEL_PATH=${_latest:-$SFT_CKPT_ROOT/global_step_0}
fi
EXP_NAME=${EXP_NAME:-grpo_coderl_qwen35_4b_sft_mt8}
PROJECT_NAME=${PROJECT_NAME:-grpo_coderl}

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
SAVE_FREQ=${SAVE_FREQ:-20}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=$REPO/swanlog
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
  echo "  swanlab key loaded (mode=local, no cloud login)"
else
  echo "WARN: no SWANLAB_API_KEY / $SWANLAB_API_KEY_FILE (local logs still OK)" >&2
fi

export TRAIN_BATCH_SIZE GROUP_SIZE VAL_DATA_SIZE
export CUDA_VISIBLE_DEVICES
export SWEBENCH_TIMING_LOG=$REPO/logs/swebench_timing_${EXP_NAME}.jsonl

cd "$REPO"
mkdir -p logs "$SWANLAB_LOG_DIR" "$HF_HOME"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: missing SFT ckpt: $MODEL_PATH" >&2
  echo "  expected under $SFT_CKPT_ROOT/global_step_*" >&2
  exit 1
fi
# Sanity: reject accidental 7B / wrong-family ckpt
if ! grep -q 'Qwen3_5\|qwen3_5\|Qwen3.5' "$MODEL_PATH/config.json" 2>/dev/null; then
  echo "ERROR: MODEL_PATH does not look like Qwen3.5: $MODEL_PATH" >&2
  echo "  architectures/model_type in config.json:" >&2
  grep -E 'architectures|model_type' "$MODEL_PATH/config.json" >&2 || true
  exit 1
fi
# SFT save may omit multimodal processor sidecars; copy from base before vLLM load.
SFT_BASE_MODEL=${SFT_BASE_MODEL:-$ROOT/models/Qwen3.5-4B}
for _f in preprocessor_config.json video_preprocessor_config.json generation_config.json merges.txt vocab.json; do
  if [[ -f "$SFT_BASE_MODEL/$_f" && ! -f "$MODEL_PATH/$_f" ]]; then
    cp -f "$SFT_BASE_MODEL/$_f" "$MODEL_PATH/$_f"
    echo "  copied sidecar $_f into $MODEL_PATH"
  fi
done
# Avoid `ls | grep -q` under pipefail (SIGPIPE → false "missing weights").
_w=()
shopt -s nullglob
_w=("$MODEL_PATH"/model*.safetensors "$MODEL_PATH"/pytorch_model*.bin)
shopt -u nullglob
if (( ${#_w[@]} == 0 )); then
  echo "ERROR: MODEL_PATH has no weight shards: $MODEL_PATH" >&2
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

LOG=$REPO/logs/${EXP_NAME}.nohup.log
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  nohup env FOREGROUND=1 \
    MODEL_PATH="$MODEL_PATH" \
    EXP_NAME="$EXP_NAME" \
    PROJECT_NAME="$PROJECT_NAME" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    N_GPUS="$N_GPUS" \
    TP_SIZE="$TP_SIZE" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    GROUP_SIZE="$GROUP_SIZE" \
    VAL_DATA_SIZE="$VAL_DATA_SIZE" \
    GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL" \
    MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
    PPO_MINI_BSZ="$PPO_MINI_BSZ" \
    PPO_MICRO_BSZ="$PPO_MICRO_BSZ" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    SAVE_FREQ="$SAVE_FREQ" \
    MAX_ACTOR_CKPT_TO_KEEP="$MAX_ACTOR_CKPT_TO_KEEP" \
    bash "$SCRIPT_PATH" >"$LOG" 2>&1 &
  echo "PID=$!  log=$LOG"
  echo "  model=$MODEL_PATH"
  echo "  devices=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS batch=$TRAIN_BATCH_SIZE group=$GROUP_SIZE tp=$TP_SIZE"
  echo "  util=$GPU_MEMORY_UTIL micro=$PPO_MICRO_BSZ save_freq=$SAVE_FREQ"
  echo "tail -f $LOG"
  exit 0
fi

ray stop --force || true
sleep 3

echo "CodeRL+ GRPO (Qwen3.5-4B from SFT) | model=$MODEL_PATH | bench=apps_train_coderl"
echo "  exp=$EXP_NAME batch=$TRAIN_BATCH_SIZE group=$GROUP_SIZE steps=$TOTAL_STEPS"
echo "  gpus=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS tp=$TP_SIZE util=$GPU_MEMORY_UTIL"
echo "  save_freq=$SAVE_FREQ max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"

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
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.total_epochs=9999 \
  trainer.val_before_train=False \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.test_freq=-1 \
  trainer.skip_val_envs=True \
  trainer.experiment_name="$EXP_NAME" \
  trainer.project_name="$PROJECT_NAME" \
  "trainer.logger=['console','swanlab']" \
  env.swebench.data_root="$ROOT/datasets" \
  env.resources_per_worker.num_cpus=0.5
