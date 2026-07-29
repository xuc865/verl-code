#!/usr/bin/env bash
# Resume DiDPO CodeRL from global_step_20 with instance_id group tracking.
# Stops relying on ephemeral uid UUIDs; pins tracked instance_ids into each batch.
#
# Usage (train host):
#   # 1) stop the old job first (free GPUs 2-7)
#   # 2) bash /mnt/z4/solariewang/verl-swe/scripts/launch_didpo_coderl_sft_mt8_resume20.sh
#
# FOREGROUND=1 bash .../launch_didpo_coderl_sft_mt8_resume20.sh

set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=${REPO:-$ROOT/verl-swe}
CONDA_ENV=${CONDA_ENV:-verl-agent}

MODEL_PATH=${MODEL_PATH:-$REPO/checkpoints/apps_mt8_sft_qwen25_coder7b_think/global_step_242}
EXP_NAME=${EXP_NAME:-didpo_coderl_qwen25_7b_sft_mt8}
PROJECT_NAME=${PROJECT_NAME:-didpo_coderl}
CKPT_DIR=${CKPT_DIR:-$REPO/checkpoints/${PROJECT_NAME}/${EXP_NAME}}
RESUME_STEP=${RESUME_STEP:-20}
RESUME_CKPT=${RESUME_CKPT:-$CKPT_DIR/global_step_${RESUME_STEP}}
GROUP_DUMP_DIR=${GROUP_DUMP_DIR:-$REPO/logs/didpo_groups}
TRACK_FILE=${TRACK_FILE:-$GROUP_DUMP_DIR/tracked_instance_ids.json}

# Continue previous local SwanLab run when possible
SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-c08qn4a8}
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
mkdir -p logs "$SWANLAB_LOG_DIR" "$HF_HOME" "$GROUP_DUMP_DIR"

# Archive stale UUID-era dumps from the previous tracker so resume starts clean.
# (Old run locked ephemeral uid UUIDs; new tracker uses stable instance_ids.)
if [[ -f "$GROUP_DUMP_DIR/didpo_prompt_groups.jsonl" ]] || \
   compgen -G "$GROUP_DUMP_DIR/prompt_*.json" >/dev/null 2>&1; then
  _arch="$GROUP_DUMP_DIR/archive_uuid_era_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$_arch"
  mv "$GROUP_DUMP_DIR/didpo_prompt_groups.jsonl" "$_arch/" 2>/dev/null || true
  mv "$GROUP_DUMP_DIR"/prompt_*.json "$_arch/" 2>/dev/null || true
  # Drop UUID-only tracked file if present (stable apps__* files are kept).
  if [[ -f "$TRACK_FILE" ]]; then
    if python3 - "$TRACK_FILE" <<'PY'
import json, re, sys
uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
p = sys.argv[1]
try:
    data = json.load(open(p))
except Exception:
    raise SystemExit(1)
ids = data.get("instance_ids") or data.get("uids") or []
stable = [x for x in ids if str(x) and not uuid_re.match(str(x))]
raise SystemExit(0 if stable else 1)
PY
    then
      echo "  keep existing stable tracked ids: $TRACK_FILE"
    else
      mv "$TRACK_FILE" "$_arch/" 2>/dev/null || true
      echo "  archived UUID-only track file -> $_arch"
    fi
  fi
  echo "  archived old UUID group dumps -> $_arch"
fi

# ---- preflight: ckpt ----
if [[ ! -d "$RESUME_CKPT/actor" ]]; then
  echo "ERROR: missing resume actor ckpt: $RESUME_CKPT/actor" >&2
  exit 1
fi
_n_ranks=$(ls "$RESUME_CKPT/actor"/model_world_size_*_rank_0.pt 2>/dev/null | wc -l)
if [[ ! -f "$RESUME_CKPT/actor/model_world_size_${N_GPUS}_rank_0.pt" ]]; then
  echo "ERROR: resume ckpt world_size mismatch; need model_world_size_${N_GPUS}_rank_*.pt under $RESUME_CKPT/actor" >&2
  ls "$RESUME_CKPT/actor"/model_world_size_*_rank_0.pt 2>/dev/null || true
  exit 1
fi
echo "  ckpt shards ok (looking for world_size=${N_GPUS}; found rank0 files: ${_n_ranks})"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: missing SFT config/tokenizer path: $MODEL_PATH" >&2
  exit 1
fi

# Force auto-resume pointer to step 20 (ignore any later in-memory progress of the old job).
mkdir -p "$CKPT_DIR"
echo "$RESUME_STEP" >"$CKPT_DIR/latest_checkpointed_iteration.txt"
echo "  resume pointer -> $CKPT_DIR/latest_checkpointed_iteration.txt = $RESUME_STEP"
echo "  track_file=$TRACK_FILE"

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

echo "==> preflight (imports + didpo + instance_id tracker)"
REPO="$REPO" SWEBENCH_DATA_ROOT="${SWEBENCH_DATA_ROOT:-$ROOT/datasets}" \
  python3 "$REPO/scripts/preflight_grpo_coderl_sft.py"
REPO="$REPO" GROUP_DUMP_DIR="$GROUP_DUMP_DIR" python3 - <<'PY'
import os, sys
repo = os.environ["REPO"]
sys.path.insert(0, repo)
from didpo.group_tracker import DidpoGroupTracker
from didpo import core_didpo  # noqa: F401
from verl.trainer.ppo.ray_trainer import AdvantageEstimator
assert AdvantageEstimator.DIDPO.value == "didpo"
td = os.environ["GROUP_DUMP_DIR"]
t = DidpoGroupTracker(dump_dir=td, track_n=3)
assert t._tracked_file.name == "tracked_instance_ids.json"
print("[preflight] didpo instance_id tracker ok; dump=", td)
PY

LOG=$REPO/logs/${EXP_NAME}_resume${RESUME_STEP}.nohup.log
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  # Refuse to start if old DiDPO train still holds GPUs (best-effort).
  if pgrep -af 'verl.trainer.main_ppo.*adv_estimator=didpo|launch_didpo_coderl' >/dev/null 2>&1; then
    echo "WARN: existing didpo/main_ppo processes detected. Kill them before resume, e.g.:" >&2
    pgrep -af 'main_ppo|launch_didpo' || true
    echo "  Continuing nohup launch anyway in 3s (ray stop will run in child)..." >&2
    sleep 3
  fi
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
    RESUME_STEP="$RESUME_STEP" \
    GROUP_DUMP_DIR="$GROUP_DUMP_DIR" \
    TRACK_FILE="$TRACK_FILE" \
    bash "$SCRIPT_PATH" >"$LOG" 2>&1 &
  echo "PID=$!  log=$LOG"
  echo "  resume=$RESUME_CKPT  swanlab_id=$SWANLAB_RUN_ID"
  echo "  devices=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS batch=$TRAIN_BATCH_SIZE"
  echo "  track_file=$TRACK_FILE"
  echo "tail -f $LOG"
  exit 0
fi

ray stop --force || true
sleep 3

echo "RESUME DiDPO from step $RESUME_STEP -> total $TOTAL_STEPS"
echo "  ckpt=$RESUME_CKPT"
echo "  model(config)=$MODEL_PATH | bench=apps_train_coderl"
echo "  batch=$TRAIN_BATCH_SIZE group=$GROUP_SIZE rollouts=$((TRAIN_BATCH_SIZE * GROUP_SIZE))"
echo "  gpus=$CUDA_VISIBLE_DEVICES n_gpus=$N_GPUS tp=$TP_SIZE util=$GPU_MEMORY_UTIL"
echo "  swanlab=$SWANLAB_MODE id=$SWANLAB_RUN_ID resume=$SWANLAB_RESUME"
echo "  instance_id tracking dump=$GROUP_DUMP_DIR file=$TRACK_FILE"

bash examples/didpo_trainer/run_swebench.sh vllm \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.max_num_gen_batches=20 \
  algorithm.didpo.overhead_mode=lightweight \
  algorithm.didpo.snippet_advantage_w=1.0 \
  algorithm.didpo.track_prompt_n=3 \
  algorithm.didpo.group_dump_dir="$GROUP_DUMP_DIR" \
  env.swebench.tracked_instance_ids_file="$TRACK_FILE" \
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
