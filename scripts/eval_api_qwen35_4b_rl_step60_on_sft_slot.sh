#!/usr/bin/env bash
# 用 Qwen3.5-4B RL ckpt（默认 SFT→GRPO tp1_g16 @ global_step_60）顶掉 SFT 评测槽：
#   GPU1 / :8000  — 停 SFT vLLM + 残留 SFT eval，merge FSDP→HF，再起 vLLM，跑全套评测。
#
# 训练机：
#   bash /mnt/z4/solariewang/verl-swe/scripts/eval_api_qwen35_4b_rl_step60_on_sft_slot.sh
#
# 可选：
#   STEP=60
#   CKPT_ROOT=.../grpo_coderl_qwen35_4b_sft_mt8_tp1_g16   # 默认
#   # 若要旧的 base GRPO：
#   CKPT_ROOT=/mnt/z4/solariewang/verl-swe/checkpoints/grpo_code_apps/grpo_qwen35_4b_apps_train_full
#   SKIP_EVAL=1          # 只起服务
#   SKIP_VLLM_RESTART=1  # 假定 HF 已在 :8000 服着
#   EVAL_PROTOCOL=train  # 默认：与 RL 训练对齐 mt8·interactive·think ON
#   EVAL_PROTOCOL=sft    # 与 SFT 评测对齐 mt12·exec·think ON·react
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-/mnt/z4/solariewang}
REPO_ROOT=${REPO_ROOT:-$ROOT/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

STEP=${STEP:-60}
CKPT_ROOT=${CKPT_ROOT:-$REPO_ROOT/checkpoints/grpo_coderl/grpo_coderl_qwen35_4b_sft_mt8_tp1_g16}
ACTOR_DIR=${ACTOR_DIR:-$CKPT_ROOT/global_step_${STEP}/actor}
HF_DIR=${HF_DIR:-$CKPT_ROOT/hf_global_step_${STEP}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-grpo_qwen35_4b_sft_mt8_tp1_g16_step${STEP}}
METHOD=${METHOD:-${SERVED_MODEL_NAME}_mt8_interactive}

PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-train} # train | sft

export REPO_ROOT DATA_ROOT LOG_DIR ROOT
export API_BASE=${API_BASE:-http://127.0.0.1:${PORT}/v1}
export MODEL=${MODEL:-$SERVED_MODEL_NAME}
export METHOD
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_kill_port() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    local pids
    pids="$(ss -lntp 2>/dev/null | awk -v p=":$p" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
    for pid in $pids; do
      echo "[$(_ts)] kill pid=$pid on :$p"
      kill "$pid" 2>/dev/null || true
    done
  fi
  fuser -k "${p}/tcp" 2>/dev/null || true
  sleep 2
  fuser -k -9 "${p}/tcp" 2>/dev/null || true
}

_stop_sft_slot() {
  echo "[$(_ts)] freeing GPU${CUDA_VISIBLE_DEVICES}/:${PORT} (SFT slot + leftover evals)"
  # stop SFT / prior qwen35 serves on this port
  QWEN35_SERVED_NAME=qwen35-4b-apps-mt8-sft PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" stop 2>/dev/null || true
  QWEN35_SERVED_NAME="$SERVED_MODEL_NAME" PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" stop 2>/dev/null || true
  SERVED_MODEL_NAME=grpo_coderl_sft_mt8_step50 PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" stop 2>/dev/null || true
  # leftover eval writers (SFT method + polluters using same METHOD prefix)
  pkill -f 'eval_api_.*qwen35_4b_sft_mt12_exec' 2>/dev/null || true
  pkill -f "eval_api_baseline.py.*${METHOD}" 2>/dev/null || true
  pkill -f 'run_qwen35_4b_sft_eval_retry' 2>/dev/null || true
  _kill_port "$PORT"
  sleep 3
}

_merge_hf() {
  if [[ -f "$HF_DIR/config.json" ]] && compgen -G "$HF_DIR"/model*.safetensors >/dev/null 2>&1; then
    echo "[$(_ts)] HF already merged: $HF_DIR"
  else
    if [[ ! -f "$ACTOR_DIR/config.json" ]]; then
      echo "[$(_ts)] ERROR: missing actor $ACTOR_DIR/config.json" >&2
      exit 1
    fi
    if ! compgen -G "$ACTOR_DIR"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
      echo "[$(_ts)] ERROR: no FSDP shards under $ACTOR_DIR" >&2
      exit 1
    fi
    echo "[$(_ts)] merging FSDP → HF"
    echo "  actor=$ACTOR_DIR"
    echo "  out=$HF_DIR"
    mkdir -p "$HF_DIR"
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-verl-agent}"
    python3 "$REPO_ROOT/scripts/model_merger.py" merge \
      --backend fsdp \
      --local_dir "$ACTOR_DIR" \
      --target_dir "$HF_DIR"
    echo "[$(_ts)] merge done"
  fi
  # Qwen3.5 sidecars from base if missing
  local base=${SFT_BASE_MODEL:-$ROOT/models/Qwen3.5-4B}
  local f
  for f in preprocessor_config.json video_preprocessor_config.json generation_config.json merges.txt vocab.json; do
    if [[ -f "$base/$f" && ! -f "$HF_DIR/$f" ]]; then
      cp -f "$base/$f" "$HF_DIR/$f"
      echo "[$(_ts)] copied sidecar $f"
    fi
  done
}

_start_vllm() {
  echo "[$(_ts)] starting vLLM on :$PORT GPU=$CUDA_VISIBLE_DEVICES"
  echo "  hf=$HF_DIR served=$SERVED_MODEL_NAME"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PORT="$PORT" \
    QWEN35_MODEL_PATH="$HF_DIR" \
    QWEN35_SERVED_NAME="$SERVED_MODEL_NAME" \
    LANGUAGE_MODEL_ONLY=1 \
    REASONING_PARSER=qwen3 \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" start
  echo "[$(_ts)] waiting for API $API_BASE ..."
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PORT="$PORT" \
    QWEN35_MODEL_PATH="$HF_DIR" \
    QWEN35_SERVED_NAME="$SERVED_MODEL_NAME" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" wait 480
}

_check_api() {
  local url="${API_BASE%/}/models"
  echo "[$(_ts)] checking $url"
  local i
  for i in $(seq 1 60); do
    if curl -fsS --max-time 10 "$url" | grep -q "$SERVED_MODEL_NAME"; then
      echo "[$(_ts)] API ok"
      curl -fsS --max-time 10 "$url" | head -c 600 || true
      echo
      return 0
    fi
    sleep 10
  done
  echo "[$(_ts)] ERROR: API not ready / model id mismatch" >&2
  exit 1
}

_set_protocol() {
  if [[ "$EVAL_PROTOCOL" == "sft" ]]; then
    METHOD=${METHOD%_mt8_interactive}_mt12_exec
    export METHOD
    export TEST_FEEDBACK_MODE=exec
    export EVAL_MAX_TURNS=12
    export EVAL_HISTORY_LENGTH=6
    export MAX_TOKENS=8192
    export DISABLE_THINKING=0
    export PROMPT_MODE=react
    export ENCOURAGE_COT=0
    echo "[$(_ts)] protocol=SFT-like mt12·exec·think ON"
  else
    export TEST_FEEDBACK_MODE=interactive
    export EVAL_MAX_TURNS=8
    export EVAL_HISTORY_LENGTH=5
    export MAX_TOKENS=4096
    export DISABLE_THINKING=0
    export PROMPT_MODE=react
    export ENCOURAGE_COT=0
    echo "[$(_ts)] protocol=train-aligned mt8·interactive·think ON"
  fi
  export API_TIMEOUT=${API_TIMEOUT:-1800}
  export API_RETRIES=${API_RETRIES:-3}
  export EVAL_RESUME=${EVAL_RESUME:-1}
  export EVAL_RETRY_ERRORS=${EVAL_RETRY_ERRORS:-1}
  export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
  export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
  export EVAL_WORKERS=${EVAL_WORKERS:-4}
  export EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}
  export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
  export SKIP_WAIT=1
  export SKIP_PEER_WAIT=1
  export WAIT_FOR_SINGLE_TURN=0
  export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
  export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
  export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*${METHOD}"
}

# ---- nohup wrapper ----
CHAIN_LOG=${CHAIN_LOG:-$LOG_DIR/eval_api_${SERVED_MODEL_NAME}_suite.nohup.log}
if [[ "${FOREGROUND:-0}" != "1" && "${_INNER:-0}" != "1" ]]; then
  nohup env _INNER=1 FOREGROUND=1 \
    STEP="$STEP" CKPT_ROOT="$CKPT_ROOT" ACTOR_DIR="$ACTOR_DIR" HF_DIR="$HF_DIR" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" METHOD="$METHOD" \
    PORT="$PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    EVAL_PROTOCOL="$EVAL_PROTOCOL" \
    SKIP_VLLM_RESTART="${SKIP_VLLM_RESTART:-0}" SKIP_EVAL="${SKIP_EVAL:-0}" \
    bash "$0" >>"$CHAIN_LOG" 2>&1 &
  echo "PID=$!  log=$CHAIN_LOG"
  echo "  ckpt=$ACTOR_DIR"
  echo "  serve $SERVED_MODEL_NAME on GPU $CUDA_VISIBLE_DEVICES :$PORT"
  echo "  then eval METHOD=$METHOD protocol=$EVAL_PROTOCOL"
  echo "  tail -f $CHAIN_LOG"
  exit 0
fi

echo "[$(_ts)] ========== Qwen3.5-4B RL step${STEP} on SFT slot =========="
echo "  CKPT_ROOT=$CKPT_ROOT"
echo "  ACTOR=$ACTOR_DIR"
echo "  HF=$HF_DIR"
echo "  served=$SERVED_MODEL_NAME port=$PORT gpu=$CUDA_VISIBLE_DEVICES"

if [[ "${SKIP_VLLM_RESTART:-0}" != "1" ]]; then
  _stop_sft_slot
  _merge_hf
  _start_vllm
fi
_check_api

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_EVAL=1 — server only"
  exit 0
fi

_set_protocol
echo "[$(_ts)] API=$API_BASE MODEL=$MODEL METHOD=$METHOD"

echo "[$(_ts)] stage 1/2: non-APPS queue"
bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" \
  >>"$LOG_DIR/eval_api_${METHOD}_queue.nohup.log" 2>&1 || \
  echo "[$(_ts)] WARNING: queue exit=$?"

echo "[$(_ts)] stage 2/2: APPS"
bash "$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh" \
  >>"$LOG_DIR/eval_api_${METHOD}_apps.nohup.log" 2>&1 || \
  echo "[$(_ts)] WARNING: apps exit=$?"

echo "[$(_ts)] ========== RL step${STEP} eval DONE =========="
