#!/usr/bin/env bash
# 顶掉 Qwen3.5-4B GRPO step60 评测槽（GPU1 / :8000），挂 DiDPO step60 并只测 contest 三套：
#   usaco → ojbench → icpc
#
# DiDPO ckpt（Qwen2.5-Coder-7B SFT→DiDPO）:
#   checkpoints/didpo_coderl/didpo_coderl_qwen25_7b_sft_mt8/global_step_60
#
# 训练机：
#   bash /mnt/z4/solariewang/verl-swe/scripts/eval_api_didpo_coderl_step60_usaco_oj_icpc_on_grpo_slot.sh
#
# 可选：
#   STEP=60
#   PORT=8000 CUDA_VISIBLE_DEVICES=1
#   SKIP_VLLM_RESTART=1   # 假定 HF 已在 :8000 服着
#   SKIP_EVAL=1           # 只起服务
#   FOREGROUND=1          # 前台跑（默认 nohup）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-/mnt/z4/solariewang}
REPO_ROOT=${REPO_ROOT:-$ROOT/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

STEP=${STEP:-60}
CKPT_ROOT=${CKPT_ROOT:-$REPO_ROOT/checkpoints/didpo_coderl/didpo_coderl_qwen25_7b_sft_mt8}
ACTOR_DIR=${ACTOR_DIR:-$CKPT_ROOT/global_step_${STEP}/actor}
HF_DIR=${HF_DIR:-$CKPT_ROOT/hf_global_step_${STEP}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-didpo_coderl_qwen25_7b_sft_mt8_step${STEP}}
METHOD=${METHOD:-${SERVED_MODEL_NAME}_mt8_interactive}

PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

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

_stop_grpo_qwen35_slot() {
  echo "[$(_ts)] freeing GPU${CUDA_VISIBLE_DEVICES}/:${PORT} (GRPO qwen35 step60 slot)"

  # stop vLLM serves that historically occupy this slot
  QWEN35_SERVED_NAME=grpo_qwen35_4b_sft_mt8_tp1_g16_step60 PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" stop 2>/dev/null || true
  QWEN35_SERVED_NAME=qwen35-4b-apps-mt8-sft PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" stop 2>/dev/null || true
  SERVED_MODEL_NAME=grpo_coderl_sft_mt8_step50 PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" stop 2>/dev/null || true
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME" PORT="$PORT" STEP="$STEP" \
    CKPT_ROOT="$CKPT_ROOT" HF_DIR="$HF_DIR" ACTOR_DIR="$ACTOR_DIR" \
    bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" stop 2>/dev/null || true

  # leftover GRPO / RL suite writers on this slot
  pkill -f 'eval_api_.*grpo_qwen35_4b_sft_mt8_tp1_g16_step60' 2>/dev/null || true
  pkill -f 'eval_api_qwen35_4b_rl_step60' 2>/dev/null || true
  pkill -f 'eval_api_baseline.py.*grpo_qwen35_4b_sft_mt8_tp1_g16_step60' 2>/dev/null || true
  pkill -f 'run_qwen35_4b_rl_step60' 2>/dev/null || true
  pkill -f "eval_api_baseline.py.*${METHOD}" 2>/dev/null || true

  _kill_port "$PORT"
  sleep 3
}

_merge_and_start_vllm() {
  echo "[$(_ts)] merge+serve DiDPO step${STEP}"
  echo "  actor=$ACTOR_DIR"
  echo "  hf=$HF_DIR"
  echo "  served=$SERVED_MODEL_NAME on GPU=$CUDA_VISIBLE_DEVICES :$PORT"

  STEP="$STEP" \
    CKPT_ROOT="$CKPT_ROOT" \
    ACTOR_DIR="$ACTOR_DIR" \
    HF_DIR="$HF_DIR" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    PORT="$PORT" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" merge

  STEP="$STEP" \
    CKPT_ROOT="$CKPT_ROOT" \
    ACTOR_DIR="$ACTOR_DIR" \
    HF_DIR="$HF_DIR" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    PORT="$PORT" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" start

  echo "[$(_ts)] waiting for API $API_BASE ..."
  STEP="$STEP" \
    CKPT_ROOT="$CKPT_ROOT" \
    HF_DIR="$HF_DIR" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" wait 480
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
  # Match DiDPO / GRPO train recipe: mt8 · interactive · think ON · react
  export TEST_FEEDBACK_MODE=interactive
  export EVAL_MAX_TURNS=8
  export EVAL_HISTORY_LENGTH=5
  export MAX_TOKENS=4096
  export DISABLE_THINKING=0
  export PROMPT_MODE=react
  export ENCOURAGE_COT=0
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
  export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-usaco,ojbench,icpc}
  export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
  export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*${METHOD}"
  echo "[$(_ts)] protocol=train-aligned mt8·interactive·think ON"
  echo "[$(_ts)] benches=$EVAL_BENCHMARKS"
}

# ---- nohup wrapper ----
CHAIN_LOG=${CHAIN_LOG:-$LOG_DIR/eval_api_${SERVED_MODEL_NAME}_usaco_oj_icpc.nohup.log}
if [[ "${FOREGROUND:-0}" != "1" && "${_INNER:-0}" != "1" ]]; then
  nohup env _INNER=1 FOREGROUND=1 \
    STEP="$STEP" CKPT_ROOT="$CKPT_ROOT" ACTOR_DIR="$ACTOR_DIR" HF_DIR="$HF_DIR" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" METHOD="$METHOD" \
    PORT="$PORT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-usaco,ojbench,icpc}" \
    SKIP_VLLM_RESTART="${SKIP_VLLM_RESTART:-0}" SKIP_EVAL="${SKIP_EVAL:-0}" \
    bash "$0" >>"$CHAIN_LOG" 2>&1 &
  echo "PID=$!  log=$CHAIN_LOG"
  echo "  ckpt=$ACTOR_DIR"
  echo "  kill GRPO qwen35 slot → serve $SERVED_MODEL_NAME on GPU $CUDA_VISIBLE_DEVICES :$PORT"
  echo "  then eval METHOD=$METHOD benches=${EVAL_BENCHMARKS:-usaco,ojbench,icpc}"
  echo "  tail -f $CHAIN_LOG"
  exit 0
fi

echo "[$(_ts)] ========== DiDPO step${STEP} contest eval on GRPO slot =========="
echo "  CKPT_ROOT=$CKPT_ROOT"
echo "  ACTOR=$ACTOR_DIR"
echo "  HF=$HF_DIR"
echo "  served=$SERVED_MODEL_NAME port=$PORT gpu=$CUDA_VISIBLE_DEVICES"

if [[ ! -f "$ACTOR_DIR/config.json" ]]; then
  echo "[$(_ts)] ERROR: missing actor $ACTOR_DIR/config.json" >&2
  exit 1
fi

if [[ "${SKIP_VLLM_RESTART:-0}" != "1" ]]; then
  _stop_grpo_qwen35_slot
  _merge_and_start_vllm
fi
_check_api

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_EVAL=1 — server only"
  exit 0
fi

_set_protocol
echo "[$(_ts)] API=$API_BASE MODEL=$MODEL METHOD=$METHOD"

echo "[$(_ts)] contest queue: $EVAL_BENCHMARKS"
bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" \
  >>"$LOG_DIR/eval_api_${METHOD}_queue.nohup.log" 2>&1 || \
  echo "[$(_ts)] WARNING: queue exit=$?"

echo "[$(_ts)] ========== DiDPO step${STEP} usaco/ojbench/icpc DONE =========="
