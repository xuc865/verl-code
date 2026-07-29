#!/usr/bin/env bash
# Deploy GRPO ckpt (FSDP shards → HF merge → vLLM OpenAI API) for eval.
#
# Why merge: global_step_N/actor has model_world_size_6_rank_*.pt (not HF).
# vLLM needs a merged HuggingFace dir.
#
# Usage (train host TENCENT64):
#   # 1) merge step-50 + serve on free port 8000 (GPU 1 by default; GPU0 often has :80 vllm)
#   bash /mnt/z4/solariewang/verl-swe/scripts/launch_vllm_grpo_sft_mt8_ckpt.sh start
#
#   # 2) wait + smoke
#   bash .../launch_vllm_grpo_sft_mt8_ckpt.sh wait
#   bash .../launch_vllm_grpo_sft_mt8_ckpt.sh test
#
#   # 3) stop
#   bash .../launch_vllm_grpo_sft_mt8_ckpt.sh stop
#
# Only merge (no serve):
#   bash .../launch_vllm_grpo_sft_mt8_ckpt.sh merge
#
# Overrides:
#   STEP=50 PORT=8000 CUDA_VISIBLE_DEVICES=1 TP=1
#   GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=32768
#   VLLM_ENV=/opt/conda/envs/vllm
#   CONDA_ENV=verl-agent   # for model_merger

set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=${REPO:-$ROOT/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO/logs}

STEP=${STEP:-50}
CKPT_ROOT=${CKPT_ROOT:-$REPO/checkpoints/grpo_coderl/grpo_coderl_qwen25_7b_sft_mt8}
ACTOR_DIR=${ACTOR_DIR:-$CKPT_ROOT/global_step_${STEP}/actor}
HF_DIR=${HF_DIR:-$CKPT_ROOT/hf_global_step_${STEP}}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-grpo_coderl_sft_mt8_step${STEP}}

VLLM_ENV=${VLLM_ENV:-/opt/conda/envs/vllm}
CONDA_ENV=${CONDA_ENV:-verl-agent}

PID_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME}.nohup.log"
API_BASE="http://127.0.0.1:${PORT}/v1"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_port_free() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ! ss -lntu 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${p}$"
  else
    python3 - "$p" <<'PY'
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
  fi
}

_activate_conda() {
  local env=$1
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "$env"
}

_activate_vllm() {
  if [[ -x "$VLLM_ENV/bin/vllm" || -x "$VLLM_ENV/bin/python" ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "$(basename "$VLLM_ENV")" 2>/dev/null || true
    export PATH="$VLLM_ENV/bin:$PATH"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: vLLM not found (VLLM_ENV=$VLLM_ENV)" >&2
  exit 1
}

merge_hf() {
  if [[ -f "$HF_DIR/config.json" ]] && compgen -G "$HF_DIR/model*.safetensors" >/dev/null 2>&1; then
    echo "[$(_ts)] HF already merged: $HF_DIR"
    return 0
  fi
  if [[ ! -f "$ACTOR_DIR/config.json" ]]; then
    echo "ERROR: missing actor ckpt: $ACTOR_DIR/config.json" >&2
    exit 1
  fi
  if ! compgen -G "$ACTOR_DIR/model_world_size_*_rank_*.pt" >/dev/null; then
    echo "ERROR: no FSDP shards under $ACTOR_DIR" >&2
    exit 1
  fi

  echo "[$(_ts)] merging FSDP → HF"
  echo "  actor=$ACTOR_DIR"
  echo "  out=$HF_DIR"
  mkdir -p "$HF_DIR"
  _activate_conda "$CONDA_ENV"
  cd "$REPO"
  python3 scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$HF_DIR"
  echo "[$(_ts)] merge done"
}

_running_pid() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  pgrep -f "vllm serve .*${HF_DIR}" 2>/dev/null | head -1 || true
}

start_server() {
  mkdir -p "$LOG_DIR"
  merge_hf
  _activate_vllm

  if ! _port_free "$PORT"; then
    echo "ERROR: PORT=$PORT is busy. Pick another:" >&2
    echo "  PORT=\$(bash $REPO/scripts/find_free_port.sh --one --start 8000) \\" >&2
    echo "    bash $0 start" >&2
    exit 1
  fi

  local old
  old="$(_running_pid)"
  if [[ -n "$old" ]]; then
    echo "[$(_ts)] already running pid=$old API=$API_BASE"
    exit 0
  fi

  echo "[$(_ts)] starting vLLM"
  echo "  hf=$HF_DIR"
  echo "  name=$SERVED_MODEL_NAME"
  echo "  api=$API_BASE"
  echo "  cuda=$CUDA_VISIBLE_DEVICES tp=$TP util=$GPU_MEM_UTIL max_len=$MAX_MODEL_LEN"
  echo "  log=$LOG_FILE"

  export CUDA_VISIBLE_DEVICES
  nohup vllm serve "$HF_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype auto \
    --trust-remote-code \
    >>"$LOG_FILE" 2>&1 &

  echo $! >"$PID_FILE"
  echo "[$(_ts)] started pid=$(cat "$PID_FILE")"
  echo "  tail -f $LOG_FILE"
  echo "  wait: bash $0 wait"
}

stop_server() {
  local pid
  pid="$(_running_pid)"
  if [[ -z "$pid" ]]; then
    echo "[$(_ts)] not running"
    rm -f "$PID_FILE"
    return 0
  fi
  echo "[$(_ts)] stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[$(_ts)] stopped"
}

wait_ready() {
  local tries=${1:-90}
  for i in $(seq 1 "$tries"); do
    if curl -fsS "$API_BASE/models" >/dev/null 2>&1; then
      echo "[$(_ts)] API ready: $API_BASE"
      curl -fsS "$API_BASE/models" | python3 -m json.tool | head -30
      return 0
    fi
    echo "[$(_ts)] waiting ($i/$tries) ..."
    sleep 5
  done
  echo "ERROR: not ready after $((tries * 5))s — check $LOG_FILE" >&2
  tail -n 40 "$LOG_FILE" 2>/dev/null || true
  return 1
}

smoke_test() {
  wait_ready 90
  echo "[$(_ts)] === chat smoke ==="
  curl -fsS "$API_BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "user", "content": "Write a Python function is_palindrome(s). Return only the function."}
  ],
  "max_tokens": 256,
  "temperature": 0.2,
}))
PY
)" | python3 -m json.tool | head -60
}

status_server() {
  local pid
  pid="$(_running_pid)"
  echo "step=$STEP hf=$HF_DIR port=$PORT cuda=$CUDA_VISIBLE_DEVICES"
  if [[ -n "$pid" ]]; then
    echo "[$(_ts)] running pid=$pid API=$API_BASE"
  else
    echo "[$(_ts)] not running"
  fi
  [[ -f "$LOG_FILE" ]] && tail -n 8 "$LOG_FILE" || true
}

usage() {
  cat <<EOF
Usage: $0 {merge|start|stop|status|wait|test}

Defaults: STEP=50 PORT=8000 CUDA_VISIBLE_DEVICES=1

Train host:
  bash $0 start
  bash $0 wait
  bash $0 test

Eval against this API:
  API_BASE=http://127.0.0.1:8000/v1 MODEL=${SERVED_MODEL_NAME} \\
    bash $REPO/scripts/eval_api_qwen25_coder_7b_pure.sh
EOF
}

cmd=${1:-start}
case "$cmd" in
  merge) merge_hf ;;
  start) start_server ;;
  stop) stop_server ;;
  status) status_server ;;
  wait) wait_ready "${2:-90}" ;;
  test) smoke_test ;;
  *) usage; exit 1 ;;
esac
