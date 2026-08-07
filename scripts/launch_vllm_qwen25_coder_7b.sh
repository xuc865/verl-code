#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible API for Qwen2.5-Coder-7B-Instruct on training host.
#
# Usage (training machine):
#   cd /mnt/z4/solariewang/verl-swe
#   bash scripts/launch_vllm_qwen25_coder_7b.sh start
#
# Smoke test after server is up:
#   bash scripts/launch_vllm_qwen25_coder_7b.sh test
#
# Stop:
#   bash scripts/launch_vllm_qwen25_coder_7b.sh stop
#
# Env overrides (training-host defaults match current working setup):
#   MODEL_PATH=/mnt/z4/solariewang/models/Qwen2.5-Coder-7B-Instruct
#   SERVED_MODEL_NAME=Qwen2.5-Coder-7B-Instruct
#   HOST=0.0.0.0  PORT=80  CUDA_VISIBLE_DEVICES=0
#   TP=1  MAX_MODEL_LEN=32768  GPU_MEM_UTIL=0.90
#   VLLM_ENV=/opt/conda/envs/vllm
#   LOG_DIR=/mnt/z4/solariewang/verl-swe/logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}

MODEL_PATH=${MODEL_PATH:-/mnt/z4/solariewang/models/Qwen2.5-Coder-7B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen2.5-Coder-7B-Instruct}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
VLLM_ENV=${VLLM_ENV:-/opt/conda/envs/vllm}

PID_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME//\//_}.pid"
LOG_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME//\//_}.nohup.log"
API_BASE="http://127.0.0.1:${PORT}/v1"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_activate_vllm() {
  if [[ -x "$VLLM_ENV/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$VLLM_ENV")/../etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$VLLM_ENV" 2>/dev/null || true
    export PATH="$VLLM_ENV/bin:$PATH"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: vLLM env not found: $VLLM_ENV" >&2
  echo "  export VLLM_ENV=/path/to/conda/env/with/vllm" >&2
  exit 1
}

_check_model() {
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: model not found: $MODEL_PATH/config.json" >&2
    exit 1
  fi
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
  pgrep -f "vllm serve .*${MODEL_PATH}" 2>/dev/null | head -1 || true
}

start_server() {
  mkdir -p "$LOG_DIR"
  _check_model
  _activate_vllm

  local old_pid
  old_pid="$(_running_pid)"
  if [[ -n "$old_pid" ]]; then
    echo "[$(_ts)] already running pid=$old_pid API=$API_BASE"
    exit 0
  fi

  echo "[$(_ts)] starting vLLM"
  echo "  model_path=$MODEL_PATH"
  echo "  served_name=$SERVED_MODEL_NAME"
  echo "  api=$API_BASE"
  echo "  cuda=$CUDA_VISIBLE_DEVICES tp=$TP max_len=$MAX_MODEL_LEN"
  echo "  log=$LOG_FILE"

  export CUDA_VISIBLE_DEVICES
  nohup vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype auto \
    --trust-remote-code \
    >> "$LOG_FILE" 2>&1 &

  echo $! > "$PID_FILE"
  echo "[$(_ts)] started pid=$(cat "$PID_FILE")"
  echo "  tail -f $LOG_FILE"
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
  local tries=${1:-60}
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$API_BASE/models" >/dev/null 2>&1; then
      echo "[$(_ts)] API ready: $API_BASE"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: API not ready after $((tries * 5))s — check $LOG_FILE" >&2
  return 1
}

smoke_test() {
  wait_ready 60
  echo "[$(_ts)] === /v1/models ==="
  curl -fsS "$API_BASE/models" | python3 -m json.tool | head -40

  echo
  echo "[$(_ts)] === chat completion smoke ==="
  curl -fsS "$API_BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "user", "content": "Write a Python function is_palindrome(s) that returns True if s is a palindrome."}
  ],
  "max_tokens": 256,
  "temperature": 0.2,
}))
PY
)" | python3 -m json.tool | head -80

  echo
  echo "[$(_ts)] === humaneval-style one-shot ==="
  curl -fsS "$API_BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<'PY'
import json
prompt = '''def has_close_elements(numbers, threshold):
    """Check if any two numbers are closer than threshold."""
'''
print(json.dumps({
  "model": "Qwen2.5-Coder-7B-Instruct",
  "messages": [{"role": "user", "content": prompt}],
  "max_tokens": 512,
  "temperature": 0.0,
}))
PY
)" | python3 - <<'PY'
import json,sys
r=json.load(sys.stdin)
text=r["choices"][0]["message"]["content"]
print(text[:1200])
PY
}

status_server() {
  local pid
  pid="$(_running_pid)"
  if [[ -n "$pid" ]]; then
    echo "[$(_ts)] running pid=$pid API=$API_BASE"
  else
    echo "[$(_ts)] not running"
  fi
  if [[ -f "$LOG_FILE" ]]; then
    echo "  log=$LOG_FILE"
    tail -n 5 "$LOG_FILE" || true
  fi
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|status|wait|test}

Quick start on training host:
  cd /mnt/z4/solariewang/verl-swe
  # defaults: VLLM_ENV=/opt/conda/envs/vllm CUDA=0 PORT=80 TP=1
  bash scripts/launch_vllm_qwen25_coder_7b.sh start
  bash scripts/launch_vllm_qwen25_coder_7b.sh wait
  bash scripts/launch_vllm_qwen25_coder_7b.sh test

Hook into existing API eval:
  # Eval from training host (or any host with igate to this node):
  #   API_BASE=http://29.163.228.59:80/v1
  #   python3 scripts/eval_api_baseline.py ...
EOF
}

cmd=${1:-start}
case "$cmd" in
  start) start_server ;;
  stop) stop_server ;;
  status) status_server ;;
  wait) wait_ready "${2:-60}" ;;
  test) smoke_test ;;
  *) usage; exit 1 ;;
esac
