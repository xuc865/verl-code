#!/usr/bin/env bash
# vLLM OpenAI API for Qwen3.5-4B (text / coding eval via --language-model-only).
#
# Single H20 (96GB) is enough: ~8GB weights (bf16) + KV for 32k context.
#
# Usage (train host):
#   cd /mnt/z4/solariewang/verl-swe
#   bash scripts/launch_vllm_qwen35_4b.sh start
#   bash scripts/launch_vllm_qwen35_4b.sh wait
#   bash scripts/launch_vllm_qwen35_4b.sh test
#
# Stop other vLLM on :80 first if needed:
#   fuser -k 80/tcp
#
# Env (script-specific — avoids inheriting generic MODEL_PATH from other jobs):
#   QWEN35_MODEL_PATH  QWEN35_SERVED_NAME  HOST  PORT  CUDA_VISIBLE_DEVICES
#   TP=1  MAX_MODEL_LEN  GPU_MEM_UTIL  VLLM_ENV=/opt/conda/envs/vllm
#   LANGUAGE_MODEL_ONLY=1   REASONING_PARSER=qwen3

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}

MODEL_PATH=${QWEN35_MODEL_PATH:-/mnt/z4/solariewang/models/Qwen3.5-4B}
SERVED_MODEL_NAME=${QWEN35_SERVED_NAME:-qwen3.5-4b}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-80}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
VLLM_ENV=${VLLM_ENV:-/opt/conda/envs/vllm}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-1}
REASONING_PARSER=${REASONING_PARSER:-qwen3}

PID_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME//\//_}.pid"
LOG_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME//\//_}.nohup.log"
API_BASE="http://127.0.0.1:${PORT}/v1"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_activate_vllm() {
  if [[ -x "$VLLM_ENV/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$VLLM_ENV")/../etc/profile.d/conda.sh" 2>/dev/null || \
      source /opt/conda/etc/profile.d/conda.sh
    conda activate "$(basename "$VLLM_ENV")" 2>/dev/null || conda activate "$VLLM_ENV"
    export PATH="$VLLM_ENV/bin:$PATH"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: vLLM env not found: $VLLM_ENV" >&2
  echo "  Qwen3.5 needs recent vLLM main/recipe — see models/Qwen3.5-4B/README.md" >&2
  exit 1
}

_check_model() {
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing $MODEL_PATH/config.json" >&2
    exit 1
  fi
}

_port_in_use() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -lntu 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${p}$"
  else
    python3 - "$p" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    raise SystemExit(0)
finally:
    s.close()
raise SystemExit(1)
PY
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

  if _port_in_use "$PORT"; then
    echo "ERROR: port $PORT already in use (stop other vLLM first)" >&2
    echo "  fuser -k ${PORT}/tcp" >&2
    exit 1
  fi

  echo "[$(_ts)] starting Qwen3.5-4B vLLM"
  echo "  model_path=$MODEL_PATH"
  echo "  served_name=$SERVED_MODEL_NAME"
  echo "  api=$API_BASE  cuda=$CUDA_VISIBLE_DEVICES  tp=$TP"
  echo "  max_len=$MAX_MODEL_LEN  gpu_util=$GPU_MEM_UTIL"
  echo "  language_model_only=$LANGUAGE_MODEL_ONLY  reasoning_parser=$REASONING_PARSER"
  echo "  log=$LOG_FILE"

  local extra_args=()
  if [[ "$LANGUAGE_MODEL_ONLY" = "1" ]]; then
    extra_args+=(--language-model-only)
  fi
  if [[ -n "$REASONING_PARSER" ]]; then
    extra_args+=(--reasoning-parser "$REASONING_PARSER")
  fi

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
    "${extra_args[@]}" \
    >> "$LOG_FILE" 2>&1 &

  echo $! > "$PID_FILE"
  echo "[$(_ts)] started pid=$(cat "$PID_FILE")"
  echo "  tail -f $LOG_FILE"
}

stop_server() {
  local pid
  pid="$(_running_pid)"
  if [[ -z "$pid" ]]; then
    if _port_in_use "$PORT"; then
      echo "[$(_ts)] no pid file but port $PORT in use — killing via fuser"
      fuser -k "${PORT}/tcp" 2>/dev/null || true
      sleep 2
      fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
    else
      echo "[$(_ts)] not running"
    fi
    rm -f "$PID_FILE"
    return 0
  fi
  echo "[$(_ts)] stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  if _port_in_use "$PORT"; then
    fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[$(_ts)] stopped"
}

_verify_served_model() {
  local want="$SERVED_MODEL_NAME"
  local got
  got="$(curl -fsS "$API_BASE/models" | python3 -c '
import json, sys
want = sys.argv[1]
data = json.load(sys.stdin)
ids = [m.get("id", "") for m in data.get("data", [])]
if want not in ids:
    sys.exit(1)
print(want)
' "$want")" || {
    echo "ERROR: $API_BASE is up but not serving model '$want'" >&2
    echo "  loaded: $(curl -fsS "$API_BASE/models" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print([m.get('id') for m in d.get('data',[])])" 2>/dev/null || echo '?')" >&2
    echo "  stop the other vLLM on port $PORT, then: bash $0 start && bash $0 wait" >&2
    return 1
  }
  echo "[$(_ts)] verified model id=$got"
}

wait_ready() {
  local tries=${1:-90}
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$API_BASE/models" >/dev/null 2>&1; then
      if _verify_served_model; then
        echo "[$(_ts)] API ready: $API_BASE"
        return 0
      fi
      return 1
    fi
    sleep 5
  done
  echo "ERROR: API not ready after $((tries * 5))s — check $LOG_FILE" >&2
  [[ -f "$LOG_FILE" ]] && tail -n 30 "$LOG_FILE" >&2 || true
  return 1
}

smoke_test() {
  wait_ready 90 || exit 1
  echo "[$(_ts)] === /v1/models ==="
  curl -fsS "$API_BASE/models" | python3 -m json.tool | head -30
  echo
  echo "[$(_ts)] === chat smoke (expect model=${SERVED_MODEL_NAME}) ==="
  curl -fsS "$API_BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "${SERVED_MODEL_NAME}",
  "messages": [{"role": "user", "content": "Write Python: def add(a,b): return a+b"}],
  "max_tokens": 128,
  "temperature": 0.0,
}))
PY
)" | python3 -c '
import json, sys
want = sys.argv[1]
r = json.load(sys.stdin)
got = r.get("model", "")
if got != want:
    print(f"ERROR: response model={got!r} expected {want!r}", file=sys.stderr)
    print("  port is still serving another vLLM — stop it first.", file=sys.stderr)
    sys.exit(1)
msg = r["choices"][0]["message"]
text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
print(json.dumps({"model": got, "preview": text[:400]}, indent=2, ensure_ascii=False))
' "${SERVED_MODEL_NAME}"
}

status_server() {
  local pid
  pid="$(_running_pid)"
  if [[ -n "$pid" ]]; then
    echo "[$(_ts)] running pid=$pid API=$API_BASE cuda=$CUDA_VISIBLE_DEVICES"
  else
    echo "[$(_ts)] not running"
  fi
  [[ -f "$LOG_FILE" ]] && tail -n 8 "$LOG_FILE" || true
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|status|wait|test}

Quick start (GPU0, port 80):
  cd /mnt/z4/solariewang/verl-swe
  bash scripts/launch_vllm_qwen35_4b.sh start
  bash scripts/launch_vllm_qwen35_4b.sh wait
  bash scripts/launch_vllm_qwen35_4b.sh test

Hook eval (example — same protocol as GRPO step50):
  API_BASE=http://127.0.0.1:80/v1 MODEL=qwen3.5-4b \\
    METHOD=qwen35_4b_mt8_interactive \\
    bash scripts/eval_api_grpo_coderl_sft_mt8_step50_after_apps.sh

Notes:
  - Uses QWEN35_MODEL_PATH / QWEN35_SERVED_NAME (not generic MODEL_PATH).
  - 1× H20 96GB is sufficient for 4B @ max-model-len=${MAX_MODEL_LEN}.
  - Port 80 conflicts with other vLLM; stop them first.
EOF
}

cmd=${1:-start}
case "$cmd" in
  start) start_server ;;
  stop) stop_server ;;
  status) status_server ;;
  wait) wait_ready "${2:-90}" ;;
  test) smoke_test ;;
  *) usage; exit 1 ;;
esac
