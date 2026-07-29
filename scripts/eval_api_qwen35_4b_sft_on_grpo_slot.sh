#!/usr/bin/env bash
# Take over the qwen25-coder SFT eval slot (GPU1 / :8000) for Qwen3.5-4B SFT.
#
# Steps:
#   1) stop whatever is on :8000 (qwen25 SFT / old GRPO serve)
#   2) serve apps_mt8_sft_qwen35_4b_think/global_step_162 on GPU1 :8000
#      (language_model_only=1)
#   3) full suite like qwen25 SFT: non-APPS queue -> APPS
#      protocol: mt12 · exec · thinking ON · react
#
# Usage (train host):
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_qwen35_4b_sft_on_grpo_slot.sh \
#     >> logs/eval_api_qwen35_4b_sft_mt12_exec_suite.nohup.log 2>&1 &
#
# Env:
#   CUDA_VISIBLE_DEVICES  PORT  MODEL_PATH  SERVED_MODEL_NAME
#   SKIP_VLLM_RESTART=1  SKIP_QUEUE=1  SKIP_APPS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
ROOT=${ROOT:-/mnt/z4/solariewang}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# ---- take over qwen25 SFT / GRPO eval slot ----
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PORT=${PORT:-8000}
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/checkpoints/apps_mt8_sft_qwen35_4b_think/global_step_162}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen35-4b-apps-mt8-sft}
SFT_BASE_MODEL=${SFT_BASE_MODEL:-$ROOT/models/Qwen3.5-4B}

export REPO_ROOT DATA_ROOT LOG_DIR
export API_BASE=${API_BASE:-http://127.0.0.1:${PORT}/v1}
export MODEL=${MODEL:-$SERVED_MODEL_NAME}
export METHOD=${METHOD:-qwen35_4b_sft_mt12_exec}

# Same protocol as qwen25 SFT suite (mt12 · exec · thinking ON)
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export DISABLE_THINKING=${DISABLE_THINKING:-0}
export PROMPT_MODE=${PROMPT_MODE:-react}
export ENCOURAGE_COT=${ENCOURAGE_COT:-0}

export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_RETRY_ERRORS=${EVAL_RETRY_ERRORS:-1}
export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export EVAL_WORKERS=${EVAL_WORKERS:-4}
export EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

CHAIN_LOG=${CHAIN_LOG:-$LOG_DIR/eval_api_${METHOD}_suite.nohup.log}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_ensure_sidecars() {
  local ckpt="$1"
  local base="$SFT_BASE_MODEL"
  local f
  for f in preprocessor_config.json video_preprocessor_config.json generation_config.json merges.txt vocab.json; do
    if [[ -f "$base/$f" && ! -f "$ckpt/$f" ]]; then
      cp -f "$base/$f" "$ckpt/$f"
      echo "[$(_ts)] copied sidecar $f -> $ckpt"
    fi
  done
}

_check_model() {
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[$(_ts)] ERROR: SFT ckpt missing: $MODEL_PATH/config.json" >&2
    exit 1
  fi
  if ! grep -q 'Qwen3_5\|qwen3_5\|Qwen3.5' "$MODEL_PATH/config.json" 2>/dev/null; then
    echo "[$(_ts)] ERROR: ckpt does not look like Qwen3.5: $MODEL_PATH" >&2
    grep -E 'architectures|model_type' "$MODEL_PATH/config.json" >&2 || true
    exit 1
  fi
  local _w=()
  shopt -s nullglob
  _w=("$MODEL_PATH"/model*.safetensors "$MODEL_PATH"/pytorch_model*.bin)
  shopt -u nullglob
  if (( ${#_w[@]} == 0 )); then
    echo "[$(_ts)] ERROR: SFT ckpt has no weight shards under $MODEL_PATH" >&2
    exit 1
  fi
  _ensure_sidecars "$MODEL_PATH"
}

_kill_port() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    local pids
    pids="$(ss -lntp 2>/dev/null | awk -v p=":$p" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
    for pid in $pids; do
      echo "[$(_ts)] killing pid=$pid on :$p"
      kill "$pid" 2>/dev/null || true
    done
  fi
  fuser -k "${p}/tcp" 2>/dev/null || true
  sleep 2
  fuser -k -9 "${p}/tcp" 2>/dev/null || true
}

_stop_old_slot_vllm() {
  echo "[$(_ts)] freeing GPU${CUDA_VISIBLE_DEVICES}/:${PORT} (stop qwen25 SFT / prior serves)"
  # previous occupant: qwen25 coder SFT
  SERVED_MODEL_NAME=qwen25-coder7b-apps-mt8-sft \
    PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_qwen25_coder_7b.sh" stop 2>/dev/null || true
  # any prior qwen35 serve with same / different name on this port
  QWEN35_SERVED_NAME="$SERVED_MODEL_NAME" PORT="$PORT" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" stop 2>/dev/null || true
  bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" stop 2>/dev/null || true
  _kill_port "$PORT"
  sleep 3
}

_start_sft_vllm() {
  if [[ "${SKIP_VLLM_RESTART:-0}" == "1" ]]; then
    echo "[$(_ts)] SKIP_VLLM_RESTART=1 — not restarting vLLM"
    return 0
  fi
  echo "[$(_ts)] starting Qwen3.5-4B SFT vLLM"
  echo "  model=$MODEL_PATH"
  echo "  served=$SERVED_MODEL_NAME"
  echo "  cuda=$CUDA_VISIBLE_DEVICES port=$PORT"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PORT="$PORT" \
    QWEN35_MODEL_PATH="$MODEL_PATH" \
    QWEN35_SERVED_NAME="$SERVED_MODEL_NAME" \
    LANGUAGE_MODEL_ONLY=1 \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" start
  # 4B load can still take a while on ceph; wait up to ~40min
  echo "[$(_ts)] waiting for API $API_BASE (up to ~40min) ..."
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PORT="$PORT" \
    QWEN35_MODEL_PATH="$MODEL_PATH" \
    QWEN35_SERVED_NAME="$SERVED_MODEL_NAME" \
    bash "$SCRIPT_DIR/launch_vllm_qwen35_4b.sh" wait 480
}

_check_api() {
  if [[ "${SKIP_API_CHECK:-0}" == "1" ]]; then
    return 0
  fi
  local url="${API_BASE%/}/models"
  echo "[$(_ts)] checking API $url"
  local i
  for i in $(seq 1 60); do
    if curl -fsS --max-time 10 "$url" >/dev/null 2>&1; then
      echo "[$(_ts)] API ok"
      curl -fsS --max-time 10 "$url" | head -c 800 || true
      echo
      # verify served id
      if ! curl -fsS --max-time 10 "$url" | grep -q "$SERVED_MODEL_NAME"; then
        echo "[$(_ts)] ERROR: API up but model id != $SERVED_MODEL_NAME" >&2
        exit 1
      fi
      return 0
    fi
    sleep 10
  done
  echo "[$(_ts)] ERROR: API not reachable: $url" >&2
  exit 1
}

_print_checkpoint_status() {
  local method=$1
  echo "[$(_ts)] checkpoints for METHOD=$method (EVAL_RESUME=$EVAL_RESUME):"
  METHOD="$method" LOG_DIR="$LOG_DIR" python3 - <<'PY'
import json, os
from collections import Counter
method = os.environ["METHOD"]
log_dir = os.environ["LOG_DIR"]
targets = {
    "apps": 5000, "humaneval": 164, "mbpp": 257, "livecodebench": 131,
    "usaco": 307, "ojbench": 159, "icpc": 106, "leetcode": 228,
}
for b, t in targets.items():
    p = os.path.join(log_dir, f"eval_api_{method}_{b}.json")
    if not os.path.isfile(p):
        print(f"  {b:13} missing  -> will run")
        continue
    arr = (json.load(open(p)).get("per_instance") or [])
    oc = Counter(r.get("outcome") for r in arr)
    w, l, e = oc.get("won", 0), oc.get("lost", 0), oc.get("error", 0)
    if len(arr) >= t:
        print(f"  {b:13} {len(arr):4d}/{t} DONE  won={w} lost={l} err={e}  -> SKIP")
    else:
        print(f"  {b:13} {len(arr):4d}/{t} partial won={w} lost={l} err={e}  -> resume")
PY
}

_run_fg() {
  local tag=$1
  local cmd=$2
  local log=$3
  echo "[$(_ts)] [$tag] start -> $log"
  set +e
  bash -c "$cmd" >>"$log" 2>&1
  local rc=$?
  set -e
  echo "[$(_ts)] [$tag] exit=$rc"
  if [[ "$rc" -ne 0 ]]; then
    echo "[$(_ts)] WARNING: [$tag] non-zero exit=$rc; continuing" >&2
  fi
  return 0
}

# ---- entry (nohup wrapper) ----
if [[ "${FOREGROUND:-0}" != "1" && "${_INNER:-0}" != "1" ]]; then
  nohup env _INNER=1 FOREGROUND=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PORT="$PORT" \
    MODEL_PATH="$MODEL_PATH" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    API_BASE="$API_BASE" MODEL="$MODEL" METHOD="$METHOD" \
    SKIP_VLLM_RESTART="${SKIP_VLLM_RESTART:-0}" \
    SKIP_QUEUE="${SKIP_QUEUE:-0}" SKIP_APPS="${SKIP_APPS:-0}" \
    bash "$0" >>"$CHAIN_LOG" 2>&1 &
  echo "PID=$!  log=$CHAIN_LOG"
  echo "  will serve $SERVED_MODEL_NAME on GPU $CUDA_VISIBLE_DEVICES :$PORT"
  echo "  then: non-APPS queue -> APPS  (METHOD=$METHOD)"
  echo "  tail -f $CHAIN_LOG"
  exit 0
fi

_check_model
if [[ "${SKIP_VLLM_RESTART:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_VLLM_RESTART=1 — keep existing vLLM on :$PORT"
else
  _stop_old_slot_vllm
  _start_sft_vllm
fi
_check_api

_think_lbl=ON
[[ "$DISABLE_THINKING" = "1" ]] && _think_lbl=OFF
echo "[$(_ts)] ========== Qwen3.5-4B SFT · full suite =========="
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
echo "  ckpt=$MODEL_PATH"
echo "  protocol=exec max_turns=$EVAL_MAX_TURNS thinking=$_think_lbl prompt=$PROMPT_MODE"
echo "  workers=$EVAL_WORKERS resume=$EVAL_RESUME"
_print_checkpoint_status "$METHOD"

if [[ "${SKIP_QUEUE:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_QUEUE=1 — skipping non-APPS"
else
  echo "[$(_ts)] ========== stage 1/2: non-APPS queue =========="
  _run_fg "qwen35-sft-queue" \
    "bash \"$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh\"" \
    "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log"
fi

if [[ "${SKIP_APPS:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_APPS=1 — skipping APPS"
else
  echo "[$(_ts)] ========== stage 2/2: APPS (last) =========="
  _print_checkpoint_status "$METHOD"
  _run_fg "qwen35-sft-apps" \
    "bash \"$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh\"" \
    "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"
fi

echo "[$(_ts)] ========== Qwen3.5-4B SFT eval DONE =========="
_print_checkpoint_status "$METHOD"
