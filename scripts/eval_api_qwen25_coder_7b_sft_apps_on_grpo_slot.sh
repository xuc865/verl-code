#!/usr/bin/env bash
# Finish Qwen2.5-Coder-7B SFT leftover evals on the GRPO step50 vLLM slot
# (default: CUDA=1, PORT=8000).
#
# SFT suite died before APPS; also re-run / resume LeetCode. This script:
#   1) stop GRPO vLLM on :8000 (optional)
#   2) serve SFT global_step_242 on GPU1 / :8000
#   3) LeetCode then APPS  (METHOD=qwen25_coder7b_sft_mt12_exec, resume-friendly)
#
# Usage (train host):
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_qwen25_coder_7b_sft_apps_on_grpo_slot.sh \
#     >> logs/eval_api_qwen25_coder7b_sft_mt12_exec_apps_only.nohup.log 2>&1 &
#
# Skip vLLM restart (API already serving the SFT name on :8000):
#   SKIP_VLLM_RESTART=1 bash scripts/eval_api_qwen25_coder_7b_sft_apps_on_grpo_slot.sh
#
# Env overrides:
#   CUDA_VISIBLE_DEVICES  PORT  MODEL_PATH  SERVED_MODEL_NAME
#   API_BASE  MODEL  METHOD  EVAL_WORKERS  EVAL_INSTANCE_TIMEOUT
#   STOP_GRPO_VLLM=0   # do not call grpo vllm stop
#   SKIP_VLLM_RESTART=1  SKIP_LEETCODE=1  SKIP_APPS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# ---- take over GRPO eval slot ----
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PORT=${PORT:-8000}
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/checkpoints/apps_mt8_sft_qwen25_coder7b_think/global_step_242}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen25-coder7b-apps-mt8-sft}

export REPO_ROOT DATA_ROOT LOG_DIR
export API_BASE=${API_BASE:-http://127.0.0.1:${PORT}/v1}
export MODEL=${MODEL:-$SERVED_MODEL_NAME}
export METHOD=${METHOD:-qwen25_coder7b_sft_mt12_exec}

# Same protocol as eval_api_qwen25_coder_7b_sft.sh (mt12 · exec · thinking ON)
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
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}

LEETCODE_JSON="$LOG_DIR/eval_api_${METHOD}_leetcode.json"
LEETCODE_LOG="$LOG_DIR/eval_api_${METHOD}_leetcode_rerun.nohup.log"
APPS_JSON="$LOG_DIR/eval_api_${METHOD}_apps.json"
APPS_LOG="$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"
CHAIN_LOG="$LOG_DIR/eval_api_${METHOD}_apps_only.nohup.log"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_check_model() {
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[$(_ts)] ERROR: SFT ckpt missing: $MODEL_PATH/config.json" >&2
    exit 1
  fi
  if ! ls "$MODEL_PATH"/model*.safetensors >/dev/null 2>&1; then
    echo "[$(_ts)] ERROR: SFT ckpt has no weight shards under $MODEL_PATH" >&2
    exit 1
  fi
}

_stop_grpo_vllm() {
  if [[ "${STOP_GRPO_VLLM:-1}" != "1" ]]; then
    echo "[$(_ts)] STOP_GRPO_VLLM=0 — leave existing GRPO vLLM alone"
    return 0
  fi
  echo "[$(_ts)] stopping GRPO vLLM (if any) to free GPU${CUDA_VISIBLE_DEVICES}/:${PORT}"
  bash "$SCRIPT_DIR/launch_vllm_grpo_sft_mt8_ckpt.sh" stop 2>/dev/null || true
  # Also kill anything still bound to PORT (best-effort)
  if command -v ss >/dev/null 2>&1; then
    local pids
    pids="$(ss -lntp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
    for pid in $pids; do
      echo "[$(_ts)] killing pid=$pid still on :$PORT"
      kill "$pid" 2>/dev/null || true
    done
  fi
  sleep 3
}

_start_sft_vllm() {
  if [[ "${SKIP_VLLM_RESTART:-0}" == "1" ]]; then
    echo "[$(_ts)] SKIP_VLLM_RESTART=1 — not restarting vLLM"
    return 0
  fi
  echo "[$(_ts)] starting SFT vLLM"
  echo "  model=$MODEL_PATH"
  echo "  served=$SERVED_MODEL_NAME"
  echo "  cuda=$CUDA_VISIBLE_DEVICES port=$PORT"
  # stop any previous SFT serve with same name
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME" bash "$SCRIPT_DIR/launch_vllm_qwen25_coder_7b.sh" stop 2>/dev/null || true
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PORT="$PORT" \
    MODEL_PATH="$MODEL_PATH" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    bash "$SCRIPT_DIR/launch_vllm_qwen25_coder_7b.sh" start
  # 7B SFT load on this host has taken ~20+ min; default wait=60 (~5min) is too short.
  echo "[$(_ts)] waiting for API $API_BASE (up to ~40min) ..."
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PORT="$PORT" \
    MODEL_PATH="$MODEL_PATH" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    bash "$SCRIPT_DIR/launch_vllm_qwen25_coder_7b.sh" wait 480
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
      curl -fsS --max-time 10 "$url" | head -c 600 || true
      echo
      return 0
    fi
    sleep 10
  done
  echo "[$(_ts)] ERROR: API not reachable: $url" >&2
  exit 1
}

_print_bench_status() {
  local bench=$1
  local target=$2
  METHOD="$METHOD" LOG_DIR="$LOG_DIR" BENCH="$bench" TARGET="$target" python3 - <<'PY'
import json, os
from collections import Counter
method, log_dir = os.environ["METHOD"], os.environ["LOG_DIR"]
bench, target = os.environ["BENCH"], int(os.environ["TARGET"])
p = os.path.join(log_dir, f"eval_api_{method}_{bench}.json")
print(f"  {bench} json: {p}")
if not os.path.isfile(p):
    print(f"  status: missing -> will run (target {target})")
else:
    arr = (json.load(open(p)).get("per_instance") or [])
    oc = Counter(r.get("outcome") for r in arr)
    w, l, e = oc.get("won", 0), oc.get("lost", 0), oc.get("error", 0)
    flag = "DONE" if len(arr) >= target else "partial/resume"
    print(f"  status: {len(arr)}/{target}  won={w} lost={l} err={e}  ({flag})")
PY
}

# ---- entry (nohup wrapper) ----
if [[ "${FOREGROUND:-0}" != "1" && "${_INNER:-0}" != "1" ]]; then
  nohup env _INNER=1 FOREGROUND=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PORT="$PORT" \
    MODEL_PATH="$MODEL_PATH" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    API_BASE="$API_BASE" MODEL="$MODEL" METHOD="$METHOD" \
    SKIP_VLLM_RESTART="${SKIP_VLLM_RESTART:-0}" \
    STOP_GRPO_VLLM="${STOP_GRPO_VLLM:-1}" \
    bash "$0" >>"$CHAIN_LOG" 2>&1 &
  echo "PID=$!  log=$CHAIN_LOG"
  echo "  will serve $SERVED_MODEL_NAME on GPU $CUDA_VISIBLE_DEVICES :$PORT"
  echo "  then: LeetCode -> APPS"
  echo "  leetcode json -> $LEETCODE_JSON"
  echo "  apps json     -> $APPS_JSON"
  echo "  tail -f $CHAIN_LOG"
  exit 0
fi

_check_model
_stop_grpo_vllm
_start_sft_vllm
_check_api

echo "[$(_ts)] ========== Qwen2.5-Coder SFT · LeetCode + APPS =========="
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
echo "  protocol=$TEST_FEEDBACK_MODE max_turns=$EVAL_MAX_TURNS thinking=$([ "$DISABLE_THINKING" = 1 ] && echo OFF || echo ON)"
echo "  workers=$EVAL_WORKERS resume=$EVAL_RESUME"
_print_bench_status leetcode 228
_print_bench_status apps 5000

rc=0

# ----- 1) LeetCode first -----
if [[ "${SKIP_LEETCODE:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_LEETCODE=1 — skipping LeetCode"
else
  echo "[$(_ts)] starting LeetCode -> $LEETCODE_LOG"
  set +e
  EVAL_BENCHMARKS=leetcode \
    STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_leetcode_queue_status.json" \
    bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" >>"$LEETCODE_LOG" 2>&1
  lc_rc=$?
  set -e
  echo "[$(_ts)] LeetCode exit=$lc_rc"
  _print_bench_status leetcode 228
  if [[ "$lc_rc" -ne 0 ]]; then
    rc=$lc_rc
    echo "[$(_ts)] WARNING: LeetCode non-zero; continuing to APPS" >&2
  fi
fi

# ----- 2) APPS last -----
if [[ "${SKIP_APPS:-0}" == "1" ]]; then
  echo "[$(_ts)] SKIP_APPS=1 — skipping APPS"
else
  echo "[$(_ts)] starting APPS -> $APPS_LOG"
  set +e
  bash "$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh" >>"$APPS_LOG" 2>&1
  apps_rc=$?
  set -e
  echo "[$(_ts)] APPS exit=$apps_rc"
  _print_bench_status apps 5000
  if [[ "$apps_rc" -ne 0 ]]; then
    rc=$apps_rc
  fi
fi

echo "[$(_ts)] done."
echo "  leetcode: $LEETCODE_JSON"
echo "  apps:     $APPS_JSON"
exit "$rc"
