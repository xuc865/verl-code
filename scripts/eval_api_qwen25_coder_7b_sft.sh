#!/usr/bin/env bash
# Eval SFT'd Qwen2.5-Coder-7B (apps_mt8 think) via local vLLM.
#
# Order (intentional):
#   1) non-APPS queue  — humaneval, mbpp, livecodebench, usaco, ojbench, icpc, leetcode
#   2) APPS            — last (longest)
#
# Protocol: mt12 + exec + thinking ON (aligned with think-injected SFT).
#
# Usage (training host, after vLLM is up on GPU1 / port 80):
#   cd /mnt/z4/solariewang/verl-swe
#   # vLLM:
#   CUDA_VISIBLE_DEVICES=1 \
#     MODEL_PATH=.../global_step_242 \
#     SERVED_MODEL_NAME=qwen25-coder7b-apps-mt8-sft \
#     PORT=80 \
#     bash scripts/launch_vllm_qwen25_coder_7b.sh start
#
#   nohup bash scripts/eval_api_qwen25_coder_7b_sft.sh \
#     >> logs/eval_api_qwen25_coder7b_sft_mt12_exec_suite.nohup.log 2>&1 &
#
# Env overrides:
#   API_BASE  MODEL  METHOD  EVAL_WORKERS  EVAL_BENCHMARKS
#   SKIP_API_CHECK=1  SKIP_APPS=1  SKIP_QUEUE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

export REPO_ROOT DATA_ROOT
export API_BASE=${API_BASE:-http://127.0.0.1:80/v1}
export MODEL=${MODEL:-qwen25-coder7b-apps-mt8-sft}
export METHOD=${METHOD:-qwen25_coder7b_sft_mt12_exec}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export DISABLE_THINKING=${DISABLE_THINKING:-0}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_RETRY_ERRORS=${EVAL_RETRY_ERRORS:-1}
export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export EVAL_WORKERS=${EVAL_WORKERS:-4}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export WAIT_EVAL_PATTERN=${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py.*qwen25_coder7b_sft_mt12_exec}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_check_api() {
  if [ "${SKIP_API_CHECK:-0}" = "1" ]; then
    return 0
  fi
  local url="${API_BASE%/}/models"
  echo "[$(_ts)] checking API $url"
  if ! curl -fsS --max-time 10 "$url" >/dev/null; then
    echo "[$(_ts)] ERROR: API not reachable: $url" >&2
    echo "  Start SFT vLLM first, e.g.:" >&2
    echo "  CUDA_VISIBLE_DEVICES=1 PORT=80 \\" >&2
    echo "    MODEL_PATH=$REPO_ROOT/checkpoints/apps_mt8_sft_qwen25_coder7b_think/global_step_242 \\" >&2
    echo "    SERVED_MODEL_NAME=qwen25-coder7b-apps-mt8-sft \\" >&2
    echo "    bash scripts/launch_vllm_qwen25_coder_7b.sh start" >&2
    exit 1
  fi
  echo "[$(_ts)] API ok"
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
  if [ "$rc" -ne 0 ]; then
    echo "[$(_ts)] WARNING: [$tag] non-zero exit=$rc; continuing" >&2
  fi
  return 0
}

_check_api
export EVAL_RESUME=${EVAL_RESUME:-1}

echo "[$(_ts)] ========== SFT eval plan =========="
echo "  1) non-APPS queue  ($EVAL_BENCHMARKS)"
echo "  2) APPS            (last)"
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
_think_lbl=ON
[ "$DISABLE_THINKING" = "1" ] && _think_lbl=OFF
echo "  protocol=exec max_turns=$EVAL_MAX_TURNS thinking=$_think_lbl"
echo "  workers=$EVAL_WORKERS EVAL_RESUME=$EVAL_RESUME"
_print_checkpoint_status "$METHOD"

# ----- 1) non-APPS first -----
if [ "${SKIP_QUEUE:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_QUEUE=1 — skipping non-APPS"
else
  echo "[$(_ts)] ========== stage 1/2: non-APPS queue =========="
  _run_fg "sft-queue" \
    "bash \"$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh\"" \
    "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log"
fi

# ----- 2) APPS last -----
if [ "${SKIP_APPS:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_APPS=1 — skipping APPS"
else
  echo "[$(_ts)] ========== stage 2/2: APPS (last) =========="
  _print_checkpoint_status "$METHOD"
  _run_fg "sft-apps" \
    "bash \"$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh\"" \
    "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"
fi

echo "[$(_ts)] ========== SFT eval DONE (non-APPS then APPS) =========="
