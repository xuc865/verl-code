#!/usr/bin/env bash
# Eval GRPO CodeRL+ (SFT→RL global_step_50) with the SAME protocol as training:
#   apps_train_coderl recipe → ReAct + interactive + max_turns=8 + history=5 + thinking ON
#
# Defaults target the served ckpt:
#   API  http://127.0.0.1:8000/v1   (on train host; or http://29.163.228.59:8000/v1)
#   MODEL grpo_coderl_sft_mt8_step50
#
# Full suite: non-APPS queue first, then APPS (5000).
#
# Usage (on the train / API host):
#   cd /mnt/z4/solariewang/verl-swe
#   # ensure vLLM is up:
#   #   bash scripts/launch_vllm_grpo_sft_mt8_ckpt.sh status
#   nohup bash scripts/eval_api_grpo_coderl_sft_mt8_step50.sh \
#     >> logs/eval_api_grpo_coderl_sft_mt8_step50_suite.nohup.log 2>&1 &
#
# Env overrides:
#   API_BASE MODEL METHOD EVAL_WORKERS EVAL_BENCHMARKS
#   SKIP_API_CHECK=1 SKIP_APPS=1 SKIP_QUEUE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

export REPO_ROOT DATA_ROOT
export API_BASE=${API_BASE:-http://127.0.0.1:8000/v1}
export MODEL=${MODEL:-grpo_coderl_sft_mt8_step50}
export METHOD=${METHOD:-grpo_coderl_sft_mt8_step50_mt8_interactive}

# ---- match launch_grpo_coderl_sft_mt8.sh training protocol ----
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-interactive}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-8}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-5}
export MAX_TOKENS=${MAX_TOKENS:-4096}          # trainer data.max_response_length
export DISABLE_THINKING=${DISABLE_THINKING:-0} # SFT/RL think-on
export PROMPT_MODE=${PROMPT_MODE:-react}       # same ReAct XML as swebench prompts
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
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export WAIT_EVAL_PATTERN=${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py.*grpo_coderl_sft_mt8_step50}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_check_api() {
  if [ "${SKIP_API_CHECK:-0}" = "1" ]; then
    return 0
  fi
  local url="${API_BASE%/}/models"
  echo "[$(_ts)] checking API $url"
  if ! curl -fsS --max-time 15 "$url" >/dev/null; then
    echo "[$(_ts)] ERROR: API not reachable: $url" >&2
    echo "  On train host, start vLLM first:" >&2
    echo "    bash $REPO_ROOT/scripts/launch_vllm_grpo_sft_mt8_ckpt.sh start" >&2
    echo "    bash $REPO_ROOT/scripts/launch_vllm_grpo_sft_mt8_ckpt.sh wait" >&2
    exit 1
  fi
  echo "[$(_ts)] API ok — models:"
  curl -fsS --max-time 15 "$url" | head -c 800 || true
  echo
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

echo "[$(_ts)] ========== GRPO step50 eval (train-aligned) =========="
echo "  1) non-APPS queue  ($EVAL_BENCHMARKS)"
echo "  2) APPS            (last, 5000)"
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
_think_lbl=ON
[ "$DISABLE_THINKING" = "1" ] && _think_lbl=OFF
echo "  protocol=$TEST_FEEDBACK_MODE max_turns=$EVAL_MAX_TURNS history=$EVAL_HISTORY_LENGTH"
echo "  prompt=$PROMPT_MODE thinking=$_think_lbl max_tokens=$MAX_TOKENS workers=$EVAL_WORKERS"
_print_checkpoint_status "$METHOD"

if [ "${SKIP_QUEUE:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_QUEUE=1 — skipping non-APPS"
else
  echo "[$(_ts)] ========== stage 1/2: non-APPS queue =========="
  _run_fg "grpo50-queue" \
    "bash \"$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh\"" \
    "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log"
fi

if [ "${SKIP_APPS:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_APPS=1 — skipping APPS"
else
  echo "[$(_ts)] ========== stage 2/2: APPS (last) =========="
  _print_checkpoint_status "$METHOD"
  _run_fg "grpo50-apps" \
    "bash \"$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh\"" \
    "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"
fi

echo "[$(_ts)] ========== GRPO step50 eval DONE =========="
echo "  results: $LOG_DIR/eval_api_${METHOD}_*.json"
echo "  progress: bash $REPO_ROOT/scripts/inspect_eval_api_progress.sh"
