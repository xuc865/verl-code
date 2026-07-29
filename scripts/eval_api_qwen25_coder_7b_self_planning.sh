#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct — Self-Planning baseline.
#
# Protocol:
#   1) Ask the model for a numbered intent plan (no code).
#   2) Ask the model to implement solution.py following that plan.
#   prompt_mode=self_planning, test_feedback_mode=exec, disable_thinking
#   Default max_turns=1 (one plan + one code attempt), like pure.
#   Set EVAL_MAX_TURNS=12 to allow plan-guided repair after failed tests.
#
#   METHOD=qwen25_coder_7b_st1_self_planning
#
# Usage (training / eval host with Qwen API up):
#   cd /mnt/z4/solariewang/verl-swe   # or /apdcephfs/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_qwen25_coder_7b_self_planning.sh \
#     >> logs/eval_api_qwen25_coder_7b_st1_self_planning_suite.nohup.log 2>&1 &
#
# Env overrides:
#   EVAL_BENCHMARKS=humaneval,mbpp,...
#   SKIP_APPS=1
#   EVAL_MAX_TURNS=12 METHOD=qwen25_coder_7b_mt12_self_planning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}

export REPO_ROOT DATA_ROOT
export API_BASE=${API_BASE:-http://29.163.228.59:80/v1}
export MODEL=${MODEL:-Qwen2.5-Coder-7B-Instruct}
export METHOD=${METHOD:-qwen25_coder_7b_st1_self_planning}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-1}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-4096}
export DISABLE_THINKING=${DISABLE_THINKING:-1}
export PROMPT_MODE=${PROMPT_MODE:-self_planning}
export ENCOURAGE_COT=${ENCOURAGE_COT:-0}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-10}
export EVAL_WORKERS=${EVAL_WORKERS:-4}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
# Prefer non-APPS first; APPS last (same as cot/codeact).
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export WAIT_EVAL_PATTERN=${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py.*qwen25_coder_7b_.*self_planning}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[$(_ts)] ========== Qwen2.5-Coder-7B Self-Planning =========="
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
echo "  protocol=exec max_turns=$EVAL_MAX_TURNS thinking=off prompt_mode=$PROMPT_MODE"
echo "  flow: plan(no code) -> code(from plan)  workers=$EVAL_WORKERS"
echo "  order: non-APPS queue -> APPS (last)"

echo "[$(_ts)] stage 1/2: non-APPS queue ($EVAL_BENCHMARKS)"
if [ "$EVAL_MAX_TURNS" -le 1 ]; then
  bash "$SCRIPT_DIR/eval_api_kimi_queue.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log" 2>&1
else
  bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log" 2>&1
fi
echo "[$(_ts)] stage 1/2 done"

if [ "${SKIP_APPS:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_APPS=1 — done after non-APPS"
  exit 0
fi

echo "[$(_ts)] stage 2/2: APPS (last)"
if [ "$EVAL_MAX_TURNS" -le 1 ]; then
  # single-turn APPS via queue-style one-bench run
  export EVAL_BENCHMARKS=apps
  bash "$SCRIPT_DIR/eval_api_kimi_queue.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log" 2>&1
else
  bash "$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log" 2>&1
fi
echo "[$(_ts)] ========== Self-Planning DONE (non-APPS then APPS) =========="
