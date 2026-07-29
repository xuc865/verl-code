#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct — CodeAct baseline: mt12_exec (same as Kimi/GLM).
#
# Order:
#   1) non-APPS queue
#   2) APPS (last)
#
# Protocol:
#   max_turns=12, test_feedback_mode=exec, disable_thinking
#   METHOD=qwen25_coder_7b_mt12_exec
#
# Usage (training host):
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_qwen25_coder_7b_codeact.sh \
#     >> logs/eval_api_qwen25_coder_7b_mt12_exec_suite.nohup.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}

export REPO_ROOT DATA_ROOT
export API_BASE=${API_BASE:-http://29.163.228.59:80/v1}
export MODEL=${MODEL:-Qwen2.5-Coder-7B-Instruct}
export METHOD=${METHOD:-qwen25_coder_7b_mt12_exec}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export DISABLE_THINKING=${DISABLE_THINKING:-1}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export EVAL_WORKERS=${EVAL_WORKERS:-4}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export WAIT_EVAL_PATTERN=${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py.*qwen25_coder_7b_mt12_exec}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[$(_ts)] ========== Qwen2.5-Coder-7B CodeAct (mt12 exec) =========="
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
echo "  protocol=exec max_turns=12 thinking=off workers=$EVAL_WORKERS"
echo "  order: non-APPS queue -> APPS (last)"

# 1) non-APPS first (foreground so APPS starts only after queue finishes)
echo "[$(_ts)] stage 1/2: non-APPS queue ($EVAL_BENCHMARKS)"
bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" \
  >> "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log" 2>&1
echo "[$(_ts)] stage 1/2 done"

# 2) APPS last
if [ "${SKIP_APPS:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_APPS=1 — done after non-APPS"
  exit 0
fi
echo "[$(_ts)] stage 2/2: APPS (last)"
bash "$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh" \
  >> "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log" 2>&1
echo "[$(_ts)] ========== CodeAct DONE (non-APPS then APPS) =========="
