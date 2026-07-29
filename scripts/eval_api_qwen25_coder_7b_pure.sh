#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct — pure baseline: single-turn + exec + freeform.
#
# Protocol:
#   max_turns=1, test_feedback_mode=exec, disable_thinking
#   prompt_mode=freeform (no ReAct XML; extract ```python``` -> solution.py)
#   METHOD=qwen25_coder_7b_st1_freeform
#
# Usage (training host):
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_qwen25_coder_7b_pure.sh \
#     >> logs/eval_api_qwen25_coder_7b_st1_freeform_suite.nohup.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}

export REPO_ROOT DATA_ROOT
export API_BASE=${API_BASE:-http://29.163.228.59:80/v1}
export MODEL=${MODEL:-Qwen2.5-Coder-7B-Instruct}
export METHOD=${METHOD:-qwen25_coder_7b_st1_freeform}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-1}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-4096}
export DISABLE_THINKING=${DISABLE_THINKING:-1}
export PROMPT_MODE=${PROMPT_MODE:-freeform}
export ENCOURAGE_COT=${ENCOURAGE_COT:-0}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-10}
export EVAL_WORKERS=${EVAL_WORKERS:-4}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export SKIP_WAIT=${SKIP_WAIT:-1}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-apps,humaneval,mbpp,livecodebench,usaco,ojbench,icpc}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export WAIT_EVAL_PATTERN=${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py.*qwen25_coder_7b_st1_freeform}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p "$LOG_DIR"

echo "[$(_ts)] ========== Qwen2.5-Coder-7B pure (st1 freeform) =========="
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
echo "  protocol=exec max_turns=1 thinking=off prompt_mode=$PROMPT_MODE workers=$EVAL_WORKERS"
echo "  benches=$EVAL_BENCHMARKS"

exec bash "$SCRIPT_DIR/eval_api_kimi_queue.sh"
