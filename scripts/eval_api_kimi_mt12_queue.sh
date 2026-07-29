#!/usr/bin/env bash
# Kimi API eval queue — multi-turn ReAct (mt12 exec), coding suite except APPS.
#
# APPS mt12 is launched separately (eval_api_kimi_apps_mt12.sh).
# This queue runs: humaneval → mbpp → livecodebench → usaco → ojbench → icpc
#
# Usage:
#   nohup bash scripts/eval_api_kimi_mt12_queue.sh >> logs/eval_api_kimi_mt12_queue.nohup.log 2>&1 &
#
# Env:
#   SKIP_WAIT=1          default: do not block on APPS or other in-flight eval
#   EVAL_WORKERS=6       per-benchmark concurrency (APPS mt12 may run in parallel)
#   EVAL_BENCHMARKS=...  override benchmark list
#   bash scripts/inspect_eval_api_progress.sh logs/eval_api_kimi_2_6_mt12_blind_humaneval.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/apdcephfs/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"

export REPO_ROOT
export DATA_ROOT=${DATA_ROOT:-/apdcephfs/z4/solariewang/datasets}
export API_BASE=${API_BASE:-http://29.163.228.8:8080/v1}
export MODEL=${MODEL:-Kimi-K2.6}
export METHOD=${METHOD:-kimi_2_6_mt12_exec}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export DISABLE_THINKING=${DISABLE_THINKING:-1}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export EVAL_WORKERS=${EVAL_WORKERS:-6}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export SKIP_WAIT=${SKIP_WAIT:-1}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"

exec bash "$SCRIPT_DIR/eval_api_kimi_queue.sh"
