#!/usr/bin/env bash
# Re-run outcome=error rows for SFT eval (keeps won/lost).
# Run on the training host where vLLM listens on :80.
#
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/retry_sft_eval_errors.sh \
#     >> logs/eval_api_qwen25_coder7b_sft_mt12_exec_retry_errors.nohup.log 2>&1 &
set -euo pipefail
cd /mnt/z4/solariewang/verl-swe

export REPO_ROOT=/mnt/z4/solariewang/verl-swe
export DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
export API_BASE=${API_BASE:-http://127.0.0.1:80/v1}
export MODEL=${MODEL:-qwen25-coder7b-apps-mt8-sft}
export METHOD=${METHOD:-qwen25_coder7b_sft_mt12_exec}
export TEST_FEEDBACK_MODE=exec
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export DISABLE_THINKING=${DISABLE_THINKING:-0}
export EVAL_RESUME=1
export EVAL_RETRY_ERRORS=1
export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
export EVAL_WORKERS=${EVAL_WORKERS:-4}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export SKIP_WAIT=1
export SKIP_PEER_WAIT=1
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$REPO_ROOT/logs/eval_api_${METHOD}_retry_errors_status.json"
export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*${METHOD}"

mkdir -p "$REPO_ROOT/logs"
echo "[$(date '+%F %T')] retry errors METHOD=$METHOD benches=$EVAL_BENCHMARKS API=$API_BASE"
exec bash "$REPO_ROOT/scripts/eval_api_kimi_mt12_queue.sh"
