#!/usr/bin/env bash
# Qwen3.5-4B SFT eval retry：假定 :8000 上已有 qwen35-4b-apps-mt8-sft。
# 只补 outcome=error（queue 非APPS -> APPS），不杀 / 不重启 vLLM。
#
# 训练机：
#   bash /mnt/z4/solariewang/verl-swe/scripts/run_qwen35_4b_sft_eval_retry.sh
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
LOG_DIR="$REPO_ROOT/logs"
PORT=${PORT:-8000}
API_BASE=${API_BASE:-http://127.0.0.1:${PORT}/v1}
MODEL=${MODEL:-qwen35-4b-apps-mt8-sft}
METHOD=${METHOD:-qwen35_4b_sft_mt12_exec}
EVAL_WORKERS=${EVAL_WORKERS:-6}

cd "$REPO_ROOT" || { echo "找不到仓库 $REPO_ROOT"; exit 1; }
mkdir -p "$LOG_DIR"

export REPO_ROOT DATA_ROOT LOG_DIR
export API_BASE MODEL METHOD
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export DISABLE_THINKING=${DISABLE_THINKING:-0}
export PROMPT_MODE=${PROMPT_MODE:-react}
export ENCOURAGE_COT=${ENCOURAGE_COT:-0}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=1
export EVAL_RETRY_ERRORS=1
export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export EVAL_WORKERS
export EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export WAIT_FOR_SINGLE_TURN=0
export SKIP_WAIT=1
export SKIP_PEER_WAIT=1
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*${METHOD}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

CHAIN_LOG="$LOG_DIR/eval_api_${METHOD}_retry_suite.nohup.log"

echo "== API check $API_BASE =="
curl -sS -m 15 "${API_BASE%/}/models" | head -c 500; echo
if ! curl -fsS -m 15 "${API_BASE%/}/models" | grep -q "$MODEL"; then
  echo "!! API 不通或模型不是 $MODEL"
  echo "   先起服务，或用全量脚本（会重启 vLLM）："
  echo "   bash $REPO_ROOT/scripts/eval_api_qwen35_4b_sft_on_grpo_slot.sh"
  exit 1
fi

# 把关键 env 写进子进程，避免被 shell 里残留的 base(:8001) 覆盖
nohup env \
  REPO_ROOT="$REPO_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  LOG_DIR="$LOG_DIR" \
  API_BASE="$API_BASE" \
  MODEL="$MODEL" \
  METHOD="$METHOD" \
  TEST_FEEDBACK_MODE="$TEST_FEEDBACK_MODE" \
  EVAL_MAX_TURNS="$EVAL_MAX_TURNS" \
  EVAL_HISTORY_LENGTH="$EVAL_HISTORY_LENGTH" \
  MAX_TOKENS="$MAX_TOKENS" \
  DISABLE_THINKING="$DISABLE_THINKING" \
  PROMPT_MODE="$PROMPT_MODE" \
  ENCOURAGE_COT="$ENCOURAGE_COT" \
  API_TIMEOUT="$API_TIMEOUT" \
  API_RETRIES="$API_RETRIES" \
  EVAL_RESUME=1 \
  EVAL_RETRY_ERRORS=1 \
  EVAL_RETRY_ERROR_TYPES="$EVAL_RETRY_ERROR_TYPES" \
  EVAL_CHECKPOINT_EVERY="$EVAL_CHECKPOINT_EVERY" \
  EVAL_WORKERS="$EVAL_WORKERS" \
  EVAL_INSTANCE_TIMEOUT="$EVAL_INSTANCE_TIMEOUT" \
  LCB_MIN_DATE="$LCB_MIN_DATE" \
  WAIT_FOR_SINGLE_TURN=0 \
  SKIP_WAIT=1 \
  SKIP_PEER_WAIT=1 \
  EVAL_BENCHMARKS="$EVAL_BENCHMARKS" \
  STATUS_JSON="$STATUS_JSON" \
  WAIT_EVAL_PATTERN="$WAIT_EVAL_PATTERN" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  bash -c '
  set -euo pipefail
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
  echo "[$(date "+%F %T")] ========== SFT retry suite start =========="
  echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
  echo "[$(date "+%F %T")] stage 1/2: non-APPS queue retry_errors=1"
  bash "$REPO_ROOT/scripts/eval_api_kimi_mt12_queue.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_queue_retry.nohup.log" 2>&1
  echo "[$(date "+%F %T")] stage 2/2: APPS retry_errors=1"
  bash "$REPO_ROOT/scripts/eval_api_kimi_apps_mt12.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_apps_retry.nohup.log" 2>&1
  echo "[$(date "+%F %T")] ========== SFT retry suite DONE =========="
' >>"$CHAIN_LOG" 2>&1 &

PID=$!
echo "PID=$PID  log=$CHAIN_LOG"
echo "  queue: logs/eval_api_${METHOD}_queue_retry.nohup.log"
echo "  apps:  logs/eval_api_${METHOD}_apps_retry.nohup.log"
echo "  tail -f $CHAIN_LOG"
