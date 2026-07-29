#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct — chain:
#   1) CoT APPS --retry-errors
#   2) CodeAct APPS --retry-errors
#   3) CodeAct non-APPS --retry-errors (remaining outcome=error rows)
#   4) Self-planning full suite (non-APPS then APPS)
#
# Usage (dev / training host):
#   cd /mnt/z4/solariewang/verl-swe   # or apdcephfs path
#   nohup bash scripts/eval_api_qwen25_coder_7b_error_retry_then_self_planning.sh \
#     >> logs/eval_api_qwen25_coder_7b_error_retry_then_self_planning.nohup.log 2>&1 &
#
# Env:
#   API_BASE=http://127.0.0.1:80/v1          # training host local vLLM
#   MODEL=Qwen2.5-Coder-7B-Instruct
#   SKIP_CODEACT_NON_APPS=1
#   SKIP_SELF_PLANNING=1
#   SKIP_COT_APPS=1 / SKIP_CODEACT_APPS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}

export REPO_ROOT DATA_ROOT
# Default to local vLLM on the training machine. Override explicitly if needed.
export API_BASE=${API_BASE:-http://127.0.0.1:80/v1}
export MODEL=${MODEL:-Qwen2.5-Coder-7B-Instruct}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_RETRY_ERRORS=${EVAL_RETRY_ERRORS:-1}
export EVAL_WORKERS=${EVAL_WORKERS:-8}
# 方案2: cot/codeact 多轮 APPS 大量打满 12 轮仍失败，砍到 6 轮提速（可用 CHAIN_MAX_TURNS 覆盖）
CHAIN_MAX_TURNS=${CHAIN_MAX_TURNS:-6}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export SKIP_WAIT=${SKIP_WAIT:-1}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export MAX_TOKENS=${MAX_TOKENS:-8192}
# Local vLLM must not go through corporate http_proxy.
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
_log() { echo "[$(_ts)] $*" | tee -a "$CHAIN_LOG"; }

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
CHAIN_LOG="$LOG_DIR/eval_api_qwen25_coder_7b_error_retry_then_self_planning.nohup.log"

_log "========== Qwen2.5-Coder-7B chain start =========="
_log "API=$API_BASE MODEL=$MODEL workers=$EVAL_WORKERS"
_log "order: cot-APPS-retry -> codeact-APPS-retry -> codeact-nonAPPS-retry -> self-planning"

_check_api() {
  local url="${API_BASE%/}/models"
  _log "API check: $url"
  # Bypass http_proxy/https_proxy — otherwise urllib may fail on 127.0.0.1
  # while curl still works.
  if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      ALL_PROXY= all_proxy= \
      python3 - <<PY
import json, sys, urllib.request
url = ${url@Q}
want = ${MODEL@Q}
# Force direct connection (ignore any inherited ProxyHandler defaults).
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    req = urllib.request.Request(url, headers={"User-Agent": "verl-swe-eval/1.0"})
    with opener.open(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
except Exception as e:
    print(f"API_UNREACHABLE: {type(e).__name__}: {e}", flush=True)
    sys.exit(2)
ids = [m.get("id") for m in data.get("data") or []]
print("models=", ids, flush=True)
if want not in ids:
    print(f"MODEL_MISSING: want={want!r} have={ids!r}", flush=True)
    sys.exit(3)
print("API_OK", flush=True)
PY
  then
    _log "ERROR: API check failed — fix vLLM / MODEL name / unset proxy, then re-run"
    return 1
  fi
  return 0
}

# Wait until Instruct base is reachable (igate / model swap).
_WAIT_SECS=${API_WAIT_SECS:-0}   # 0 = check once and fail; >0 poll that many seconds
if [ "$_WAIT_SECS" -gt 0 ]; then
  _deadline=$((SECONDS + _WAIT_SECS))
  until _check_api; do
    if [ "$SECONDS" -ge "$_deadline" ]; then
      _log "ERROR: timed out waiting for API (${_WAIT_SECS}s)"
      exit 1
    fi
    _log "waiting for API / correct MODEL ... sleep 60"
    sleep 60
  done
else
  # Soft check: warn but still attempt stages (user asked to launch now).
  if ! _check_api; then
    _log "WARNING: API not OK right now; stages will keep failing until Instruct is up"
  fi
fi

############################################
# 1) CoT APPS retry-errors
############################################
if [ "${SKIP_COT_APPS:-0}" != "1" ]; then
  _log "stage 1/4: CoT APPS retry-errors"
  export METHOD=qwen25_coder_7b_mt12_cot_freeform
  export PROMPT_MODE=freeform
  export ENCOURAGE_COT=1
  export DISABLE_THINKING=0
  export EVAL_MAX_TURNS=$CHAIN_MAX_TURNS
  export EVAL_HISTORY_LENGTH=6
  export EVAL_RETRY_ERRORS=1
  bash "$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log" 2>&1 \
    || _log "WARNING: cot APPS stage exited $?"
  _log "stage 1/4 done"
else
  _log "stage 1/4 SKIP_COT_APPS=1"
fi

############################################
# 2) CodeAct APPS retry-errors
############################################
if [ "${SKIP_CODEACT_APPS:-0}" != "1" ]; then
  _log "stage 2/4: CodeAct APPS retry-errors"
  export METHOD=qwen25_coder_7b_mt12_exec
  unset PROMPT_MODE || true
  unset ENCOURAGE_COT || true
  export ENCOURAGE_COT=0
  export DISABLE_THINKING=1
  export EVAL_MAX_TURNS=$CHAIN_MAX_TURNS
  export EVAL_HISTORY_LENGTH=6
  export EVAL_RETRY_ERRORS=1
  bash "$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log" 2>&1 \
    || _log "WARNING: codeact APPS stage exited $?"
  _log "stage 2/4 done"
else
  _log "stage 2/4 SKIP_CODEACT_APPS=1"
fi

############################################
# 3) CodeAct non-APPS retry-errors
############################################
if [ "${SKIP_CODEACT_NON_APPS:-0}" != "1" ]; then
  _log "stage 3/4: CodeAct non-APPS retry-errors"
  export METHOD=qwen25_coder_7b_mt12_exec
  unset PROMPT_MODE || true
  export ENCOURAGE_COT=0
  export DISABLE_THINKING=1
  export EVAL_MAX_TURNS=$CHAIN_MAX_TURNS
  export EVAL_RETRY_ERRORS=1
  export EVAL_BENCHMARKS=${CODEACT_NON_APPS_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc}
  export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_nonapps_retry_queue_status.json"
  bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" \
    >> "$LOG_DIR/eval_api_${METHOD}_nonapps_retry_queue.nohup.log" 2>&1 \
    || _log "WARNING: codeact non-APPS retry exited $?"
  _log "stage 3/4 done"
else
  _log "stage 3/4 SKIP_CODEACT_NON_APPS=1"
fi

############################################
# 4) Self-planning suite
############################################
if [ "${SKIP_SELF_PLANNING:-0}" != "1" ]; then
  _log "stage 4/4: Self-planning suite"
  # fresh env for self-planning defaults inside its script
  env -u PROMPT_MODE -u ENCOURAGE_COT \
    API_BASE="$API_BASE" \
    MODEL="$MODEL" \
    REPO_ROOT="$REPO_ROOT" \
    DATA_ROOT="$DATA_ROOT" \
    EVAL_RESUME=1 \
    EVAL_RETRY_ERRORS=0 \
    SKIP_PEER_WAIT=1 \
    SKIP_WAIT=1 \
    SKIP_APPS="${SKIP_SELF_PLANNING_APPS:-0}" \
    bash "$SCRIPT_DIR/eval_api_qwen25_coder_7b_self_planning.sh" \
      >> "$LOG_DIR/eval_api_qwen25_coder_7b_st1_self_planning_suite.nohup.log" 2>&1 \
      || _log "WARNING: self-planning stage exited $?"
  _log "stage 4/4 done"
else
  _log "stage 4/4 SKIP_SELF_PLANNING=1"
fi

_log "========== Qwen2.5-Coder-7B chain FINISHED =========="
