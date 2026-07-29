#!/usr/bin/env bash
# Kimi APPS eval — multi-turn ReAct aligned with GRPO training (one.sh defaults).
#
#   blind, max_turns=12, history_length=6, disable_thinking
#   Output: logs/eval_api_${METHOD}_apps.json  (default METHOD=kimi_2_6_mt12_exec)
#
# Usage (after single-turn APPS finishes):
#   nohup bash scripts/eval_api_kimi_apps_mt12.sh >> logs/eval_api_kimi_2_6_mt12_apps.nohup.log 2>&1 &
#
# Env:
#   API_BASE  MODEL  METHOD  DATA_ROOT  REPO_ROOT
#   WAIT_FOR_SINGLE_TURN=0   do not block on single-turn APPS (default off)
#   EVAL_WORKERS=8           concurrent instances (mt12 = many API calls each)
#   EVAL_CHECKPOINT_EVERY=5  write JSON every N finished instances (mid-run inspect)
#   MAX_TOKENS=8192          match RL MAX_RESPONSE_LENGTH
#   EVAL_RESUME=1
#
# Mid-run progress:
#   bash scripts/inspect_eval_api_progress.sh
#   bash scripts/inspect_eval_api_progress.sh --errors 20

set -euo pipefail

# 训练机用 /mnt；开发机用 /apdcephfs。若误带 /apdcephfs 且本机有 /mnt，自动纠正。
if [ -z "${REPO_ROOT:-}" ]; then
  if [ -d /mnt/z4/solariewang/verl-swe ]; then
    REPO_ROOT=/mnt/z4/solariewang/verl-swe
  else
    REPO_ROOT=/apdcephfs/z4/solariewang/verl-swe
  fi
fi
if [ -z "${DATA_ROOT:-}" ]; then
  if [ -d /mnt/z4/solariewang/datasets ]; then
    DATA_ROOT=/mnt/z4/solariewang/datasets
  else
    DATA_ROOT=/apdcephfs/z4/solariewang/datasets
  fi
fi
case "$REPO_ROOT" in
  /apdcephfs/*)
    if [ -d /mnt/z4/solariewang/verl-swe ] && [ ! -f "$REPO_ROOT/scripts/eval_api_baseline.py" ]; then
      REPO_ROOT=/mnt/z4/solariewang/verl-swe
      DATA_ROOT=/mnt/z4/solariewang/datasets
    fi
    ;;
esac
LOG_DIR="$REPO_ROOT/logs"
EVAL_PY="$REPO_ROOT/scripts/eval_api_baseline.py"

API_BASE=${API_BASE:-http://29.163.228.8:8080/v1}
MODEL=${MODEL:-Kimi-K2.6}
METHOD=${METHOD:-kimi_2_6_mt12_exec}
TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
MAX_TOKENS=${MAX_TOKENS:-8192}
DISABLE_THINKING=${DISABLE_THINKING:-1}
API_TIMEOUT=${API_TIMEOUT:-1800}
API_RETRIES=${API_RETRIES:-3}
EVAL_RESUME=${EVAL_RESUME:-1}
EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
EVAL_WORKERS=${EVAL_WORKERS:-8}
WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}
EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}

OUT_JSON="$LOG_DIR/eval_api_${METHOD}_apps.json"
OUT_LOG="$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"
# Single-turn APPS json for WAIT_FOR_SINGLE_TURN (derive from METHOD if *_mt12_exec).
_st_method="${METHOD_ST:-${METHOD/_mt12_exec/_exec}}"
SINGLE_TURN_JSON=${SINGLE_TURN_JSON:-"$LOG_DIR/eval_api_${_st_method}_apps.json"}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_wait_single_turn_done() {
  if [ "$WAIT_FOR_SINGLE_TURN" != "1" ]; then
    return 0
  fi
  echo "[$(_ts)] waiting for single-turn APPS: $SINGLE_TURN_JSON"
  while true; do
    if python3 - <<PY
import json, os, sys
path = ${SINGLE_TURN_JSON@Q}
if not os.path.isfile(path):
    sys.exit(1)
d = json.load(open(path))
rows = d.get("per_instance") or []
errors = sum(1 for r in rows if r.get("outcome") == "error")
graded = sum(1 for r in rows if r.get("outcome") in ("won", "lost"))
proto = d.get("protocol") or {}
ok = (
    len(rows) >= 5000
    and errors == 0
    and graded >= 5000
    and int(proto.get("max_turns", 1)) == 1
)
sys.exit(0 if ok else 1)
PY
    then
      echo "[$(_ts)] single-turn APPS complete — starting multi-turn"
      return 0
    fi
    # also wait if single-turn eval still running
    if pgrep -f 'eval_api_baseline.py.*--benchmark apps' >/dev/null 2>&1; then
      echo "[$(_ts)] single-turn APPS eval still running ..."
    else
      python3 - <<PY
import json, os
path = ${SINGLE_TURN_JSON@Q}
if os.path.isfile(path):
    d = json.load(open(path))
    rows = d.get("per_instance") or []
    g = sum(1 for r in rows if r.get("outcome") in ("won","lost"))
    e = sum(1 for r in rows if r.get("outcome")=="error")
    print(f"  checkpoint: rows={len(rows)} graded={g} errors={e}")
PY
    fi
    sleep 120
  done
}

_wait_single_turn_done

if [ "${SKIP_PEER_WAIT:-0}" != "1" ]; then
  _peer_pat="${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py}"
  while pgrep -f "$_peer_pat" >/dev/null 2>&1; do
    echo "[$(_ts)] waiting for other eval_api_baseline.py to finish ..."
    sleep 30
  done
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[$(_ts)] ========== Kimi APPS multi-turn (mt12) =========="
echo "  API=$API_BASE  MODEL=$MODEL  METHOD=$METHOD"
echo "  protocol=$TEST_FEEDBACK_MODE  max_turns=$EVAL_MAX_TURNS  history=$EVAL_HISTORY_LENGTH"
echo "  max_tokens=$MAX_TOKENS  workers=$EVAL_WORKERS  out=$OUT_JSON"

args=(
  --api-base "$API_BASE"
  --model "$MODEL"
  --method-name "$METHOD"
  --benchmark apps
  --data-root "$DATA_ROOT"
  --max-turns "$EVAL_MAX_TURNS"
  --history-length "$EVAL_HISTORY_LENGTH"
  --max-tokens "$MAX_TOKENS"
  --test-feedback-mode "$TEST_FEEDBACK_MODE"
  --api-timeout "$API_TIMEOUT"
  --api-retries "$API_RETRIES"
  --checkpoint-every "$EVAL_CHECKPOINT_EVERY"
  --workers "$EVAL_WORKERS"
  --instance-timeout "$EVAL_INSTANCE_TIMEOUT"
  --out "$OUT_JSON"
)
if [ "$DISABLE_THINKING" = 1 ]; then
  args+=(--disable-thinking)
else
  args+=(--enable-thinking)
fi
if [ -n "${PROMPT_MODE:-}" ]; then
  args+=(--prompt-mode "$PROMPT_MODE")
fi
if [ "${ENCOURAGE_COT:-0}" = "1" ]; then
  args+=(--encourage-cot)
fi
if [ -n "${API_KEY:-}" ]; then
  args+=(--api-key "$API_KEY")
fi
if [ "$EVAL_RESUME" = 1 ]; then
  args+=(--resume)
else
  args+=(--no-resume)
fi
if [ "${EVAL_RETRY_ERRORS:-0}" = 1 ]; then
  args+=(--retry-errors)
fi

export PYTHONUNBUFFERED=1
echo "[$(_ts)] RUN mt12 apps workers=$EVAL_WORKERS retry_errors=${EVAL_RETRY_ERRORS:-0}" | tee -a "$OUT_LOG"
# 不用 exec ... | tee：管道下 exec 无效，且脚本若在长跑期间被改写，结束后 bash 续读会报诡异 syntax error。
python3 "$EVAL_PY" "${args[@]}" 2>&1 | tee -a "$OUT_LOG"
exit "${PIPESTATUS[0]}"
