#!/usr/bin/env bash
# After the in-flight APPS eval finishes, backfill skipped non-APPS benches
# for grpo_coderl_sft_mt8_step50 (train-aligned mt8 interactive + thinking).
#
# Skipped earlier when HumanEval was killed mid-queue:
#   humaneval (resume 125/164), mbpp, livecodebench, usaco, ojbench, icpc, leetcode
# Does NOT re-run APPS.
#
# Usage (train host — default: do NOT wait for APPS):
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_grpo_coderl_sft_mt8_step50_after_apps.sh \
#     >> logs/eval_api_grpo_coderl_sft_mt8_step50_after_apps.nohup.log 2>&1 &
#
#   # If you really want to wait for APPS first:
#   SKIP_WAIT_APPS=0 nohup bash scripts/eval_api_grpo_coderl_sft_mt8_step50_after_apps.sh \
#     >> logs/eval_api_grpo_coderl_sft_mt8_step50_after_apps.nohup.log 2>&1 &
#
# Env:
#   API_BASE MODEL METHOD EVAL_WORKERS EVAL_INSTANCE_TIMEOUT
#   APPS_TARGET=5000  POLL_SECS=120
#   SKIP_WAIT_APPS=1 (default) — do not wait for APPS; run other benches now
#   SKIP_WAIT_APPS=0 — wait until APPS JSON hits APPS_TARGET and apps proc exits

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

export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-interactive}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-8}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-5}
export MAX_TOKENS=${MAX_TOKENS:-4096}
export DISABLE_THINKING=${DISABLE_THINKING:-0}
export PROMPT_MODE=${PROMPT_MODE:-react}
export ENCOURAGE_COT=${ENCOURAGE_COT:-0}

export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_RESUME=${EVAL_RESUME:-1}
export EVAL_RETRY_ERRORS=${EVAL_RETRY_ERRORS:-1}
export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export EVAL_WORKERS=${EVAL_WORKERS:-2}
export EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}
export LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
export SKIP_WAIT=${SKIP_WAIT:-1}
export SKIP_PEER_WAIT=${SKIP_PEER_WAIT:-1}
export WAIT_FOR_SINGLE_TURN=${WAIT_FOR_SINGLE_TURN:-0}

# Remaining benches only (APPS already running / done)
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_after_apps_queue_status.json"
export WAIT_EVAL_PATTERN=${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py.*grpo_coderl_sft_mt8_step50}

APPS_JSON=${APPS_JSON:-$LOG_DIR/eval_api_${METHOD}_apps.json}
APPS_TARGET=${APPS_TARGET:-5000}
POLL_SECS=${POLL_SECS:-120}
SKIP_WAIT_APPS=${SKIP_WAIT_APPS:-1}
APPS_PROC_PATTERN=${APPS_PROC_PATTERN:-eval_api_baseline.py.*grpo_coderl_sft_mt8_step50_mt8_interactive.*--benchmark apps}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_apps_json_done() {
  APPS_JSON="$APPS_JSON" APPS_TARGET="$APPS_TARGET" python3 - <<'PY'
import json, os, sys
path = os.environ["APPS_JSON"]
target = int(os.environ["APPS_TARGET"])
if not os.path.isfile(path):
    sys.exit(1)
d = json.load(open(path))
rows = d.get("per_instance") or []
# finished when we have target rows (won/lost/error all count toward coverage)
sys.exit(0 if len(rows) >= target else 1)
PY
}

_apps_proc_running() {
  pgrep -f "$APPS_PROC_PATTERN" >/dev/null 2>&1
}

_wait_apps() {
  if [ "${SKIP_WAIT_APPS:-0}" = "1" ]; then
    echo "[$(_ts)] SKIP_WAIT_APPS=1 — not waiting for APPS"
    return 0
  fi
  echo "[$(_ts)] waiting for APPS to finish"
  echo "  json=$APPS_JSON  target=$APPS_TARGET  poll=${POLL_SECS}s"
  while true; do
    local n=0
    if [ -f "$APPS_JSON" ]; then
      n="$(python3 -c "import json; print(len(json.load(open('$APPS_JSON')).get('per_instance') or []))")"
    fi
    local proc=no
    _apps_proc_running && proc=yes
    echo "[$(_ts)] APPS progress rows=$n/$APPS_TARGET  apps_proc=$proc"
    if _apps_json_done && ! _apps_proc_running; then
      echo "[$(_ts)] APPS complete — starting backfill queue"
      return 0
    fi
    # JSON hit target but process still flushing / last workers
    if _apps_json_done && _apps_proc_running; then
      echo "[$(_ts)] APPS json complete but process still alive — waiting for exit"
    fi
    sleep "$POLL_SECS"
  done
}

_check_api() {
  if [ "${SKIP_API_CHECK:-0}" = "1" ]; then
    return 0
  fi
  local url="${API_BASE%/}/models"
  echo "[$(_ts)] checking API $url"
  if ! curl -fsS --max-time 15 "$url" >/dev/null; then
    echo "[$(_ts)] ERROR: API not reachable: $url" >&2
    exit 1
  fi
  echo "[$(_ts)] API ok"
}

_print_status() {
  echo "[$(_ts)] backfill targets:"
  METHOD="$METHOD" LOG_DIR="$LOG_DIR" python3 - <<'PY'
import json, os
from collections import Counter
method = os.environ["METHOD"]
log_dir = os.environ["LOG_DIR"]
targets = {
    "humaneval": 164, "mbpp": 257, "livecodebench": 131,
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
        print(f"  {b:13} {len(arr):4d}/{t} DONE  won={w} lost={l} err={e}")
    else:
        print(f"  {b:13} {len(arr):4d}/{t} partial won={w} lost={l} err={e}  -> resume")
PY
}

echo "[$(_ts)] ========== after-APPS backfill =========="
echo "  API=$API_BASE MODEL=$MODEL METHOD=$METHOD"
echo "  protocol=$TEST_FEEDBACK_MODE turns=$EVAL_MAX_TURNS history=$EVAL_HISTORY_LENGTH"
echo "  workers=$EVAL_WORKERS instance_timeout=${EVAL_INSTANCE_TIMEOUT}s"
echo "  benches=$EVAL_BENCHMARKS"
_print_status

_wait_apps
_check_api
_print_status

echo "[$(_ts)] ========== running backfill queue =========="
# Avoid colliding with any leftover apps tee; peer wait skipped via SKIP_PEER_WAIT
bash "$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh" \
  >>"$LOG_DIR/eval_api_${METHOD}_after_apps_queue.nohup.log" 2>&1

echo "[$(_ts)] ========== after-APPS backfill DONE =========="
_print_status
echo "  apps json (untouched): $APPS_JSON"
echo "  queue log: $LOG_DIR/eval_api_${METHOD}_after_apps_queue.nohup.log"
