#!/usr/bin/env bash
# Kimi API eval queue — full coding suite (exec, single-turn).
#
# Default model: Kimi-K2.6 (override MODEL / METHOD / API_BASE).
# Output: logs/eval_api_${METHOD}_${bench}.json  (e.g. eval_api_kimi_2_6_apps.json)
#
# Order: apps → humaneval → mbpp → livecodebench → usaco → ojbench → icpc
#
# Usage:
#   nohup bash scripts/eval_api_kimi_queue.sh >> logs/eval_api_kimi_queue.nohup.log 2>&1 &
#
# Env overrides:
#   EVAL_BENCHMARKS=apps,humaneval,...   (default: full list below)
#   API_BASE  MODEL  METHOD  DATA_ROOT  REPO_ROOT
#   TEST_FEEDBACK_MODE=exec  EVAL_MAX_TURNS=1  EVAL_HISTORY_LENGTH=6  DISABLE_THINKING=1
#   LCB_MIN_DATE=2025-02-01  API_TIMEOUT=1800  API_RETRIES=3
#   EVAL_RESUME=1  EVAL_CHECKPOINT_EVERY=10  EVAL_WORKERS=8

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/apdcephfs/z4/solariewang/verl-swe}
DATA_ROOT=${DATA_ROOT:-/apdcephfs/z4/solariewang/datasets}
API_BASE=${API_BASE:-http://29.163.228.8:8080/v1}
MODEL=${MODEL:-Kimi-K2.6}
METHOD=${METHOD:-kimi_2_6_exec}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
DISABLE_THINKING=${DISABLE_THINKING:-1}
EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-1}
EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
LCB_MIN_DATE=${LCB_MIN_DATE:-2025-02-01}
API_TIMEOUT=${API_TIMEOUT:-1800}
API_RETRIES=${API_RETRIES:-3}
EVAL_RESUME=${EVAL_RESUME:-1}
EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-10}
EVAL_WORKERS=${EVAL_WORKERS:-8}
EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}
EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-apps,humaneval,mbpp,livecodebench,usaco,ojbench,icpc}

LOG_DIR="$REPO_ROOT/logs"
STATUS_JSON=${STATUS_JSON:-$LOG_DIR/eval_api_kimi_queue_status.json}
EVAL_PY="$REPO_ROOT/scripts/eval_api_baseline.py"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

wait_for_eval() {
  if [ "${SKIP_WAIT:-0}" = "1" ]; then
    return 0
  fi
  local pattern="${WAIT_EVAL_PATTERN:-python3.*eval_api_baseline.py}"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    echo "[$(_ts)] waiting for in-flight eval ($pattern) ..."
    sleep 30
  done
}

count_benchmark_instances() {
  local bench="$1"
  python3 - <<PY
import importlib.util, os, sys
from pathlib import Path
repo = Path(${REPO_ROOT@Q})
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("eval_api", repo / "scripts/eval_api_baseline.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
bench = ${bench@Q}
min_date = os.environ.get("LCB_MIN_DATE", "2025-02-01")
rows, _ = mod._load_benchmark_instances(
    bench, Path(${DATA_ROOT@Q}),
    min_date=min_date if bench == "livecodebench" else "",
)
print(len(rows))
PY
}

benchmark_is_complete() {
  local bench="$1"
  local out_json="$2"
  local expected_n="$3"
  export OUT_JSON="$out_json" EXPECTED_N="$expected_n"
  export TEST_FEEDBACK_MODE DISABLE_THINKING EVAL_WORKERS EVAL_MAX_TURNS
  export EVAL_RETRY_ERRORS=${EVAL_RETRY_ERRORS:-0}
  export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
  python3 <<'PY' 2>/dev/null
import json, os, sys
path = os.environ["OUT_JSON"]
if not os.path.isfile(path):
    sys.exit(1)
d = json.load(open(path))
proto = d.get("protocol") or {}
mode = proto.get("test_feedback_mode", "")
if mode == "oracle":
    mode = "interactive"
want_mode = os.environ.get("TEST_FEEDBACK_MODE", "exec")
want_think_off = os.environ.get("DISABLE_THINKING", "1") == "1"
want_workers = int(os.environ.get("EVAL_WORKERS", "8"))
want_max_turns = int(os.environ.get("EVAL_MAX_TURNS", "1"))
think_off = proto.get("disable_thinking", False)
workers = int(proto.get("workers", 1))
max_turns = int(proto.get("max_turns", 12))
rows = d.get("per_instance") or []
done = len(rows)
expected = int(os.environ["EXPECTED_N"])
ok = (
    done >= expected
    and mode == want_mode
    and think_off == want_think_off
    and workers == want_workers
    and max_turns == want_max_turns
)
# With EVAL_RETRY_ERRORS=1, leave "incomplete" while retryable errors remain
# so the queue re-enters the benchmark and --retry-errors re-runs them.
if ok and os.environ.get("EVAL_RETRY_ERRORS", "0") == "1":
    retry_types = {t.strip() for t in os.environ.get("EVAL_RETRY_ERROR_TYPES", "api_http_error,eval_error").split(",") if t.strip()}
    n_err = sum(
        1 for r in rows
        if r.get("outcome") == "error"
        and str(r.get("error_type", "eval_error")) in retry_types
    )
    if n_err > 0:
        sys.exit(1)
sys.exit(0 if ok else 1)
PY
}

update_status() {
  local current="$1"
  export STATUS_JSON
  python3 - <<PY
import json, os, time
path = os.environ["STATUS_JSON"]
data = json.load(open(path)) if os.path.isfile(path) else {}
data["current"] = "$current" if "$current" else None
data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(data, open(path, "w"), indent=2)
PY
}

run_benchmark() {
  local bench="$1"
  local out_json="$LOG_DIR/eval_api_${METHOD}_${bench}.json"
  local out_log="$LOG_DIR/eval_api_${METHOD}_${bench}.nohup.log"

  wait_for_eval

  local expected_n
  expected_n="$(count_benchmark_instances "$bench")"
  if [ "$expected_n" -le 0 ]; then
    echo "[$(_ts)] SKIP $bench — 0 instances (data missing?)"
    return 0
  fi

  if benchmark_is_complete "$bench" "$out_json" "$expected_n"; then
    echo "[$(_ts)] SKIP $bench — already complete ($expected_n/$expected_n): $out_json"
    return 0
  fi

  update_status "$bench"
  echo "[$(_ts)] START $bench n=$expected_n max_turns=$EVAL_MAX_TURNS -> $out_json"
  echo "[$(_ts)] log -> $out_log"

  local args=(
    --api-base "$API_BASE"
    --model "$MODEL"
    --method-name "$METHOD"
    --benchmark "$bench"
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
    --out "$out_json"
  )
  if [ "$bench" = "livecodebench" ]; then
    args+=(--min-date "$LCB_MIN_DATE")
  fi
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

  export PYTHONUNBUFFERED=1 EVAL_RESUME
  echo "[$(_ts)] RUN $bench workers=$EVAL_WORKERS" | tee -a "$out_log"

  set +e
  python3 "$EVAL_PY" "${args[@]}" 2>&1 | tee -a "$out_log"
  local _py_rc=("${PIPESTATUS[@]}")
  local rc="${_py_rc[0]:-1}"
  set -e

  if [ "$rc" -eq 0 ]; then
    echo "[$(_ts)] DONE $bench -> $out_json"
    export STATUS_JSON
    python3 - <<PY
import json, os, time
path = os.environ["STATUS_JSON"]
data = json.load(open(path))
data.setdefault("completed", [])
entry = {
    "benchmark": "$bench",
    "out": "$out_json",
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
if entry not in data["completed"]:
    data["completed"].append(entry)
data["current"] = None
json.dump(data, open(path, "w"), indent=2)
PY
    return 0
  fi

  echo "[$(_ts)] FAIL $bench (exit $rc) — partial checkpoint: $out_json"
  export STATUS_JSON
  python3 - <<PY
import json, os, time
path = os.environ["STATUS_JSON"]
data = json.load(open(path))
data.setdefault("failed", [])
data["failed"].append({
    "benchmark": "$bench",
    "exit_code": $rc,
    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
data["current"] = None
json.dump(data, open(path, "w"), indent=2)
PY
  return "$rc"
}

echo "[$(_ts)] ========== Kimi API eval queue start =========="
echo "  API=$API_BASE  MODEL=$MODEL  METHOD=$METHOD  DATA_ROOT=$DATA_ROOT"
echo "  benchmarks=$EVAL_BENCHMARKS"
echo "  protocol=$TEST_FEEDBACK_MODE  max_turns=$EVAL_MAX_TURNS  thinking=$([ "$DISABLE_THINKING" = 1 ] && echo off || echo on)  workers=$EVAL_WORKERS"

export STATUS_JSON MODEL METHOD API_BASE TEST_FEEDBACK_MODE DISABLE_THINKING EVAL_MAX_TURNS LCB_MIN_DATE DATA_ROOT REPO_ROOT EVAL_BENCHMARKS

python3 - <<PY
import json, os, time
path = os.environ["STATUS_JSON"]
method = os.environ.get("METHOD", "kimi")
queue = [b.strip() for b in os.environ.get("EVAL_BENCHMARKS", "").split(",") if b.strip()]
data = {
    "model": os.environ["MODEL"],
    "method": method,
    "api_base": os.environ["API_BASE"],
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "protocol": {
        "test_feedback_mode": os.environ.get("TEST_FEEDBACK_MODE", "exec"),
        "max_turns": int(os.environ.get("EVAL_MAX_TURNS", "1")),
        "disable_thinking": os.environ.get("DISABLE_THINKING", "1") == "1",
        "workers": int(os.environ.get("EVAL_WORKERS", "8")),
        "lcb_min_date": os.environ.get("LCB_MIN_DATE", "2025-02-01"),
    },
    "queue": queue,
    "completed": [],
    "failed": [],
    "current": None,
}
# preserve completed only for same method (switching 2.7 -> 2.6 starts fresh)
if os.path.isfile(path):
    try:
        old = json.load(open(path))
        if old.get("method") == method:
            data["completed"] = old.get("completed") or []
    except Exception:
        pass
json.dump(data, open(path, "w"), indent=2)
PY

IFS=',' read -ra BENCH_LIST <<< "$EVAL_BENCHMARKS"
fail_rc=0
for bench in "${BENCH_LIST[@]}"; do
  bench="${bench// /}"
  [ -n "$bench" ] || continue
  if ! run_benchmark "$bench"; then
    fail_rc=$?
    echo "[$(_ts)] queue halted on $bench (exit $fail_rc); fix and re-run to continue"
    exit "$fail_rc"
  fi
done

echo "[$(_ts)] ========== Kimi API eval queue finished =========="
