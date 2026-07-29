#!/usr/bin/env bash
# Qwen2.5-Coder-7B orchestrator:
#   1) CoT freeform  — non-APPS queue
#   2) CodeAct       — non-APPS queue
#   3) CoT freeform  — APPS
#   4) CodeAct       — APPS
#
# Non-APPS first so humaneval/mbpp/... finish before the long APPS jobs.
#
# Usage (training host):
#   cd /mnt/z4/solariewang/verl-swe
#   nohup bash scripts/eval_api_qwen25_coder_7b_cot_then_codeact.sh \
#     >> logs/eval_api_qwen25_coder_7b_cot_then_codeact.nohup.log 2>&1 &
#
# Env overrides (optional):
#   API_BASE  MODEL  EVAL_WORKERS  EVAL_BENCHMARKS
#   SKIP_API_CHECK=1  SKIP_APPS=1  (skip stages 3–4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

export REPO_ROOT DATA_ROOT
export API_BASE=${API_BASE:-http://29.163.228.59:80/v1}
export MODEL=${MODEL:-Qwen2.5-Coder-7B-Instruct}
export TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-12}
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export MAX_TOKENS=${MAX_TOKENS:-8192}
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

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_check_api() {
  if [ "${SKIP_API_CHECK:-0}" = "1" ]; then
    return 0
  fi
  local url="${API_BASE%/}/models"
  echo "[$(_ts)] checking API $url"
  if ! curl -fsS --max-time 10 "$url" >/dev/null; then
    echo "[$(_ts)] ERROR: API not reachable: $url" >&2
    echo "  Start vLLM first, e.g.:" >&2
    echo "  VLLM_ENV=/opt/conda/envs/vllm CUDA_VISIBLE_DEVICES=0 PORT=80 \\" >&2
    echo "    bash scripts/launch_vllm_qwen25_coder_7b.sh start" >&2
    exit 1
  fi
  echo "[$(_ts)] API ok"
}

# Show existing checkpoints so already-finished benches are visibly skipped.
_print_checkpoint_status() {
  local method=$1
  echo "[$(_ts)] checkpoints for METHOD=$method (EVAL_RESUME=$EVAL_RESUME):"
  METHOD="$method" LOG_DIR="$LOG_DIR" python3 - <<'PY'
import json, os
from collections import Counter
method = os.environ["METHOD"]
log_dir = os.environ["LOG_DIR"]
targets = {
    "apps": 5000, "humaneval": 164, "mbpp": 257, "livecodebench": 131,
    "usaco": 307, "ojbench": 159, "icpc": 106,
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
        print(f"  {b:13} {len(arr):4d}/{t} DONE  won={w} lost={l} err={e}  -> SKIP")
    else:
        print(f"  {b:13} {len(arr):4d}/{t} partial won={w} lost={l} err={e}  -> resume")
PY
}

_run_fg() {
  local tag=$1
  local cmd=$2
  local log=$3
  echo "[$(_ts)] [$tag] start -> $log"
  set +e
  bash -c "$cmd" >>"$log" 2>&1
  local rc=$?
  set -e
  echo "[$(_ts)] [$tag] exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(_ts)] WARNING: [$tag] non-zero exit=$rc; continuing" >&2
  fi
  return 0
}

_export_cot() {
  export METHOD=qwen25_coder_7b_mt12_cot_freeform
  export DISABLE_THINKING=0
  export PROMPT_MODE=freeform
  export ENCOURAGE_COT=1
  export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
  export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*qwen25_coder_7b_mt12_cot_freeform"
}

_export_codeact() {
  export METHOD=qwen25_coder_7b_mt12_exec
  export DISABLE_THINKING=1
  unset PROMPT_MODE ENCOURAGE_COT || true
  export STATUS_JSON="$LOG_DIR/eval_api_${METHOD}_queue_status.json"
  export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*qwen25_coder_7b_mt12_exec"
}

_check_api

# Never wipe finished work unless the user explicitly disables resume.
export EVAL_RESUME=${EVAL_RESUME:-1}
if [ "$EVAL_RESUME" != "1" ]; then
  echo "[$(_ts)] WARNING: EVAL_RESUME=$EVAL_RESUME — finished instances may be re-run" >&2
fi

echo "[$(_ts)] ========== plan =========="
echo "  1) cot freeform  non-APPS ($EVAL_BENCHMARKS)"
echo "  2) codeact       non-APPS ($EVAL_BENCHMARKS)"
echo "  3) cot freeform  APPS"
echo "  4) codeact       APPS"
echo "  API=$API_BASE MODEL=$MODEL workers=$EVAL_WORKERS EVAL_RESUME=$EVAL_RESUME"
echo "  Already-complete benches are SKIP'd; partial benches resume."

# ----- 1) CoT non-APPS -----
echo "[$(_ts)] ========== stage 1/4: CoT non-APPS queue =========="
_export_cot
_print_checkpoint_status "$METHOD"
echo "  METHOD=$METHOD prompt_mode=$PROMPT_MODE encourage_cot=$ENCOURAGE_COT thinking=ON"
_run_fg "cot-queue" \
  "bash \"$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh\"" \
  "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log"

# ----- 2) CodeAct non-APPS -----
echo "[$(_ts)] ========== stage 2/4: CodeAct non-APPS queue =========="
_export_codeact
_print_checkpoint_status "$METHOD"
echo "  METHOD=$METHOD thinking=off (ReAct)"
_run_fg "codeact-queue" \
  "bash \"$SCRIPT_DIR/eval_api_kimi_mt12_queue.sh\"" \
  "$LOG_DIR/eval_api_${METHOD}_queue.nohup.log"

if [ "${SKIP_APPS:-0}" = "1" ]; then
  echo "[$(_ts)] SKIP_APPS=1 — skipping stages 3–4"
  echo "[$(_ts)] ========== cot/codeact non-APPS DONE =========="
  exit 0
fi

# ----- 3) CoT APPS -----
echo "[$(_ts)] ========== stage 3/4: CoT APPS =========="
_export_cot
_print_checkpoint_status "$METHOD"
echo "  METHOD=$METHOD APPS mt12 (resume if partial)"
_run_fg "cot-apps" \
  "bash \"$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh\"" \
  "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"

# ----- 4) CodeAct APPS -----
echo "[$(_ts)] ========== stage 4/4: CodeAct APPS =========="
_export_codeact
_print_checkpoint_status "$METHOD"
echo "  METHOD=$METHOD APPS mt12 (resume if partial)"
_run_fg "codeact-apps" \
  "bash \"$SCRIPT_DIR/eval_api_kimi_apps_mt12.sh\"" \
  "$LOG_DIR/eval_api_${METHOD}_apps.nohup.log"

echo "[$(_ts)] ========== ALL DONE (non-APPS then APPS) =========="
