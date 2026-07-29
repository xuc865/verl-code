#!/usr/bin/env bash
# Periodic SwanLab cloud sync for DiDPO offline/local runs (dev box with net).
# Usage:
#   nohup bash /mnt/z4/solariewang/verl-swe/scripts/swanlab_sync_didpo_coderl_daemon.sh \
#     >/dev/null 2>&1 &
set -euo pipefail
ROOT=${ROOT:-/mnt/z4/solariewang}
if [[ ! -f "$ROOT/verl-swe/scripts/swanlab_sync_watch.sh" && -f /apdcephfs/z4/solariewang/verl-swe/scripts/swanlab_sync_watch.sh ]]; then
  ROOT=/apdcephfs/z4/solariewang
fi
REPO=${REPO:-$ROOT/verl-swe}
cd "$REPO"

export SWANLAB_SYNC_ALL_RUNS=${SWANLAB_SYNC_ALL_RUNS:-1}
# Sync to both projects: native DiDPO space + grpo_coderl for side-by-side curves
SWANLAB_SYNC_PROJECTS=${SWANLAB_SYNC_PROJECTS:-didpo_coderl,grpo_coderl}
# Live resume dir only (same run_id also exists on an older frozen dir)
export SWANLAB_SYNC_RUN_IDS=${SWANLAB_SYNC_RUN_IDS:-c08qn4a8}
export SWANLAB_SYNC_RUN_NAMES=${SWANLAB_SYNC_RUN_NAMES:-run-20260723_211312-c08qn4a8}
export SWANLAB_SYNC_EXP_NAME=${SWANLAB_SYNC_EXP_NAME:-didpo_coderl_qwen25_7b_sft_mt8}
export SWANLAB_SYNC_INTERVAL=${SWANLAB_SYNC_INTERVAL:-1800}
export SWANLAB_DISABLE_PROXY=${SWANLAB_DISABLE_PROXY:-1}
# .swanlab races with sync: always rebuild scalars from logs/ before upload
LIVE_RUN="$REPO/swanlog/$SWANLAB_SYNC_RUN_NAMES"
if [[ -d "$LIVE_RUN" ]]; then
  touch "$LIVE_RUN/.rebuild_from_logs"
fi

LOG="$REPO/logs/swanlab_sync_didpo_coderl.log"
PIDFILE="$REPO/logs/swanlab_sync_didpo_coderl.pid"
mkdir -p "$REPO/logs"

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "[swanlab-daemon $(date '+%F %T')] already running pid=$old" >>"$LOG"
    exit 0
  fi
fi
echo $$ >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "[swanlab-daemon $(date '+%F %T')] start interval=${SWANLAB_SYNC_INTERVAL}s projects=$SWANLAB_SYNC_PROJECTS run_ids=$SWANLAB_SYNC_RUN_IDS names=$SWANLAB_SYNC_RUN_NAMES" >>"$LOG"
while true; do
  echo "[swanlab-daemon $(date '+%F %T')] tick" >>"$LOG"
  IFS=',' read -r -a _projects <<< "$SWANLAB_SYNC_PROJECTS"
  for _proj in "${_projects[@]}"; do
    _proj="${_proj// /}"
    [ -n "$_proj" ] || continue
    echo "[swanlab-daemon $(date '+%F %T')] sync project=$_proj" >>"$LOG"
    # Per-project state so uploading to A does not skip B
    SWANLAB_SYNC_PROJECT="$_proj" \
      SWANLAB_SYNC_STATE="$REPO/swanlog/.sync_state_didpo_${_proj}.tsv" \
      bash "$REPO/scripts/swanlab_sync_watch.sh" once >>"$LOG" 2>&1 || true
  done
  sleep "$SWANLAB_SYNC_INTERVAL"
done
