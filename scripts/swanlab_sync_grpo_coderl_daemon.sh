#!/usr/bin/env bash
# Periodic SwanLab cloud sync for offline/local GRPO runs (dev box with net).
# Usage:
#   nohup bash /mnt/z4/solariewang/verl-swe/scripts/swanlab_sync_grpo_coderl_daemon.sh \
#     >/dev/null 2>&1 &
set -euo pipefail
ROOT=${ROOT:-/mnt/z4/solariewang}
if [[ ! -f "$ROOT/verl-swe/scripts/swanlab_sync_watch.sh" && -f /apdcephfs/z4/solariewang/verl-swe/scripts/swanlab_sync_watch.sh ]]; then
  ROOT=/apdcephfs/z4/solariewang
fi
REPO=${REPO:-$ROOT/verl-swe}
cd "$REPO"

# All offline runs under swanlog/ (not only .registry entries)
export SWANLAB_SYNC_ALL_RUNS=${SWANLAB_SYNC_ALL_RUNS:-1}
export SWANLAB_SYNC_PROJECT=${SWANLAB_SYNC_PROJECT:-grpo_coderl}
export SWANLAB_SYNC_INTERVAL=${SWANLAB_SYNC_INTERVAL:-1800}
# Train nodes often set HTTP proxies that break api.swanlab.cn
export SWANLAB_DISABLE_PROXY=${SWANLAB_DISABLE_PROXY:-1}

LOG="$REPO/logs/swanlab_sync_grpo_coderl.log"
PIDFILE="$REPO/logs/swanlab_sync_grpo_coderl.pid"
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

echo "[swanlab-daemon $(date '+%F %T')] start interval=${SWANLAB_SYNC_INTERVAL}s project=$SWANLAB_SYNC_PROJECT all_runs=$SWANLAB_SYNC_ALL_RUNS" >>"$LOG"
while true; do
  echo "[swanlab-daemon $(date '+%F %T')] tick" >>"$LOG"
  bash "$REPO/scripts/swanlab_sync_watch.sh" once >>"$LOG" 2>&1 || true
  sleep "$SWANLAB_SYNC_INTERVAL"
done
