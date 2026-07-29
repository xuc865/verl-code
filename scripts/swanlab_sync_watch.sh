#!/bin/bash
# Watch local SwanLab runs (offline mode) and sync new/changed logs to cloud.
#
# One-shot (for cron):
#   bash /mnt/z4/solariewang/verl-swe/scripts/swanlab_sync_watch.sh once
#
# Daemon on DEV machine only (training nodes must not run sync):
#   nohup bash /mnt/z4/solariewang/verl-swe/scripts/swanlab_sync_watch.sh loop &
#
# Cron (hourly at :05):
#   5 * * * * bash /mnt/z4/solariewang/verl-swe/scripts/swanlab_sync_watch.sh once
#
# Optional env:
#   SWANLAB_SYNC_INTERVAL=3600   # loop sleep seconds (default 1h)
#   SWANLAB_SYNC_PROJECT=grpo_coderl
#   SWANLAB_SYNC_ALL_RUNS=1      # sync every run-* (recommended for offline SFT→GRPO)
#   SWANLAB_SYNC_RUN_IDS=id1,id2 # only sync these run ids (comma-separated)
#   SWANLAB_SYNC_RUN_NAMES=run-... # only these run dir basenames (comma-separated)
#   CONDA_ENV=verl-agent          # only if conda is available; dev box can use system python3
set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
if [ ! -f "$ROOT/verl-swe/scripts/swanlab_env.sh" ] && [ -f "/apdcephfs/z4/solariewang/verl-swe/scripts/swanlab_env.sh" ]; then
    ROOT=/apdcephfs/z4/solariewang
fi
REPO=${REPO:-$ROOT/verl-swe}
SYNC_LOG=${SWANLAB_SYNC_LOG:-$REPO/logs/swanlab_sync.log}
mkdir -p "$REPO/logs"
CONDA_ENV=${CONDA_ENV:-verl-agent}
MODE=${1:-once}
INTERVAL=${SWANLAB_SYNC_INTERVAL:-3600}
PROJECT=${SWANLAB_SYNC_PROJECT:-grpo_coderl}
LOCK_FILE=${SWANLAB_SYNC_LOCK:-/tmp/swanlab_sync_watch.lock}
STATE_FILE=${SWANLAB_SYNC_STATE:-}
LOG_TAG="[swanlab-sync $(date '+%F %T')]"

# shellcheck source=/dev/null
source "$REPO/scripts/swanlab_env.sh"
STATE_FILE=${STATE_FILE:-$SWANLAB_LOG_DIR/.sync_state.tsv}

# Training writes absolute paths under /mnt/z4/solariewang; on dev box the same
# Ceph data may live under /apdcephfs/z4/solariewang. SwanLab sync reads those
# recorded paths for config/metadata upload — symlink so they resolve.
_ensure_train_path_aliases() {
    local train_root=/mnt/z4/solariewang
    local train_repo="$train_root/verl-swe"
    local train_swanlog="$train_repo/swanlog"
    if [ "$ROOT" = "$train_root" ]; then
        return 0
    fi
    if [ -f "$train_repo/scripts/swanlab_env.sh" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$train_root")"
    if [ -e "$train_root" ] && [ ! -L "$train_root" ]; then
        if [ -d "$train_root/verl-swe" ] && [ ! -f "$train_repo/scripts/swanlab_env.sh" ]; then
            rm -rf "$train_root"
            ln -sfn "$ROOT" "$train_root"
            echo "$LOG_TAG linked $train_root -> $ROOT"
            return 0
        fi
    fi
    if [ ! -e "$train_root" ]; then
        ln -sfn "$ROOT" "$train_root"
        echo "$LOG_TAG linked $train_root -> $ROOT"
        return 0
    fi
    mkdir -p "$train_repo"
    if [ ! -e "$train_swanlog" ] || [ -L "$train_swanlog" ]; then
        ln -sfn "$SWANLAB_LOG_DIR" "$train_swanlog"
        echo "$LOG_TAG linked $train_swanlog -> $SWANLAB_LOG_DIR"
    fi
}

_setup_python() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    fi
    if python3 -c "import swanlab" 2>/dev/null; then
        PY=python3
    elif python -c "import swanlab" 2>/dev/null; then
        PY=python
    else
        echo "$LOG_TAG installing swanlab via pip ..."
        python3 -m pip install -q 'swanlab>=0.4.0' || pip install -q 'swanlab>=0.4.0'
        PY=python3
    fi
    if ! "$PY" -c "import swanlab" 2>/dev/null; then
        echo "$LOG_TAG ERROR: swanlab not available (install: pip install swanlab)" >&2
        return 1
    fi
    if ! command -v swanlab >/dev/null 2>&1; then
        echo "$LOG_TAG ERROR: swanlab CLI not on PATH" >&2
        return 1
    fi
}

_extract_run_id() {
    local run_name="$1"
    # run-YYYYMMDD_HHMMSS-<run_id>
    if [[ "$run_name" =~ ^run-[0-9]{8}_[0-9]{6}-(.+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

_collect_registry_runs() {
    # Prints lines: run_dir<TAB>run_id<TAB>experiment_name
    local reg_dir="$SWANLAB_LOG_DIR/.registry"
    [ -d "$reg_dir" ] || return 0
    local f
    for f in "$reg_dir"/*.json; do
        [ -f "$f" ] || continue
        "$PY" -c "
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
run_dir = data.get('run_dir', '')
run_id = data.get('run_id', '')
name = data.get('experiment_name', '')
if run_dir and run_id:
    print(f'{run_dir}\t{run_id}\t{name}')
" "$f" 2>/dev/null || true
    done
}

_run_once() {
    mkdir -p "$SWANLAB_LOG_DIR" "$(dirname "$STATE_FILE")" "$REPO/logs"
    touch "$STATE_FILE"

    if [ -z "${SWANLAB_API_KEY:-}" ]; then
        echo "$LOG_TAG ERROR: SWANLAB_API_KEY missing (need cloud sync credentials)" >&2
        return 1
    fi

    _setup_python || return 1
    _ensure_train_path_aliases

    "$PY" -c "import os, swanlab; swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=False)"

    local synced=0 skipped=0 failed=0
    local -a targets=()

    if [ "${SWANLAB_SYNC_ALL_RUNS:-0}" != "1" ]; then
        # Prefer canonical runs from .registry (one curve per experiment on cloud).
        while IFS=$'\t' read -r run_dir run_id _exp; do
            [ -n "$run_dir" ] && [ -d "$run_dir" ] && targets+=("${run_dir}|${run_id}")
        done < <(_collect_registry_runs)
    fi

    if [ ${#targets[@]} -eq 0 ]; then
        shopt -s nullglob
        local runs=("$SWANLAB_LOG_DIR"/run-*)
        shopt -u nullglob
        for run_dir in "${runs[@]}"; do
            [ -d "$run_dir" ] || continue
            local run_name run_id
            run_name="$(basename "$run_dir")"
            run_id="$(_extract_run_id "$run_name" || true)"
            [ -n "$run_id" ] || continue
            targets+=("${run_dir}|${run_id}")
        done
    fi

    if [ -n "${SWANLAB_SYNC_RUN_IDS:-}" ]; then
        local -a filtered=()
        local want entry_f rid_f
        for entry_f in "${targets[@]}"; do
            rid_f="${entry_f##*|}"
            # strip optional .fix suffix used by remapped dirs
            rid_f="${rid_f%.fix}"
            IFS=',' read -r -a want <<< "$SWANLAB_SYNC_RUN_IDS"
            local id
            for id in "${want[@]}"; do
                id="${id// /}"
                [ -n "$id" ] || continue
                if [ "$rid_f" = "$id" ]; then
                    filtered+=("$entry_f")
                    break
                fi
            done
        done
        targets=("${filtered[@]}")
    fi

    if [ -n "${SWANLAB_SYNC_RUN_NAMES:-}" ]; then
        local -a filtered=()
        local entry_f run_dir_f run_name_f
        IFS=',' read -r -a want_names <<< "$SWANLAB_SYNC_RUN_NAMES"
        for entry_f in "${targets[@]}"; do
            run_dir_f="${entry_f%%|*}"
            run_name_f="$(basename "$run_dir_f")"
            local nm
            for nm in "${want_names[@]}"; do
                nm="${nm// /}"
                [ -n "$nm" ] || continue
                if [ "$run_name_f" = "$nm" ]; then
                    filtered+=("$entry_f")
                    break
                fi
            done
        done
        targets=("${filtered[@]}")
    fi

    if [ ${#targets[@]} -eq 0 ]; then
        echo "$LOG_TAG no runs to sync under $SWANLAB_LOG_DIR"
        return 0
    fi

    local entry run_dir run_id run_name mtime last_mtime
    for entry in "${targets[@]}"; do
        run_dir="${entry%%|*}"
        run_id="${entry##*|}"
        if [ ! -d "$run_dir" ]; then
            local cand
            for cand in \
                "$run_dir" \
                "${run_dir//\/mnt\/z4\/solariewang/$ROOT}" \
                "${run_dir//\/apdcephfs\/z4\/solariewang/$ROOT}" \
                "${run_dir//$ROOT/\/mnt\/z4\/solariewang}" \
                "${run_dir//$ROOT/\/apdcephfs\/z4\/solariewang}"; do
                if [ -n "$cand" ] && [ -d "$cand" ]; then
                    run_dir="$cand"
                    break
                fi
            done
        fi
        [ -d "$run_dir" ] || continue
        run_name="$(basename "$run_dir")"
        # Optional remap: cloud run id may differ from local folder suffix when a
        # full re-upload created a fresh id (e.g. zbavplj0 -> zbavplj0full).
        # File format (TSV): local_id<TAB>cloud_id
        local id_map="${SWANLAB_SYNC_ID_MAP:-$SWANLAB_LOG_DIR/.sync_id_map.tsv}"
        if [ -f "$id_map" ]; then
            local mapped
            mapped="$(awk -F'\t' -v r="$run_id" '$1 == r {print $2; exit}' "$id_map")"
            if [ -n "${mapped:-}" ]; then
                echo "$LOG_TAG remap run_id $run_id -> $mapped"
                run_id="$mapped"
            fi
        fi
        mtime="$(find "$run_dir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
        mtime="${mtime:-0}"
        last_mtime="$(awk -F'\t' -v r="$run_name" '$1 == r {print $2}' "$STATE_FILE" | tail -1)"
        last_mtime="${last_mtime:-0}"

        if awk -v a="$mtime" -v b="$last_mtime" 'BEGIN { exit !(a > b + 1) }'; then
            :
        else
            echo "$LOG_TAG skip $run_name (mtime=$mtime last=$last_mtime)"
            skipped=$((skipped + 1))
            continue
        fi

        # Snapshot before sync: live writers append to *.swanlab; concurrent
        # `swanlab sync` races and can stop uploading training metrics (only a
        # few hardware records thereafter). Copy then sync the frozen tree.
        local snap_root snap_dir sync_src
        snap_root="${SWANLAB_SYNC_SNAP_DIR:-/tmp/swanlab_sync_snap}"
        snap_dir="$snap_root/$run_name.$$"
        sync_src="$run_dir"
        mkdir -p "$snap_root"
        rm -rf "$snap_dir"

        # If .swanlab datastore is corrupt but logs/ still grow, rebuild a clean
        # offline run from logs/<col>/1000.log before syncing.
        if [ -f "$run_dir/.rebuild_from_logs" ]; then
            local rebuild_dir exp_name
            rebuild_dir="$snap_root/rebuild-$run_id.$$"
            exp_name="$(
                "$PY" -c "
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'files'/'swanlab-metadata.json'
name=''
if p.exists():
    try: name=json.loads(p.read_text()).get('name','') or ''
    except Exception: pass
print(name)
" "$run_dir" 2>/dev/null || true
            )"
            exp_name="${SWANLAB_SYNC_EXP_NAME:-${exp_name:-$run_name}}"
            echo "$LOG_TAG rebuild-from-logs $run_name -> $rebuild_dir id=$run_id name=$exp_name"
            if "$PY" "$REPO/scripts/swanlab_rebuild_from_logs.py" \
                --src "$run_dir" --out "$rebuild_dir" --run-id "$run_id" \
                --project "$PROJECT" --name "$exp_name"; then
                sync_src="$rebuild_dir"
            else
                echo "$LOG_TAG WARN: rebuild-from-logs failed; falling back to snapshot" >&2
            fi
        fi

        if [ "$sync_src" = "$run_dir" ]; then
            if mkdir -p "$snap_dir" && cp -a "$run_dir"/. "$snap_dir"/ 2>/dev/null; then
                sync_src="$snap_dir"
                echo "$LOG_TAG snapshot $run_name -> $snap_dir"
            else
                echo "$LOG_TAG WARN: snapshot failed for $run_name; syncing live dir (race-prone)" >&2
            fi
        fi

        echo "$LOG_TAG syncing $run_name id=$run_id -> project=$PROJECT (src=$sync_src)"
        if swanlab sync "$sync_src" -p "$PROJECT" -i "$run_id"; then
            grep -v "^${run_name}	" "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null || true
            mv "${STATE_FILE}.tmp" "$STATE_FILE"
            # Re-stat after sync so we don't immediately re-upload unchanged bytes;
            # next tick will pick up further writer growth.
            mtime="$(find "$run_dir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
            mtime="${mtime:-0}"
            printf '%s\t%s\t%s\tok\n' "$run_name" "$mtime" "$(date +%s)" >> "$STATE_FILE"
            synced=$((synced + 1))
        else
            printf '%s\t%s\t%s\tfail\n' "$run_name" "$mtime" "$(date +%s)" >> "$STATE_FILE"
            echo "$LOG_TAG FAIL $run_name" >&2
            failed=$((failed + 1))
        fi
        rm -rf "$snap_dir" 2>/dev/null || true
        rm -rf "$snap_root/rebuild-$run_id.$$" 2>/dev/null || true
    done

    echo "$LOG_TAG done synced=$synced skipped=$skipped failed=$failed logdir=$SWANLAB_LOG_DIR"
    [ "$failed" -eq 0 ]
}

_with_lock() {
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$LOCK_FILE"
        if ! flock -n 9; then
            holder="$(fuser "$LOCK_FILE" 2>/dev/null | tr -s ' ' || true)"
            if [ -n "$holder" ]; then
                echo "$LOG_TAG another sync is running (pid:$holder), exit"
                exit 0
            fi
            echo "$LOG_TAG stale lock at $LOCK_FILE, clearing"
            rm -f "$LOCK_FILE"
            exec 9>"$LOCK_FILE"
            flock -n 9 || {
                echo "$LOG_TAG failed to acquire lock after cleanup, exit"
                exit 0
            }
        fi
    fi
    "$@"
}

case "$MODE" in
    once)
        _with_lock _run_once
        ;;
    loop)
        _pids=($(pgrep -f "swanlab_sync_watch.sh loop" 2>/dev/null || true))
        for _p in "${_pids[@]}"; do
            if [ "$_p" != "$$" ]; then
                echo "$LOG_TAG sync daemon already running (pid $_p) — not starting a second one"
                exit 0
            fi
        done
        mkdir -p "$(dirname "$SYNC_LOG")"
        # tee when attached to tmux/terminal; log-only when fully detached (nohup/cron).
        if [ -t 1 ]; then
            exec > >(tee -a "$SYNC_LOG") 2>&1
        else
            exec >>"$SYNC_LOG" 2>&1
        fi
        echo "$LOG_TAG daemon start interval=${INTERVAL}s logdir=$SWANLAB_LOG_DIR log=$SYNC_LOG"
        while true; do
            _with_lock _run_once || true
            echo "$LOG_TAG sleeping ${INTERVAL}s until next sync ..."
            sleep "$INTERVAL"
        done
        ;;
    *)
        echo "Usage: $0 [once|loop]" >&2
        exit 2
        ;;
esac
