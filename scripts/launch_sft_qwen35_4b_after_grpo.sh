#!/usr/bin/env bash
# Wait for CodeRL+ GRPO (grpo_coderl_qwen25_7b_sft_mt8) to finish, then start
# Qwen3.5-4B APPS mt8 SFT on GPUs 2–7.
#
# Usage (train host — can start while GRPO is still running):
#   nohup bash /mnt/z4/solariewang/verl-swe/scripts/launch_sft_qwen35_4b_after_grpo.sh \
#     >> /mnt/z4/solariewang/verl-swe/logs/launch_sft_qwen35_4b_after_grpo.nohup.log 2>&1 &
#
# Foreground:
#   FOREGROUND=1 bash .../launch_sft_qwen35_4b_after_grpo.sh
#
# Env:
#   GRPO_EXP_NAME   default grpo_coderl_qwen25_7b_sft_mt8
#   GRPO_TOTAL_STEPS default 100
#   POLL_SECONDS    default 120
#   SKIP_WAIT=1     start SFT immediately (no GRPO wait)
#   CUDA_VISIBLE_DEVICES default 2,3,4,5,6,7
set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=${REPO:-$ROOT/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR"

GRPO_EXP_NAME=${GRPO_EXP_NAME:-grpo_coderl_qwen25_7b_sft_mt8}
GRPO_TOTAL_STEPS=${GRPO_TOTAL_STEPS:-100}
GRPO_CKPT_DIR=${GRPO_CKPT_DIR:-$REPO/checkpoints/grpo_coderl/$GRPO_EXP_NAME}
GRPO_LOG_GLOB=${GRPO_LOG_GLOB:-$LOG_DIR/${GRPO_EXP_NAME}_resume*.nohup.log}
POLL_SECONDS=${POLL_SECONDS:-120}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}

SFT_SCRIPT=${SFT_SCRIPT:-$REPO/examples/sft/apps_mt8/run_apps_mt8_sft_qwen35_4b.sh}
SFT_LOG=${SFT_LOG:-$LOG_DIR/apps_mt8_sft_qwen35_4b_think.nohup.log}
CHAIN_LOG=${CHAIN_LOG:-$LOG_DIR/launch_sft_qwen35_4b_after_grpo.nohup.log}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_grpo_proc_alive() {
  # Prefer the trainer driver cmdline (contains trainer.experiment_name=...).
  if pgrep -af "verl.trainer.main_ppo" 2>/dev/null | grep -F "$GRPO_EXP_NAME" | grep -v grep >/dev/null 2>&1; then
    return 0
  fi
  # Fallback: the FOREGROUND launch wrapper (resume50 / mt8) stays alive for the whole run.
  if pgrep -af "launch_grpo_coderl_sft_mt8" 2>/dev/null \
    | grep -v grep | grep -v after_grpo | grep -v launch_sft_qwen35 >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

_grpo_reached_target() {
  local latest="" ptr=""
  if [[ -f "$GRPO_CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
    ptr="$(tr -d '[:space:]' <"$GRPO_CKPT_DIR/latest_checkpointed_iteration.txt" || true)"
  fi
  # Prefer pointer; also accept final step folder.
  if [[ -n "$ptr" && "$ptr" =~ ^[0-9]+$ ]] && (( ptr >= GRPO_TOTAL_STEPS )); then
    return 0
  fi
  if [[ -d "$GRPO_CKPT_DIR/global_step_${GRPO_TOTAL_STEPS}" ]]; then
    return 0
  fi
  # Log fallback: last "Training Progress: …| N/TOTAL"
  # shellcheck disable=SC2086
  latest="$(ls -t $GRPO_LOG_GLOB 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" && -f "$latest" ]]; then
    python3 - "$latest" "$GRPO_TOTAL_STEPS" <<'PY'
import re, sys
path, target = sys.argv[1], int(sys.argv[2])
txt = open(path, errors="ignore").read()
steps = [int(a) for a, b in re.findall(r"Training Progress:.*?\|?\s*(\d+)/(\d+)", txt)]
# also step:N metrics lines
steps += [int(x) for x in re.findall(r"\bstep:(\d+)\s+-", txt)]
sys.exit(0 if steps and max(steps) >= target else 1)
PY
    return $?
  fi
  return 1
}

_target_gpus_still_busy() {
  # After GRPO exits, Ray workers can briefly hold GPUs 2–7. Don't start SFT until clear
  # (ignore vLLM on 0/1 used by eval).
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local gpu_csv="$CUDA_VISIBLE_DEVICES"
  python3 - "$gpu_csv" <<'PY'
import subprocess, sys
want = {int(x) for x in sys.argv[1].split(',') if x.strip() != ''}
try:
    q = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,process_name', '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.DEVNULL,
    )
    idx = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,gpu_bus_id', '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.DEVNULL,
    )
except Exception:
    sys.exit(1)
bus2idx = {}
for line in idx.splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) >= 2 and parts[0].isdigit():
        bus2idx[parts[1]] = int(parts[0])
busy = set()
for line in q.splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 3:
        continue
    bus, pid, name = parts[0], parts[1], parts[2].lower()
    gi = bus2idx.get(bus)
    if gi is None or gi not in want:
        continue
    if 'vllm' in name or 'placeholder' in name:
        continue
    busy.add(gi)
sys.exit(0 if busy else 1)
PY
}

wait_for_grpo() {
  if [[ "${SKIP_WAIT:-0}" == "1" ]]; then
    echo "[$(_ts)] SKIP_WAIT=1 — not waiting for GRPO"
    return 0
  fi

  echo "[$(_ts)] waiting for GRPO exp=$GRPO_EXP_NAME to finish (target_steps=$GRPO_TOTAL_STEPS)"
  echo "  ckpt_dir=$GRPO_CKPT_DIR"
  echo "  poll every ${POLL_SECONDS}s"

  local saw_alive=0
  while true; do
    local alive=0 done=0
    if _grpo_proc_alive; then
      alive=1
      saw_alive=1
    fi
    if _grpo_reached_target; then
      done=1
    fi

    if (( done == 1 && alive == 0 )); then
      echo "[$(_ts)] GRPO finished (reached step>=$GRPO_TOTAL_STEPS and no trainer proc)."
      break
    fi

    # If we never saw the job but ckpt already at target, proceed.
    if (( done == 1 && saw_alive == 0 )); then
      echo "[$(_ts)] GRPO already complete (ckpt/log shows step>=$GRPO_TOTAL_STEPS)."
      break
    fi

    # Trainer died before target — abort rather than silently SFT on partial RL.
    if (( saw_alive == 1 && alive == 0 && done == 0 )); then
      echo "[$(_ts)] ERROR: GRPO process exited before step $GRPO_TOTAL_STEPS." >&2
      echo "  Check $GRPO_LOG_GLOB and $GRPO_CKPT_DIR/latest_checkpointed_iteration.txt" >&2
      echo "  Override with SKIP_WAIT=1 if you still want to start SFT." >&2
      exit 2
    fi

    local ptr="?"
    [[ -f "$GRPO_CKPT_DIR/latest_checkpointed_iteration.txt" ]] \
      && ptr="$(tr -d '[:space:]' <"$GRPO_CKPT_DIR/latest_checkpointed_iteration.txt")"
    echo "[$(_ts)] still waiting… alive=$alive ptr=$ptr target=$GRPO_TOTAL_STEPS"
    sleep "$POLL_SECONDS"
  done

  # Give Ray/GPU time to release cards 2–7 before torchrun.
  echo "[$(_ts)] cool-down / wait for GPUs $CUDA_VISIBLE_DEVICES to free…"
  local i
  for i in $(seq 1 30); do
    if ! _target_gpus_still_busy; then
      echo "[$(_ts)] target GPUs look free (after ${i} checks)."
      break
    fi
    echo "[$(_ts)] target GPUs still busy ($i/30), sleep 20s…"
    sleep 20
  done
  sleep 15
}

start_sft() {
  if [[ ! -x "$SFT_SCRIPT" && ! -f "$SFT_SCRIPT" ]]; then
    echo "ERROR: missing SFT script: $SFT_SCRIPT" >&2
    exit 1
  fi
  chmod +x "$SFT_SCRIPT" 2>/dev/null || true

  echo "[$(_ts)] starting Qwen3.5-4B SFT"
  echo "  script=$SFT_SCRIPT"
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "  log=$SFT_LOG"

  nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    bash "$SFT_SCRIPT" >>"$SFT_LOG" 2>&1 &
  local pid=$!
  echo "[$(_ts)] SFT PID=$pid  log=$SFT_LOG"
  echo "  tail -f $SFT_LOG"
}

# ---- entry ----
if [[ "${FOREGROUND:-0}" != "1" && "${_CHAIN_INNER:-0}" != "1" ]]; then
  nohup env _CHAIN_INNER=1 FOREGROUND=1 \
    ROOT="$ROOT" REPO="$REPO" \
    GRPO_EXP_NAME="$GRPO_EXP_NAME" GRPO_TOTAL_STEPS="$GRPO_TOTAL_STEPS" \
    POLL_SECONDS="$POLL_SECONDS" SKIP_WAIT="${SKIP_WAIT:-0}" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    bash "$0" >>"$CHAIN_LOG" 2>&1 &
  echo "PID=$!  chain_log=$CHAIN_LOG"
  echo "  waits for GRPO($GRPO_EXP_NAME)->${GRPO_TOTAL_STEPS}, then SFT on GPUs $CUDA_VISIBLE_DEVICES"
  echo "  tail -f $CHAIN_LOG"
  exit 0
fi

cd "$REPO"
wait_for_grpo
start_sft
echo "[$(_ts)] chain launcher done (SFT detached)."
