#!/usr/bin/env bash
# Wait for Qwen3.5-4B APPS mt8 SFT to finish, then start CodeRL+ GRPO on that ckpt.
#
# Usage (train host — can start while SFT is still running):
#   nohup bash /mnt/z4/solariewang/verl-swe/scripts/launch_grpo_qwen35_4b_after_sft.sh \
#     >> /mnt/z4/solariewang/verl-swe/logs/launch_grpo_qwen35_4b_after_sft.nohup.log 2>&1 &
#
# Foreground:
#   FOREGROUND=1 bash .../launch_grpo_qwen35_4b_after_sft.sh
#
# Env:
#   SFT_SAVE_DIR     default checkpoints/apps_mt8_sft_qwen35_4b_think
#   SFT_LOG          default logs/apps_mt8_sft_qwen35_4b_think.nohup.log
#   SFT_MIN_EPOCHS   default 2  (require this many global_step_* dirs, or "SFT done" in log)
#   POLL_SECONDS     default 120
#   SKIP_WAIT=1      start GRPO immediately (still resolves latest SFT ckpt)
#   CUDA_VISIBLE_DEVICES default 2,3,4,5,6,7
set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
REPO=${REPO:-$ROOT/verl-swe}
LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR"

SFT_SAVE_DIR=${SFT_SAVE_DIR:-$REPO/checkpoints/apps_mt8_sft_qwen35_4b_think}
SFT_LOG=${SFT_LOG:-$LOG_DIR/apps_mt8_sft_qwen35_4b_think.nohup.log}
SFT_MIN_EPOCHS=${SFT_MIN_EPOCHS:-2}
POLL_SECONDS=${POLL_SECONDS:-120}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}

GRPO_SCRIPT=${GRPO_SCRIPT:-$REPO/scripts/launch_grpo_coderl_qwen35_4b_sft_mt8.sh}
CHAIN_LOG=${CHAIN_LOG:-$LOG_DIR/launch_grpo_qwen35_4b_after_sft.nohup.log}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_sft_proc_alive() {
  if pgrep -af "verl.trainer.fsdp_sft_trainer" 2>/dev/null \
    | grep -E 'qwen35-4b-mt8-think|apps_mt8_sft_qwen35_4b' \
    | grep -v grep >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -af "run_apps_mt8_sft_qwen35_4b" 2>/dev/null | grep -v grep >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

_count_sft_ckpts() {
  local n=0
  shopt -s nullglob
  local d
  for d in "$SFT_SAVE_DIR"/global_step_*; do
    [[ -f "$d/config.json" ]] && n=$((n + 1))
  done
  shopt -u nullglob
  echo "$n"
}

_latest_sft_ckpt() {
  # Full paths contain many '_' (e.g. apps_mt8_sft_...); sort by trailing step number only.
  local best="" best_n=-1 d n
  shopt -s nullglob
  for d in "$SFT_SAVE_DIR"/global_step_*; do
    n="${d##*_}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if (( n > best_n )); then
      best_n=$n
      best=$d
    fi
  done
  shopt -u nullglob
  printf '%s' "$best"
}

_sft_log_says_done() {
  [[ -f "$SFT_LOG" ]] || return 1
  grep -q 'SFT done\. Checkpoint under:' "$SFT_LOG" 2>/dev/null
}

_sft_ckpt_is_qwen35() {
  local p="$1"
  [[ -f "$p/config.json" ]] || return 1
  grep -q 'Qwen3_5\|qwen3_5\|Qwen3.5' "$p/config.json" 2>/dev/null
}

_sft_finished_ok() {
  if _sft_log_says_done; then
    return 0
  fi
  local n
  n="$(_count_sft_ckpts)"
  (( n >= SFT_MIN_EPOCHS ))
}

_target_gpus_still_busy() {
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

wait_for_sft() {
  if [[ "${SKIP_WAIT:-0}" == "1" ]]; then
    echo "[$(_ts)] SKIP_WAIT=1 — not waiting for SFT"
    return 0
  fi

  echo "[$(_ts)] waiting for SFT to finish"
  echo "  save_dir=$SFT_SAVE_DIR  min_ckpts=$SFT_MIN_EPOCHS"
  echo "  sft_log=$SFT_LOG  poll=${POLL_SECONDS}s"

  local saw_alive=0
  while true; do
    local alive=0 done=0
    if _sft_proc_alive; then
      alive=1
      saw_alive=1
    fi
    if _sft_finished_ok; then
      done=1
    fi

    if (( done == 1 && alive == 0 )); then
      echo "[$(_ts)] SFT finished (ckpts/log OK, no trainer proc)."
      break
    fi

    if (( done == 1 && saw_alive == 0 )); then
      echo "[$(_ts)] SFT already complete (found >=$SFT_MIN_EPOCHS ckpts or done marker)."
      break
    fi

    if (( saw_alive == 1 && alive == 0 && done == 0 )); then
      echo "[$(_ts)] ERROR: SFT process exited before completion marker / enough ckpts." >&2
      echo "  ckpts=$(_count_sft_ckpts) need>=$SFT_MIN_EPOCHS  log=$SFT_LOG" >&2
      echo "  Override with SKIP_WAIT=1 if you still want to start GRPO." >&2
      exit 2
    fi

    echo "[$(_ts)] still waiting… alive=$alive ckpts=$(_count_sft_ckpts) target>=$SFT_MIN_EPOCHS"
    sleep "$POLL_SECONDS"
  done

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

_ensure_qwen35_sidecars() {
  # SFT save may omit multimodal processor files; copy from base before GRPO/vLLM load.
  local ckpt="$1"
  local base="${SFT_BASE_MODEL:-$ROOT/models/Qwen3.5-4B}"
  local f
  for f in preprocessor_config.json video_preprocessor_config.json generation_config.json merges.txt vocab.json; do
    if [[ -f "$base/$f" && ! -f "$ckpt/$f" ]]; then
      cp -f "$base/$f" "$ckpt/$f"
      echo "[$(_ts)] copied sidecar $f -> $ckpt"
    fi
  done
  # Minimal loadability: weights + config + tokenizer
  if [[ ! -f "$ckpt/config.json" ]]; then
    echo "ERROR: ckpt missing config.json: $ckpt" >&2
    return 1
  fi
  # Avoid `ls | grep -q` under pipefail: SIGPIPE makes ls non-zero even when weights exist.
  local _w=()
  shopt -s nullglob
  _w=("$ckpt"/model*.safetensors "$ckpt"/pytorch_model*.bin)
  shopt -u nullglob
  if (( ${#_w[@]} == 0 )); then
    echo "ERROR: ckpt missing model weights under $ckpt" >&2
    return 1
  fi
  if [[ ! -f "$ckpt/tokenizer.json" && ! -f "$ckpt/tokenizer_config.json" ]]; then
    echo "ERROR: ckpt missing tokenizer files: $ckpt" >&2
    return 1
  fi
  return 0
}

start_grpo() {
  local ckpt
  ckpt="$(_latest_sft_ckpt)"
  if [[ -z "$ckpt" || ! -f "$ckpt/config.json" ]]; then
    echo "ERROR: no SFT ckpt under $SFT_SAVE_DIR" >&2
    exit 1
  fi
  if ! _sft_ckpt_is_qwen35 "$ckpt"; then
    echo "ERROR: latest SFT ckpt is not Qwen3.5 — refusing to start GRPO: $ckpt" >&2
    grep -E 'architectures|model_type' "$ckpt/config.json" >&2 || true
    echo "  (Current SFT may have been started with a wrong MODEL_PATH; restart SFT from base Qwen3.5-4B.)" >&2
    exit 3
  fi
  if ! _ensure_qwen35_sidecars "$ckpt"; then
    exit 3
  fi

  chmod +x "$GRPO_SCRIPT" 2>/dev/null || true
  echo "[$(_ts)] starting GRPO from SFT ckpt"
  echo "  model=$ckpt"
  echo "  script=$GRPO_SCRIPT"
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "  (vLLM language_model_only=True is set in $GRPO_SCRIPT)"

  # Launch GRPO via its own nohup wrapper (FOREGROUND unset).
  env MODEL_PATH="$ckpt" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    FOREGROUND=0 \
    bash "$GRPO_SCRIPT"
}

# ---- entry ----
if [[ "${FOREGROUND:-0}" != "1" && "${_CHAIN_INNER:-0}" != "1" ]]; then
  nohup env _CHAIN_INNER=1 FOREGROUND=1 \
    ROOT="$ROOT" REPO="$REPO" \
    SFT_SAVE_DIR="$SFT_SAVE_DIR" SFT_LOG="$SFT_LOG" \
    SFT_MIN_EPOCHS="$SFT_MIN_EPOCHS" POLL_SECONDS="$POLL_SECONDS" \
    SKIP_WAIT="${SKIP_WAIT:-0}" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    bash "$0" >>"$CHAIN_LOG" 2>&1 &
  echo "PID=$!  chain_log=$CHAIN_LOG"
  echo "  waits for SFT($SFT_SAVE_DIR)->${SFT_MIN_EPOCHS} ckpts, then GRPO on GPUs $CUDA_VISIBLE_DEVICES"
  echo "  tail -f $CHAIN_LOG"
  exit 0
fi

cd "$REPO"
wait_for_sft
start_grpo
echo "[$(_ts)] chain launcher done (GRPO detached)."
