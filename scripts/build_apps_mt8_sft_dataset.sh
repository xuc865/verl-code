#!/usr/bin/env bash
# Build verl multi-turn SFT parquet from collected API trajectories.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
LOG_DIR="$REPO_ROOT/logs"
OUT_DIR="$REPO_ROOT/data/sft/apps_mt8_mix"

APPS_BENCHMARK=${APPS_BENCHMARK:-apps_train}
INPUTS=()
for f in \
  "$LOG_DIR/sft_collect_${APPS_BENCHMARK}.json" \
  "$LOG_DIR/sft_collect_${APPS_BENCHMARK}_topup_introductory.json" \
  "$LOG_DIR/sft_collect_${APPS_BENCHMARK}_topup_competition.json" \
  "$LOG_DIR/sft_collect_humaneval.json" \
  "$LOG_DIR/sft_collect_mbpp.json"
do
  if [ -f "$f" ]; then
    INPUTS+=("$f")
  else
    echo "WARN: skip missing $f" >&2
  fi
done
if [ "${#INPUTS[@]}" -eq 0 ]; then
  echo "ERROR: no collection JSON found under $LOG_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python3 "$REPO_ROOT/scripts/convert_sft_trajectories_to_sharegpt.py" \
  --inputs "${INPUTS[@]}" \
  --out "$OUT_DIR/train.jsonl" \
  --val-out "$OUT_DIR/val.jsonl" \
  --parquet-out "$OUT_DIR/train.parquet" \
  --val-parquet-out "$OUT_DIR/val.parquet" \
  --outcomes won \
  --granularity step \
  --min-turns 1 \
  --max-turns 8 \
  --val-ratio 0.05

if [ ! -f "$OUT_DIR/train.parquet" ]; then
  echo "ERROR: train.parquet not created (no won episodes?)" >&2
  exit 1
fi
if [ ! -f "$OUT_DIR/val.parquet" ]; then
  echo "ERROR: val.parquet not created" >&2
  exit 1
fi

echo "SFT dataset ready (verl multiturn parquet):"
ls -lh "$OUT_DIR"/train.parquet "$OUT_DIR"/val.parquet "$OUT_DIR"/train.meta.json 2>/dev/null || true
