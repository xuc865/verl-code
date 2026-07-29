#!/usr/bin/env bash
# Top-up APPS SFT collection for introductory / competition to balance won distribution.
#
# Excludes instance_ids that already have outcome=won in the base collection JSON
# (per difficulty tier), then random-samples fresh problems from the remaining pool.
#
# Usage (training machine):
#   cd /apdcephfs/z4/solariewang/verl-swe   # or /mnt/z4/...
#   TOPUP_DIFFICULTIES=competition \
#     nohup bash scripts/collect_api_sft_apps_topup.sh \
#       >> logs/collect_api_sft_apps_topup_competition.nohup.log 2>&1 &

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/apdcephfs/z4/solariewang/verl-swe}
# Prefer /mnt bind if present
if [ -d /mnt/z4/solariewang/verl-swe ]; then
  REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
fi
DATA_ROOT=${DATA_ROOT:-/apdcephfs/z4/solariewang/datasets}
if [ -d /mnt/z4/solariewang/datasets ]; then
  DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
fi
LOG_DIR="$REPO_ROOT/logs"
EVAL_PY="$REPO_ROOT/scripts/eval_api_baseline.py"

API_BASE=${API_BASE:-http://29.116.237.141:8080/v1}
MODEL=${MODEL:-Qwen3.6-27B}
METHOD=${METHOD:-qwen36_mt8_sft_topup}

TEST_FEEDBACK_MODE=${TEST_FEEDBACK_MODE:-exec}
EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-8}
EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-5}
MAX_TOKENS=${MAX_TOKENS:-4096}
DISABLE_THINKING=${DISABLE_THINKING:-1}
API_TIMEOUT=${API_TIMEOUT:-1800}
API_RETRIES=${API_RETRIES:-3}
EVAL_WORKERS=${EVAL_WORKERS:-6}
EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
SFT_MAX_ROLLOUTS=${SFT_MAX_ROLLOUTS:-5}

APPS_BENCHMARK=${APPS_BENCHMARK:-apps_train}
BASE_JSON=${BASE_JSON:-$LOG_DIR/sft_collect_apps_train.json}
EXCLUDE_FROM_MODE=${EXCLUDE_FROM_MODE:-won}
TOPUP_DIFFICULTIES=${TOPUP_DIFFICULTIES:-introductory,competition}

INTRO_TOPUP_SIZE=${INTRO_TOPUP_SIZE:-1500}
INTRO_TOPUP_SEED=${INTRO_TOPUP_SEED:-1042}
COMP_TOPUP_SIZE=${COMP_TOPUP_SIZE:-220}
COMP_TOPUP_SEED=${COMP_TOPUP_SEED:-2042}

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_topup() {
  local diff="$1"
  local sample_size="$2"
  local sample_seed="$3"
  local out_json="$LOG_DIR/sft_collect_${APPS_BENCHMARK}_topup_${diff}.json"

  echo "[$(_ts)] TOPUP difficulty=$diff sample=$sample_size seed=$sample_seed"
  echo "  exclude_from=$BASE_JSON mode=$EXCLUDE_FROM_MODE"
  echo "  out=$out_json rollouts=$SFT_MAX_ROLLOUTS"

  if [ ! -f "$EVAL_PY" ]; then
    echo "ERROR: missing $EVAL_PY — restore scripts/eval_api_baseline.py first" >&2
    exit 1
  fi

  export PYTHONUNBUFFERED=1
  export EXPORT_FULL_TRANSCRIPT=1
  export SFT_MAX_ROLLOUTS
  export EVAL_EXCLUDE_FROM="$BASE_JSON"
  export EVAL_EXCLUDE_FROM_MODE="$EXCLUDE_FROM_MODE"
  # Keep fn_name -> stdin conversion on (apps adapter)
  export APPS_CONVERT_FN_NAME_TO_STDIN=${APPS_CONVERT_FN_NAME_TO_STDIN:-1}

  python3 "$EVAL_PY" \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --method-name "${METHOD}_${diff}" \
    --benchmark "$APPS_BENCHMARK" \
    --data-root "$DATA_ROOT" \
    --difficulty-filter "$diff" \
    --sample-size "$sample_size" \
    --sample-seed "$sample_seed" \
    --max-turns "$EVAL_MAX_TURNS" \
    --history-length "$EVAL_HISTORY_LENGTH" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --test-feedback-mode "$TEST_FEEDBACK_MODE" \
    --api-timeout "$API_TIMEOUT" \
    --api-retries "$API_RETRIES" \
    --checkpoint-every "$EVAL_CHECKPOINT_EVERY" \
    --workers "$EVAL_WORKERS" \
    --out "$out_json" \
    --export-full-transcript \
    --max-rollouts-per-instance "$SFT_MAX_ROLLOUTS" \
    --disable-thinking \
    --resume
}

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" data/sft

if [ ! -f "$BASE_JSON" ]; then
  echo "ERROR: base collection not found: $BASE_JSON" >&2
  exit 1
fi

echo "[$(_ts)] ========== APPS SFT top-up collection =========="
echo "  teacher=$MODEL @ $API_BASE"
echo "  repo=$REPO_ROOT"
echo "  base=$BASE_JSON exclude_mode=$EXCLUDE_FROM_MODE"
echo "  tiers=$TOPUP_DIFFICULTIES workers=$EVAL_WORKERS rollouts=$SFT_MAX_ROLLOUTS"

IFS=',' read -r -a _tiers <<< "$TOPUP_DIFFICULTIES"
for diff in "${_tiers[@]}"; do
  diff="$(echo "$diff" | xargs)"
  [ -z "$diff" ] && continue
  case "$diff" in
    introductory)
      run_topup "$diff" "$INTRO_TOPUP_SIZE" "$INTRO_TOPUP_SEED"
      ;;
    competition)
      run_topup "$diff" "$COMP_TOPUP_SIZE" "$COMP_TOPUP_SEED"
      ;;
    *)
      echo "WARN: skip unknown difficulty tier: $diff" >&2
      ;;
  esac
done

echo "[$(_ts)] ========== top-up done — run build_apps_mt8_sft_dataset.sh =========="
