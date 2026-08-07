#!/usr/bin/env bash
# Download CodeRL+ data (if needed) and stage into apps_prime_extra for multi-turn RL/SFT collect.
#
# Train host:
#   bash /mnt/z4/solariewang/verl-swe/scripts/prepare_coderlplus_for_multiturn.sh
#
# After this, RL preset apps_train_coderl = APPS train + extras automatically.
# To grow multi-turn SFT parquet, collect won trajectories on the extras, e.g.:
#   EVAL_BENCHMARKS=apps_prime_extra ...  (or custom collect script)

set -euo pipefail

ROOT=${ROOT:-/mnt/z4/solariewang}
if [[ ! -d "$ROOT" ]]; then
  ROOT=/apdcephfs/z4/solariewang
fi
REPO=${REPO:-$ROOT/verl-swe}
CODERLPLUS_DATA=${CODERLPLUS_DATA:-$ROOT/CODERLPLUS/data}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$ROOT/datasets/hf_cache}
export SWEBENCH_DATA_ROOT=${SWEBENCH_DATA_ROOT:-$ROOT/datasets}
export ROOT CODERLPLUS_DATA

mkdir -p "$CODERLPLUS_DATA" "$REPO/logs"

if [[ ! -f "$CODERLPLUS_DATA/train_set.parquet" ]]; then
  echo "==> downloading xueniki/data_CodeRLPLUS → $CODERLPLUS_DATA"
  hf download xueniki/data_CodeRLPLUS --repo-type dataset --local-dir "$CODERLPLUS_DATA"
else
  echo "==> found $CODERLPLUS_DATA/train_set.parquet"
fi

echo "==> converting CodeRL+ → apps_prime_extra / apps_coderl_train"
python3 "$REPO/scripts/build_apps_train_coderl.py" "$@"

echo "==> done"
echo "  extras:  \$SWEBENCH_DATA_ROOT/codeparrot_apps_prime_extra/train.jsonl"
echo "  merged:  \$SWEBENCH_DATA_ROOT/codeparrot_apps_coderl_train/train.jsonl"
echo "  RL:      env.swebench.benchmark=apps_train_coderl"
echo "  note:    CodeRL+ has IO tests but NO gold solutions → multi-turn SFT still needs trajectory collect"
