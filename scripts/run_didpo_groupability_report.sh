#!/usr/bin/env bash
# Enable GRPO/DIDPO multi-rollout dumps for offline DIDPO groupability analysis.
# Appends trainer.rollout_data_dir — note: default verl dump is prompt/response/score
# per *step row*, not yet uid-grouped. Prefer collecting via:
#   python3 scripts/analyze_didpo_groupability.py --groups-json <groups.json>
# where groups.json follows the schema in analyze_didpo_groupability.py.
#
# For the next DIDPO train run, also log SwanLab; after a few steps, export
# agent transcripts into groups.json (helper TBD once DIDPO train is up).
#
# Quick methodology check (no GPU):
#   python3 scripts/analyze_didpo_groupability.py --demo --out logs/didpo_groupability_report.json

set -euo pipefail
ROOT=${ROOT:-/mnt/z4/solariewang/verl-swe}
cd "$ROOT"
python3 scripts/analyze_didpo_groupability.py --demo --group-size "${GROUP_SIZE:-32}" \
  ${EVAL_JSON:+--eval-json "$EVAL_JSON"} \
  --out "${OUT:-$ROOT/logs/didpo_groupability_report.json}"
echo "Open / refresh canvas: DIDPO Groupability (or re-run analyzer after groups dump)."
