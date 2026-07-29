#!/usr/bin/env bash
set -euo pipefail
cd /mnt/z4/solariewang/verl-swe
export EVAL_BENCHMARKS=ojbench,icpc,leetcode
nohup bash scripts/retry_sft_eval_errors.sh \
  >> logs/eval_api_qwen25_coder7b_sft_mt12_exec_retry_errors.nohup.log 2>&1 &
echo "pid=$!"
echo "tail -f logs/eval_api_qwen25_coder7b_sft_mt12_exec_retry_errors.nohup.log"
