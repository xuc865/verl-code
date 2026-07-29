#!/usr/bin/env bash
# qwen3.5-4b — codeact (ReAct thought-action, exec)。max_turns 默认 6(可覆盖为12)。
# 训练机上跑: bash scripts/eval_api_qwen35_4b_codeact.sh
export API_BASE=${API_BASE:-http://127.0.0.1:80/v1}
export MODEL=${MODEL:-qwen3.5-4b}
export METHOD=${METHOD:-qwen35_4b_mt12_exec}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-6}
export EVAL_WORKERS=${EVAL_WORKERS:-8}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_api_qwen25_coder_7b_codeact.sh"
