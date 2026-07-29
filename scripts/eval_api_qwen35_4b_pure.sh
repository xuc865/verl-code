#!/usr/bin/env bash
# qwen3.5-4b — pure (单轮 freeform)。复用 qwen25coder pure wrapper，仅覆盖 MODEL/METHOD/API/地址。
# 训练机上跑: bash scripts/eval_api_qwen35_4b_pure.sh
export API_BASE=${API_BASE:-http://127.0.0.1:80/v1}
export MODEL=${MODEL:-qwen3.5-4b}
export METHOD=${METHOD:-qwen35_4b_st1_freeform}
export EVAL_WORKERS=${EVAL_WORKERS:-8}
export EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-apps,humaneval,mbpp,livecodebench,usaco,ojbench,icpc,leetcode}
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_api_qwen25_coder_7b_pure.sh"
