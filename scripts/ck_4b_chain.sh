#!/usr/bin/env bash
# 确认 qwen3.5-4b 串行总脚本(run_qwen35_4b_all.sh)的父进程是否还活着,
# 以判断 pure 跑完后是否会自动续 cot/codeact/self-planning。
# 训练机上一行运行: bash /mnt/z4/solariewang/verl-swe/scripts/ck_4b_chain.sh
echo "== 1) 串行父进程 run_qwen35_4b_all.sh 是否在? =="
if pgrep -af 'run_qwen35_4b_all\.sh'; then
  echo ">> 父进程在 —— pure 完会自动续 cot->codeact->self-planning ✓"
else
  echo "!! 没有 run_qwen35_4b_all.sh 父进程 —— pure 完不会自动续,需要手动挂后面的 !!"
fi
echo
echo "== 2) 当前在跑的 qwen35_4b 评测(看跑到哪个方法/bench) =="
pgrep -af 'eval_api_baseline\.py.*qwen35_4b' || echo "(当前没有 qwen35_4b 评测进程)"