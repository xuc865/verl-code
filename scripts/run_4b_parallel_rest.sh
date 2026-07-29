#!/usr/bin/env bash
# 在【训练机】上: 停掉串行链的调度(避免它稍后重复跑codeact/selfplan撞车),
# 然后把剩下的 cot / codeact / self-planning 三个方法【并行】挂起来加速。
# pure 的 leetcode 子进程不受影响,会自己跑完(resume)。
# 一行运行: bash /mnt/z4/solariewang/verl-swe/scripts/run_4b_parallel_rest.sh
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=/mnt/z4/solariewang/verl-swe
cd "$REPO_ROOT"

echo "== 1) 停掉串行链调度 parent(只杀 run_qwen35_4b_all.sh 本身,不动正在跑的 eval 子进程) =="
pkill -f 'run_qwen35_4b_all\.sh' 2>/dev/null && echo "  已杀串行链 parent" || echo "  没有串行链 parent(可能已退) —— 继续"
sleep 2

echo "== 2) API check =="
curl -sS http://127.0.0.1:80/v1/models | head -c 200; echo; echo

echo "== 3) 并行挂起 cot / codeact / self-planning (各自 resume, 互不写同一文件) =="
for m in cot codeact self_planning; do
  nohup bash "$SD/eval_api_qwen35_4b_${m}.sh" \
    >> "logs/qwen35_4b_${m}_parallel.nohup.log" 2>&1 &
  echo "  $m PID=$!  log=logs/qwen35_4b_${m}_parallel.nohup.log"
done
echo
echo "全部并行挂起。看进度: tail -f logs/qwen35_4b_{cot,codeact,self_planning}_parallel.nohup.log"
echo "注意: 3方法(各workers=8)+leetcode 同打一张4B卡,会有争抢;4B小,通常还能提吞吐。"
