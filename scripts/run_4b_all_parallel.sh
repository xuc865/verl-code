#!/usr/bin/env bash
# 在【训练机】上:
#   1) 外科手术式 kill 当前 qwen3.5-4b 测试(串行链 + 其 queue/apps 驱动 + python worker)
#      —— 只针对 qwen35_4b 相关进程; 决不碰 SFT/RL 训练进程或别的模型的 eval。
#   2) 并行挂: pure(resume,补剩余leetcode)/cot/codeact/self-planning, 各写各文件, 真并行。
#
# 先预览会杀谁(不杀不挂):   DRY_RUN=1 bash scripts/run_4b_all_parallel.sh
# 确认无误后真跑:           bash scripts/run_4b_all_parallel.sh
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=/mnt/z4/solariewang/verl-swe
cd "$REPO_ROOT" || { echo "找不到仓库"; exit 1; }
DRY_RUN=${DRY_RUN:-0}
# 训练进程关键词护栏(命中则跳过,决不杀)。故意不含裸 'verl'(仓库路径 verl-swe 会误命中)。
TRAIN_RE='torchrun|main_ppo|megatron|deepspeed|sft_trainer|ppo_trainer|fsdp_sft|verl\.trainer|accelerate launch|ray[_ ]|raylet'

# 自动探测当前 qwen35_4b 评测正在用的 API_BASE(比如 8001), 重挂时沿用, 避免写死端口连错。
DETECTED_API=$(pgrep -af 'eval_api_baseline\.py.*qwen35_4b' 2>/dev/null | grep -oE -- '--api-base [^ ]+' | awk '{print $2}' | head -1)
API_BASE_USE=${API_BASE:-${DETECTED_API:-http://127.0.0.1:8001/v1}}
echo ">> 检测到当前评测 API_BASE = ${DETECTED_API:-(未检测到,将用默认)} ; 重挂将使用 = $API_BASE_USE"
echo

# 收集待杀 PID(只限 qwen35_4b 相关):
mapfile -t PIDS < <(
  pgrep -f 'run_qwen35_4b_all\.sh'            2>/dev/null
  pgrep -f 'run_4b_parallel_rest\.sh'         2>/dev/null
  pgrep -f 'eval_api_qwen35_4b_'              2>/dev/null
  pgrep -f 'eval_api_baseline\.py.*qwen35_4b' 2>/dev/null
  # queue/apps 驱动: 仅当其环境变量里带 qwen35_4b 才算(surgical, 不误伤别的模型 eval)
  for pid in $(pgrep -f 'eval_api_kimi_queue\.sh|eval_api_kimi_mt12_queue\.sh|eval_api_kimi_apps_mt12\.sh' 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q 'qwen35_4b'; then echo "$pid"; fi
  done
)
# 去重
mapfile -t PIDS < <(printf '%s\n' "${PIDS[@]}" | sort -un)

echo "===== 1) 待处理进程(DRY_RUN=$DRY_RUN) ====="
if [ "${#PIDS[@]}" -eq 0 ] || [ -z "${PIDS[0]:-}" ]; then
  echo "  没有 qwen35_4b 相关进程。"
else
  for pid in "${PIDS[@]}"; do
    [ -z "$pid" ] && continue
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null) || { echo "  PID=$pid 已退出"; continue; }
    if printf '%s' "$cmd" | grep -qiE "$TRAIN_RE"; then
      echo "  ⚠️ 跳过(疑似训练进程!): PID=$pid | $cmd"; continue
    fi
    echo "  将kill PID=$pid | $cmd"
    [ "$DRY_RUN" = 1 ] || { kill "$pid" 2>/dev/null || true; }
  done
fi

if [ "$DRY_RUN" = 1 ]; then
  echo; echo "== DRY_RUN: 未杀任何进程、未挂任何任务。确认上面没有训练进程后, 去掉 DRY_RUN 再跑。=="
  exit 0
fi

# 强杀残留的 qwen35_4b python
sleep 3
for pid in $(pgrep -f 'eval_api_baseline\.py.*qwen35_4b' 2>/dev/null); do
  cmd=$(ps -o cmd= -p "$pid" 2>/dev/null) || continue
  printf '%s' "$cmd" | grep -qiE "$TRAIN_RE" && continue
  kill -9 "$pid" 2>/dev/null || true
done
sleep 1
if pgrep -f 'eval_api_baseline\.py.*qwen35_4b' >/dev/null 2>&1; then
  echo "!! 仍有 qwen35_4b python 残留:"; pgrep -af 'eval_api_baseline\.py.*qwen35_4b'; echo "手动处理后再跑。"; exit 1
fi
echo "  已清干净"

echo; echo "===== 2) API check ($API_BASE_USE) ====="
curl -sS "${API_BASE_USE%/}/models" | head -c 220; echo; echo

echo "===== 3) 并行挂起 pure / cot / codeact / self-planning ====="
export EVAL_RESUME=1 SKIP_PEER_WAIT=1 SKIP_WAIT=1 WAIT_FOR_SINGLE_TURN=0
export API_BASE=$API_BASE_USE
export MODEL=${MODEL:-qwen3.5-4b}
export EVAL_WORKERS=${EVAL_WORKERS:-8}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
launch(){ nohup bash "$SD/$2" >> "logs/qwen35_4b_$1_parallel.nohup.log" 2>&1 & echo "  $1 PID=$! -> logs/qwen35_4b_$1_parallel.nohup.log"; sleep 1; }
launch pure          eval_api_qwen35_4b_pure.sh
launch cot           eval_api_qwen35_4b_cot.sh
launch codeact       eval_api_qwen35_4b_codeact.sh
launch self_planning eval_api_qwen35_4b_self_planning.sh
echo; echo "===== 已全部并行挂起 ====="
echo "看进度: tail -f logs/qwen35_4b_{pure,cot,codeact,self_planning}_parallel.nohup.log"
