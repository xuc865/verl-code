#!/usr/bin/env bash
# 在【训练机】上重挂 Qwen2.5-Coder 链条(resume)，已应用 方案1(workers=8) + 方案2(max_turns=6)。
# 一行运行:  bash /mnt/z4/solariewang/verl-swe/scripts/rsq.sh
set -uo pipefail
REPO_ROOT=/mnt/z4/solariewang/verl-swe
cd "$REPO_ROOT" || { echo "找不到仓库"; exit 1; }

echo "===== 1) 杀掉旧 chain / 旧 qwen25 eval ====="
pkill -f 'eval_api_qwen25_coder_7b_error_retry_then_self_planning' 2>/dev/null || true
pkill -f 'eval_api_baseline.py.*qwen25'                            2>/dev/null || true
pkill -f 'eval_api_kimi_apps_mt12'                                 2>/dev/null || true
pkill -f 'eval_api_kimi_mt12_queue'                                2>/dev/null || true
sleep 3
if pgrep -af 'error_retry_then_self_planning|eval_api_baseline.py.*qwen25'; then
  echo "!! 还有残留进程，等几秒或手动 kill 上面这些 PID 后再跑本脚本"; exit 1
fi
echo "已清干净"

echo; echo "===== 2) 确认本地 Instruct 基座 ====="
curl -sS http://127.0.0.1:80/v1/models; echo
# 期望 id = Qwen2.5-Coder-7B-Instruct

echo; echo "===== 3) resume 重挂 (workers=8, max_turns=6) ====="
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export REPO_ROOT DATA_ROOT=/mnt/z4/solariewang/datasets
export API_BASE=http://127.0.0.1:80/v1
export MODEL=Qwen2.5-Coder-7B-Instruct
export API_WAIT_SECS=0 SKIP_PEER_WAIT=1
export EVAL_RESUME=1
export EVAL_WORKERS=8
export CHAIN_MAX_TURNS=6
LOG=logs/eval_api_qwen25_coder_7b_chain_w8t6.nohup.log

nohup env API_BASE="$API_BASE" MODEL="$MODEL" EVAL_WORKERS=8 CHAIN_MAX_TURNS=6 \
  EVAL_RESUME=1 SKIP_PEER_WAIT=1 API_WAIT_SECS=0 \
  bash scripts/eval_api_qwen25_coder_7b_error_retry_then_self_planning.sh \
  >> "$LOG" 2>&1 &
echo "chain_pid=$!  log=$LOG"
sleep 6
echo "----- 日志开头 -----"
tail -n 30 "$LOG"
echo "----"
echo "成功标志: API_OK -> stage 1/4: CoT APPS -> pending 行里 workers=8 且新完成的题 turns<=6"
echo "看进度:  tail -f $REPO_ROOT/$LOG"
