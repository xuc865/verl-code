#!/usr/bin/env bash
# 在【开发机本地】重跑 Kimi APPS 的 error 行（api_http_error + eval_error）。
# 保留原有 won/lost，只补 error 的实例；用 Kimi-K2.7-Code-node3 @ 29.163.228.8:8080。
# 一行启动（脚本内部会自动 nohup 后台跑真正的评测）：
#   bash /apdcephfs/z4/solariewang/verl-swe/scripts/run_kimi_apps_retry.sh
set -euo pipefail
REPO_ROOT=/apdcephfs/z4/solariewang/verl-swe
cd "$REPO_ROOT"
J="logs/eval_api_kimi_2_6_mt12_exec_apps.json"
LOG="logs/eval_api_kimi_2_6_mt12_exec_apps.retry_k27.nohup.log"

# 1) 备份原结果（含 K2.6 的 won/lost）
BAK="${J}.bak.$(date +%Y%m%d_%H%M%S)"
cp "$J" "$BAK"
echo "已备份原文件 -> $BAK"

# 2) API 预检
echo "== API check 29.163.228.8:8080 =="
curl -sS -m 15 http://29.163.228.8:8080/v1/models | head -c 300; echo

# 3) 后台挂起 retry-errors（保留 won/lost，只跑 error 的实例）
export REPO_ROOT DATA_ROOT=/apdcephfs/z4/solariewang/datasets
export API_BASE=http://29.163.228.8:8080/v1
export MODEL=Kimi-K2.7-Code-node3
export METHOD=kimi_2_6_mt12_exec
export EVAL_RESUME=1
export EVAL_RETRY_ERRORS=1          # 只重跑 outcome=error 的行
export EVAL_RETRY_ERROR_TYPES=api_http_error,eval_error
export WAIT_FOR_SINGLE_TURN=0
export SKIP_PEER_WAIT=1
export EVAL_WORKERS=${EVAL_WORKERS:-8}
export EVAL_CHECKPOINT_EVERY=5
export MAX_TOKENS=8192
export API_TIMEOUT=1800
export API_RETRIES=3

nohup bash scripts/eval_api_kimi_apps_mt12.sh >> "$LOG" 2>&1 &
PID=$!
echo "kimi_apps_retry PID=$PID  log=$LOG"
sleep 6
echo "===== 日志开头 ====="
tail -n 25 "$LOG"
echo "----"
echo "看进度:  tail -f $REPO_ROOT/$LOG"
