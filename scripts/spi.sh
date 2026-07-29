#!/usr/bin/env bash
# 在【训练机】上并行挂起两个任务(都打本地 127.0.0.1:80 的 Qwen Instruct):
#   A) pure 只补 APPS introductory 子集(备份->删intro行->resume补回)
#   B) self-planning 全数据集(非APPS队列 -> APPS)
# 一行运行:  bash /mnt/z4/solariewang/verl-swe/scripts/spi.sh
set -uo pipefail
REPO_ROOT=/mnt/z4/solariewang/verl-swe
cd "$REPO_ROOT" || { echo "找不到仓库"; exit 1; }
API=http://127.0.0.1:80/v1
MODEL=Qwen2.5-Coder-7B-Instruct

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export REPO_ROOT DATA_ROOT=/mnt/z4/solariewang/datasets

echo "===== API check ====="
curl -sS http://127.0.0.1:80/v1/models | head -c 200; echo; echo

########## A) pure 补 intro ##########
PJ=logs/eval_api_qwen25_coder_7b_st1_freeform_apps.json
cp "$PJ" "${PJ}.bak.$(date +%Y%m%d_%H%M%S)"
python3 - <<'PY'
import json
p='logs/eval_api_qwen25_coder_7b_st1_freeform_apps.json'
d=json.load(open(p)); a=d['per_instance']
keep=[r for r in a if r.get('difficulty')!='introductory']
d['per_instance']=keep
json.dump(d,open(p,'w'))
print(f"[pure-intro] 删掉 intro {len(a)-len(keep)} 行, 剩 {len(keep)} 行 -> resume 只补 intro")
PY
A_LOG=logs/eval_api_qwen25_coder_7b_st1_freeform_apps_introRetest.nohup.log
API_BASE=$API MODEL=$MODEL METHOD=qwen25_coder_7b_st1_freeform \
  EVAL_BENCHMARKS=apps EVAL_RESUME=1 EVAL_WORKERS=8 \
  SKIP_PEER_WAIT=1 SKIP_WAIT=1 WAIT_FOR_SINGLE_TURN=0 \
  nohup bash scripts/eval_api_qwen25_coder_7b_pure.sh >> "$A_LOG" 2>&1 &
echo "A) pure-intro PID=$!  log=$A_LOG"

########## B) self-planning 全数据集 ##########
B_LOG=logs/eval_api_qwen25_coder_7b_st1_self_planning_suite.nohup.log
API_BASE=$API MODEL=$MODEL EVAL_WORKERS=8 \
  SKIP_PEER_WAIT=1 SKIP_WAIT=1 WAIT_FOR_SINGLE_TURN=0 \
  nohup bash scripts/eval_api_qwen25_coder_7b_self_planning.sh >> "$B_LOG" 2>&1 &
echo "B) self-planning PID=$!  log=$B_LOG"

sleep 8
echo; echo "===== A) pure-intro 日志 ====="; tail -n 10 "$A_LOG"
echo; echo "===== B) self-planning 日志 ====="; tail -n 10 "$B_LOG"
echo "----"
echo "看进度: tail -f $REPO_ROOT/$A_LOG   /   tail -f $REPO_ROOT/$B_LOG"
