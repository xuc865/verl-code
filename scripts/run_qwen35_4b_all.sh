#!/usr/bin/env bash
# 在【训练机】上串行跑 qwen3.5-4b 的 pure -> cot -> codeact -> self-planning。
# 每个方法内部: 非APPS队列 -> APPS。串行避免单卡互抢。
# 一行挂起(训练机):
#   nohup bash /mnt/z4/solariewang/verl-swe/scripts/run_qwen35_4b_all.sh \
#     >> /mnt/z4/solariewang/verl-swe/logs/qwen35_4b_all.nohup.log 2>&1 &
#
# 覆盖: MODEL=... EVAL_MAX_TURNS=6 EVAL_WORKERS=8  SKIP_PURE=1 SKIP_COT=1 ...
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
cd "$REPO_ROOT"
API_BASE=${API_BASE:-http://127.0.0.1:80/v1}
MODEL=${MODEL:-qwen3.5-4b}
export API_BASE MODEL REPO_ROOT
export EVAL_WORKERS=${EVAL_WORKERS:-8}
export EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-6}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
_ts(){ date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(_ts)] ===== qwen3.5-4b ALL: API=$API_BASE MODEL=$MODEL turns=$EVAL_MAX_TURNS workers=$EVAL_WORKERS ====="

# --- 校验 API 且 MODEL 名必须精确匹配 served id ---
echo "[$(_ts)] API check ${API_BASE%/}/models"
if ! python3 - <<PY
import json,sys,urllib.request
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    d=json.loads(op.open(urllib.request.Request("${API_BASE%/}/models"),timeout=15).read())
except Exception as e:
    print("API_UNREACHABLE:",e); sys.exit(2)
ids=[m.get("id") for m in d.get("data") or []]
print("served ids =",ids)
if "${MODEL}" not in ids:
    print(f"!! MODEL '${MODEL}' 不在 served id 里；请用上面真实 id 重跑: MODEL=<id> bash .../run_qwen35_4b_all.sh")
    sys.exit(3)
print("MODEL_OK")
PY
then
  echo "[$(_ts)] ABORT: API/模型名校验失败(见上)。修好再跑。"; exit 1
fi

run(){ # $1=方法名 $2=脚本
  if [ "${2:-}" ] && [ "${SKIP:-0}" != 1 ]; then :; fi
  local name="$1" script="$2" var="SKIP_${1^^}"
  if [ "${!var:-0}" = 1 ]; then echo "[$(_ts)] SKIP $name"; return; fi
  echo "[$(_ts)] >>>> START $name"
  bash "$SD/$script" >> "logs/qwen35_4b_${name}.nohup.log" 2>&1 || echo "[$(_ts)] WARN $name 退出码 $?"
  echo "[$(_ts)] <<<< DONE  $name"
}
run pure     eval_api_qwen35_4b_pure.sh
run cot      eval_api_qwen35_4b_cot.sh
run codeact  eval_api_qwen35_4b_codeact.sh
run selfplan eval_api_qwen35_4b_self_planning.sh
echo "[$(_ts)] ===== qwen3.5-4b ALL FINISHED ====="
