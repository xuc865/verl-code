#!/usr/bin/env bash
# Qwen3.5-4B base：把 cot / codeact 的 APPS error 行并行重挂，和正在跑的 self-planning 一起打 :8001。
# 保留已有 won/lost，只补 outcome=error；不杀 selfplan。
#
# 训练机一行：
#   bash /mnt/z4/solariewang/verl-swe/scripts/run_qwen35_4b_base_apps_retry_parallel.sh
#
# 可选：
#   EVAL_WORKERS=8 bash ...   # 默认 6；三路并行会抢同一张卡
#   SKIP_COT=1 / SKIP_CODEACT=1
set -euo pipefail

# 训练机必须用 /mnt；误用 /apdcephfs 会找不到 eval_api_baseline.py（上次 retry 已踩坑）
if [ -d /mnt/z4/solariewang/verl-swe ]; then
  REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
  DATA_ROOT=${DATA_ROOT:-/mnt/z4/solariewang/datasets}
elif [ -d /apdcephfs/z4/solariewang/verl-swe ]; then
  REPO_ROOT=${REPO_ROOT:-/apdcephfs/z4/solariewang/verl-swe}
  DATA_ROOT=${DATA_ROOT:-/apdcephfs/z4/solariewang/datasets}
else
  echo "!! 找不到 verl-swe 仓库" >&2
  exit 1
fi
# 强制纠正：禁止在训练机把 REPO_ROOT 指到 /apdcephfs
case "$REPO_ROOT" in
  /apdcephfs/*)
    if [ -d /mnt/z4/solariewang/verl-swe ]; then
      echo "!! REPO_ROOT=$REPO_ROOT 在训练机不可用，改用 /mnt/z4/..."
      REPO_ROOT=/mnt/z4/solariewang/verl-swe
      DATA_ROOT=/mnt/z4/solariewang/datasets
    fi
    ;;
esac
LOG_DIR="$REPO_ROOT/logs"
API_BASE=${API_BASE:-http://127.0.0.1:8001/v1}
MODEL=${MODEL:-qwen3.5-4b}
EVAL_WORKERS=${EVAL_WORKERS:-6}
EVAL_MAX_TURNS=${EVAL_MAX_TURNS:-6}

cd "$REPO_ROOT" || { echo "找不到仓库 $REPO_ROOT"; exit 1; }
mkdir -p "$LOG_DIR"
if [ ! -f "$REPO_ROOT/scripts/eval_api_baseline.py" ]; then
  echo "!! 缺少 $REPO_ROOT/scripts/eval_api_baseline.py" >&2
  exit 1
fi

export REPO_ROOT DATA_ROOT
export API_BASE MODEL
export EVAL_RESUME=1
export EVAL_RETRY_ERRORS=1
export EVAL_RETRY_ERROR_TYPES=${EVAL_RETRY_ERROR_TYPES:-api_http_error,eval_error}
export WAIT_FOR_SINGLE_TURN=0
export SKIP_PEER_WAIT=1
export EVAL_WORKERS
export EVAL_MAX_TURNS
export EVAL_HISTORY_LENGTH=${EVAL_HISTORY_LENGTH:-6}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-5}
export MAX_TOKENS=${MAX_TOKENS:-8192}
export API_TIMEOUT=${API_TIMEOUT:-1800}
export API_RETRIES=${API_RETRIES:-3}
export EVAL_INSTANCE_TIMEOUT=${EVAL_INSTANCE_TIMEOUT:-3600}
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

echo "== REPO_ROOT=$REPO_ROOT =="
echo "== API check $API_BASE =="
curl -sS -m 15 "$API_BASE/models" | head -c 400; echo; echo

_launch() {
  local tag="$1" method="$2"
  shift 2
  local log="$LOG_DIR/eval_api_${method}_apps_retry.nohup.log"
  local json="$LOG_DIR/eval_api_${method}_apps.json"
  if [ ! -f "$json" ]; then
    echo "!! 缺少 $json，跳过 $tag" >&2
    return 1
  fi

  # 子 shell 里强制带上 /mnt 路径，避免 kimi 脚本默认落到 /apdcephfs
  (
    export REPO_ROOT="$REPO_ROOT"
    export DATA_ROOT="$DATA_ROOT"
    export METHOD="$method"
    export WAIT_EVAL_PATTERN="python3.*eval_api_baseline.py.*${method}"
    for kv in "$@"; do
      # shellcheck disable=SC2163
      export "$kv"
    done
    echo "[$(date '+%F %T')] launch $tag METHOD=$METHOD REPO_ROOT=$REPO_ROOT workers=$EVAL_WORKERS retry_errors=1" >>"$log"
    nohup bash "$REPO_ROOT/scripts/eval_api_kimi_apps_mt12.sh" >>"$log" 2>&1 &
    echo $!
  )
}

echo "== 并行挂起 cot + codeact APPS retry（不碰 selfplan） =="
PIDS=()

if [ "${SKIP_CODEACT:-0}" != "1" ]; then
  pid=$(_launch codeact qwen35_4b_mt12_exec \
    DISABLE_THINKING=1 \
    TEST_FEEDBACK_MODE=exec)
  PIDS+=("codeact:$pid")
  echo "  codeact  PID=$pid  log=logs/eval_api_qwen35_4b_mt12_exec_apps_retry.nohup.log"
fi

if [ "${SKIP_COT:-0}" != "1" ]; then
  pid=$(_launch cot qwen35_4b_mt12_cot_freeform \
    DISABLE_THINKING=0 \
    TEST_FEEDBACK_MODE=exec \
    PROMPT_MODE=freeform \
    ENCOURAGE_COT=1)
  PIDS+=("cot:$pid")
  echo "  cot      PID=$pid  log=logs/eval_api_qwen35_4b_mt12_cot_freeform_apps_retry.nohup.log"
fi

sleep 8
echo
echo "===== 启动日志 ====="
for tag_pid in "${PIDS[@]}"; do
  tag="${tag_pid%%:*}"
  case "$tag" in
    codeact) f="$LOG_DIR/eval_api_qwen35_4b_mt12_exec_apps_retry.nohup.log" ;;
    cot)     f="$LOG_DIR/eval_api_qwen35_4b_mt12_cot_freeform_apps_retry.nohup.log" ;;
  esac
  echo "--- $tag ---"
  tail -n 20 "$f" 2>/dev/null || echo "(log 尚未写出)"
  echo
done

echo "看进度:"
echo "  tail -f $LOG_DIR/eval_api_qwen35_4b_st1_self_planning_apps.nohup.log"
echo "  tail -f $LOG_DIR/eval_api_qwen35_4b_mt12_exec_apps_retry.nohup.log"
echo "  tail -f $LOG_DIR/eval_api_qwen35_4b_mt12_cot_freeform_apps_retry.nohup.log"
echo "三路并行打同一张 4B(:8001)；卡顿可降 workers: EVAL_WORKERS=4 bash $0"
