#!/usr/bin/env bash
# 实测 Qwen2.5-Coder CoT APPS 当前推进速率 + ETA。
# 在【训练机】上一行运行：
#   bash /mnt/z4/solariewang/verl-swe/scripts/rateq.sh
# 可选：SAMPLE_SECS=300 bash .../rateq.sh   # 改采样间隔，默认 180s
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
cd "$REPO_ROOT" || { echo "找不到仓库 $REPO_ROOT"; exit 1; }
L="logs/eval_api_qwen25_coder_7b_mt12_cot_freeform_apps.nohup.log"
SAMPLE_SECS=${SAMPLE_SECS:-180}
TOTAL=5000

_cur() { grep -oE '\[[0-9]+/5000\] apps__idx' "$L" 2>/dev/null \
           | grep -oE '^\[[0-9]+' | tr -d '[' | sort -n | tail -1; }

c1=$(_cur); [ -z "$c1" ] && { echo "日志里还没有 [N/5000] 记录，稍后再试"; exit 1; }
echo "采样1: $c1/$TOTAL  ($(date '+%H:%M:%S'))  —— 等 ${SAMPLE_SECS}s ..."
sleep "$SAMPLE_SECS"
c2=$(_cur)
echo "采样2: $c2/$TOTAL  ($(date '+%H:%M:%S'))"

d=$((c2 - c1))
echo "----"
if [ "$d" -le 0 ]; then
  echo "!! ${SAMPLE_SECS}s 内进度没变（+$d）—— 基本卡住/极慢，建议查 vLLM 负载或重挂"
  exit 0
fi
python3 - "$c2" "$d" "$SAMPLE_SECS" "$TOTAL" <<'PY'
import sys
c2,d,secs,total=map(int,sys.argv[1:5])
per_hr=d*3600/secs
remain=total-c2
eta_h=remain/per_hr
print(f"速率 = {per_hr:.1f} 题/小时  (本段 CoT APPS)")
print(f"剩余 = {remain} 题  ->  ETA ≈ {eta_h:.1f} 小时 ≈ {eta_h/24:.1f} 天")
print("注意：这只是 stage1；后面还有 codeact APPS(再5000) + self-planning APPS。")
PY
