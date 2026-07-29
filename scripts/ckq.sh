#!/usr/bin/env bash
# 检查训练机上 Qwen2.5-Coder 串行链（error_retry_then_self_planning）是否还活着。
# 在【训练机】上一行运行：
#   bash /mnt/z4/solariewang/verl-swe/scripts/check_qwen25_coder_chain_alive.sh
REPO_ROOT=${REPO_ROOT:-/mnt/z4/solariewang/verl-swe}
cd "$REPO_ROOT" || { echo "找不到仓库 $REPO_ROOT"; exit 1; }

echo "===== 1) 进程是否活着 ====="
if pgrep -af 'error_retry_then_self_planning|eval_api_kimi_apps_mt12|eval_api_baseline\.py'; then
  echo ">> 有进程在跑"
else
  echo "!!! 没有相关进程 —— 链条已死，需要重挂 !!!"
fi

echo; echo "===== 2) stage1 CoT APPS 真实输出日志（尾部 + 最后修改时间）====="
L="logs/eval_api_qwen25_coder_7b_mt12_cot_freeform_apps.nohup.log"
if [ -f "$L" ]; then
  stat -c '最后修改: %y  (%n)' "$L"
  tail -n 12 "$L"
else
  echo "无日志: $L"
fi

echo; echo "===== 3) 结果 json 行数 + 最后修改时间 ====="
J="logs/eval_api_qwen25_coder_7b_mt12_cot_freeform_apps.json"
if [ -f "$J" ]; then
  stat -c '最后修改: %y  (%n)' "$J"
  python3 - "$J" <<'PY'
import json,sys
from collections import Counter
a=json.load(open(sys.argv[1])).get('per_instance',[])
oc=Counter(r.get('outcome') for r in a)
w,l=oc.get('won',0),oc.get('lost',0)
print(f"rows = {len(a)}  won={w} lost={l} err={oc.get('error',0)}"
      + (f"  sr(excl.err)={w/(w+l):.1%}" if w+l else ""))
PY
else
  echo "无 json: $J"
fi

echo; echo "===== 判断 ====="
echo "第1步有进程 + 第2/3步最后修改在最近几分钟内 = 正常在跑；"
echo "打印'链条已死' 或 时间停在 02:05 附近没动 = 挂了，把本输出贴回来。"
