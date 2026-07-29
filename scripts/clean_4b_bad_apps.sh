#!/usr/bin/env bash
# 清理被 01:44 无-pyarrow 启动污染的 codeact/cot APPS 结果(全 eval_error)。
# 安全前提: 当前 cot/codeact 都在跑【非APPS】,没有 APPS eval 进程在写。脚本先校验再挪走(备份,可回滚)。
# 训练机一行运行: bash /mnt/z4/solariewang/verl-swe/scripts/clean_4b_bad_apps.sh
set -uo pipefail
cd /mnt/z4/solariewang/verl-swe/logs || exit 1

echo "== 校验: 有没有 qwen35_4b 的 APPS eval 进程在写 =="
if pgrep -af 'eval_api_baseline\.py.*qwen35_4b.*--benchmark apps' >/dev/null 2>&1; then
  echo "!! 有 APPS 进程在跑,先别删(否则删活文件)。当前:"
  pgrep -af 'eval_api_baseline\.py.*qwen35_4b.*--benchmark apps'
  exit 1
fi
echo "  没有 APPS 进程在写,可安全清理"

ts=$(date +%Y%m%d_%H%M%S)
for f in eval_api_qwen35_4b_mt12_cot_freeform_apps.json eval_api_qwen35_4b_mt12_exec_apps.json; do
  if [ -f "$f" ]; then
    n=$(python3 -c "import json;a=json.load(open('$f'))['per_instance'];e=sum(1 for r in a if r.get('outcome')=='error');print(f'{len(a)}行,{e}个error')" 2>/dev/null || echo '读失败')
    mv "$f" "$f.bad_$ts.bak"
    echo "  已挪走(备份 -> $f.bad_$ts.bak)  [$n]"
  fi
done
echo "完成。当前 cot/codeact 跑完非APPS 后会用 pyarrow 新建【干净】的 APPS(5000 全新跑)。"
