#!/usr/bin/env bash
# Qwen2.5-Coder 三/四个 baseline 的实时总准确率 + error 残留统计。
# 只读共享盘，任意机器都能跑：
#   bash /mnt/z4/solariewang/verl-swe/scripts/rptq.sh      (训练机)
#   bash /apdcephfs/z4/solariewang/verl-swe/scripts/rptq.sh (开发机)
REPO_ROOT=${REPO_ROOT:-/apdcephfs/z4/solariewang/verl-swe}
[ -d "$REPO_ROOT/logs" ] || REPO_ROOT=/mnt/z4/solariewang/verl-swe
cd "$REPO_ROOT/logs" || { echo "找不到 logs 目录"; exit 1; }

python3 - <<'PY'
import json,os,glob
from collections import Counter
BENCH=[('apps',5000),('humaneval',164),('mbpp',257),('livecodebench',131),
       ('usaco',307),('ojbench',159),('icpc',106)]
BASE=[('pure (st1_freeform)','eval_api_qwen25_coder_7b_st1_freeform_%s.json'),
      ('cot  (mt12_cot_freeform)','eval_api_qwen25_coder_7b_mt12_cot_freeform_%s.json'),
      ('codeact (mt12_exec)','eval_api_qwen25_coder_7b_mt12_exec_%s.json')]
def rows(p):
    try: return json.load(open(p)).get('per_instance',[]) or []
    except: return []
for name,pat in BASE:
    print('='*64); print(name); print('-'*64)
    print(f"{'bench':14}{'done':>10}{'won':>6}{'lost':>6}{'err':>6}{'sr_incl':>9}{'sr_excl':>9}")
    tw=tl=te=tn=0
    for b,tot in BENCH:
        p=pat%b
        if not os.path.exists(p):
            print(f"{b:14}{'-':>10}"); continue
        a=rows(p); oc=Counter(r.get('outcome') for r in a)
        w,l,e=oc.get('won',0),oc.get('lost',0),oc.get('error',0)
        n=len(a); tw+=w; tl+=l; te+=e; tn+=n
        incl=f"{w/(w+l+e):.1%}" if w+l+e else "-"     # 把 error 算失败
        excl=f"{w/(w+l):.1%}" if w+l else "-"          # 排除 error
        print(f"{b:14}{n:>7}/{tot:<3}{w:>6}{l:>6}{e:>6}{incl:>9}{excl:>9}")
    ti=f"{tw/(tw+tl+te):.1%}" if tw+tl+te else "-"
    tx=f"{tw/(tw+tl):.1%}" if tw+tl else "-"
    print('-'*64)
    print(f"{'总计(micro)':14}{tn:>10}{tw:>6}{tl:>6}{te:>6}{ti:>9}{tx:>9}")
    print(f"   sr_incl=把error当失败 ; sr_excl=排除error ; micro=按题数加权")
print('='*64)
print("说明: chain 现在在 stage1(CoT APPS)。codeact 的 error 重跑是 stage3，尚未开始。")
PY
