#!/usr/bin/env python3
"""SFT eval accuracy report with subcategory breakdowns."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LOG = Path("/mnt/z4/solariewang/verl-swe/logs")
DATA = Path("/mnt/z4/solariewang/datasets")
METHOD = "qwen25_coder7b_sft_mt12_exec"


def load(b):
    p = LOG / f"eval_api_{METHOD}_{b}.json"
    return json.loads(p.read_text()) if p.exists() else None


def rate(w, g):
    return f"{w}/{g} = {100 * w / g:.2f}%" if g else "n/a"


def show(title, by, order=None):
    print(f"  【{title}】")
    keys = list(order) if order else sorted(by.keys(), key=lambda x: str(x))
    seen = set()
    for k in keys:
        if k not in by:
            continue
        seen.add(k)
        c = by[k]
        w, l, e = c.get("won", 0), c.get("lost", 0), c.get("error", 0)
        print(
            f"    {str(k):18s}  n={w + l + e:<4d}  won={w:<4d} lost={l:<4d} err={e:<3d}  "
            f"pass {rate(w, w + l)}"
        )
    for k in sorted(by.keys(), key=lambda x: str(x)):
        if k in seen:
            continue
        c = by[k]
        w, l, e = c.get("won", 0), c.get("lost", 0), c.get("error", 0)
        print(
            f"    {str(k):18s}  n={w + l + e:<4d}  won={w:<4d} lost={l:<4d} err={e:<3d}  "
            f"pass {rate(w, w + l)}"
        )


def slim_json_fields(line: str, fields):
    """Extract a few string fields without parsing huge nested blobs."""
    out = {}
    for f in fields:
        m = re.search(rf'"{re.escape(f)}"\s*:\s*"((?:\\.|[^"\\])*)"', line)
        if m:
            out[f] = m.group(1)
    return out


def main():
    # --- light metas ---
    lcd = {}
    lcd_tags = {}
    for line in open(DATA / "newfacade_LeetCodeDataset/LeetCodeDataset-test.jsonl"):
        o = json.loads(line)
        lcd[o["task_id"]] = o.get("difficulty", "unknown")
        lcd_tags[o["task_id"]] = o.get("tags") or []

    lcb = {}
    for fp in (DATA / "livecodebench_code_generation_lite").glob("test*.jsonl"):
        for line in open(fp):
            s = slim_json_fields(
                line, ["question_id", "difficulty", "platform", "contest_date"]
            )
            qid = s.get("question_id")
            if not qid:
                continue
            lcb[qid] = {
                "difficulty": str(s.get("difficulty", "unknown")).lower(),
                "platform": s.get("platform", "unknown"),
                "date": str(s.get("contest_date", ""))[:10],
            }

    oj = {}
    for line in open(DATA / "OJBench_testdata/prompts/full.jsonl"):
        o = json.loads(line)
        pid = str(o.get("id", o.get("problem_id", "")))
        ds = str(o.get("dataset", "ojbench")).lower()
        oj[f"{ds}_{pid}"] = {
            "difficulty": str(o.get("difficulty", "unknown")).lower(),
            "dataset": ds,
        }

    print("=" * 78)
    print("SFT 后模型准确率全量报告")
    print("模型: qwen25-coder7b-apps-mt8-sft  (global_step_242)")
    print("协议: mt12 · exec · thinking ON · react")
    print(f"METHOD: {METHOD}")
    print("=" * 78)

    print(f"\n{'数据集':14s} {'完成':>10s} {'won':>5s} {'lost':>5s} {'err':>4s}  pass@1(graded)")
    print("-" * 64)
    for b, name, expect in [
        ("humaneval", "HumanEval", 164),
        ("mbpp", "MBPP", 257),
        ("livecodebench", "LiveCodeBench", 131),
        ("usaco", "USACO", 307),
        ("ojbench", "OJBench", 159),
        ("icpc", "ICPC", 106),
        ("leetcode", "LeetCode", 228),
        ("apps", "APPS", 5000),
    ]:
        d = load(b)
        if not d:
            print(f"{name:14s} {'未跑':>10s}")
            continue
        arr = d["per_instance"]
        oc = Counter(r["outcome"] for r in arr)
        w, l, e = oc.get("won", 0), oc.get("lost", 0), oc.get("error", 0)
        print(
            f"{name:14s} {f'{len(arr)}/{expect}':>10s} {w:5d} {l:5d} {e:4d}  "
            f"{rate(w, w + l)}"
        )

    print("\n" + "-" * 78)
    print("分项明细")
    print("-" * 78)

    for b, name in [("humaneval", "HumanEval"), ("mbpp", "MBPP")]:
        d = load(b)
        arr = d["per_instance"]
        oc = Counter(r["outcome"] for r in arr)
        print(f"\n## {name}")
        print(
            f"  overall: {rate(oc['won'], oc['won'] + oc['lost'])}  "
            f"(n={len(arr)}, err={oc.get('error', 0)})"
        )
        print("  小项: 无官方难度分档（函数级单档题）")
        ts = d.get("turn_stats") or {}
        print(
            f"  turns: avg_all={ts.get('avg_turns_all')}  "
            f"avg_won={ts.get('avg_turns_won')}  median_won={ts.get('median_turns_won')}"
        )

    d = load("livecodebench")
    arr = d["per_instance"]
    oc = Counter(r["outcome"] for r in arr)
    print("\n## LiveCodeBench (LCB_MIN_DATE=2025-02-01)")
    print(f"  overall: {rate(oc['won'], oc['won'] + oc['lost'])}  (n={len(arr)})")
    by_d, by_p, by_m = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    miss = 0
    for r in arr:
        qid = r["instance_id"].split("__", 1)[-1]
        m = lcb.get(qid)
        if not m:
            miss += 1
            diff, plat, month = "unknown", "unknown", "unknown"
        else:
            diff, plat = m["difficulty"], m["platform"]
            month = (m["date"] or "")[:7] or "unknown"
        by_d[diff][r["outcome"]] += 1
        by_p[plat][r["outcome"]] += 1
        by_m[month][r["outcome"]] += 1
    show("difficulty", by_d, ["easy", "medium", "hard", "unknown"])
    show("platform", by_p)
    show("contest_month", by_m)
    if miss:
        print(f"  (meta miss {miss})")

    d = load("usaco")
    arr = d["per_instance"]
    oc = Counter(r["outcome"] for r in arr)
    print("\n## USACO")
    print(f"  overall: {rate(oc['won'], oc['won'] + oc['lost'])}  (n={len(arr)})")
    by = defaultdict(Counter)
    for r in arr:
        iid = r["instance_id"]
        tier = "other"
        for t in ("bronze", "silver", "gold", "platinum"):
            if f"_{t}_" in iid:
                tier = t
                break
        by[tier][r["outcome"]] += 1
    show("tier", by, ["bronze", "silver", "gold", "platinum", "other"])

    d = load("ojbench")
    arr = d["per_instance"]
    oc = Counter(r["outcome"] for r in arr)
    print("\n## OJBench")
    print(f"  overall: {rate(oc['won'], oc['won'] + oc['lost'])}  (n={len(arr)})")
    by_d, by_ds = defaultdict(Counter), defaultdict(Counter)
    miss = 0
    for r in arr:
        rest = r["instance_id"].split("__", 1)[-1]
        m = oj.get(rest)
        if not m:
            miss += 1
            ds = rest.split("_", 1)[0]
            diff = "unknown"
        else:
            ds, diff = m["dataset"], m["difficulty"]
        by_d[diff][r["outcome"]] += 1
        by_ds[ds][r["outcome"]] += 1
    show("difficulty", by_d, ["easy", "medium", "hard", "unknown"])
    show("dataset", by_ds)
    if miss:
        print(f"  (meta miss {miss})")

    d = load("icpc")
    arr = d["per_instance"]
    oc = Counter(r["outcome"] for r in arr)
    print("\n## ICPC")
    print(f"  overall: {rate(oc['won'], oc['won'] + oc['lost'])}  (n={len(arr)})")
    print("  小项: 结果 JSON 未落 contest/difficulty；全量 0 won")

    d = load("leetcode")
    arr = d["per_instance"]
    oc = Counter(r["outcome"] for r in arr)
    print("\n## LeetCode  (225/228，差 3 题未完)")
    print(f"  overall: {rate(oc['won'], oc['won'] + oc['lost'])}  (n={len(arr)}/228)")
    by_d = defaultdict(Counter)
    for r in arr:
        slug = r["instance_id"].split("__", 1)[-1]
        by_d[lcd.get(slug, "unknown")][r["outcome"]] += 1
    show("difficulty", by_d, ["Easy", "Medium", "Hard", "unknown"])
    tag_w, tag_n = Counter(), Counter()
    for r in arr:
        slug = r["instance_id"].split("__", 1)[-1]
        for t in lcd_tags.get(slug) or ["(no tag)"]:
            tag_n[t] += 1
            if r.get("outcome") == "won":
                tag_w[t] += 1
    print("  【tag — 出现最多 top10】")
    for t, n in tag_n.most_common(10):
        print(f"    {t:40s} won={tag_w[t]:<3d}/n={n:<3d}  pass {rate(tag_w[t], n)}")
    print("  【tag — won 最多】")
    for t, w in tag_w.most_common(10):
        print(f"    {t:40s} won={w:<3d}/n={tag_n[t]:<3d}  pass {rate(w, tag_n[t])}")

    print("\n## APPS")
    print("  未跑")
    print("\n" + "=" * 78)
    print("pass@1 = won/(won+lost)；上表各集当前 err=0（LeetCode 未满 228）")


if __name__ == "__main__":
    main()
