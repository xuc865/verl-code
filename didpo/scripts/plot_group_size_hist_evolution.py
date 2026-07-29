#!/usr/bin/env python3
"""DiDPO **sub-diff** group-size histogram evolution.

Sources
-------
- ``jsonl`` (fine ``k=2..20`` + ``21+``): tracked dumps in
  ``logs/didpo_groups/didpo_prompt_groups.jsonl``. Only the pinned cases
  (default 3 APPS ids) have exact sizes.
- ``logs`` (coarse ``2..8``, ``9-16``, ``17+``): full-batch hist from nohup —
  every prompt in the training batch each step (~100 groups/step).

**count vs mass**
- count: each group contributes 1 → “% of groups”
- mass: each group of size ``k`` contributes ``k`` → “% of group members /
  item mass” (large groups weigh more)

Default: emit both sources, both modes. No per-window single-panel files.

Usage:
  python3 didpo/scripts/plot_group_size_hist_evolution.py
  python3 didpo/scripts/plot_group_size_hist_evolution.py --source jsonl --pool macro
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NINTENDO_BLUE = "#0066CC"
NINTENDO_RED = "#E60012"

FINE_KS = list(range(2, 21)) + ["21+"]
COARSE_BUCKETS = ["2", "3", "4", "5", "6", "7", "8", "9-16", "17+"]
COARSE_MID = {
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "7": 7.0,
    "8": 8.0,
    "9-16": 12.0,
    "17+": 24.0,
}
WINDOWS = [
    (1, 20),
    (21, 40),
    (41, 60),
    (61, 80),
    (81, 100),
]
DEFAULT_LOGS = [
    "logs/didpo_coderl_qwen25_7b_sft_mt8.nohup.log",
    "logs/didpo_coderl_qwen25_7b_sft_mt8_resume20.nohup.log",
]


def _setup_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.titlesize": 11,
        }
    )


def parse_hist_by_step_logs(log_paths: list[Path]) -> dict[int, dict[str, float]]:
    by: dict[int, dict[str, float]] = {}
    prefix = "didpo/group_size_hist/"
    for path in log_paths:
        if not path.exists():
            continue
        for line in path.open(errors="ignore"):
            if "step:" not in line or prefix not in line:
                continue
            m = re.search(r"step:(\d+)\s+-\s+(.*)", line)
            if not m:
                continue
            step = int(m.group(1))
            metrics = {
                km.group(1): float(km.group(2))
                for km in re.finditer(
                    r"([\w./+-]+):(-?\d+\.?\d*(?:e[+-]?\d+)?)", m.group(2)
                )
            }
            hist = {
                k[len(prefix) :]: v
                for k, v in metrics.items()
                if k.startswith(prefix)
            }
            if hist:
                by[step] = hist
    return by


def load_sizes_by_instance_jsonl(
    jsonl: Path,
    *,
    instance_ids: set[str] | None = None,
) -> dict[str, dict[int, list[int]]]:
    """Exact sub-diff group sizes: instance_id -> step -> sizes (>=2)."""
    by: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r.get("instance_id")
        if not iid:
            continue
        if instance_ids is not None and iid not in instance_ids:
            continue
        step = int(r["step"])
        for g in r.get("groups") or []:
            sz = int(g.get("size") or 0)
            if sz >= 2:
                by[iid][step].append(sz)
    return {iid: dict(steps) for iid, steps in by.items()}


def fine_mass_pct(sizes: list[int]) -> dict:
    """Mass share: group of size k adds mass k."""
    mass = {k: 0.0 for k in range(2, 21)}
    mass["21+"] = 0.0
    for sz in sizes:
        if 2 <= sz <= 20:
            mass[sz] += float(sz)
        elif sz >= 21:
            mass["21+"] += float(sz)
    tot = sum(mass.values()) or 1.0
    return {k: 100.0 * mass[k] / tot for k in FINE_KS}


def fine_count_pct(sizes: list[int]) -> dict:
    """Count share: each group adds 1."""
    counts = {k: 0.0 for k in range(2, 21)}
    counts["21+"] = 0.0
    for sz in sizes:
        if 2 <= sz <= 20:
            counts[sz] += 1.0
        elif sz >= 21:
            counts["21+"] += 1.0
    tot = sum(counts.values()) or 1.0
    return {k: 100.0 * counts[k] / tot for k in FINE_KS}


def window_sizes(by_step: dict[int, list[int]], lo: int, hi: int) -> list[int]:
    out: list[int] = []
    for s, xs in by_step.items():
        if lo <= s <= hi:
            out.extend(xs)
    return out


def aggregate_fine_window(
    by_inst: dict[str, dict[int, list[int]]],
    lo: int,
    hi: int,
    *,
    mode: str,
    pool: str,
) -> tuple[dict, int, int]:
    """Return (pct, n_groups_total, n_instances_with_data)."""
    pct_fn = fine_mass_pct if mode == "mass" else fine_count_pct
    per_inst: list[tuple[str, list[int]]] = []
    for iid, steps in by_inst.items():
        sizes = window_sizes(steps, lo, hi)
        if sizes:
            per_inst.append((iid, sizes))
    if not per_inst:
        return ({k: 0.0 for k in FINE_KS}, 0, 0)

    n_groups = sum(len(s) for _, s in per_inst)
    n_inst = len(per_inst)
    if pool == "micro":
        all_sizes = [sz for _, s in per_inst for sz in s]
        return pct_fn(all_sizes), n_groups, n_inst

    # macro: equal weight per case (avoids one frequent dump dominating)
    pcts = [pct_fn(s) for _, s in per_inst]
    out = {k: float(np.mean([p[k] for p in pcts])) for k in FINE_KS}
    # renormalize tiny float drift
    s = sum(out.values()) or 1.0
    out = {k: 100.0 * out[k] / s for k in FINE_KS}
    return out, n_groups, n_inst


def coarse_window_pct(
    by_step: dict[int, dict[str, float]],
    lo: int,
    hi: int,
    *,
    mode: str,
) -> tuple[dict[str, float], int]:
    rows = [by_step[s] for s in by_step if lo <= s <= hi]
    if not rows:
        return {b: 0.0 for b in COARSE_BUCKETS}, 0
    mean_counts = {
        b: float(np.mean([r.get(b, 0.0) for r in rows])) for b in COARSE_BUCKETS
    }
    n_g = int(round(sum(mean_counts.values())))
    if mode == "count":
        tot = sum(mean_counts.values()) or 1.0
        return {b: 100.0 * mean_counts[b] / tot for b in COARSE_BUCKETS}, n_g
    mass = {b: mean_counts[b] * COARSE_MID[b] for b in COARSE_BUCKETS}
    tot = sum(mass.values()) or 1.0
    return {b: 100.0 * mass[b] / tot for b in COARSE_BUCKETS}, n_g


def plot_panels(
    series: list[tuple[str, dict, int, int]],
    *,
    bins: list,
    out: Path,
    mode: str,
    title: str,
    dpi: int = 180,
    annotate: bool = False,
    x_rotation: float = 0.0,
) -> pd.DataFrame:
    """series: (window_label, pct_dict, n_groups, n_cases)."""
    sns.set_theme(style="whitegrid", context="paper")
    _setup_font()

    series = [(lab, pct, n_g, n_c) for lab, pct, n_g, n_c in series if n_g > 0]
    if not series:
        raise SystemExit("no non-empty windows to plot")

    n = len(series)
    fig_w = max(3.0, 0.42 * len(bins)) * n
    fig_h = 4.2 if annotate else 3.6
    fig, axes = plt.subplots(1, n, figsize=(min(fig_w, 22), fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    color = NINTENDO_BLUE if mode == "mass" else NINTENDO_RED
    ylabel = (
        "Percentage of group mass %"
        if mode == "mass"
        else "Percentage of groups %"
    )
    rows_csv: list[dict] = []
    ymax = max(max(pct.values()) if pct else 0.0 for _, pct, _, _ in series)
    y_top = max(20.0, ymax * (1.22 if annotate else 1.12))

    bin_labels = [str(b) for b in bins]
    for ax, (lab, pct, n_g, n_c) in zip(axes, series):
        vals = [pct[b] for b in bins]
        s = float(sum(vals))
        df = pd.DataFrame({"k": bin_labels, "pct": vals})
        sns.barplot(
            data=df,
            x="k",
            y="pct",
            ax=ax,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            order=bin_labels,
        )
        # Keep categorical order left→right as k=2..21+ (no invert).
        ax.set_xlim(-0.5, len(bin_labels) - 0.5)
        if annotate:
            for patch, v in zip(ax.patches, vals):
                x = patch.get_x() + patch.get_width() / 2.0
                y = patch.get_height()
                label = f"{v:.1f}" if v < 10 else f"{v:.0f}"
                ax.text(
                    x,
                    max(y, 0.0) + y_top * 0.012,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                    color="#222222",
                    clip_on=False,
                )
        case_bit = f", n_cases={n_c}" if n_c > 0 else ""
        ax.set_title(f"steps {lab}\n(n_groups={n_g}{case_bit}, Σ={s:.1f}%)")
        ax.set_xlabel(r"Group size $k$")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=x_rotation)
        ax.set_ylim(0, y_top)
        ax.set_axisbelow(True)
        for b, v in zip(bins, vals):
            rows_csv.append(
                {
                    "window": lab,
                    "group_size": b,
                    "pct": v,
                    "mode": mode,
                    "n_groups": n_g,
                    "n_cases": n_c,
                }
            )

    fig.suptitle(title, y=1.06, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.svg')}")
    plt.close(fig)

    csv_df = pd.DataFrame(rows_csv)
    csv_path = out.with_suffix(".csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    sums = csv_df.groupby("window")["pct"].sum()
    print("column sums:\n", sums.to_string())
    return csv_df


def run_jsonl(args, modes: list[str]) -> None:
    by_inst = load_sizes_by_instance_jsonl(args.jsonl)
    if not by_inst:
        raise SystemExit(f"no groups in {args.jsonl}")
    ids = sorted(by_inst)
    n_g_total = sum(
        len(xs) for steps in by_inst.values() for xs in steps.values()
    )
    print(
        f"[jsonl/sub-diff] cases={ids} "
        f"n_cases={len(ids)} n_groups={n_g_total} pool={args.pool}"
    )
    for mode in modes:
        series = []
        for lo, hi in WINDOWS:
            pct, n_g, n_c = aggregate_fine_window(
                by_inst, lo, hi, mode=mode, pool=args.pool
            )
            series.append((f"{lo}–{hi}", pct, n_g, n_c))
        out = args.out_dir / f"group_size_hist_fine_evolution_{mode}.png"
        pool_note = "macro avg over cases" if args.pool == "macro" else "micro pool"
        plot_panels(
            series,
            bins=FINE_KS,
            out=out,
            mode=mode,
            title=(
                "DiDPO sub-diff group-size hist evolution "
                f"({len(ids)} tracked cases, fine $k$=2..20, {mode}, {pool_note})"
            ),
            dpi=args.dpi,
            annotate=(mode == "count"),
            x_rotation=0.0,
        )


def run_logs(args, modes: list[str]) -> None:
    by_log = parse_hist_by_step_logs(list(args.logs))
    if not by_log:
        raise SystemExit(f"no hist in {args.logs}")
    print(
        f"[logs/sub-diff full-batch] steps {min(by_log)}..{max(by_log)} "
        f"n_steps={len(by_log)} (coarse bins only)"
    )
    for mode in modes:
        series = []
        for lo, hi in WINDOWS:
            pct, n_g = coarse_window_pct(by_log, lo, hi, mode=mode)
            # n_cases unknown for full batch; pass 0 → omit from title
            series.append((f"{lo}–{hi}", pct, n_g, 0))
        out = args.out_dir / f"group_size_hist_evolution_{mode}.png"
        plot_panels(
            series,
            bins=COARSE_BUCKETS,
            out=out,
            mode=mode,
            title=(
                "DiDPO sub-diff group-size hist evolution "
                f"(full-batch all prompts, coarse, {mode})"
            ),
            dpi=args.dpi,
        )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=["jsonl", "logs", "both"],
        default="both",
        help="jsonl=fine tracked cases; logs=full-batch coarse; both=default",
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=repo / "logs" / "didpo_groups" / "didpo_prompt_groups.jsonl",
    )
    ap.add_argument(
        "--logs",
        nargs="*",
        type=Path,
        default=[repo / p for p in DEFAULT_LOGS],
    )
    ap.add_argument("--out-dir", type=Path, default=repo / "didpo" / "plots")
    ap.add_argument("--mode", choices=["mass", "count", "both"], default="both")
    ap.add_argument(
        "--pool",
        choices=["macro", "micro"],
        default="macro",
        help="jsonl only: macro=equal weight per case; micro=pool all groups",
    )
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    modes = ["mass", "count"] if args.mode == "both" else [args.mode]
    if args.source in ("jsonl", "both"):
        run_jsonl(args, modes)
    if args.source in ("logs", "both"):
        run_logs(args, modes)


if __name__ == "__main__":
    main()
