#!/usr/bin/env python3
"""Scatter: each DiDPO group as a point (x=step, y=group mass/size).

Default: tracked task ``apps__idx3836``, steps in [0, 60].
Tracked dumps currently start at the resume-20 era (step >= 21).

Usage:
  python3 didpo/scripts/plot_group_mass_by_step.py
  python3 didpo/scripts/plot_group_mass_by_step.py --max-step 60 --out didpo/plots/group_mass_by_step.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def family(preview: str | None) -> str:
    p = (preview or "").replace("\\n", "\n")
    pl = " ".join(p.split())
    if "return None" in pl:
        return "stub"
    if re.search(
        r"all\s*\(\s*arr|ascending\s*=\s*True|is_ascending|ascending\s*=\s*all", pl
    ):
        return "scan"
    if "class Solution" in pl and "sorted(arr)" in pl:
        return "Sol+sorted"
    if "def is_sorted" in pl and "sorted(arr)" in pl:
        return "bare+sorted"
    if "class Solution" in pl:
        return "Sol other"
    if "def is_sorted" in pl:
        return "bare other"
    if "__main__" in pl or "stdin" in pl or "json.loads" in pl:
        return "harness"
    return "other"


# Distinct, print-friendly colors (no purple-on-white / cream-serif defaults).
FAM_STYLE = {
    "Sol+sorted": ("#0B6E4F", "o", 36),
    "bare+sorted": ("#1B4F72", "s", 32),
    "scan": ("#B36A00", "^", 34),
    "Sol other": ("#2E86AB", "D", 28),
    "bare other": ("#5D6D7E", "v", 28),
    "stub": ("#C0392B", "x", 40),
    "harness": ("#7F8C8D", "+", 30),
    "other": ("#566573", ".", 24),
}


def load_points(
    jsonl: Path,
    *,
    instance_id: str,
    max_step: int,
    min_step: int = 0,
) -> list[dict]:
    rows = [
        json.loads(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    pts: list[dict] = []
    for r in rows:
        if r.get("instance_id") != instance_id:
            continue
        step = int(r["step"])
        if step < min_step or step > max_step:
            continue
        for g in r.get("groups") or []:
            size = int(g.get("size") or 0)
            pts.append(
                {
                    "step": step,
                    "mass": size,
                    "family": family(g.get("preview")),
                    "gs": float(g["gs"]) if g.get("gs") is not None else None,
                }
            )
    return pts


def plot(
    pts: list[dict],
    *,
    out: Path,
    instance_id: str,
    max_step: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Jitter x slightly so overlapping groups at the same step are visible.
    rng = np.random.default_rng(0)
    by_fam: dict[str, list[dict]] = {}
    for p in pts:
        by_fam.setdefault(p["family"], []).append(p)

    order = [
        "Sol+sorted",
        "bare+sorted",
        "scan",
        "Sol other",
        "bare other",
        "other",
        "harness",
        "stub",
    ]
    for fam in order:
        bucket = by_fam.get(fam) or []
        if not bucket:
            continue
        color, marker, size = FAM_STYLE[fam]
        xs = np.array([p["step"] for p in bucket], dtype=float)
        xs = xs + rng.uniform(-0.18, 0.18, size=len(xs))
        ys = np.array([p["mass"] for p in bucket], dtype=float)
        filled = marker not in {"x", "+", ".", "1", "2", "3", "4"}
        ax.scatter(
            xs,
            ys,
            c=color,
            marker=marker,
            s=size,
            alpha=0.78,
            label=fam,
            linewidths=0.9 if not filled else 0.5,
            edgecolors="white" if filled else color,
            zorder=3 if fam != "stub" else 4,
        )

    ax.set_xlim(-0.5, max_step + 0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Group mass (rollouts in group)")
    ax.set_title(
        f"DiDPO group mass vs step — {instance_id}\n"
        f"each point = one group · steps 0–{max_step}"
    )
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title="strategy family",
        loc="upper left",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
        ncol=2,
        framealpha=0.95,
    )

    steps_present = sorted({p["step"] for p in pts})
    note = (
        f"n_groups={len(pts)} · snapshots={len(steps_present)} "
        f"· available steps {steps_present[0]}–{steps_present[-1]} "
        f"(tracked dumps start after resume)"
    )
    fig.text(0.01, 0.01, note, fontsize=8, color="#555555")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    # companion svg if png requested (and vice versa handled by caller)
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_jsonl = repo / "logs" / "didpo_groups" / "didpo_prompt_groups.jsonl"
    default_out = repo / "didpo" / "plots" / "group_mass_by_step_0_60.png"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=default_jsonl)
    ap.add_argument("--instance-id", default="apps__idx3836")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=60)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    pts = load_points(
        args.jsonl,
        instance_id=args.instance_id,
        max_step=args.max_step,
        min_step=args.min_step,
    )
    if not pts:
        raise SystemExit(
            f"no groups for {args.instance_id} in steps "
            f"[{args.min_step}, {args.max_step}] under {args.jsonl}"
        )
    plot(
        pts,
        out=args.out,
        instance_id=args.instance_id,
        max_step=args.max_step,
        dpi=args.dpi,
    )
    # also write svg alongside png/svg choice
    if args.out.suffix.lower() == ".png":
        plot(
            pts,
            out=args.out.with_suffix(".svg"),
            instance_id=args.instance_id,
            max_step=args.max_step,
            dpi=args.dpi,
        )
    # CSV for reuse
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,mass,family,gs\n")
        for p in sorted(pts, key=lambda x: (x["step"], -x["mass"])):
            gs = "" if p["gs"] is None else f"{p['gs']:.6f}"
            f.write(f"{p['step']},{p['mass']},{p['family']},{gs}\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
