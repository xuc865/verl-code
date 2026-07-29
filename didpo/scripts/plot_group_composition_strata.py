#!/usr/bin/env python3
"""100% stacked strata of DiDPO group-mass composition vs step.

Pools all tracked instances. Cross-task macro bands (not problem-specific
algorithms like sorted/scan):

  block     — substantive multi-line solution bodies (mean_size >= 8)
  fragment  — short real patches / local edits
  harness   — IO / typing / __main__ scaffolding
  stub      — ``return None`` / trivial abort

Usage:
  python3 didpo/scripts/plot_group_composition_strata.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


# Bottom → top.
CLASSES = [
    "block",
    "fragment",
    "harness",
    "stub",
]

COLORS = {
    # Morandi / muted earth tones (soft, distinguishable)
    "block": "#A3B18A",      # sage green
    "fragment": "#8E9AAF",   # dusty blue-gray
    "harness": "#D4C4B0",    # warm sand
    "stub": "#C9A9A6",       # dusty rose
}

HATCHES = {
    "block": "...",
    "fragment": "///",
    "harness": "xxx",
    "stub": "\\\\\\",
}

LABELS = {
    "block": "block",
    "fragment": "fragment",
    "harness": "harness",
    "stub": "stub",
}

DEFAULT_IDS = ("apps__idx3836", "apps__idx417", "apps__idx46")
BLOCK_MEAN_SIZE = 8.0


def macro_family(preview: str | None, mean_size: float) -> str:
    """Cross-task structural family."""
    p = (preview or "").replace("\\n", "\n")
    pl = " ".join(p.split())
    if "return None" in pl or pl.strip() in {"pass", "return", "...", "return None"}:
        return "stub"
    if re.search(
        r"__main__|stdin|stdout|json\.loads|\binput\s*\(|\bprint\s*\(|from typing",
        pl,
    ):
        return "harness"
    if float(mean_size) >= BLOCK_MEAN_SIZE:
        return "block"
    return "fragment"


def load_mass_by_step(
    jsonl: Path,
    *,
    instance_ids: list[str] | None,
    min_step: int,
    max_step: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    want = set(instance_ids) if instance_ids else None
    rows = [
        json.loads(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    mass: dict[int, dict[str, float]] = {}
    seen_ids: set[str] = set()
    for r in rows:
        iid = r.get("instance_id")
        if iid is None:
            continue
        if want is not None and iid not in want:
            continue
        step = int(r["step"])
        if step < min_step or step > max_step:
            continue
        seen_ids.add(str(iid))
        bucket = mass.setdefault(step, {c: 0.0 for c in CLASSES})
        for g in r.get("groups") or []:
            fam = macro_family(g.get("preview"), float(g.get("mean_size") or 0.0))
            bucket[fam] += float(g.get("size") or 0)

    if not mass:
        raise SystemExit(
            f"no dumps for {sorted(want) if want else 'ALL'} "
            f"in steps [{min_step}, {max_step}]"
        )

    steps = np.array(sorted(mass), dtype=float)
    fracs = np.zeros((len(CLASSES), len(steps)), dtype=float)
    for j, step in enumerate(steps):
        tot = sum(mass[int(step)].values()) or 1.0
        for i, c in enumerate(CLASSES):
            fracs[i, j] = mass[int(step)][c] / tot
    return steps, fracs, sorted(seen_ids)


def smooth_strata(
    steps: np.ndarray,
    fracs: np.ndarray,
    *,
    min_step: int,
    max_step: int,
    grid_dx: float = 0.25,
    sigma_steps: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(min_step, max_step + 1e-9, grid_dx)
    dense = np.zeros((fracs.shape[0], grid.size), dtype=float)
    for i in range(fracs.shape[0]):
        dense[i] = np.interp(grid, steps, fracs[i])

    sigma = max(0.5, sigma_steps / grid_dx)
    for i in range(dense.shape[0]):
        dense[i] = gaussian_filter1d(dense[i], sigma=sigma, mode="nearest")

    dense = np.clip(dense, 0.0, None)
    col = dense.sum(axis=0, keepdims=True)
    col = np.where(col > 0, col, 1.0)
    dense = dense / col
    return grid, dense


def plot(
    steps: np.ndarray,
    fracs: np.ndarray,
    *,
    out: Path,
    title: str,
    min_step: int,
    max_step: int,
    dpi: int,
) -> None:
    from matplotlib.patches import Patch

    fs = 11
    plt.rcParams.update(
        {
            "font.size": fs,
            # Prefer Times New Roman; fall back to STIX (Times-like, bundled with mpl).
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "hatch.linewidth": 0.55,
            "hatch.color": "#5C5C5C",
        }
    )
    # Legend sits on top; keep 4x3 canvas for paper layout.
    fig, ax = plt.subplots(figsize=(4, 3))

    polys = ax.stackplot(
        steps,
        fracs,
        colors=[COLORS[c] for c in CLASSES],
        alpha=0.88,
        linewidth=0.6,
        edgecolor="#F7F5F2",
    )
    for poly, cls in zip(polys, CLASSES):
        poly.set_hatch(HATCHES[cls])
        poly.set_edgecolor("#F7F5F2")
        poly.set_linewidth(0.6)

    ax.set_xlim(min_step, max_step)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=fs, length=3, pad=2)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6, color="#B0B0B0")
    ax.set_axisbelow(True)
    ax.set_facecolor("#FBF9F6")

    legend_handles = [
        Patch(
            facecolor=COLORS[c],
            edgecolor="#6E6E6E",
            linewidth=0.5,
            hatch=HATCHES[c],
            label=LABELS[c],
            alpha=0.9,
        )
        for c in reversed(CLASSES)  # stub … block, matching top→bottom reading
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=fs,
        handlelength=1.6,
        handleheight=1.1,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_jsonl = repo / "logs" / "didpo_groups" / "didpo_prompt_groups.jsonl"
    default_out = repo / "didpo" / "plots" / "group_composition_strata_20_60.png"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=default_jsonl)
    ap.add_argument("--instance-ids", nargs="*", default=list(DEFAULT_IDS))
    ap.add_argument("--min-step", type=int, default=20)
    ap.add_argument("--max-step", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--dpi", type=int, default=240)
    args = ap.parse_args()

    steps, fracs, used = load_mass_by_step(
        args.jsonl,
        instance_ids=list(args.instance_ids) if args.instance_ids else None,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    shorts = ",".join(s.replace("apps__", "") for s in used)
    title = f"pooled · {shorts}"
    print(f"pooled instances: {used}")

    grid, smooth = smooth_strata(
        steps,
        fracs,
        min_step=args.min_step,
        max_step=args.max_step,
        sigma_steps=args.sigma,
    )
    plot(
        grid,
        smooth,
        out=args.out,
        title=title,
        min_step=args.min_step,
        max_step=args.max_step,
        dpi=args.dpi,
    )
    if args.out.suffix.lower() == ".png":
        plot(
            grid,
            smooth,
            out=args.out.with_suffix(".svg"),
            title=title,
            min_step=args.min_step,
            max_step=args.max_step,
            dpi=args.dpi,
        )

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step," + ",".join(CLASSES) + "\n")
        for j, step in enumerate(steps):
            vals = ",".join(f"{fracs[i, j]:.6f}" for i in range(len(CLASSES)))
            f.write(f"{int(step)},{vals}\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
