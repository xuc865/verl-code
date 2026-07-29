#!/usr/bin/env python3
"""Bar chart: item-mass fraction vs group size (whole-diff vs DiDPO sub-diff).

Data matches the canvas ``didpo-group-size-mass`` comparison:
  - same similarity threshold 0.8
  - whole-diff: multi-prompt / multi-config controlled batches
  - sub-diff: real training dumps in ``logs/didpo_groups/didpo_prompt_groups.jsonl``

Usage:
  python3 didpo/scripts/plot_group_size_mass.py
  python3 didpo/scripts/plot_group_size_mass.py --out didpo/plots/group_size_mass.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Embedded histogram (percent of item mass in groups of size k)
# k = 1 .. 33
# --------------------------------------------------------------------------- #
THRESH = 0.8
plt.rcParams["font.size"] = 15
FULL_PCT = [
    5.28, 6.64, 3.17, 7.24, 18.3, 11.76, 2.11, 9.65, 2.71, 6.03, 3.32, 3.62, 0.0,
    4.78, 0.0, 7.24, 0.0, 8.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
]
REAL_PCT = [
    0.0, 2.15, 2.31, 3.69, 4.03, 3.0, 4.57, 3.38, 3.8, 4.23, 3.8, 3.23, 4.0, 2.15,
    2.88, 3.07, 3.92, 5.53, 8.03, 4.61, 4.03, 2.54, 5.3, 2.77, 1.92, 1.0, 2.07,
    3.23, 1.11, 0.0, 2.38, 0.0, 1.27,
]

LABEL_FULL = "Whole-diff"
LABEL_SUB = "Sub-diff"


def build_frame() -> pd.DataFrame:
    assert len(FULL_PCT) == len(REAL_PCT)
    ks = list(range(1, len(FULL_PCT) + 1))
    rows = []
    for k, a, b in zip(ks, FULL_PCT, REAL_PCT):
        rows.append({"group_size": k, "item_mass_pct": a, "method": LABEL_FULL})
        rows.append({"group_size": k, "item_mass_pct": b, "method": LABEL_SUB})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 5.5))

    sns.barplot(
        data=df,
        x="group_size",
        y="item_mass_pct",
        hue="method",
        ax=ax,
        palette=["#0066CC", "#E60012"],  # Nintendo blue / red (swapped)
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("Group size $k$")
    ax.set_ylabel("Percentage of Group Mass %")
    # ax.set_title(
    #     f"Group-size mass distribution (sim_thresh={THRESH})\n"
    #     "Whole-diff saturates at smaller $k$; training sub-diff extends to large $k$"
    # )

    # Categorical bar positions are 0..n-1 for k=1..33.
    xtick_ks =  [i for i in range(1, 34)]# [1, 5, 10, 15, 20, 25, 30, 33]
    ax.set_xticks([k - 1 for k in xtick_ks])
    ax.set_xticklabels([str(k) for k in xtick_ks])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylim(0, 20)
    # Tighten left/right padding: categorical positions are 0 .. n_cats-1.
    n_cats = len(FULL_PCT)
    pad = 0.95  # small gap past the outer bars (was ~0.5–1.0 by default)
    ax.set_xlim(-pad, n_cats - 1 + pad)
    ax.margins(x=0)
    ax.legend(
        title="",
        loc="upper right",
        frameon=True,
        fontsize=18,
        ncol=2,
        markerscale=1.0,
        handlelength=2.8,
        handleheight=0.65,  # vertically flatter patches
        handletextpad=0.6,
        columnspacing=1.6,
        borderpad=0.35,     # less vertical padding inside frame
        labelspacing=0.15,
        borderaxespad=0.5,
        fancybox=False,
        framealpha=0.95,
    )

    # Light emphasis on the large-k region where whole-diff is empty.
    ax.axvspan(18.5 - 1, n_cats - 1 + pad, color="#E60012", alpha=0.06, zorder=0)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    # Also save CSV next to the figure for reuse.
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_out = repo / "didpo" / "plots" / "group_size_mass.svg"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default_out, help="output PNG path")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    df = build_frame()
    plot(df, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
