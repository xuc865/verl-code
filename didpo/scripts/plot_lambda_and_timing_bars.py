#!/usr/bin/env python3
"""Two horizontal seaborn bar charts:

1) lambda sensitivity (APPS) — Nintendo red
2) per-operation step time — Nintendo blue

Usage:
  python3 didpo/scripts/plot_lambda_and_timing_bars.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NINTENDO_RED = "#E60012"
NINTENDO_BLUE = "#0066CC"

LAMBDA_APPS = [
    (0.0, 23.8),
    (0.2, 27.9),
    (0.4, 30.1),
    (0.6, 31.0),
    (1.0, 30.4),
    (1.2, 31.3),
    (1.5, 31.0),
    (2.0, 30.8),
    (3.0, 26.6),
]

# Mean per-step times (seconds), user table (×0.1 of raw log seconds is fine —
# we plot the provided numbers as-is).
OP_TIME_S = [
    ("rollout", 211.60),
    ("update", 67.90),
    ("ref", 18.30),
    ("prob_old", 18.40),
    ("prob_adv", 7.60),
    ("anchor calc", 7.50),
    ("grouping", 0.06),
    (r"$A^E$", 0.03),
    (r"$A^D$", 0.03),
]


def _setup_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlesize": 13,
        }
    )


def plot_lambda(out: Path) -> None:
    df = pd.DataFrame(LAMBDA_APPS, columns=["lambda", "APPS"])
    # categorical y so bars keep lambda order (small → large top→bottom or reverse)
    df["lambda_label"] = df["lambda"].map(lambda x: f"{x:g}")
    order = list(df["lambda_label"])

    sns.set_theme(style="whitegrid", context="paper")
    _setup_font()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        data=df,
        y="lambda_label",
        x="APPS",
        order=order,
        orient="h",
        color=NINTENDO_RED,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_xlabel("APPS")
    ax.set_ylabel(r"$\lambda$")
    # value-axis lower bound at 20 (horizontal bars → xlim)
    ax.set_xlim(20, 35)
    ax.grid(axis="x", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for patch, val in zip(ax.patches, df["APPS"]):
        ax.text(
            val + 0.25,
            patch.get_y() + patch.get_height() / 2,
            f"{val:.1f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.svg')}")
    plt.close(fig)


def plot_timing(out: Path) -> None:
    df = pd.DataFrame(OP_TIME_S, columns=["component", "time_s"])
    # Keep table order top→bottom (rollout first)
    order = list(df["component"])

    sns.set_theme(style="whitegrid", context="paper")
    _setup_font()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        data=df,
        y="component",
        x="time_s",
        order=order,
        orient="h",
        color=NINTENDO_BLUE,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    # log-x compresses the long rollout/update bars so tiny AE/Ad are visible
    ax.set_xscale("log")
    ax.set_xlim(0.02, 400)
    ax.grid(axis="x", which="both", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for patch, val in zip(ax.patches, df["time_s"]):
        ax.text(
            val * 1.2,
            patch.get_y() + patch.get_height() / 2,
            f"{val:g}",
            va="center",
            ha="left",
            fontsize=10,
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.svg')}")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    plot_dir = repo / "didpo" / "plots"
    plot_lambda(plot_dir / "lambda_sensitivity_apps.png")
    plot_timing(plot_dir / "step_component_timing.png")


if __name__ == "__main__":
    main()
