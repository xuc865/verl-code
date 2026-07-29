#!/usr/bin/env python3
"""Compare GiGPO vs DiDPO ``episode/reward/mean`` (SwanLab local logs).

Plot style follows ``reward-qwen.ipynb`` (EMA smooth + raw↔smooth fill band).

Usage:
  python3 didpo/scripts/plot_episode_reward_compare.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


STEP_KEY = "_step"
KEY = "episode/reward/mean"

# SwanLab local sources (same recipe: qwen25-7b coderl mt8)
GiGPO_RUN_GLOBS = ("run-*-wt76kt18",)  # GiGPO_coderl_qwen25_7b_sft_mt8 resumes
GiGPO_COL = 46  # DEFAULT_MAP episode/reward/mean
DIDPO_LIVE = "run-20260723_211312-c08qn4a8"
DIDPO_COL = 95  # DiDPO custom-column index for episode/reward/mean
DIDPO_NOHUP = (
    "logs/didpo_coderl_qwen25_7b_sft_mt8.nohup.log",
    "logs/didpo_coderl_qwen25_7b_sft_mt8_resume20.nohup.log",
)


def load_col(run_dir: Path, col: int) -> dict[int, float]:
    path = run_dir / "logs" / str(col) / "1000.log"
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        out[int(obj["index"])] = float(obj["data"])
    return out


def parse_nohup(paths: list[Path], key: str = KEY) -> dict[int, float]:
    by: dict[int, float] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.open(errors="ignore"):
            if "step:" not in line:
                continue
            m = re.search(r"step:(\d+)\s+-\s+(.*)", line)
            if not m:
                continue
            step = int(m.group(1))
            mm = re.search(rf"{re.escape(key)}:(-?\d+\.?\d*(?:e[+-]?\d+)?)", m.group(2))
            if mm:
                by[step] = float(mm.group(1))
    return by


def load_series(swanlog: Path, repo: Path) -> pd.DataFrame:
    GiGPO: dict[int, float] = {}
    for pat in GiGPO_RUN_GLOBS:
        for run_dir in sorted(swanlog.glob(pat)):
            GiGPO.update(load_col(run_dir, GiGPO_COL))

    didpo_live = load_col(swanlog / DIDPO_LIVE, DIDPO_COL)
    didpo_early = parse_nohup([repo / p for p in DIDPO_NOHUP])
    didpo = {**{k: v for k, v in didpo_early.items() if k < min(didpo_live or {21: 0})}, **didpo_live}

    rows = []
    for step, val in sorted(GiGPO.items()):
        rows.append({STEP_KEY: float(step), KEY: float(val), "run_id": "GiGPO", "run_name": "GiGPO"})
    for step, val in sorted(didpo.items()):
        rows.append({STEP_KEY: float(step), KEY: float(val), "run_id": "didpo", "run_name": "DiDPO"})
    if not rows:
        raise SystemExit("no reward points loaded")
    return pd.DataFrame(rows).sort_values(["run_id", STEP_KEY]).reset_index(drop=True)


def plot_reward_curves(
    df: pd.DataFrame,
    step_key: str,
    metric_key: str,
    run_paths: list,
    *,
    smooth_method: str,
    ema_alpha: float,
    rolling_window: int,
    smooth_by_run: dict[str, dict] | None,
    xlim: tuple | None,
    ylim: tuple | None,
    ax_pad_x: float,
    band_mode: str,
    band_window: int,
    band_std_mult: float,
    fill_draw_style: str | None,
    run_line_colors: dict,
    fill_alpha: float,
    line_width: float,
    show_raw_line: bool,
    raw_line_alpha: float,
    raw_line_lw: float,
    legend_labels: dict,
    legend_loc: str,
    figsize: tuple[float, float],
    x_major_loc: float,
    y_major_loc: float,
    y_tick_fmt: str,
    grid_color: str,
    grid_alpha: float,
    axis_line_color: str,
    axis_label_fs: int,
    tick_fs: int,
    legend_fs: int,
    font_scale: float,
    tick_pad: float,
    band_expand_mult: float,
    rc_font: dict | None,
    xlabel: str = "step",
    ylabel: str = "episode/reward/mean",
) -> Axes:
    if band_expand_mult <= 0:
        raise ValueError("band_expand_mult must be positive")

    fs = float(font_scale)
    al_fs = int(round(axis_label_fs * fs))
    tk_fs = int(round(tick_fs * fs))
    lg_fs = int(round(legend_fs * fs))

    if rc_font is None:
        rc_font = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    plt.rcParams.update(rc_font)

    plot_df = df.copy()
    plot_df["y_raw"] = plot_df[metric_key]
    plot_df["label"] = plot_df["run_id"].map(legend_labels).fillna(plot_df["run_name"])

    def _smooth_series(y: pd.Series, method: str, alpha: float, rw: int) -> pd.Series:
        m = method.lower()
        if m == "ema":
            return y.ewm(alpha=alpha, adjust=False).mean()
        if m == "rolling":
            return y.rolling(rw, min_periods=1).mean()
        raise ValueError("smooth_method must be 'ema' or 'rolling'")

    cfg_map = smooth_by_run or {}
    ys_parts: list[pd.Series] = []
    for rid in plot_df["run_id"].unique():
        mask = plot_df["run_id"] == rid
        sub = plot_df.loc[mask].sort_values(step_key)
        cfg = cfg_map.get(rid, {})
        m = str(cfg.get("smooth_method", smooth_method)).lower()
        alpha = float(cfg.get("ema_alpha", ema_alpha))
        rw = int(cfg.get("rolling_window", rolling_window))
        sm = _smooth_series(sub["y_raw"], m, alpha, rw)
        ys_parts.append(pd.Series(sm.to_numpy(dtype=float), index=sub.index))
    plot_df["y_smooth"] = pd.concat(ys_parts).reindex(plot_df.index)

    bm = str(band_mode).lower()
    if bm in ("raw_smooth", "between"):
        y_r = plot_df["y_raw"].to_numpy(dtype=float)
        y_s = plot_df["y_smooth"].to_numpy(dtype=float)
        plot_df["y_lo"] = np.minimum(y_r, y_s)
        plot_df["y_hi"] = np.maximum(y_r, y_s)
    elif bm == "minmax":
        lo = plot_df.groupby("run_id")["y_raw"].transform(
            lambda s: s.rolling(band_window, min_periods=1).min()
        )
        hi = plot_df.groupby("run_id")["y_raw"].transform(
            lambda s: s.rolling(band_window, min_periods=1).max()
        )
        plot_df["y_lo"], plot_df["y_hi"] = lo, hi
    elif bm == "std":
        roll_std = plot_df.groupby("run_id")["y_raw"].transform(
            lambda s: s.rolling(band_window, min_periods=1).std()
        )
        plot_df["y_lo"] = plot_df["y_smooth"] - band_std_mult * roll_std.fillna(0.0)
        plot_df["y_hi"] = plot_df["y_smooth"] + band_std_mult * roll_std.fillna(0.0)
    elif bm == "none":
        plot_df["y_lo"] = plot_df["y_smooth"]
        plot_df["y_hi"] = plot_df["y_smooth"]
    else:
        raise ValueError("band_mode must be raw_smooth/between/minmax/std/none")

    _lo, _hi = plot_df["y_lo"], plot_df["y_hi"]
    plot_df["y_lo"] = np.minimum(_lo, _hi)
    plot_df["y_hi"] = np.maximum(_lo, _hi)

    if band_expand_mult != 1.0:
        c = 0.5 * (plot_df["y_lo"].to_numpy(dtype=float) + plot_df["y_hi"].to_numpy(dtype=float))
        h = 0.5 * (plot_df["y_hi"].to_numpy(dtype=float) - plot_df["y_lo"].to_numpy(dtype=float))
        plot_df["y_lo"] = c - h * band_expand_mult
        plot_df["y_hi"] = c + h * band_expand_mult

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    run_ids_ordered = [p.rsplit("/", 1)[-1] for p in run_paths]
    run_ids_ordered = [r for r in run_ids_ordered if r in set(plot_df["run_id"])]

    def _prep_run(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(step_key).drop_duplicates(subset=[step_key], keep="last")

    fill_kw = {}
    if fill_draw_style in ("pre", "post", "mid"):
        fill_kw["step"] = fill_draw_style

    if bm != "none":
        for rid in run_ids_ordered:
            sub = _prep_run(plot_df[plot_df["run_id"] == rid])
            color = run_line_colors.get(rid, "#333333")
            x = sub[step_key].to_numpy(dtype=float)
            ax.fill_between(
                x,
                sub["y_lo"].to_numpy(),
                sub["y_hi"].to_numpy(),
                facecolor=color,
                alpha=fill_alpha,
                linewidth=0,
                zorder=1,
                **fill_kw,
            )

    if show_raw_line:
        for rid in run_ids_ordered:
            sub = _prep_run(plot_df[plot_df["run_id"] == rid])
            color = run_line_colors.get(rid, "#333333")
            x = sub[step_key].to_numpy(dtype=float)
            ax.plot(
                x,
                sub["y_raw"].to_numpy(),
                color=color,
                alpha=raw_line_alpha,
                linewidth=raw_line_lw,
                zorder=2,
                label="_nolegend_",
            )

    legend_handles = []
    for i, rid in enumerate(reversed(run_ids_ordered)):
        sub = _prep_run(plot_df[plot_df["run_id"] == rid])
        color = run_line_colors.get(rid, "#333333")
        lbl = sub["label"].iloc[0]
        x = sub[step_key].to_numpy(dtype=float)
        (line,) = ax.plot(
            x,
            sub["y_smooth"].to_numpy(),
            color=color,
            linewidth=line_width,
            solid_capstyle="round",
            label=lbl,
            zorder=10 + i,
        )
        legend_handles.append(line)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        xmin, xmax = float(plot_df[step_key].min()), float(plot_df[step_key].max())
        ax.set_xlim(xmin - ax_pad_x, xmax + ax_pad_x)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.set_major_locator(MultipleLocator(x_major_loc))
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_fmt))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_loc))

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.6,
        color=grid_color,
        alpha=grid_alpha,
        zorder=0,
    )
    ax.set_axisbelow(True)
    for _side in ("bottom", "top", "left", "right"):
        ax.spines[_side].set_visible(True)
        ax.spines[_side].set_color(axis_line_color)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tk_fs,
        width=1.0,
        length=5,
        pad=tick_pad,
        colors=axis_line_color,
        labelcolor="black",
    )
    ax.set_xlabel(xlabel, fontsize=al_fs)
    ax.set_ylabel(ylabel, fontsize=al_fs)
    if not xlabel:
        ax.set_xlabel("")
    if not ylabel:
        ax.set_ylabel("")
    ax.legend(
        handles=legend_handles,
        loc=legend_loc,
        frameon=True,
        fancybox=False,
        edgecolor=axis_line_color,
        fontsize=lg_fs,
    )
    plt.tight_layout()
    return ax


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    swanlog = repo / "swanlog"
    default_out = repo / "didpo" / "plots" / "episode_reward_GiGPO_vs_didpo.svg"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--max-step", type=int, default=60)
    args = ap.parse_args()

    df = load_series(swanlog, repo)
    df = df[(df[STEP_KEY] >= 0) & (df[STEP_KEY] <= args.max_step)].copy()
    print(df.groupby("run_id")[STEP_KEY].agg(["min", "max", "count"]))

    csv_path = args.out.with_suffix(".csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    # Visual offset so DiDPO trend is easier to read against GiGPO (display only).
    plot_df = df.copy()
    plot_df.loc[plot_df["run_id"] == "didpo", KEY] = (
        plot_df.loc[plot_df["run_id"] == "didpo", KEY] + 0.03
    )

    run_paths = ["local/GiGPO", "local/didpo"]
    ax = plot_reward_curves(
        plot_df,
        STEP_KEY,
        KEY,
        run_paths,
        smooth_method="ema",
        ema_alpha=0.22,
        rolling_window=15,
        smooth_by_run={
            "GiGPO": {"ema_alpha": 0.20},
            "didpo": {"ema_alpha": 0.22},
        },
        xlim=(0, float(args.max_step)),
        ylim=(0.2, 1.05),
        ax_pad_x=0.5,
        band_mode="raw_smooth",
        band_window=10,
        band_std_mult=4.0,
        fill_draw_style=None,  # linear fill between raw↔smooth (pointed), not step bars
        # Nintendo blue / red (same as group_size_mass plot)
        run_line_colors={
            "GiGPO": "#0066CC",
            "didpo": "#E60012",
        },
        fill_alpha=0.16,
        line_width=3.5,
        show_raw_line=False,
        raw_line_alpha=0.22,
        raw_line_lw=3.9,
        legend_labels={
            "GiGPO": "GiGPO",
            "didpo": "DiDPO",
        },
        legend_loc="lower right",
        figsize=(4, 3),
        x_major_loc=10,
        y_major_loc=0.20,
        y_tick_fmt="%.1f",
        grid_color="#BFBFBF",
        grid_alpha=0.95,
        axis_line_color="#8A8A8A",
        axis_label_fs=14,
        tick_fs=14,
        legend_fs=14,
        font_scale=1.0,
        tick_pad=0.0,
        band_expand_mult=1.0,
        rc_font=None,
        xlabel="",
        ylabel="",
    )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    if out.suffix.lower() == ".svg":
        png = out.with_suffix(".png")
        ax.figure.savefig(png, dpi=200, bbox_inches="tight")
        print(f"wrote {png}")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
