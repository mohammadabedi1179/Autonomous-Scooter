#!/usr/bin/env python3
"""
Depth Accuracy Plotter — Thesis Nude Contrast Style

Reads depth_experiment_results.csv and creates a thesis-ready depth accuracy figure:
1) Measured distance vs true distance
2) Signed relative error vs true distance

Usage:
    python plot_depth_accuracy_thesis_style.py --csv depth_experiment_results.csv --out plots_depth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THEME = {
    "BG": "#FAF6F0",
    "BG_AXES": "#FAF6F0",
    "GRID_CLR": "#E8DDD2",
    "TEXT_CLR": "#463F3A",
    "TICK_CLR": "#6B625B",
    "EDGE_CLR": "#D8CABD",
    "PAPER_WHITE": "#FAF6F0",
    "PRIMARY": "#7D9D9C",
    "SECONDARY": "#C9A27E",
    "ACCENT": "#B7A99A",
    "GOOD": "#87986A",
    "MARGINAL": "#C98245",
    "BAD": "#A44A3F",
}


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": THEME["BG"],
        "axes.facecolor": THEME["BG_AXES"],
        "axes.edgecolor": THEME["EDGE_CLR"],
        "axes.grid": True,
        "grid.color": THEME["GRID_CLR"],
        "grid.alpha": 0.8,
        "grid.linewidth": 0.45,
        "grid.linestyle": "-",
        "axes.labelcolor": THEME["TEXT_CLR"],
        "text.color": THEME["TEXT_CLR"],
        "xtick.color": THEME["TICK_CLR"],
        "ytick.color": THEME["TICK_CLR"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.facecolor": THEME["PAPER_WHITE"],
        "legend.edgecolor": THEME["EDGE_CLR"],
        "legend.labelcolor": THEME["TEXT_CLR"],
        "legend.framealpha": 0.95,
        "savefig.facecolor": THEME["BG"],
        "savefig.dpi": 300,
    })


def line_with_shadow(ax, x, y, color, lw=2.2, label=None, alpha=0.95):
    # Matplotlib with newer pandas versions may fail when a pandas Series is passed
    # because it internally tries multi-dimensional indexing. Convert explicitly.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return

    x = x[valid]
    y = y[valid]

    ax.plot(x, y, linewidth=lw + 1.2, color="#bbbbbb", alpha=0.25, solid_capstyle="round")
    ax.plot(x, y, linewidth=lw, color=color, alpha=alpha, solid_capstyle="round", label=label)


def add_legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=9, frameon=True)
    leg.get_frame().set_linewidth(0.9)
    leg.get_frame().set_alpha(0.95)


def plot_depth_accuracy(csv_path: str, out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).copy()
    df = df.sort_values("true_m").reset_index(drop=True)

    required = {"true_m", "measured_m", "error_m", "error_pct", "depth_std_m", "coverage_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    quality = df["quality"].fillna("UNKNOWN").astype(str) if "quality" in df.columns else pd.Series(["UNKNOWN"] * len(df))
    color_map = {
        "GOOD": THEME["GOOD"],
        "MARGINAL": THEME["MARGINAL"],
        "BAD": THEME["BAD"],
        "UNKNOWN": THEME["ACCENT"],
    }
    colors = [color_map.get(q.upper(), THEME["ACCENT"]) for q in quality]

    mae = df["error_m"].abs().mean()
    rmse = np.sqrt((df["error_m"] ** 2).mean())
    mape = df["error_pct"].abs().mean()
    bias = df["error_m"].mean()
    mean_cov = df["coverage_pct"].mean()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11.5, 8.2),
        gridspec_kw={"height_ratios": [2.2, 1.15], "hspace": 0.30}
    )

    ax = axes[0]
    x = df["true_m"].to_numpy()
    y = df["measured_m"].to_numpy()
    yerr = df["depth_std_m"].to_numpy()

    lim_min = max(0, min(x.min(), y.min()) - 0.20)
    lim_max = max(x.max(), y.max()) + 0.30

    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        linestyle="--", linewidth=1.2, color=THEME["TICK_CLR"], alpha=0.7,
        label="Ideal measurement"
    )
    ax.fill_between(
        [lim_min, lim_max],
        [lim_min - 0.10, lim_max - 0.10],
        [lim_min + 0.10, lim_max + 0.10],
        color=THEME["PRIMARY"], alpha=0.10, linewidth=0,
        label="±0.10 m band"
    )
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="none",
        ecolor=THEME["ACCENT"],
        elinewidth=0.9,
        capsize=2.5,
        alpha=0.55,
        zorder=1,
    )

    for q in sorted(set(quality)):
        mask = quality == q
        ax.scatter(
            df.loc[mask, "true_m"],
            df.loc[mask, "measured_m"],
            s=62,
            color=color_map.get(str(q).upper(), THEME["ACCENT"]),
            edgecolors=THEME["PAPER_WHITE"],
            linewidths=1.0,
            alpha=0.92,
            label=str(q).capitalize(),
            zorder=3,
        )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Measured distance (m)")
    ax.set_title("Stereo Depth Accuracy — Measured vs True Distance", fontsize=13, fontweight="bold", pad=12)

    metrics_txt = (
        f"n = {len(df)}\n"
        f"MAE = {mae:.3f} m\n"
        f"RMSE = {rmse:.3f} m\n"
        f"MAPE = {mape:.2f}%\n"
        f"Bias = {bias:+.3f} m\n"
        f"Coverage = {mean_cov:.1f}%"
    )
    ax.text(
        0.03, 0.97, metrics_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor=THEME["PAPER_WHITE"],
            edgecolor=THEME["EDGE_CLR"],
            alpha=0.95,
        ),
    )
    add_legend(ax, loc="lower right")

    ax2 = axes[1]
    ax2.axhline(0, color=THEME["TICK_CLR"], linewidth=1.0, alpha=0.65)
    ax2.fill_between(
        [lim_min, lim_max], [-5, -5], [5, 5],
        color=THEME["GOOD"], alpha=0.10, linewidth=0,
        label="±5% band"
    )
    ax2.scatter(
        df["true_m"], df["error_pct"],
        s=54,
        color=colors,
        edgecolors=THEME["PAPER_WHITE"],
        linewidths=0.9,
        alpha=0.92,
    )
    line_with_shadow(
        ax2,
        df["true_m"],
        df["error_pct"].rolling(window=3, center=True, min_periods=1).mean(),
        THEME["SECONDARY"],
        lw=2.1,
        label="Smoothed error"
    )
    ax2.set_xlabel("True distance (m)")
    ax2.set_ylabel("Error (%)")
    ax2.set_title("Signed Relative Error Across Test Distances", fontsize=12, fontweight="bold", pad=10)
    add_legend(ax2, loc="best")

    png_path = out_dir / "depth_accuracy_thesis_style.png"
    pdf_path = out_dir / "depth_accuracy_thesis_style.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300, facecolor=THEME["BG"])
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=THEME["BG"])
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to depth_experiment_results.csv")
    parser.add_argument("--out", default="depth_accuracy_plots", help="Output directory")
    args = parser.parse_args()

    apply_theme()
    plot_depth_accuracy(args.csv, args.out)


if __name__ == "__main__":
    main()
