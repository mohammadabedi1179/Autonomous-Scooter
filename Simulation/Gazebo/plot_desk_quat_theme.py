#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desk Quaternion / IMU Plotter
Thesis-style plots for desk_quat_test.csv

Usage:
    python3 plot_desk_quat_theme.py --csv desk_quat_test.csv --out desk_quat_plots --rolling 15

Outputs:
    01_heading_comparison.png
    02_heading_smoothed.png
    03_relative_heading_change.png
    04_gyro_rates.png
    05_quaternion_components.png
    06_euler_angles.png
    07_magnetometer_norm.png
    08_heading_delta.png
    09_heading_error_vs_quaternion.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Theme: thesis_nude_contrast
# ============================================================
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

    "GREEN": "#87986A",
    "RED": "#A44A3F",
    "ORANGE": "#C98245",
    "DARK": "#5E6472",
    "GRAY": "#9A928B",
}


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": THEME["BG"],
        "axes.facecolor": THEME["BG_AXES"],
        "axes.edgecolor": THEME["EDGE_CLR"],
        "axes.grid": True,
        "grid.color": THEME["GRID_CLR"],
        "grid.alpha": 0.85,
        "grid.linewidth": 0.45,
        "grid.linestyle": "-",

        "axes.labelcolor": THEME["TEXT_CLR"],
        "text.color": THEME["TEXT_CLR"],
        "xtick.color": THEME["TICK_CLR"],
        "ytick.color": THEME["TICK_CLR"],

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,

        "font.size": 10,
        "font.family": "DejaVu Sans",

        "legend.facecolor": THEME["PAPER_WHITE"],
        "legend.edgecolor": THEME["EDGE_CLR"],
        "legend.labelcolor": THEME["TEXT_CLR"],
        "legend.framealpha": 0.95,

        "savefig.facecolor": THEME["BG"],
        "savefig.dpi": 300,
    })


# ============================================================
# Helper functions
# ============================================================
def to_numeric_array(x):
    """
    Convert pandas Series/list/array to clean numpy float array.
    This avoids matplotlib/pandas version incompatibility.
    """
    return np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)


def get_col(df, col):
    """
    Return numeric column as numpy array.
    If missing, return None.
    """
    if col not in df.columns:
        print(f"[skip] Missing column: {col}")
        return None
    return to_numeric_array(df[col])


def has_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[skip] Missing columns: {missing}")
        return False
    return True


def rolling_mean(y, window=15):
    y = to_numeric_array(y)
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def valid_xy(x, y):
    x = to_numeric_array(x)
    y = to_numeric_array(y)

    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def line_with_shadow(ax, x, y, color, label, lw=2.0, alpha=0.95):
    x, y = valid_xy(x, y)

    if len(x) == 0:
        print(f"[skip] No valid data for: {label}")
        return

    ax.plot(
        x, y,
        color="#BBBBBB",
        linewidth=lw + 1.15,
        alpha=0.25,
        solid_capstyle="round"
    )

    ax.plot(
        x, y,
        color=color,
        linewidth=lw,
        alpha=alpha,
        label=label,
        solid_capstyle="round"
    )


def scatter_light(ax, x, y, color, label=None, s=14, alpha=0.22):
    x, y = valid_xy(x, y)

    if len(x) == 0:
        return

    ax.scatter(
        x, y,
        s=s,
        color=color,
        alpha=alpha,
        edgecolors="none",
        label=label
    )


def finish_plot(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True)

    if legend:
        leg = ax.legend(loc="best", fontsize=9)
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_linewidth(0.8)


def savefig(outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=THEME["BG"])
    plt.close()

    print(f"saved: {path}")


def unwrap_deg(angle_deg):
    """
    Unwrap angle in degrees to avoid jumps near +/-180 degrees.
    Useful for yaw/heading visualization.
    """
    a = to_numeric_array(angle_deg)
    return np.rad2deg(np.unwrap(np.deg2rad(a)))


def print_data_summary(df):
    print("\n=== CSV columns ===")
    for c in df.columns:
        valid_count = pd.to_numeric(df[c], errors="coerce").notna().sum()
        print(f"{c}: {valid_count}/{len(df)} numeric")
    print("===================\n")


# ============================================================
# Plot functions
# ============================================================
def plot_heading_comparison(df, t, outdir):
    required = ["quat_heading_deg", "euler_yaw_deg", "gyro_heading_deg"]
    if not has_cols(df, required):
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(
        ax, t, df["quat_heading_deg"],
        THEME["PRIMARY"],
        "Quaternion heading",
        lw=2.3
    )

    line_with_shadow(
        ax, t, df["euler_yaw_deg"],
        THEME["SECONDARY"],
        "Euler yaw",
        lw=2.0
    )

    line_with_shadow(
        ax, t, df["gyro_heading_deg"],
        THEME["DARK"],
        "Gyro integrated heading",
        lw=1.9
    )

    finish_plot(
        ax,
        "Heading Comparison",
        "Time (s)",
        "Heading / yaw (deg)"
    )

    savefig(outdir, "01_heading_comparison.png")


def plot_heading_smoothed(df, t, outdir, rolling):
    required = ["quat_heading_deg", "euler_yaw_deg", "gyro_heading_deg"]
    if not has_cols(df, required):
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(
        ax, t, rolling_mean(df["quat_heading_deg"], rolling),
        THEME["PRIMARY"],
        f"Quaternion heading rolling mean (w={rolling})",
        lw=2.4
    )

    line_with_shadow(
        ax, t, rolling_mean(df["euler_yaw_deg"], rolling),
        THEME["SECONDARY"],
        f"Euler yaw rolling mean (w={rolling})",
        lw=2.2
    )

    line_with_shadow(
        ax, t, rolling_mean(df["gyro_heading_deg"], rolling),
        THEME["DARK"],
        f"Gyro heading rolling mean (w={rolling})",
        lw=2.0
    )

    finish_plot(
        ax,
        "Smoothed Heading Comparison",
        "Time (s)",
        "Heading / yaw (deg)"
    )

    savefig(outdir, "02_heading_smoothed.png")


def plot_relative_heading_change(df, t, outdir, unwrap=False):
    required = ["quat_heading_deg", "euler_yaw_deg", "gyro_heading_deg"]
    if not has_cols(df, required):
        return

    q = get_col(df, "quat_heading_deg")
    e = get_col(df, "euler_yaw_deg")
    g = get_col(df, "gyro_heading_deg")

    if unwrap:
        q = unwrap_deg(q)
        e = unwrap_deg(e)
        g = unwrap_deg(g)

    q_delta = q - q[np.isfinite(q)][0]
    e_delta = e - e[np.isfinite(e)][0]
    g_delta = g - g[np.isfinite(g)][0]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(ax, t, q_delta, THEME["PRIMARY"], "Quaternion relative change", lw=2.3)
    line_with_shadow(ax, t, e_delta, THEME["SECONDARY"], "Euler relative change", lw=2.0)
    line_with_shadow(ax, t, g_delta, THEME["DARK"], "Gyro relative change", lw=1.9)

    ax.axhline(0, color=THEME["EDGE_CLR"], linewidth=1.1, alpha=0.9)

    finish_plot(
        ax,
        "Heading Change Relative to Initial Pose",
        "Time (s)",
        "Relative heading change (deg)"
    )

    savefig(outdir, "03_relative_heading_change.png")


def plot_gyro_rates(df, t, outdir):
    required = ["gx_rad_s", "gy_rad_s", "gz_rad_s"]
    if not has_cols(df, required):
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(ax, t, df["gx_rad_s"], THEME["PRIMARY"], "gx", lw=1.8)
    line_with_shadow(ax, t, df["gy_rad_s"], THEME["SECONDARY"], "gy", lw=1.8)
    line_with_shadow(ax, t, df["gz_rad_s"], THEME["RED"], "gz", lw=1.8)

    ax.axhline(0, color=THEME["EDGE_CLR"], linewidth=1.0, alpha=0.8)

    finish_plot(
        ax,
        "Gyroscope Angular Rates",
        "Time (s)",
        "Angular rate (rad/s)"
    )

    savefig(outdir, "04_gyro_rates.png")


def plot_quaternion_components(df, t, outdir):
    required = ["qx", "qy", "qz", "qw"]
    if not has_cols(df, required):
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(ax, t, df["qx"], THEME["PRIMARY"], "qx", lw=1.7)
    line_with_shadow(ax, t, df["qy"], THEME["SECONDARY"], "qy", lw=1.7)
    line_with_shadow(ax, t, df["qz"], THEME["GREEN"], "qz", lw=1.7)
    line_with_shadow(ax, t, df["qw"], THEME["DARK"], "qw", lw=1.9)

    finish_plot(
        ax,
        "Quaternion Components",
        "Time (s)",
        "Quaternion value"
    )

    savefig(outdir, "05_quaternion_components.png")


def plot_euler_angles(df, t, outdir):
    required = ["roll_rad", "pitch_rad", "yaw_rad"]
    if not has_cols(df, required):
        return

    roll_deg = np.rad2deg(get_col(df, "roll_rad"))
    pitch_deg = np.rad2deg(get_col(df, "pitch_rad"))
    yaw_deg = np.rad2deg(get_col(df, "yaw_rad"))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(ax, t, roll_deg, THEME["PRIMARY"], "Roll", lw=1.9)
    line_with_shadow(ax, t, pitch_deg, THEME["SECONDARY"], "Pitch", lw=1.9)
    line_with_shadow(ax, t, yaw_deg, THEME["DARK"], "Yaw", lw=1.9)

    ax.axhline(0, color=THEME["EDGE_CLR"], linewidth=1.0, alpha=0.8)

    finish_plot(
        ax,
        "Euler Angles",
        "Time (s)",
        "Angle (deg)"
    )

    savefig(outdir, "06_euler_angles.png")


def plot_magnetometer_norm(df, t, outdir, rolling):
    required = ["mag_norm_gauss"]
    if not has_cols(df, required):
        return

    mag = get_col(df, "mag_norm_gauss")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    scatter_light(
        ax, t, mag,
        THEME["GREEN"],
        label="Raw magnetic norm",
        s=12,
        alpha=0.25
    )

    line_with_shadow(
        ax, t, rolling_mean(mag, rolling),
        THEME["DARK"],
        f"Rolling mean (w={rolling})",
        lw=2.2
    )

    finish_plot(
        ax,
        "Magnetometer Norm",
        "Time (s)",
        "Magnetic field norm (gauss)"
    )

    savefig(outdir, "07_magnetometer_norm.png")


def plot_heading_delta(df, t, outdir):
    required = ["quat_delta_deg", "euler_delta_deg"]
    if not has_cols(df, required):
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(
        ax, t, df["quat_delta_deg"],
        THEME["PRIMARY"],
        "Quaternion delta",
        lw=2.1
    )

    line_with_shadow(
        ax, t, df["euler_delta_deg"],
        THEME["SECONDARY"],
        "Euler delta",
        lw=2.1
    )

    ax.axhline(0, color=THEME["EDGE_CLR"], linewidth=1.0, alpha=0.8)

    finish_plot(
        ax,
        "Step-wise Heading Delta",
        "Time (s)",
        "Delta (deg)"
    )

    savefig(outdir, "08_heading_delta.png")


def plot_heading_error_vs_quaternion(df, t, outdir, rolling):
    required = ["quat_heading_deg", "euler_yaw_deg", "gyro_heading_deg"]
    if not has_cols(df, required):
        return

    q = get_col(df, "quat_heading_deg")
    e = get_col(df, "euler_yaw_deg")
    g = get_col(df, "gyro_heading_deg")

    # Compare relative changes, not absolute offsets
    q_rel = q - q[np.isfinite(q)][0]
    e_rel = e - e[np.isfinite(e)][0]
    g_rel = g - g[np.isfinite(g)][0]

    e_err = e_rel - q_rel
    g_err = g_rel - q_rel

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_with_shadow(
        ax, t, rolling_mean(e_err, rolling),
        THEME["SECONDARY"],
        f"Euler - Quaternion error (rolling w={rolling})",
        lw=2.2
    )

    line_with_shadow(
        ax, t, rolling_mean(g_err, rolling),
        THEME["RED"],
        f"Gyro - Quaternion error (rolling w={rolling})",
        lw=2.2
    )

    ax.axhline(0, color=THEME["EDGE_CLR"], linewidth=1.1, alpha=0.9)

    finish_plot(
        ax,
        "Heading Error Relative to Quaternion Estimate",
        "Time (s)",
        "Relative error (deg)"
    )

    savefig(outdir, "09_heading_error_vs_quaternion.png")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot desk quaternion / IMU CSV with thesis nude contrast theme."
    )

    parser.add_argument(
        "--csv",
        default="desk_quat_test.csv",
        help="Path to input CSV file."
    )

    parser.add_argument(
        "--out",
        default="desk_quat_plots",
        help="Output directory."
    )

    parser.add_argument(
        "--rolling",
        type=int,
        default=15,
        help="Rolling window size."
    )

    parser.add_argument(
        "--unwrap",
        action="store_true",
        help="Use angle unwrapping for relative heading plot."
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print numeric availability summary for all columns."
    )

    args = parser.parse_args()

    apply_theme()

    csv_path = Path(args.csv)
    outdir = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.summary:
        print_data_summary(df)

    if "time_s" in df.columns:
        t = get_col(df, "time_s")
    elif "time" in df.columns:
        t = get_col(df, "time")
    else:
        print("[warning] No time column found. Using sample index as x-axis.")
        t = np.arange(len(df), dtype=float)

    print(f"Loaded CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Output directory: {outdir.resolve()}")
    print("Generating plots...\n")

    plot_heading_comparison(df, t, outdir)
    plot_heading_smoothed(df, t, outdir, args.rolling)
    plot_relative_heading_change(df, t, outdir, unwrap=args.unwrap)
    plot_gyro_rates(df, t, outdir)
    plot_quaternion_components(df, t, outdir)
    plot_euler_angles(df, t, outdir)
    plot_magnetometer_norm(df, t, outdir, args.rolling)
    plot_heading_delta(df, t, outdir)
    plot_heading_error_vs_quaternion(df, t, outdir, args.rolling)

    print("\nDone.")
    print(f"Figures saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
