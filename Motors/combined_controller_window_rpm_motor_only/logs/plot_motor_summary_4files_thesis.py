#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_motor_summary_4files_thesis.py

Compact thesis-style motor sweep summary from four separate sweep CSV files.

File order example:
  file 1 -> M0
  file 2 -> M1
  file 3 -> M2
  file 4 -> M3

Expected CSV columns:
  time_s, sweep_rpm
  rpm_m0, rpm_m1, rpm_m2, rpm_m3
  pwm_m0, pwm_m1, pwm_m2, pwm_m3

Example:
python3 plot_motor_summary_4files_thesis.py \
  monitor_sweep_20260627_193405.csv \
  monitor_sweep_20260627_193716.csv \
  monitor_sweep_20260627_193857.csv \
  monitor_sweep_20260627_194036.csv \
  --motor-order 0 1 2 3 \
  --save \
  --out motor_response_summary_slide.png \
  --dpi 260
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# =========================================================
# Thesis theme
# =========================================================
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
    "DANGER": "#A44A3F",
    "CRITICAL": "#5E6472",
    "BANDS": ["#F1E8DE", "#EADBCB", "#E3D1BF", "#F6EEE6"],
}

MOTOR_NAMES = ["M0", "M1", "M2", "M3"]


def apply_theme() -> None:
    globals().update(THEME)

    global MOTOR_COLORS
    MOTOR_COLORS = [PRIMARY, SECONDARY, ACCENT, CRITICAL]

    matplotlib.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG_AXES,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "axes.edgecolor": EDGE_CLR,
        "axes.labelcolor": TEXT_CLR,
        "axes.titlecolor": TEXT_CLR,
        "text.color": TEXT_CLR,
        "xtick.color": TICK_CLR,
        "ytick.color": TICK_CLR,
        "axes.grid": True,
        "grid.color": GRID_CLR,
        "grid.alpha": 0.82,
        "grid.linewidth": 0.45,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.facecolor": PAPER_WHITE,
        "legend.edgecolor": EDGE_CLR,
        "legend.labelcolor": TEXT_CLR,
        "legend.framealpha": 0.96,
    })


# =========================================================
# Visual helpers
# =========================================================
def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_AXES)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_CLR, alpha=0.82, linewidth=0.45)
    ax.tick_params(colors=TICK_CLR)

    ax.spines["left"].set_color(EDGE_CLR)
    ax.spines["bottom"].set_color(EDGE_CLR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def line_with_shadow(
    ax: plt.Axes,
    x,
    y,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.45,
    linestyle: str = "-",
    alpha: float = 0.95,
    drawstyle: str = "default",
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return

    if linestyle == "-":
        ax.plot(
            x,
            y,
            color=EDGE_CLR,
            linewidth=linewidth + 1.0,
            alpha=0.24,
            solid_capstyle="round",
            drawstyle=drawstyle,
            zorder=2,
        )

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        solid_capstyle="round",
        drawstyle=drawstyle,
        label=label,
        zorder=3,
    )


def add_motor_legend(
    ax: plt.Axes,
    *,
    loc: str = "best",
    ncol: int = 4,
    fontsize: float = 8.0,
) -> None:
    handles, labels = ax.get_legend_handles_labels()

    kept_handles = []
    kept_labels = []
    seen = set()

    for handle, label in zip(handles, labels):
        if label in MOTOR_NAMES and label not in seen:
            kept_handles.append(handle)
            kept_labels.append(label)
            seen.add(label)

    if not kept_handles:
        return

    legend = ax.legend(
        kept_handles,
        kept_labels,
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        facecolor=PAPER_WHITE,
        edgecolor=EDGE_CLR,
        labelcolor=TEXT_CLR,
        framealpha=0.96,
        handlelength=2.0,
        borderpad=0.55,
        columnspacing=0.9,
    )

    legend.get_frame().set_linewidth(0.85)


def step_boundaries(df: pd.DataFrame) -> list[tuple[int, int, float]]:
    if "sweep_rpm" not in df.columns:
        raise ValueError("Missing required column: sweep_rpm")

    sweep = pd.to_numeric(df["sweep_rpm"], errors="coerce").to_numpy(dtype=float)

    if len(sweep) == 0:
        return []

    steps: list[tuple[int, int, float]] = []

    i = 0
    while i < len(sweep):
        value = sweep[i]
        j = i

        while j < len(sweep) and np.isclose(sweep[j], value, equal_nan=True):
            j += 1

        if np.isfinite(value):
            steps.append((i, j - 1, float(value)))

        i = j

    return steps


def shade_steps(
    ax: plt.Axes,
    t: np.ndarray,
    steps: list[tuple[int, int, float]],
    *,
    annotate: bool = False,
) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0

    for index, (start, end, setpoint) in enumerate(steps):
        start = max(0, min(start, len(t) - 1))
        end = max(0, min(end, len(t) - 1))

        x0 = float(t[start])
        x1 = float(t[end])

        ax.axvspan(
            x0,
            x1,
            color=BANDS[index % len(BANDS)],
            alpha=0.58,
            zorder=0,
        )

        if annotate:
            ax.text(
                0.5 * (x0 + x1),
                ymax - 0.06 * span,
                f"{setpoint:.0f}",
                ha="center",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color=TICK_CLR,
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor=PAPER_WHITE,
                    edgecolor=EDGE_CLR,
                    linewidth=0.6,
                    alpha=0.86,
                ),
                zorder=6,
            )

    ax.set_ylim(ymin, ymax)


# =========================================================
# Data handling
# =========================================================
def read_sweep_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in ["time_s", "sweep_rpm"]:
        if col not in df.columns:
            raise ValueError(f"{path.name}: missing required column: {col}")

    return df


def get_series(df: pd.DataFrame, motor: int, kind: str) -> np.ndarray:
    column = f"{kind}_m{motor}"

    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)

    if kind == "rpm":
        values = np.abs(values)

    return values


def compute_stats(
    dfs: list[pd.DataFrame],
    motor_order: list[int],
    settle_frac: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for df, motor in zip(dfs, motor_order):
        rpm_col = f"rpm_m{motor}"
        pwm_col = f"pwm_m{motor}"

        if rpm_col not in df.columns:
            raise ValueError(f"Missing column: {rpm_col}")

        for start, end, setpoint in step_boundaries(df):
            count = end - start + 1
            skip = min(count - 1, int(np.floor(settle_frac * count)))

            subset = df.iloc[start + skip:end + 1]

            if subset.empty:
                continue

            rpm = pd.to_numeric(
                subset[rpm_col],
                errors="coerce"
            ).abs().to_numpy(dtype=float)

            rpm = rpm[np.isfinite(rpm)]

            if rpm.size == 0:
                continue

            if pwm_col in subset.columns:
                pwm = pd.to_numeric(
                    subset[pwm_col],
                    errors="coerce"
                ).to_numpy(dtype=float)
                pwm = pwm[np.isfinite(pwm)]
            else:
                pwm = np.array([], dtype=float)

            rows.append({
                "motor": int(motor),
                "setpoint": float(setpoint),
                "rpm_mean": float(np.mean(rpm)),
                "rpm_std": float(np.std(rpm, ddof=1)) if rpm.size > 1 else 0.0,
                "mae": float(np.mean(np.abs(rpm - setpoint))),
                "rmse": float(np.sqrt(np.mean((rpm - setpoint) ** 2))),
                "pwm_mean": float(np.mean(pwm)) if pwm.size else np.nan,
                "pwm_std": float(np.std(pwm, ddof=1)) if pwm.size > 1 else 0.0,
                "samples": int(rpm.size),
            })

    stats = pd.DataFrame(rows)

    if stats.empty:
        return stats

    grouped_rows = []

    for (motor, setpoint), group in stats.groupby(["motor", "setpoint"], sort=True):
        grouped_rows.append({
            "motor": int(motor),
            "setpoint": float(setpoint),
            "rpm_mean": float(np.average(group["rpm_mean"], weights=group["samples"])),
            "rpm_std": float(np.mean(group["rpm_std"])),
            "mae": float(np.average(group["mae"], weights=group["samples"])),
            "rmse": float(np.average(group["rmse"], weights=group["samples"])),
            "pwm_mean": float(np.nanmean(group["pwm_mean"])),
            "pwm_std": float(np.nanmean(group["pwm_std"])),
            "samples": int(group["samples"].sum()),
        })

    return pd.DataFrame(grouped_rows)


# =========================================================
# Plot
# =========================================================
def plot_summary(
    csv_paths: list[Path],
    *,
    out: Path | None,
    dpi: int,
    dz_kick: float,
    settle_frac: float,
    motor_order: list[int],
    title: str | None = None,
) -> pd.DataFrame:
    if len(csv_paths) != len(motor_order):
        raise ValueError("Number of CSV files must match number of motor indices.")

    dfs = [read_sweep_csv(path) for path in csv_paths]

    stats = compute_stats(
        dfs,
        motor_order=motor_order,
        settle_frac=settle_frac,
    )

    ref_df = dfs[0]
    ref_t = pd.to_numeric(ref_df["time_s"], errors="coerce").to_numpy(dtype=float)
    ref_t = ref_t - ref_t[0]

    ref_sweep = pd.to_numeric(
        ref_df["sweep_rpm"],
        errors="coerce"
    ).to_numpy(dtype=float)

    ref_steps = step_boundaries(ref_df)

    fig = plt.figure(figsize=(14.4, 8.6), facecolor=BG)

    gs = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.985,
        top=0.91 if title else 0.96,
        bottom=0.08,
        hspace=0.32,
        wspace=0.22,
    )

    # -----------------------------------------------------
    # Panel 1: RPM timeline
    # -----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    line_with_shadow(
        ax1,
        ref_t,
        ref_sweep,
        color=TICK_CLR,
        linewidth=1.25,
        linestyle="--",
        alpha=0.82,
        drawstyle="steps-post",
    )

    for df, motor in zip(dfs, motor_order):
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
        t = t - t[0]

        rpm = get_series(df, motor, "rpm")

        line_with_shadow(
            ax1,
            t,
            rpm,
            color=MOTOR_COLORS[motor],
            label=MOTOR_NAMES[motor],
            linewidth=1.35,
        )

    ax1.set_title("Measured RPM vs Setpoint", fontsize=12, fontweight="bold", pad=9)
    ax1.set_xlabel("Time from run start (s)")
    ax1.set_ylabel("Absolute speed (RPM)")

    style_axis(ax1)
    shade_steps(ax1, ref_t, ref_steps, annotate=True)
    add_motor_legend(ax1, loc="upper left", ncol=4, fontsize=7.6)

    # -----------------------------------------------------
    # Panel 2: PWM timeline
    # -----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    for df, motor in zip(dfs, motor_order):
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
        t = t - t[0]

        pwm = get_series(df, motor, "pwm")

        line_with_shadow(
            ax2,
            t,
            pwm,
            color=MOTOR_COLORS[motor],
            label=MOTOR_NAMES[motor],
            linewidth=1.30,
        )

    ax2.axhline(
        dz_kick,
        color=DANGER,
        linestyle="--",
        linewidth=1.0,
        alpha=0.82,
        zorder=2,
    )

    ax2.set_title("PWM Output", fontsize=12, fontweight="bold", pad=9)
    ax2.set_xlabel("Time from run start (s)")
    ax2.set_ylabel("PWM command")

    style_axis(ax2)
    shade_steps(ax2, ref_t, ref_steps, annotate=False)
    add_motor_legend(ax2, loc="upper left", ncol=4, fontsize=7.6)

    # -----------------------------------------------------
    # Panel 3: steady-state RPM
    # -----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    if not stats.empty:
        x_min = float(stats["setpoint"].min())
        x_max = float(stats["setpoint"].max())

        x_line = np.linspace(x_min, x_max, 100)

        ax3.plot(
            x_line,
            x_line,
            color=TICK_CLR,
            linestyle="--",
            linewidth=1.15,
            alpha=0.82,
            zorder=2,
        )

        for motor in motor_order:
            subset = stats[stats["motor"] == motor].sort_values("setpoint")

            if subset.empty:
                continue

            ax3.errorbar(
                subset["setpoint"],
                subset["rpm_mean"],
                yerr=subset["rpm_std"],
                fmt="o-",
                color=MOTOR_COLORS[motor],
                ecolor=EDGE_CLR,
                elinewidth=1.0,
                capsize=3,
                linewidth=1.35,
                markersize=5,
                label=MOTOR_NAMES[motor],
                alpha=0.96,
                zorder=4,
            )

    ax3.set_title("Steady-State Tracking", fontsize=12, fontweight="bold", pad=9)
    ax3.set_xlabel("Setpoint (RPM)")
    ax3.set_ylabel("Measured RPM")

    style_axis(ax3)
    add_motor_legend(ax3, loc="upper left", ncol=4, fontsize=7.6)

    # -----------------------------------------------------
    # Panel 4: tracking error
    # -----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    if not stats.empty:
        for motor in motor_order:
            subset = stats[stats["motor"] == motor].sort_values("setpoint")

            if subset.empty:
                continue

            line_with_shadow(
                ax4,
                subset["setpoint"],
                subset["mae"],
                color=MOTOR_COLORS[motor],
                label=MOTOR_NAMES[motor],
                linewidth=1.45,
            )

            ax4.scatter(
                subset["setpoint"],
                subset["mae"],
                color=MOTOR_COLORS[motor],
                edgecolor=PAPER_WHITE,
                linewidth=0.7,
                s=42,
                zorder=5,
            )

    ax4.set_title("Mean Absolute Tracking Error", fontsize=12, fontweight="bold", pad=9)
    ax4.set_xlabel("Setpoint (RPM)")
    ax4.set_ylabel("MAE (RPM)")

    style_axis(ax4)
    add_motor_legend(ax4, loc="upper left", ncol=4, fontsize=7.6)

    if title:
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            color=TEXT_CLR,
            y=0.982,
        )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=BG,
            edgecolor=BG,
        )

        print(f"Saved → {out}")

        summary_path = out.with_suffix(".csv")
        stats.to_csv(summary_path, index=False)

        print(f"Saved → {summary_path}")
    else:
        plt.show()

    plt.close(fig)

    return stats


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact thesis-style motor summary from four sweep CSV files."
    )

    parser.add_argument(
        "csv",
        nargs="+",
        help="Sweep CSV files. Default order maps to M0, M1, M2, M3.",
    )

    parser.add_argument(
        "--motor-order",
        nargs="+",
        type=int,
        default=None,
        help="Motor index for each CSV. Default: 0 1 2 3 for four files.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figure instead of showing it.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("motor_response_summary_slide.png"),
        help="Output image path.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=260,
        help="Output DPI.",
    )

    parser.add_argument(
        "--dz-kick",
        type=float,
        default=12.0,
        help="Dead-zone kick PWM reference.",
    )

    parser.add_argument(
        "--settle-frac",
        type=float,
        default=0.30,
        help="Fraction of each step discarded before steady-state statistics.",
    )

    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title. Leave empty for slide use.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    apply_theme()

    csv_paths = [Path(path) for path in args.csv]

    if args.motor_order is None:
        if len(csv_paths) == 4:
            motor_order = [0, 1, 2, 3]
        else:
            raise ValueError(
                "For non-four-file input, provide --motor-order explicitly."
            )
    else:
        motor_order = args.motor_order

    if len(csv_paths) != len(motor_order):
        raise ValueError(
            "Number of CSV files and --motor-order values must match."
        )

    if any(motor < 0 or motor > 3 for motor in motor_order):
        raise ValueError("Motor indices must be between 0 and 3.")

    if not 0 <= args.settle_frac < 1:
        raise ValueError("--settle-frac must satisfy 0 <= value < 1.")

    out = args.out if args.save else None

    plot_summary(
        csv_paths,
        out=out,
        dpi=args.dpi,
        dz_kick=args.dz_kick,
        settle_frac=args.settle_frac,
        motor_order=motor_order,
        title=args.title.strip() or None,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
