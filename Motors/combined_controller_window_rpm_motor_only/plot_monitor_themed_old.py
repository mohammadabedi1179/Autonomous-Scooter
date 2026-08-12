#!/usr/bin/env python3
"""
plot_monitor_themed.py
======================
Thesis-style visualisation for CSV logs produced by motor_monitor.py.

The script automatically detects fixed-setpoint and sweep logs and preserves
all three workflows from the original script:

  1. Plot one or more fixed logs
  2. Plot one or more sweep logs
  3. Overlay multiple fixed logs for before/after comparison

Examples
--------
python3 plot_monitor_themed.py logs/monitor_fixed_5rpm.csv

python3 plot_monitor_themed.py logs/monitor_fixed_5rpm.csv \
    --save --theme thesis_nude_contrast --dpi 220

python3 plot_monitor_themed.py logs/monitor_sweep.csv \
    --save --out plots --theme muji

python3 plot_monitor_themed.py before.csv after.csv \
    --overlay --save --theme thesis_nude_contrast

Dependencies
------------
pip install numpy pandas matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# THEMES — aligned with the plotting suite used in the thesis
# ══════════════════════════════════════════════════════════════
THEMES = {
    "muji": {
        "BG": "#f7f3ee",
        "BG_AXES": "#f7f3ee",
        "GRID_CLR": "#e6dfd5",
        "TEXT_CLR": "#5f5a55",
        "TICK_CLR": "#7a746e",
        "EDGE_CLR": "#d8d1c7",
        "PAPER_WHITE": "#f7f3ee",
        "PRIMARY": "#a8c5d1",
        "SECONDARY": "#d8b4a0",
        "ACCENT": "#b7c4b2",
        "DANGER": "#bc6c25",
        "CRITICAL": "#8d99ae",
        "BANDS": ["#f0ebe5", "#ebe6dd", "#e8e1d8", "#f2ede6"],
    },
    "soft_linen": {
        "BG": "#F8F5F1",
        "BG_AXES": "#F8F5F1",
        "GRID_CLR": "#E7E0D8",
        "TEXT_CLR": "#5E5A56",
        "TICK_CLR": "#7C746D",
        "EDGE_CLR": "#D6CCC2",
        "PAPER_WHITE": "#F8F5F1",
        "PRIMARY": "#D8C3A5",
        "SECONDARY": "#C9ADA7",
        "ACCENT": "#B7B7A4",
        "DANGER": "#C97C5D",
        "CRITICAL": "#9A8C98",
        "BANDS": ["#f1ebe4", "#ece5dc", "#e8ddd1", "#f3eee8"],
    },
    "nude_editorial": {
        "BG": "#FAF7F2",
        "BG_AXES": "#FAF7F2",
        "GRID_CLR": "#ECE6DE",
        "TEXT_CLR": "#605A54",
        "TICK_CLR": "#7D756E",
        "EDGE_CLR": "#DDD2C7",
        "PAPER_WHITE": "#FAF7F2",
        "PRIMARY": "#B7A99A",
        "SECONDARY": "#C7A9A3",
        "ACCENT": "#A68A64",
        "DANGER": "#B56576",
        "CRITICAL": "#6D6875",
        "BANDS": ["#f3eee7", "#eee5db", "#e9ddd2", "#f7f2eb"],
    },
    "japanese_earth": {
        "BG": "#F6F1EB",
        "BG_AXES": "#F6F1EB",
        "GRID_CLR": "#E4DBD0",
        "TEXT_CLR": "#5C5752",
        "TICK_CLR": "#756F69",
        "EDGE_CLR": "#D8CEC2",
        "PAPER_WHITE": "#F6F1EB",
        "PRIMARY": "#A3B18A",
        "SECONDARY": "#DDBEA9",
        "ACCENT": "#B08968",
        "DANGER": "#9C6644",
        "CRITICAL": "#7F5539",
        "BANDS": ["#efe6dc", "#e7ddd1", "#ddd1c3", "#f3ece3"],
    },
    "warm_gray": {
        "BG": "#FBFAF8",
        "BG_AXES": "#FBFAF8",
        "GRID_CLR": "#ECE8E2",
        "TEXT_CLR": "#57534E",
        "TICK_CLR": "#78716C",
        "EDGE_CLR": "#D6D3D1",
        "PAPER_WHITE": "#FBFAF8",
        "PRIMARY": "#A8A29E",
        "SECONDARY": "#CBBBA0",
        "ACCENT": "#78716C",
        "DANGER": "#BC6C25",
        "CRITICAL": "#6B705C",
        "BANDS": ["#f2f0ec", "#ece8e2", "#e6e1da", "#f7f5f1"],
    },
    "porcelain_sage": {
        "BG": "#FAF9F6",
        "BG_AXES": "#FAF9F6",
        "GRID_CLR": "#E8E4DD",
        "TEXT_CLR": "#4F514A",
        "TICK_CLR": "#74766F",
        "EDGE_CLR": "#D9D5CC",
        "PAPER_WHITE": "#FAF9F6",
        "PRIMARY": "#8FA998",
        "SECONDARY": "#D7BFAE",
        "ACCENT": "#A7B8C2",
        "DANGER": "#B76E65",
        "CRITICAL": "#6F7480",
        "BANDS": ["#F1EFEA", "#EBE8E1", "#E5E0D7", "#F5F2EC"],
    },
    "thesis_nude_contrast": {
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
    },
    "oat_blueprint": {
        "BG": "#FAF8F2",
        "BG_AXES": "#FAF8F2",
        "GRID_CLR": "#E3E0D8",
        "TEXT_CLR": "#50545A",
        "TICK_CLR": "#73777E",
        "EDGE_CLR": "#D5D2CA",
        "PAPER_WHITE": "#FAF8F2",
        "PRIMARY": "#8EA4B8",
        "SECONDARY": "#CDBBA7",
        "ACCENT": "#A9A58F",
        "DANGER": "#B56B65",
        "CRITICAL": "#687284",
        "BANDS": ["#F0EEE8", "#E9E7E0", "#E2DFD6", "#F5F3ED"],
    },
}

CURRENT_THEME = THEMES["thesis_nude_contrast"]

MOTOR_NAMES = [
    "M0 Front Left",
    "M1 Front Right",
    "M2 Back Left",
    "M3 Back Right",
]


def apply_theme(theme_name: str) -> None:
    global CURRENT_THEME, MOTOR_COLORS

    if theme_name not in THEMES:
        raise ValueError(
            f"Unknown theme '{theme_name}'. Available: {list(THEMES)}"
        )

    CURRENT_THEME = THEMES[theme_name]
    globals().update(CURRENT_THEME)

    # Four related but visibly distinct thesis colours.
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
        "font.size": 9,
        "legend.facecolor": PAPER_WHITE,
        "legend.edgecolor": EDGE_CLR,
        "legend.labelcolor": TEXT_CLR,
        "legend.framealpha": 0.97,
    })


# ══════════════════════════════════════════════════════════════
# CSV detection and validation
# ══════════════════════════════════════════════════════════════
def detect_mode(df: pd.DataFrame) -> str:
    if "sweep_rpm" in df.columns:
        return "sweep"
    if "cmd_m0" in df.columns:
        return "fixed"
    raise ValueError(
        "Unrecognised CSV format. Expected fixed columns such as cmd_m0 "
        "or the sweep_rpm column."
    )


def validate_common_columns(df: pd.DataFrame) -> None:
    if "time_s" not in df.columns:
        raise ValueError("Missing required column: time_s")


def active_motors(df: pd.DataFrame) -> list[int]:
    """Return motor indices that appear active in the log."""
    active: list[int] = []

    if "cmd_m0" in df.columns:
        for i in range(4):
            col = f"cmd_m{i}"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.abs().max() > 0.5:
                    active.append(i)
    else:
        for i in range(4):
            col = f"rpm_m{i}"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.abs().max() > 0.5:
                    active.append(i)

    return active or list(range(4))


def step_boundaries(df: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Return (start index, end index, setpoint) for each sweep step."""
    if "sweep_rpm" not in df.columns:
        return []

    sp = pd.to_numeric(df["sweep_rpm"], errors="coerce").to_numpy()
    boundaries: list[tuple[int, int, float]] = []
    i = 0
    while i < len(sp):
        value = sp[i]
        j = i
        while j < len(sp) and sp[j] == value:
            j += 1
        boundaries.append((i, j - 1, float(value)))
        i = j
    return boundaries


# ══════════════════════════════════════════════════════════════
# Visual helpers
# ══════════════════════════════════════════════════════════════
def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_AXES)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_CLR, alpha=0.82, linewidth=0.45)
    ax.tick_params(colors=TICK_CLR)
    ax.spines["left"].set_color(EDGE_CLR)
    ax.spines["bottom"].set_color(EDGE_CLR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_title(ax: plt.Axes, text: str, fontsize: float = 10.5) -> None:
    ax.set_title(text, fontsize=fontsize, fontweight="bold", color=TEXT_CLR, pad=9)


def add_legend(
    ax: plt.Axes,
    *,
    loc: str = "best",
    fontsize: float = 7.5,
    ncol: int = 1,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    legend = ax.legend(
        loc=loc,
        fontsize=fontsize,
        ncol=ncol,
        facecolor=PAPER_WHITE,
        edgecolor=EDGE_CLR,
        labelcolor=TEXT_CLR,
        framealpha=0.97,
        handlelength=2.3,
        handletextpad=0.7,
        borderpad=0.7,
        columnspacing=1.0,
    )
    legend.get_frame().set_linewidth(0.9)

    legend_handles = getattr(
        legend, "legend_handles", getattr(legend, "legendHandles", [])
    )
    for handle in legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)
        if hasattr(handle, "get_sizes") and hasattr(handle, "set_sizes"):
            handle.set_sizes([48])
            if hasattr(handle, "set_edgecolors"):
                handle.set_edgecolors(TEXT_CLR)
            if hasattr(handle, "set_linewidths"):
                handle.set_linewidths([0.7])


def line_with_shadow(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 1.35,
    linestyle: str = "-",
    alpha: float = 0.94,
    drawstyle: str = "default",
) -> None:
    x_values = np.asarray(x)
    y_values = np.asarray(y)

    if linestyle == "-":
        ax.plot(
            x_values,
            y_values,
            color=EDGE_CLR,
            linewidth=linewidth + 1.0,
            alpha=0.26,
            solid_capstyle="round",
            drawstyle=drawstyle,
            zorder=2,
        )

    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        solid_capstyle="round",
        drawstyle=drawstyle,
        label=label,
        zorder=3,
    )


def reference_line(
    ax: plt.Axes,
    value: float,
    *,
    orientation: str = "horizontal",
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.0,
) -> None:
    c = color or TICK_CLR
    if orientation == "horizontal":
        ax.axhline(
            value,
            color=c,
            linestyle="--",
            linewidth=linewidth,
            alpha=0.78,
            label=label,
            zorder=2,
        )
    else:
        ax.axvline(
            value,
            color=c,
            linestyle="--",
            linewidth=linewidth,
            alpha=0.78,
            label=label,
            zorder=2,
        )


def time_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "thesis_time", [PAPER_WHITE, PRIMARY, ACCENT, CRITICAL], N=256
    )


def style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, color=TEXT_CLR, fontsize=8)
    colorbar.ax.tick_params(colors=TICK_CLR, labelsize=7)
    colorbar.outline.set_edgecolor(EDGE_CLR)


def shade_sweep_steps(
    ax: plt.Axes,
    t: np.ndarray,
    steps: list[tuple[int, int, float]],
    *,
    annotate: bool,
) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin

    for index, (start, end, setpoint) in enumerate(steps):
        end_index = min(end, len(t) - 1)
        x0 = t[start]
        x1 = t[end_index]
        color = BANDS[index % len(BANDS)]
        ax.axvspan(x0, x1, color=color, alpha=0.62, zorder=0)

        if annotate:
            ax.text(
                0.5 * (x0 + x1),
                ymax - 0.05 * span,
                f"{setpoint:.0f}",
                ha="center",
                va="top",
                fontsize=7,
                fontweight="bold",
                color=TICK_CLR,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=PAPER_WHITE,
                    edgecolor=EDGE_CLR,
                    linewidth=0.6,
                    alpha=0.82,
                ),
                zorder=5,
            )

    ax.set_ylim(ymin, ymax)


def finish_figure(
    fig: plt.Figure,
    save_path: str | Path | None,
    *,
    dpi: int,
) -> None:
    fig.patch.set_facecolor(BG)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=BG,
            edgecolor=BG,
        )
        print(f"Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Fixed-setpoint analysis
# ══════════════════════════════════════════════════════════════
def plot_fixed(
    df: pd.DataFrame,
    title_text: str,
    *,
    save_path: str | Path | None,
    dpi: int,
    dz_kick: float,
) -> None:
    t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    fig.suptitle(
        title_text,
        fontsize=14,
        fontweight="bold",
        color=TEXT_CLR,
        y=0.993,
    )
    grid = gridspec.GridSpec(
        5,
        4,
        figure=fig,
        hspace=0.58,
        wspace=0.38,
        top=0.965,
        bottom=0.045,
    )

    # Row 0 — RPM command and measured response.
    ax_rpm = fig.add_subplot(grid[0, :])
    for motor in range(4):
        command_column = f"cmd_m{motor}"
        rpm_column = f"rpm_m{motor}"
        color = MOTOR_COLORS[motor]

        if command_column in df.columns:
            command = pd.to_numeric(
                df[command_column], errors="coerce"
            ).abs().to_numpy()
            line_with_shadow(
                ax_rpm,
                t,
                command,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.63,
                label=f"{MOTOR_NAMES[motor]} command",
                drawstyle="steps-post",
            )

        if rpm_column in df.columns:
            rpm = pd.to_numeric(
                df[rpm_column], errors="coerce"
            ).abs().to_numpy()
            line_with_shadow(
                ax_rpm,
                t,
                rpm,
                color=color,
                linewidth=1.35,
                label=f"{MOTOR_NAMES[motor]} measured",
            )

    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_ylabel("Absolute speed (RPM)")
    set_title(ax_rpm, "Motor-Speed Tracking: Commanded vs Measured RPM", 11.5)
    style_axis(ax_rpm)
    add_legend(ax_rpm, loc="upper right", fontsize=7, ncol=4)

    # Row 1 — PWM outputs.
    ax_pwm = fig.add_subplot(grid[1, :])
    for motor in range(4):
        column = f"pwm_m{motor}"
        if column in df.columns:
            pwm = pd.to_numeric(df[column], errors="coerce").to_numpy()
            line_with_shadow(
                ax_pwm,
                t,
                pwm,
                color=MOTOR_COLORS[motor],
                linewidth=1.25,
                label=MOTOR_NAMES[motor],
            )

    reference_line(
        ax_pwm,
        dz_kick,
        label=f"Dead-zone kick = {dz_kick:g}",
        color=DANGER,
    )
    ax_pwm.set_xlabel("Time (s)")
    ax_pwm.set_ylabel("PWM command")
    set_title(ax_pwm, "PWM Output and Dead-Zone Activity", 11.5)
    style_axis(ax_pwm)
    add_legend(ax_pwm, loc="upper right", fontsize=7.2, ncol=5)

    # Row 2 — tracking error.
    ax_error = fig.add_subplot(grid[2, :])
    for motor in range(4):
        column = f"err_m{motor}"
        if column in df.columns:
            error = pd.to_numeric(df[column], errors="coerce").to_numpy()
            line_with_shadow(
                ax_error,
                t,
                error,
                color=MOTOR_COLORS[motor],
                linewidth=1.25,
                label=MOTOR_NAMES[motor],
            )

    reference_line(ax_error, 0.0)
    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel("Tracking error (RPM)")
    set_title(ax_error, "Tracking Error: Setpoint − Measured Speed", 11.5)
    style_axis(ax_error)
    add_legend(ax_error, loc="upper right", fontsize=7.2, ncol=4)

    # Row 3 — PWM histograms.
    for column_index, motor in enumerate(range(4)):
        ax = fig.add_subplot(grid[3, column_index])
        pwm_column = f"pwm_m{motor}"

        if pwm_column in df.columns:
            values = pd.to_numeric(
                df[pwm_column], errors="coerce"
            ).dropna().to_numpy()
            zero_count = int(np.sum(np.isclose(values, 0.0)))
            total_count = len(values)
            zero_fraction = zero_count / max(total_count, 1)

            ax.hist(
                values,
                bins=30,
                color=MOTOR_COLORS[motor],
                alpha=0.76,
                edgecolor=PAPER_WHITE,
                linewidth=0.75,
            )
            reference_line(
                ax,
                dz_kick,
                orientation="vertical",
                label=f"DZ kick = {dz_kick:g}",
                color=DANGER,
                linewidth=1.0,
            )
            set_title(
                ax,
                f"{MOTOR_NAMES[motor]}\nPWM zeros: {zero_count}/{total_count} "
                f"({zero_fraction:.0%})",
                8.7,
            )
            ax.set_xlabel("PWM output")
            ax.set_ylabel("Count")
            add_legend(ax, loc="best", fontsize=6.6)

        style_axis(ax)

    # Row 4 — RPM/PWM mapping.
    colormap = time_colormap()
    for column_index, motor in enumerate(range(4)):
        ax = fig.add_subplot(grid[4, column_index])
        rpm_column = f"rpm_m{motor}"
        pwm_column = f"pwm_m{motor}"

        if rpm_column in df.columns and pwm_column in df.columns:
            rpm = pd.to_numeric(
                df[rpm_column], errors="coerce"
            ).abs().to_numpy()
            pwm = pd.to_numeric(df[pwm_column], errors="coerce").to_numpy()
            valid = np.isfinite(t) & np.isfinite(rpm) & np.isfinite(pwm)

            scatter = ax.scatter(
                pwm[valid],
                rpm[valid],
                c=t[valid],
                cmap=colormap,
                s=13,
                alpha=0.72,
                edgecolors=PAPER_WHITE,
                linewidths=0.35,
                zorder=3,
            )
            colorbar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.055)
            style_colorbar(colorbar, "Time (s)")

            fit_mask = valid & (pwm > 2) & (rpm > 1)
            if np.sum(fit_mask) > 5:
                slope, intercept = np.polyfit(pwm[fit_mask], rpm[fit_mask], 1)
                x_fit = np.linspace(
                    float(np.min(pwm[fit_mask])),
                    float(np.max(pwm[fit_mask])),
                    80,
                )
                line_with_shadow(
                    ax,
                    x_fit,
                    slope * x_fit + intercept,
                    color=DANGER,
                    linestyle="--",
                    linewidth=1.15,
                    label=f"Fit: {slope:.2f}·PWM {intercept:+.1f}",
                )
                add_legend(ax, fontsize=6.5)

        ax.set_xlabel("PWM output")
        ax.set_ylabel("Absolute speed (RPM)")
        set_title(ax, f"{MOTOR_NAMES[motor]}\nRPM–PWM Relationship", 8.7)
        style_axis(ax)

    finish_figure(fig, save_path, dpi=dpi)


# ══════════════════════════════════════════════════════════════
# Sweep analysis
# ══════════════════════════════════════════════════════════════
def plot_sweep(
    df: pd.DataFrame,
    title_text: str,
    *,
    save_path: str | Path | None,
    dpi: int,
    dz_kick: float,
) -> None:
    t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    steps = step_boundaries(df)
    setpoints = sorted(
        pd.to_numeric(df["sweep_rpm"], errors="coerce").dropna().unique()
    )
    motors = active_motors(df)

    if not setpoints:
        raise ValueError("Sweep log contains no finite sweep_rpm values.")

    figure_width = max(16.0, 3.2 * len(setpoints))
    fig = plt.figure(figsize=(figure_width, 18), facecolor=BG)
    fig.suptitle(
        title_text,
        fontsize=14,
        fontweight="bold",
        color=TEXT_CLR,
        y=0.993,
    )
    grid = gridspec.GridSpec(
        5,
        len(setpoints),
        figure=fig,
        hspace=0.62,
        wspace=0.38,
        top=0.965,
        bottom=0.045,
    )

    # Row 0 — full speed timeline.
    ax_rpm = fig.add_subplot(grid[0, :])
    sweep = pd.to_numeric(df["sweep_rpm"], errors="coerce").to_numpy()
    line_with_shadow(
        ax_rpm,
        t,
        sweep,
        color=TICK_CLR,
        linestyle="--",
        linewidth=1.35,
        alpha=0.80,
        label="Sweep setpoint",
        drawstyle="steps-post",
    )

    for motor in range(4):
        column = f"rpm_m{motor}"
        if column in df.columns:
            rpm = pd.to_numeric(df[column], errors="coerce").abs().to_numpy()
            line_with_shadow(
                ax_rpm,
                t,
                rpm,
                color=MOTOR_COLORS[motor],
                linewidth=1.2,
                label=MOTOR_NAMES[motor],
            )

    shade_sweep_steps(ax_rpm, t, steps, annotate=True)
    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_ylabel("Absolute speed (RPM)")
    set_title(ax_rpm, "Sweep Timeline: Measured RPM vs Setpoint", 11.5)
    style_axis(ax_rpm)
    add_legend(ax_rpm, loc="upper right", fontsize=7.2, ncol=5)

    # Row 1 — full PWM timeline.
    ax_pwm = fig.add_subplot(grid[1, :])
    for motor in range(4):
        column = f"pwm_m{motor}"
        if column in df.columns:
            pwm = pd.to_numeric(df[column], errors="coerce").to_numpy()
            line_with_shadow(
                ax_pwm,
                t,
                pwm,
                color=MOTOR_COLORS[motor],
                linewidth=1.15,
                label=MOTOR_NAMES[motor],
            )

    reference_line(
        ax_pwm,
        dz_kick,
        label=f"Dead-zone kick = {dz_kick:g}",
        color=DANGER,
    )
    shade_sweep_steps(ax_pwm, t, steps, annotate=False)
    ax_pwm.set_xlabel("Time (s)")
    ax_pwm.set_ylabel("PWM command")
    set_title(ax_pwm, "Sweep Timeline: PWM Output", 11.5)
    style_axis(ax_pwm)
    add_legend(ax_pwm, loc="upper right", fontsize=7.2, ncol=5)

    # Rows 2–4 — per-setpoint diagnostics.
    for column_index, setpoint in enumerate(setpoints):
        mask = np.isclose(sweep, setpoint, equal_nan=False)
        subset = df.loc[mask].copy()
        if subset.empty:
            continue

        subset_time = pd.to_numeric(
            subset["time_s"], errors="coerce"
        ).to_numpy()
        relative_time = subset_time - subset_time[0]

        # Per-step measured RPM.
        ax_speed = fig.add_subplot(grid[2, column_index])
        for motor in range(4):
            column = f"rpm_m{motor}"
            if column in subset.columns:
                rpm = pd.to_numeric(
                    subset[column], errors="coerce"
                ).abs().to_numpy()
                line_with_shadow(
                    ax_speed,
                    relative_time,
                    rpm,
                    color=MOTOR_COLORS[motor],
                    linewidth=1.05,
                    label=MOTOR_NAMES[motor],
                )

        reference_line(
            ax_speed,
            float(setpoint),
            label="Setpoint",
            color=TICK_CLR,
        )
        ax_speed.set_xlabel("Time within step (s)")
        ax_speed.set_ylabel("Absolute speed (RPM)")
        set_title(ax_speed, f"Setpoint = {setpoint:.0f} RPM\nSpeed Response", 8.7)
        style_axis(ax_speed)
        if column_index == 0:
            add_legend(ax_speed, fontsize=6.2)

        # Per-step PWM.
        ax_step_pwm = fig.add_subplot(grid[3, column_index])
        for motor in range(4):
            column = f"pwm_m{motor}"
            if column in subset.columns:
                pwm = pd.to_numeric(
                    subset[column], errors="coerce"
                ).to_numpy()
                line_with_shadow(
                    ax_step_pwm,
                    relative_time,
                    pwm,
                    color=MOTOR_COLORS[motor],
                    linewidth=1.05,
                    label=MOTOR_NAMES[motor],
                )

        reference_line(
            ax_step_pwm,
            dz_kick,
            label=f"DZ kick = {dz_kick:g}",
            color=DANGER,
            linewidth=0.9,
        )
        ax_step_pwm.set_xlabel("Time within step (s)")
        ax_step_pwm.set_ylabel("PWM")
        set_title(ax_step_pwm, f"Setpoint = {setpoint:.0f} RPM\nPWM Response", 8.7)
        style_axis(ax_step_pwm)
        if column_index == 0:
            add_legend(ax_step_pwm, fontsize=6.2)

        # Per-step PWM histogram.
        ax_hist = fig.add_subplot(grid[4, column_index])
        for motor in motors:
            column = f"pwm_m{motor}"
            if column not in subset.columns:
                continue

            values = pd.to_numeric(
                subset[column], errors="coerce"
            ).dropna().to_numpy()
            zero_count = int(np.sum(np.isclose(values, 0.0)))
            ax_hist.hist(
                values,
                bins=20,
                color=MOTOR_COLORS[motor],
                alpha=0.56,
                edgecolor=PAPER_WHITE,
                linewidth=0.65,
                label=f"M{motor}: zero={zero_count}",
            )

        reference_line(
            ax_hist,
            dz_kick,
            orientation="vertical",
            label=f"DZ = {dz_kick:g}",
            color=DANGER,
            linewidth=0.9,
        )
        ax_hist.set_xlabel("PWM output")
        ax_hist.set_ylabel("Count")
        set_title(ax_hist, f"Setpoint = {setpoint:.0f} RPM\nPWM Distribution", 8.7)
        style_axis(ax_hist)
        add_legend(ax_hist, fontsize=5.8)

    finish_figure(fig, save_path, dpi=dpi)


# ══════════════════════════════════════════════════════════════
# Multi-file fixed-mode overlay
# ══════════════════════════════════════════════════════════════
def plot_overlay(
    dataframes: list[pd.DataFrame],
    labels: list[str],
    *,
    save_path: str | Path | None,
    dpi: int,
    dz_kick: float,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        gridspec_kw={"hspace": 0.46},
        facecolor=BG,
    )
    fig.suptitle(
        "Multi-Run Motor-Control Comparison",
        fontsize=14,
        fontweight="bold",
        color=TEXT_CLR,
        y=0.985,
    )

    run_palette = [
        PRIMARY,
        SECONDARY,
        ACCENT,
        CRITICAL,
        DANGER,
        TICK_CLR,
        BANDS[1],
        BANDS[2],
    ]
    motor_linestyles = ["-", "--", ":", "-."]

    for run_index, (df, label) in enumerate(zip(dataframes, labels)):
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
        run_color = run_palette[run_index % len(run_palette)]

        for motor in active_motors(df):
            linestyle = motor_linestyles[motor]
            rpm_column = f"rpm_m{motor}"
            pwm_column = f"pwm_m{motor}"
            error_column = f"err_m{motor}"

            if rpm_column in df.columns:
                rpm = pd.to_numeric(
                    df[rpm_column], errors="coerce"
                ).abs().to_numpy()
                axes[0].plot(
                    t,
                    rpm,
                    color=run_color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    alpha=0.88,
                    label=f"{label} · M{motor}",
                )

            if pwm_column in df.columns:
                pwm = pd.to_numeric(df[pwm_column], errors="coerce").to_numpy()
                axes[1].plot(
                    t,
                    pwm,
                    color=run_color,
                    linestyle=linestyle,
                    linewidth=1.15,
                    alpha=0.88,
                )

            if error_column in df.columns:
                error = pd.to_numeric(
                    df[error_column], errors="coerce"
                ).to_numpy()
                axes[2].plot(
                    t,
                    error,
                    color=run_color,
                    linestyle=linestyle,
                    linewidth=1.15,
                    alpha=0.88,
                )

    axes[0].set_ylabel("Absolute speed (RPM)")
    set_title(axes[0], "Measured RPM", 11)
    add_legend(axes[0], fontsize=6.4, ncol=4)

    axes[1].set_ylabel("PWM command")
    set_title(axes[1], "PWM Output", 11)
    reference_line(
        axes[1],
        dz_kick,
        label=f"DZ kick = {dz_kick:g}",
        color=DANGER,
    )
    add_legend(axes[1], fontsize=6.7)

    axes[2].set_ylabel("Tracking error (RPM)")
    set_title(axes[2], "Tracking Error", 11)
    reference_line(axes[2], 0.0)

    for axis in axes:
        axis.set_xlabel("Time (s)")
        style_axis(axis)

    finish_figure(fig, save_path, dpi=dpi)


# ══════════════════════════════════════════════════════════════
# Command line
# ══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-style plots from motor_monitor.py CSV logs."
    )
    parser.add_argument(
        "csv",
        nargs="+",
        help="One or more fixed- or sweep-mode CSV logs.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures instead of opening an interactive window.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay multiple fixed-mode logs for before/after comparison.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Without this option, saved figures are "
            "written beside the input CSV."
        ),
    )
    parser.add_argument(
        "--theme",
        choices=list(THEMES.keys()),
        default="thesis_nude_contrast",
        help="Scientific visual theme.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved image resolution. Default: 220.",
    )
    parser.add_argument(
        "--dz-kick",
        type=float,
        default=12.0,
        help="Dead-zone kick PWM reference value. Default: 12.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Saved figure format. Default: png.",
    )
    return parser.parse_args()


def output_path_for_file(
    csv_path: str | Path,
    *,
    output_directory: Path | None,
    suffix: str,
    file_format: str,
) -> Path:
    csv_path = Path(csv_path)
    directory = output_directory if output_directory else csv_path.parent
    return directory / f"{csv_path.stem}{suffix}.{file_format}"


def main() -> int:
    args = parse_args()
    apply_theme(args.theme)

    dataframes: list[pd.DataFrame] = []
    labels: list[str] = []
    modes: list[str] = []

    for csv_name in args.csv:
        path = Path(csv_name)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1

        try:
            dataframe = pd.read_csv(path)
            dataframe.columns = dataframe.columns.str.strip()
            validate_common_columns(dataframe)
            mode = detect_mode(dataframe)
        except Exception as exc:
            print(f"ERROR in {path}: {exc}", file=sys.stderr)
            return 1

        dataframes.append(dataframe)
        labels.append(path.stem)
        modes.append(mode)
        print(f"  {path} → {mode} mode, {len(dataframe)} rows")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    if args.overlay:
        if len(dataframes) < 2:
            print("ERROR: --overlay requires at least two CSV files.", file=sys.stderr)
            return 1
        if any(mode != "fixed" for mode in modes):
            print("ERROR: --overlay only supports fixed-mode logs.", file=sys.stderr)
            return 1

        save_path = None
        if args.save:
            first_path = Path(args.csv[0])
            directory = args.out if args.out else first_path.parent
            save_path = directory / f"{first_path.stem}_overlay.{args.format}"

        plot_overlay(
            dataframes,
            labels,
            save_path=save_path,
            dpi=args.dpi,
            dz_kick=args.dz_kick,
        )
        return 0

    for dataframe, label, mode, csv_name in zip(
        dataframes, labels, modes, args.csv
    ):
        save_path = None
        if args.save:
            save_path = output_path_for_file(
                csv_name,
                output_directory=args.out,
                suffix=f"_{mode}_themed",
                file_format=args.format,
            )

        title_text = f"{label} · {mode.capitalize()} Mode · {args.theme}"
        print(f"Plotting: {title_text}")

        if mode == "fixed":
            plot_fixed(
                dataframe,
                title_text,
                save_path=save_path,
                dpi=args.dpi,
                dz_kick=args.dz_kick,
            )
        else:
            plot_sweep(
                dataframe,
                title_text,
                save_path=save_path,
                dpi=args.dpi,
                dz_kick=args.dz_kick,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
