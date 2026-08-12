#!/usr/bin/env python3
"""
plot_monitor.py
===============
Plots CSV files produced by motor_monitor.py.

Automatically detects whether the file is a fixed-setpoint log or a sweep log
and chooses the right layout.

FIXED LOG  (columns: time_s, cmd_m0..3, rpm_m0..3, err_m0..3, pwm_m0..3)
────────────────────────────────────────────────────────────────────────────
  Row 1 — RPM commanded vs measured (all 4 motors)
  Row 2 — PWM output (all 4 motors)
  Row 3 — Tracking error (all 4 motors)
  Row 4 — PWM histogram per motor  (shows limit-cycle bimodal distribution)
  Row 5 — RPM vs PWM scatter per motor  (characterises dead-zone / linearity)

SWEEP LOG  (columns: time_s, sweep_rpm, rpm_m0..3, err_m0..3, pwm_m0..3)
────────────────────────────────────────────────────────────────────────────
  Row 1 — Full timeline: RPM measured vs sweep setpoint
  Row 2 — Full timeline: PWM output per motor
  Row 3 — Per-step summary: PWM min/mean/max bar chart
  Row 4 — Per-step summary: RPM mean ± std bar chart
  Row 5 — Per-step: PWM zero-count fraction (limit-cycle severity)

USAGE
─────
  python3 plot_monitor.py logs/monitor_fixed_5rpm_20260627_123456.csv
  python3 plot_monitor.py logs/monitor_sweep_20260627_123456.csv
  python3 plot_monitor.py logs/monitor_fixed_5rpm_*.csv   # overlay multiple runs
  python3 plot_monitor.py logs/monitor_*.csv --save       # save PNG instead of show
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

MOTOR_NAMES  = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]
MOTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ─────────────────────────────────────────────────────────────
#  CSV detection
# ─────────────────────────────────────────────────────────────
def detect_mode(df):
    if "sweep_rpm" in df.columns:
        return "sweep"
    if "cmd_m0" in df.columns:
        return "fixed"
    raise ValueError("Unrecognised CSV format — expected motor_monitor.py output.")

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def active_motors(df):
    """Return list of motor indices that received a nonzero command."""
    if "cmd_m0" not in df.columns:
        # sweep mode — all motors present but only one driven
        active = []
        for i in range(4):
            col = f"rpm_m{i}"
            if col in df.columns and df[col].abs().max() > 0.5:
                active.append(i)
        return active or list(range(4))
    active = []
    for i in range(4):
        col = f"cmd_m{i}"
        if col in df.columns and df[col].abs().max() > 0.5:
            active.append(i)
    return active or list(range(4))


def step_boundaries(df):
    """For sweep logs: return list of (start_idx, end_idx, setpoint) per step."""
    if "sweep_rpm" not in df.columns:
        return []
    sp = df["sweep_rpm"].values
    boundaries = []
    i = 0
    while i < len(sp):
        val = sp[i]
        j = i
        while j < len(sp) and sp[j] == val:
            j += 1
        boundaries.append((i, j - 1, val))
        i = j
    return boundaries


# ─────────────────────────────────────────────────────────────
#  FIXED mode plot
# ─────────────────────────────────────────────────────────────
def plot_fixed(df, title, save_path=None):
    t = df["time_s"].values
    motors = active_motors(df)

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(5, 4, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.97, bottom=0.04)

    # ── Row 0: RPM commanded vs measured ─────────────────────
    ax_rpm = fig.add_subplot(gs[0, :])
    for i in range(4):
        cmd_col = f"cmd_m{i}"
        rpm_col = f"rpm_m{i}"
        c = MOTOR_COLORS[i]
        if cmd_col in df.columns:
            ax_rpm.step(t, df[cmd_col].abs().values, color=c,
                        linestyle="--", alpha=0.5, linewidth=1.2,
                        label=f"{MOTOR_NAMES[i]} CMD")
        if rpm_col in df.columns:
            ax_rpm.plot(t, df[rpm_col].abs().values, color=c,
                        linewidth=1.0, label=f"{MOTOR_NAMES[i]} RPM")
    ax_rpm.set_ylabel("|RPM|")
    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_title("RPM: commanded (dashed) vs measured (solid)")
    ax_rpm.legend(ncol=4, fontsize=7, loc="upper right")

    # ── Row 1: PWM output ────────────────────────────────────
    ax_pwm = fig.add_subplot(gs[1, :])
    for i in range(4):
        col = f"pwm_m{i}"
        if col in df.columns:
            ax_pwm.plot(t, df[col].values, color=MOTOR_COLORS[i],
                        linewidth=1.0, label=MOTOR_NAMES[i])
    ax_pwm.set_ylabel("PWM output")
    ax_pwm.set_xlabel("Time (s)")
    ax_pwm.set_title("PWM output per motor  ← watch for oscillation between 0 and floor")
    ax_pwm.legend(ncol=4, fontsize=7, loc="upper right")

    # ── Row 2: Tracking error ────────────────────────────────
    ax_err = fig.add_subplot(gs[2, :])
    for i in range(4):
        col = f"err_m{i}"
        if col in df.columns:
            ax_err.plot(t, df[col].values, color=MOTOR_COLORS[i],
                        linewidth=1.0, label=MOTOR_NAMES[i])
    ax_err.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax_err.set_ylabel("Error (RPM)")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_title("Tracking error (setpoint − measured)")
    ax_err.legend(ncol=4, fontsize=7, loc="upper right")

    # ── Row 3: PWM histogram per motor ───────────────────────
    for col_idx, i in enumerate(range(4)):
        ax = fig.add_subplot(gs[3, col_idx])
        pwm_col = f"pwm_m{i}"
        if pwm_col in df.columns:
            vals = df[pwm_col].values
            n_zero = (vals == 0).sum()
            n_tot  = len(vals)
            ax.hist(vals, bins=30, color=MOTOR_COLORS[i], alpha=0.75, edgecolor="white")
            ax.set_title(f"{MOTOR_NAMES[i]}\nzeros: {n_zero}/{n_tot} "
                         f"({100*n_zero/max(n_tot,1):.0f}%)", fontsize=8)
            ax.set_xlabel("PWM")
            ax.set_ylabel("Count")
            # Mark the DZ_KICK_PWM floor
            ax.axvline(12, color="red", linestyle="--", linewidth=1.0,
                       label="DZ_KICK=12")
            ax.legend(fontsize=7)

    # ── Row 4: RPM vs PWM scatter ────────────────────────────
    for col_idx, i in enumerate(range(4)):
        ax = fig.add_subplot(gs[4, col_idx])
        rpm_col = f"rpm_m{i}"
        pwm_col = f"pwm_m{i}"
        if rpm_col in df.columns and pwm_col in df.columns:
            rpm_v = df[rpm_col].abs().values
            pwm_v = df[pwm_col].values
            # colour by time so you can see evolution
            sc = ax.scatter(pwm_v, rpm_v, c=t, cmap="viridis",
                            s=8, alpha=0.6)
            plt.colorbar(sc, ax=ax, label="time (s)", pad=0.02)
            # Fit line through nonzero points
            mask = (pwm_v > 2) & (rpm_v > 1)
            if mask.sum() > 5:
                m, b = np.polyfit(pwm_v[mask], rpm_v[mask], 1)
                xs = np.linspace(pwm_v[mask].min(), pwm_v[mask].max(), 50)
                ax.plot(xs, m * xs + b, "r--", linewidth=1.0,
                        label=f"fit: {m:.2f}×PWM+{b:.1f}")
                ax.legend(fontsize=7)
        ax.set_xlabel("PWM output")
        ax.set_ylabel("|RPM|")
        ax.set_title(f"{MOTOR_NAMES[i]}\nRPM vs PWM", fontsize=8)

    _finish(fig, save_path)


# ─────────────────────────────────────────────────────────────
#  SWEEP mode plot
# ─────────────────────────────────────────────────────────────
def plot_sweep(df, title, save_path=None):
    t      = df["time_s"].values
    steps  = step_boundaries(df)
    sp_vals = sorted(set(df["sweep_rpm"].values))
    motors  = active_motors(df)

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(5, len(sp_vals), figure=fig,
                           hspace=0.6, wspace=0.35,
                           top=0.97, bottom=0.04)

    # ── Row 0: Full timeline RPM ─────────────────────────────
    ax_rpm = fig.add_subplot(gs[0, :])
    ax_rpm.step(t, df["sweep_rpm"].values, color="black",
                linestyle="--", linewidth=1.2, label="Setpoint", alpha=0.6)
    for i in range(4):
        col = f"rpm_m{i}"
        if col in df.columns:
            ax_rpm.plot(t, df[col].abs().values, color=MOTOR_COLORS[i],
                        linewidth=0.9, label=MOTOR_NAMES[i])
    # shade each step
    for start, end, sp in steps:
        ax_rpm.axvspan(t[start], t[min(end, len(t)-1)],
                       alpha=0.06, color="blue")
        ax_rpm.text(0.5*(t[start]+t[min(end,len(t)-1)]),
                    ax_rpm.get_ylim()[1] * 0.92,
                    f"{sp:.0f}", fontsize=7, ha="center", color="navy")
    ax_rpm.set_ylabel("|RPM|")
    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_title("Full sweep timeline — RPM measured vs setpoint")
    ax_rpm.legend(ncol=5, fontsize=7, loc="upper right")

    # ── Row 1: Full timeline PWM ─────────────────────────────
    ax_pwm = fig.add_subplot(gs[1, :])
    for i in range(4):
        col = f"pwm_m{i}"
        if col in df.columns:
            ax_pwm.plot(t, df[col].values, color=MOTOR_COLORS[i],
                        linewidth=0.9, label=MOTOR_NAMES[i])
    for start, end, sp in steps:
        ax_pwm.axvspan(t[start], t[min(end, len(t)-1)],
                       alpha=0.06, color="blue")
    ax_pwm.axhline(12, color="red", linestyle="--", linewidth=0.9,
                   label="DZ_KICK=12")
    ax_pwm.set_ylabel("PWM output")
    ax_pwm.set_xlabel("Time (s)")
    ax_pwm.set_title("Full sweep timeline — PWM output  (red dashed = DZ_KICK floor)")
    ax_pwm.legend(ncol=5, fontsize=7, loc="upper right")

    # ── Rows 2-4: Per-step breakdowns (one column per setpoint) ──
    for col_idx, sp in enumerate(sp_vals):
        mask = df["sweep_rpm"] == sp
        sub  = df[mask]
        t_sub = sub["time_s"].values - sub["time_s"].values[0]

        # Row 2: RPM time series for this step
        ax2 = fig.add_subplot(gs[2, col_idx])
        for i in range(4):
            col = f"rpm_m{i}"
            if col in sub.columns:
                ax2.plot(t_sub, sub[col].abs().values,
                         color=MOTOR_COLORS[i], linewidth=1.0,
                         label=MOTOR_NAMES[i])
        ax2.axhline(sp, color="black", linestyle="--",
                    linewidth=0.9, label="SP")
        ax2.set_title(f"SP = {sp:.0f} RPM\nRPM", fontsize=8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("|RPM|")
        if col_idx == 0:
            ax2.legend(fontsize=6)

        # Row 3: PWM time series for this step
        ax3 = fig.add_subplot(gs[3, col_idx])
        for i in range(4):
            col = f"pwm_m{i}"
            if col in sub.columns:
                ax3.plot(t_sub, sub[col].values,
                         color=MOTOR_COLORS[i], linewidth=1.0)
        ax3.axhline(12, color="red", linestyle="--",
                    linewidth=0.8, label="DZ_KICK=12")
        ax3.set_title(f"SP = {sp:.0f} RPM\nPWM", fontsize=8)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("PWM")
        if col_idx == 0:
            ax3.legend(fontsize=6)

        # Row 4: PWM histogram for this step — reveals limit cycles
        ax4 = fig.add_subplot(gs[4, col_idx])
        for i in motors:
            col = f"pwm_m{i}"
            if col in sub.columns:
                vals = sub[col].values
                n_zero = (vals == 0).sum()
                ax4.hist(vals, bins=20, color=MOTOR_COLORS[i],
                         alpha=0.55, edgecolor="white",
                         label=f"{MOTOR_NAMES[i][:5]} z={n_zero}")
        ax4.axvline(12, color="red", linestyle="--",
                    linewidth=0.8, label="DZ=12")
        ax4.set_title(f"SP = {sp:.0f} RPM\nPWM histogram", fontsize=8)
        ax4.set_xlabel("PWM")
        ax4.set_ylabel("Count")
        ax4.legend(fontsize=6)

    _finish(fig, save_path)


# ─────────────────────────────────────────────────────────────
#  Multi-file overlay (fixed mode only)
# ─────────────────────────────────────────────────────────────
def plot_overlay(dfs, labels, save_path=None):
    """Overlay multiple fixed-mode runs — useful for before/after comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"hspace": 0.45})
    fig.suptitle("Multi-run overlay — RPM / PWM / Error", fontsize=11,
                 fontweight="bold")

    cmap = plt.cm.tab10
    for run_idx, (df, label) in enumerate(zip(dfs, labels)):
        t  = df["time_s"].values
        c  = cmap(run_idx % 10)
        motors = active_motors(df)
        for i in motors:
            ls = ["-", "--", ":", "-."][i % 4]
            axes[0].plot(t, df[f"rpm_m{i}"].abs().values,
                         color=c, linestyle=ls, linewidth=0.9,
                         label=f"{label} M{i}")
            axes[1].plot(t, df[f"pwm_m{i}"].values,
                         color=c, linestyle=ls, linewidth=0.9)
            axes[2].plot(t, df[f"err_m{i}"].values,
                         color=c, linestyle=ls, linewidth=0.9)

    axes[0].set_ylabel("|RPM|"); axes[0].set_title("RPM measured")
    axes[0].legend(fontsize=6, ncol=4)
    axes[1].set_ylabel("PWM"); axes[1].set_title("PWM output")
    axes[1].axhline(12, color="red", linestyle="--",
                    linewidth=0.8, label="DZ_KICK=12")
    axes[1].legend(fontsize=6)
    axes[2].set_ylabel("Error (RPM)"); axes[2].set_title("Tracking error")
    axes[2].axhline(0, color="black", linewidth=0.7, linestyle="--")
    for ax in axes:
        ax.set_xlabel("Time (s)")

    _finish(fig, save_path)


# ─────────────────────────────────────────────────────────────
#  Shared finish
# ─────────────────────────────────────────────────────────────
def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Plot motor_monitor.py CSV logs")
    p.add_argument("csv", nargs="+", help="One or more CSV log files")
    p.add_argument("--save", action="store_true",
                   help="Save PNG next to CSV instead of showing interactively")
    p.add_argument("--overlay", action="store_true",
                   help="Overlay multiple fixed-mode files on one figure "
                        "(useful for before/after firmware comparison)")
    return p.parse_args()


def main():
    args = parse_args()

    dfs    = []
    labels = []
    modes  = []

    for path in args.csv:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        try:
            mode = detect_mode(df)
        except ValueError as e:
            print(f"ERROR in {path}: {e}")
            sys.exit(1)
        dfs.append(df)
        labels.append(Path(path).stem)
        modes.append(mode)
        print(f"  {path}  →  {mode} mode, {len(df)} rows")

    if args.overlay:
        if any(m != "fixed" for m in modes):
            print("ERROR: --overlay only works with fixed-mode logs")
            sys.exit(1)
        save_path = None
        if args.save:
            save_path = str(Path(args.csv[0]).with_suffix("")) + "_overlay.png"
        plot_overlay(dfs, labels, save_path)

    else:
        for df, label, mode, path in zip(dfs, labels, modes, args.csv):
            save_path = None
            if args.save:
                save_path = str(Path(path).with_suffix("")) + "_plot.png"
            title = f"{label}  ({mode} mode)"
            print(f"Plotting: {title}")
            if mode == "fixed":
                plot_fixed(df, title, save_path)
            else:
                plot_sweep(df, title, save_path)


if __name__ == "__main__":
    main()
