#!/usr/bin/env python3
"""
plot_simultaneous_4motor_lifted_test.py
=======================================

Plot the CSV produced by:
  simultaneous_4motor_lifted_test.py

Outputs:
  00_overview_rpm_setpoints.png
  01_rpm_tracking_all_motors.png
  02_motor_speed_spread.png
  03_pwm_outputs_all_motors.png
  04_pwm_effort_spread.png
  05_error_all_motors.png
  06_phase_summary.csv
  diagnostic_report.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MOTOR_NAMES = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]


def arr(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def load_csv(path):
    df = pd.read_csv(path)
    numeric_cols = ["time_s"]
    numeric_cols += [f"cmd_m{i}" for i in range(4)]
    numeric_cols += [f"rpm_m{i}" for i in range(4)]
    numeric_cols += [f"err_m{i}" for i in range(4)]
    numeric_cols += [f"pwm_m{i}" for i in range(4)]
    numeric_cols += ["rpm_mean_abs", "rpm_spread_abs", "pwm_mean_abs", "pwm_spread_abs"]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_s"]).copy()
    df["t_s"] = df["time_s"] - df["time_s"].min()

    for i in range(4):
        df[f"rpm_abs_m{i}"] = df[f"rpm_m{i}"].abs()
        df[f"cmd_abs_m{i}"] = df[f"cmd_m{i}"].abs()
        df[f"pwm_abs_m{i}"] = df[f"pwm_m{i}"].abs()

    return df


def make_outdir(csv_path, out_arg=None):
    if out_arg:
        out = Path(out_arg)
    else:
        out = Path(str(csv_path).replace(".csv", "_plots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_overview(df, out):
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    t = arr(df["t_s"])

    for i, ax in enumerate(axes):
        ax.plot(t, arr(df[f"cmd_m{i}"]), linestyle="--", label="Commanded setpoint")
        ax.plot(t, arr(df[f"rpm_m{i}"]), label="Measured RPM")
        ax.plot(t, arr(df[f"pwm_m{i}"]), alpha=0.45, label="PWM out")
        ax.set_title(MOTOR_NAMES[i])
        ax.set_ylabel("RPM / PWM")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Simultaneous 4-motor lifted test overview", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "00_overview_rpm_setpoints.png", dpi=160)
    plt.close(fig)


def plot_rpm_tracking(df, out):
    fig, ax = plt.subplots(figsize=(16, 7))
    t = arr(df["t_s"])
    ax.plot(t, arr(df["cmd_m0"]), linestyle="--", linewidth=2.0, label="Command")
    for i in range(4):
        ax.plot(t, arr(df[f"rpm_m{i}"]), label=MOTOR_NAMES[i])
    ax.set_title("Measured RPM of all motors during simultaneous commands")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_rpm_tracking_all_motors.png", dpi=160)
    plt.close(fig)


def plot_speed_spread(df, out):
    fig, ax = plt.subplots(figsize=(16, 6))
    t = arr(df["t_s"])
    ax.plot(t, arr(df["rpm_mean_abs"]), label="Mean |RPM|")
    ax.plot(t, arr(df["rpm_spread_abs"]), label="Spread max(|RPM|)-min(|RPM|)")
    ax.set_title("Motor speed matching during simultaneous test")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_motor_speed_spread.png", dpi=160)
    plt.close(fig)


def plot_pwm(df, out):
    fig, ax = plt.subplots(figsize=(16, 7))
    t = arr(df["t_s"])
    for i in range(4):
        ax.plot(t, arr(df[f"pwm_m{i}"]), label=MOTOR_NAMES[i])
    ax.set_title("PWM outputs during simultaneous test")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM out")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_pwm_outputs_all_motors.png", dpi=160)
    plt.close(fig)


def plot_pwm_spread(df, out):
    fig, ax = plt.subplots(figsize=(16, 6))
    t = arr(df["t_s"])
    ax.plot(t, arr(df["pwm_mean_abs"]), label="Mean |PWM|")
    ax.plot(t, arr(df["pwm_spread_abs"]), label="Spread max(|PWM|)-min(|PWM|)")
    ax.set_title("PWM effort matching during simultaneous test")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_pwm_effort_spread.png", dpi=160)
    plt.close(fig)


def plot_errors(df, out):
    fig, ax = plt.subplots(figsize=(16, 7))
    t = arr(df["t_s"])
    for i in range(4):
        ax.plot(t, arr(df[f"err_m{i}"]), label=MOTOR_NAMES[i])
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("RPM tracking error during simultaneous test")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_error_all_motors.png", dpi=160)
    plt.close(fig)


def phase_summary(df, out):
    rows = []
    for phase, d in df.groupby("phase"):
        active = d[[f"cmd_m{i}" for i in range(4)]].abs().max(axis=1) > 5
        da = d[active].copy()
        if da.empty:
            continue

        row = {"phase": phase, "samples": int(len(da))}
        row["rpm_spread_median"] = float(da["rpm_spread_abs"].median())
        row["rpm_spread_max"] = float(da["rpm_spread_abs"].max())
        row["pwm_spread_median"] = float(da["pwm_spread_abs"].median())
        row["pwm_spread_max"] = float(da["pwm_spread_abs"].max())

        for i in range(4):
            row[f"m{i}_rpm_abs_median"] = float(da[f"rpm_abs_m{i}"].median())
            row[f"m{i}_pwm_abs_median"] = float(da[f"pwm_abs_m{i}"].median())

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "06_phase_summary.csv", index=False)
    return summary


def write_report(df, summary, out):
    lines = []
    lines.append("Simultaneous 4-motor lifted test report")
    lines.append("=" * 48)
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append("")
    lines.append("Phase summary:")
    if summary.empty:
        lines.append("  No active phases found.")
    else:
        for _, r in summary.iterrows():
            lines.append(
                f"  {r['phase']}: median speed spread={r['rpm_spread_median']:.1f} rpm, "
                f"max speed spread={r['rpm_spread_max']:.1f} rpm, "
                f"median PWM spread={r['pwm_spread_median']:.1f}"
            )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Small speed spread means the four motors are matching well.")
    lines.append("  - Large PWM spread at similar speed means one motor/driver needs more effort.")
    lines.append("  - Reverse/reversal direction still needs visual confirmation because Hall RPM can be direction-blind.")
    lines.append("  - If all lifted simultaneous tests are smooth, the next test can be very low-speed ground motion.")
    (out / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Plot simultaneous 4-motor lifted test CSV.")
    p.add_argument("csv")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_csv(csv_path)
    out = make_outdir(csv_path, args.out)

    plot_overview(df, out)
    plot_rpm_tracking(df, out)
    plot_speed_spread(df, out)
    plot_pwm(df, out)
    plot_pwm_spread(df, out)
    plot_errors(df, out)
    summary = phase_summary(df, out)
    write_report(df, summary, out)

    print(f"Saved plots/report to: {out}")
    print("Key plots:")
    print(f"  {out / '00_overview_rpm_setpoints.png'}")
    print(f"  {out / '01_rpm_tracking_all_motors.png'}")
    print(f"  {out / '02_motor_speed_spread.png'}")
    print(f"  {out / '03_pwm_outputs_all_motors.png'}")
    print(f"  {out / 'diagnostic_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
