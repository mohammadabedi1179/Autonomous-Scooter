#!/usr/bin/env python3
"""
plot_keyboard_signed_drive_log.py
=================================

Plot CSV logs created by:
  keyboard_drive_test_signed_wheels.py --log

Expected CSV columns:
  time_s, vehicle_cmd,
  cmd_m0, cmd_m1, cmd_m2, cmd_m3,
  rpm_m0, rpm_m1, rpm_m2, rpm_m3,
  err_m0, err_m1, err_m2, err_m3,
  pwm_m0, pwm_m1, pwm_m2, pwm_m3

Important:
  Your wheel geometry uses opposite signs:
    forward vehicle command => M0 +, M1 -, M2 +, M3 -

  However, your Hall RPM feedback can be direction-blind. Therefore this plotter
  focuses mostly on speed magnitude matching, PWM effort, and command signs.
  Physical forward/backward direction still needs visual confirmation.

Outputs:
  00_overview_vehicle_speed.png
  01_motor_command_rpm_pwm.png
  02_speed_matching_spread.png
  03_pwm_outputs.png
  04_pwm_effort_matching.png
  05_tracking_error_magnitude.png
  06_left_right_balance.png
  07_phase_scatter_pwm_vs_rpm.png
  08_summary_by_command.csv
  diagnostic_report.txt

Usage:
  python3 plot_keyboard_signed_drive_log.py logs/keyboard_motor_only_signed_YYYYMMDD_HHMMSS.csv

Optional:
  python3 plot_keyboard_signed_drive_log.py logs/file.csv --out keyboard_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MOTOR_NAMES = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]
WHEEL_SIGN = np.array([+1, -1, +1, -1], dtype=float)


def arr(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def load_csv(path):
    df = pd.read_csv(path)

    required = ["time_s", "vehicle_cmd"]
    required += [f"cmd_m{i}" for i in range(4)]
    required += [f"rpm_m{i}" for i in range(4)]
    required += [f"err_m{i}" for i in range(4)]
    required += [f"pwm_m{i}" for i in range(4)]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_s"]).copy()
    df["t_s"] = df["time_s"] - df["time_s"].min()

    # Magnitudes are more reliable for direction-blind Hall RPM.
    for i in range(4):
        df[f"cmd_abs_m{i}"] = df[f"cmd_m{i}"].abs()
        df[f"rpm_abs_m{i}"] = df[f"rpm_m{i}"].abs()
        df[f"pwm_abs_m{i}"] = df[f"pwm_m{i}"].abs()
        df[f"mag_error_m{i}"] = df[f"cmd_abs_m{i}"] - df[f"rpm_abs_m{i}"]

    rpm_abs_cols = [f"rpm_abs_m{i}" for i in range(4)]
    pwm_abs_cols = [f"pwm_abs_m{i}" for i in range(4)]
    mag_err_cols = [f"mag_error_m{i}" for i in range(4)]

    df["vehicle_cmd_abs"] = df["vehicle_cmd"].abs()
    df["mean_rpm_abs"] = df[rpm_abs_cols].mean(axis=1)
    df["rpm_spread_abs"] = df[rpm_abs_cols].max(axis=1) - df[rpm_abs_cols].min(axis=1)
    df["mean_pwm_abs"] = df[pwm_abs_cols].mean(axis=1)
    df["pwm_spread_abs"] = df[pwm_abs_cols].max(axis=1) - df[pwm_abs_cols].min(axis=1)
    df["mean_mag_error"] = df[mag_err_cols].mean(axis=1)

    # Direction-blind but useful display: assign vehicle sign to average speed magnitude.
    # This is NOT a true signed velocity measurement; it is just for plotting against keyboard command.
    df["estimated_vehicle_rpm_signed"] = np.sign(df["vehicle_cmd"]) * df["mean_rpm_abs"]

    # Left/right and front/back matching by speed magnitude.
    df["left_mean_abs"] = df[["rpm_abs_m0", "rpm_abs_m2"]].mean(axis=1)
    df["right_mean_abs"] = df[["rpm_abs_m1", "rpm_abs_m3"]].mean(axis=1)
    df["front_mean_abs"] = df[["rpm_abs_m0", "rpm_abs_m1"]].mean(axis=1)
    df["back_mean_abs"] = df[["rpm_abs_m2", "rpm_abs_m3"]].mean(axis=1)
    df["left_minus_right_abs"] = df["left_mean_abs"] - df["right_mean_abs"]
    df["front_minus_back_abs"] = df["front_mean_abs"] - df["back_mean_abs"]

    return df


def make_outdir(csv_path, out_arg):
    if out_arg:
        out = Path(out_arg)
    else:
        out = Path(str(csv_path).replace(".csv", "_plots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_overview(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(t, arr(df["vehicle_cmd"]), linestyle="--", linewidth=2.0, label="Keyboard vehicle command")
    ax.plot(t, arr(df["estimated_vehicle_rpm_signed"]), label="Estimated vehicle speed = sign(cmd) × mean |RPM|")
    ax.plot(t, arr(df["rpm_spread_abs"]), alpha=0.7, label="Motor speed spread |RPM|")

    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("Keyboard drive overview")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "00_overview_vehicle_speed.png", dpi=160)
    plt.close(fig)


def plot_motor_panels(df, out):
    t = arr(df["t_s"])
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, arr(df[f"cmd_m{i}"]), linestyle="--", label="Commanded wheel RPM")
        ax.plot(t, arr(df[f"rpm_m{i}"]), label="Measured RPM")
        ax.plot(t, arr(df[f"pwm_m{i}"]), alpha=0.45, label="PWM out")
        ax.set_title(MOTOR_NAMES[i])
        ax.set_ylabel("RPM / PWM")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Per-motor command, RPM, and PWM", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "01_motor_command_rpm_pwm.png", dpi=160)
    plt.close(fig)


def plot_speed_matching(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(16, 7))

    for i in range(4):
        ax.plot(t, arr(df[f"rpm_abs_m{i}"]), label=f"{MOTOR_NAMES[i]} |RPM|")

    ax.plot(t, arr(df["vehicle_cmd_abs"]), linestyle="--", linewidth=2.0, label="|vehicle command|")
    ax.set_title("Motor speed magnitude matching")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|RPM|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_speed_matching_spread.png", dpi=160)
    plt.close(fig)


def plot_pwm_outputs(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(16, 7))

    for i in range(4):
        ax.plot(t, arr(df[f"pwm_m{i}"]), label=MOTOR_NAMES[i])

    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("PWM outputs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM out")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_pwm_outputs.png", dpi=160)
    plt.close(fig)


def plot_pwm_effort(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(t, arr(df["mean_pwm_abs"]), label="Mean |PWM|")
    ax.plot(t, arr(df["pwm_spread_abs"]), label="PWM spread max-min")
    ax.plot(t, arr(df["rpm_spread_abs"]), label="RPM spread max-min")
    ax.set_title("PWM effort and speed spread")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM / RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_pwm_effort_matching.png", dpi=160)
    plt.close(fig)


def plot_tracking_error(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(16, 7))

    for i in range(4):
        ax.plot(t, arr(df[f"mag_error_m{i}"]), label=f"{MOTOR_NAMES[i]}: |cmd|-|rpm|")

    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("Magnitude tracking error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_tracking_error_magnitude.png", dpi=160)
    plt.close(fig)


def plot_balance(df, out):
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(t, arr(df["left_mean_abs"]), label="Left mean |RPM|: M0,M2")
    axes[0].plot(t, arr(df["right_mean_abs"]), label="Right mean |RPM|: M1,M3")
    axes[0].plot(t, arr(df["left_minus_right_abs"]), label="Left - Right")
    axes[0].axhline(0, linestyle="--", linewidth=0.8)
    axes[0].set_title("Left/right speed balance")
    axes[0].set_ylabel("RPM")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, arr(df["front_mean_abs"]), label="Front mean |RPM|: M0,M1")
    axes[1].plot(t, arr(df["back_mean_abs"]), label="Back mean |RPM|: M2,M3")
    axes[1].plot(t, arr(df["front_minus_back_abs"]), label="Front - Back")
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_title("Front/back speed balance")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RPM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out / "06_left_right_balance.png", dpi=160)
    plt.close(fig)


def plot_pwm_vs_rpm(df, out):
    active = df[df["vehicle_cmd_abs"] > 1].copy()
    if active.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes = axes.ravel()

    sc = None
    for i, ax in enumerate(axes):
        sc = ax.scatter(
            arr(active[f"rpm_abs_m{i}"]),
            arr(active[f"pwm_abs_m{i}"]),
            c=arr(active["vehicle_cmd_abs"]),
            s=14,
            alpha=0.65,
        )
        ax.set_title(MOTOR_NAMES[i])
        ax.set_xlabel("|RPM|")
        ax.set_ylabel("|PWM|")
        ax.grid(True, alpha=0.3)

    fig.colorbar(sc, ax=axes.tolist(), label="|Vehicle command| RPM")
    fig.suptitle("PWM effort versus measured wheel speed", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "07_phase_scatter_pwm_vs_rpm.png", dpi=160)
    plt.close(fig)


def command_summary(df, out):
    active = df[df["vehicle_cmd_abs"] > 1].copy()
    if active.empty:
        summary = pd.DataFrame()
        summary.to_csv(out / "08_summary_by_command.csv", index=False)
        return summary

    # Bin by rounded keyboard command magnitude and sign.
    active["vehicle_cmd_rounded"] = active["vehicle_cmd"].round().astype(int)

    rows = []
    for cmd, d in active.groupby("vehicle_cmd_rounded"):
        row = {
            "vehicle_cmd": int(cmd),
            "samples": int(len(d)),
            "mean_abs_rpm": float(d["mean_rpm_abs"].mean()),
            "median_abs_rpm": float(d["mean_rpm_abs"].median()),
            "median_rpm_spread": float(d["rpm_spread_abs"].median()),
            "max_rpm_spread": float(d["rpm_spread_abs"].max()),
            "median_pwm_abs": float(d["mean_pwm_abs"].median()),
            "median_pwm_spread": float(d["pwm_spread_abs"].median()),
            "median_left_minus_right_abs": float(d["left_minus_right_abs"].median()),
            "median_front_minus_back_abs": float(d["front_minus_back_abs"].median()),
        }

        for i in range(4):
            row[f"m{i}_median_abs_rpm"] = float(d[f"rpm_abs_m{i}"].median())
            row[f"m{i}_median_abs_pwm"] = float(d[f"pwm_abs_m{i}"].median())

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("vehicle_cmd")
    summary.to_csv(out / "08_summary_by_command.csv", index=False)
    return summary


def write_report(df, summary, out):
    lines = []
    lines.append("Keyboard signed-wheel drive log report")
    lines.append("=" * 48)
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append("")
    lines.append("Sign convention:")
    lines.append("  Forward command: M0 +, M1 -, M2 +, M3 -")
    lines.append("  Backward command: M0 -, M1 +, M2 -, M3 +")
    lines.append("")
    lines.append("Important:")
    lines.append("  Hall RPM may be direction-blind, so reverse direction must be visually confirmed.")
    lines.append("")

    if not summary.empty:
        lines.append("Command summary:")
        for _, r in summary.iterrows():
            lines.append(
                f"  cmd {int(r.vehicle_cmd):+4d} rpm: "
                f"median mean |RPM|={r.median_abs_rpm:.1f}, "
                f"median spread={r.median_rpm_spread:.1f}, "
                f"median |PWM|={r.median_pwm_abs:.1f}, "
                f"L-R={r.median_left_minus_right_abs:+.1f} rpm"
            )
        lines.append("")

        max_spread = float(summary["median_rpm_spread"].max()) if "median_rpm_spread" in summary else np.nan
        if np.isfinite(max_spread) and max_spread > 8:
            lines.append("Warning:")
            lines.append("  Median motor speed spread exceeded about 8 rpm for at least one command.")
            lines.append("  Check whether one side pulls or one wheel lags under the same command.")
            lines.append("")
        else:
            lines.append("General interpretation:")
            lines.append("  Motor speed spread looks acceptable for this log.")
            lines.append("")

    lines.append("Most useful plots:")
    lines.append("  - 00_overview_vehicle_speed.png")
    lines.append("  - 02_speed_matching_spread.png")
    lines.append("  - 04_pwm_effort_matching.png")
    lines.append("  - 06_left_right_balance.png")
    lines.append("  - 07_phase_scatter_pwm_vs_rpm.png")

    (out / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Plot keyboard signed-wheel motor-only log.")
    p.add_argument("csv")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_csv(csv_path)
    out = make_outdir(csv_path, args.out)

    plot_overview(df, out)
    plot_motor_panels(df, out)
    plot_speed_matching(df, out)
    plot_pwm_outputs(df, out)
    plot_pwm_effort(df, out)
    plot_tracking_error(df, out)
    plot_balance(df, out)
    plot_pwm_vs_rpm(df, out)

    summary = command_summary(df, out)
    write_report(df, summary, out)

    print(f"Saved plots/report to: {out}")
    print("Key files:")
    print(f"  {out / '00_overview_vehicle_speed.png'}")
    print(f"  {out / '02_speed_matching_spread.png'}")
    print(f"  {out / '06_left_right_balance.png'}")
    print(f"  {out / 'diagnostic_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
