#!/usr/bin/env python3
"""
plot_keyboard_directional_odometry.py

Plot logs from keyboard_directional_drive_odometry.py --log.
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
    required = ["time_s", "linear_cmd_rpm", "turn_cmd_rpm", "left_physical_cmd_rpm", "right_physical_cmd_rpm", "left_rpm_est", "right_rpm_est", "v_mps", "omega_radps", "x_m", "y_m", "theta_deg", "path_m"]
    required += [f"cmd_m{i}" for i in range(4)] + [f"rpm_m{i}" for i in range(4)] + [f"pwm_m{i}" for i in range(4)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_s"]).copy()
    df["t_s"] = df["time_s"] - df["time_s"].min()
    for i in range(4):
        df[f"rpm_abs_m{i}"] = df[f"rpm_m{i}"].abs()
        df[f"pwm_abs_m{i}"] = df[f"pwm_m{i}"].abs()
    rpm_cols = [f"rpm_abs_m{i}" for i in range(4)]
    pwm_cols = [f"pwm_abs_m{i}" for i in range(4)]
    df["mean_abs_rpm"] = df[rpm_cols].mean(axis=1)
    df["rpm_spread_abs"] = df[rpm_cols].max(axis=1) - df[rpm_cols].min(axis=1)
    df["mean_abs_pwm"] = df[pwm_cols].mean(axis=1)
    df["pwm_spread_abs"] = df[pwm_cols].max(axis=1) - df[pwm_cols].min(axis=1)
    return df


def make_outdir(csv_path, out_arg):
    out = Path(out_arg) if out_arg else Path(str(csv_path).replace(".csv", "_plots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_xy(df, out):
    x, y = arr(df["x_m"]), arr(df["y_m"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, marker=".", markersize=2, label="Estimated path")
    ax.scatter([x[0]], [y[0]], marker="o", s=80, label="Start")
    ax.scatter([x[-1]], [y[-1]], marker="x", s=80, label="End")
    ax.axis("equal")
    ax.set_title("Estimated XY path")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "00_xy_path.png", dpi=160)
    plt.close(fig)


def plot_pose(df, out):
    t = arr(df["t_s"])
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(t, arr(df["x_m"]), label="x")
    axes[0].plot(t, arr(df["y_m"]), label="y")
    axes[0].set_ylabel("m")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["theta_deg"]))
    axes[1].set_ylabel("heading deg")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, arr(df["path_m"]))
    axes[2].set_ylabel("path m")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, arr(df["v_mps"]), label="v m/s")
    axes[3].plot(t, arr(df["omega_radps"]), label="omega rad/s")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    fig.suptitle("Estimated pose over time", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "01_pose_over_time.png", dpi=160)
    plt.close(fig)


def plot_commands(df, out):
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t, arr(df["linear_cmd_rpm"]), linestyle="--", label="linear command rpm")
    axes[0].plot(t, arr(df["turn_cmd_rpm"]), linestyle="--", label="turn command rpm")
    axes[0].plot(t, arr(df["mean_abs_rpm"]), label="mean |wheel rpm|")
    axes[0].set_ylabel("RPM")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["v_mps"]), label="v m/s")
    axes[1].plot(t, arr(df["omega_radps"]), label="omega rad/s")
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Commands and estimated motion", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "02_commands_and_estimated_speed.png", dpi=160)
    plt.close(fig)


def plot_side(df, out):
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t, arr(df["left_physical_cmd_rpm"]), linestyle="--", label="left side cmd")
    axes[0].plot(t, arr(df["right_physical_cmd_rpm"]), linestyle="--", label="right side cmd")
    axes[0].plot(t, arr(df["left_rpm_est"]), label="left estimated")
    axes[0].plot(t, arr(df["right_rpm_est"]), label="right estimated")
    axes[0].set_ylabel("side RPM")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["right_rpm_est"]) - arr(df["left_rpm_est"]), label="right-left rpm")
    axes[1].plot(t, 2.0 * arr(df["turn_cmd_rpm"]), linestyle="--", label="2*turn command")
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RPM difference")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Left/right side speed", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "03_left_right_side_speed.png", dpi=160)
    plt.close(fig)


def plot_matching(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(4):
        ax.plot(t, arr(df[f"rpm_abs_m{i}"]), label=f"{MOTOR_NAMES[i]} |RPM|")
    ax.plot(t, arr(df["rpm_spread_abs"]), alpha=0.7, label="spread")
    ax.set_title("Motor RPM magnitude matching")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|RPM|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_motor_rpm_magnitude_matching.png", dpi=160)
    plt.close(fig)


def plot_pwm(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(4):
        ax.plot(t, arr(df[f"pwm_m{i}"]), label=MOTOR_NAMES[i])
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("PWM outputs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_pwm_outputs.png", dpi=160)
    plt.close(fig)


def plot_turning(df, out):
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t, arr(df["turn_cmd_rpm"]), linestyle="--", label="turn command")
    axes[0].plot(t, arr(df["omega_radps"]), label="estimated omega")
    axes[0].axhline(0, linestyle="--", linewidth=0.8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["theta_deg"]), label="heading")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("deg")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Turning and heading estimate", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "06_turning_omega.png", dpi=160)
    plt.close(fig)


def summary(df, out):
    row = {
        "duration_s": float(df["t_s"].max() - df["t_s"].min()),
        "final_x_m": float(df["x_m"].iloc[-1]),
        "final_y_m": float(df["y_m"].iloc[-1]),
        "final_theta_deg": float(df["theta_deg"].iloc[-1]),
        "path_length_m": float(df["path_m"].iloc[-1]),
        "max_abs_v_mps": float(df["v_mps"].abs().max()),
        "max_abs_omega_radps": float(df["omega_radps"].abs().max()),
        "median_rpm_spread": float(df["rpm_spread_abs"].median()),
        "max_rpm_spread": float(df["rpm_spread_abs"].max()),
        "median_pwm_spread": float(df["pwm_spread_abs"].median()),
        "max_pwm_spread": float(df["pwm_spread_abs"].max()),
    }
    pd.DataFrame([row]).to_csv(out / "07_odometry_summary.csv", index=False)
    lines = [
        "Directional keyboard odometry report",
        "=" * 44,
        f"Duration: {row['duration_s']:.2f} s",
        f"Estimated final x: {row['final_x_m']:.3f} m",
        f"Estimated final y: {row['final_y_m']:.3f} m",
        f"Estimated final heading: {row['final_theta_deg']:.1f} deg",
        f"Estimated path length: {row['path_length_m']:.3f} m",
        f"Max |v|: {row['max_abs_v_mps']:.3f} m/s",
        f"Max |omega|: {row['max_abs_omega_radps']:.3f} rad/s",
        f"Median RPM spread: {row['median_rpm_spread']:.2f} rpm",
        "",
        "Caveat: this is approximate skid-steer odometry. Turning on ground includes slip.",
    ]
    (out / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_csv(csv_path)
    out = make_outdir(csv_path, args.out)

    plot_xy(df, out)
    plot_pose(df, out)
    plot_commands(df, out)
    plot_side(df, out)
    plot_matching(df, out)
    plot_pwm(df, out)
    plot_turning(df, out)
    summary(df, out)

    print(f"Saved plots/report to: {out}")
    print("Key files:")
    print(f"  {out / '00_xy_path.png'}")
    print(f"  {out / '01_pose_over_time.png'}")
    print(f"  {out / '03_left_right_side_speed.png'}")
    print(f"  {out / '06_turning_omega.png'}")
    print(f"  {out / 'diagnostic_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
