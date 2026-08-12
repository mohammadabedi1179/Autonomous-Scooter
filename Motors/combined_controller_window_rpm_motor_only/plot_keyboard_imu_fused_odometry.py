#!/usr/bin/env python3
"""
plot_keyboard_imu_fused_odometry.py

Plot logs from keyboard_drive_imu_fused_odometry.py --log.

Produces the same kind of XY path / pose-over-time report as the earlier
plot_keyboard_directional_odometry.py, plus new plots specific to what
that script now computes:
  - v_hall_raw vs v_corrected (turning-scrub slip correction)
  - front/rear intra-side slip ratios
  - IMU-fused heading vs raw IMU yaw vs "what encoder-only would have given"
  - which heading source (IMU / encoder fallback) was active over time
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MOTOR_NAMES = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]


def arr(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_csv(path):
    df = pd.read_csv(path)

    required = [
        "time_s", "linear_cmd_rpm", "turn_cmd_rpm",
        "left_physical_cmd_rpm", "right_physical_cmd_rpm",
        "left_rpm_est", "right_rpm_est", "v_hall_raw_mps", "v_corrected_mps",
        "slip_k", "slip_k_raw", "fr_slip_ratio_left", "fr_slip_ratio_right",
        "left_mag_min_rpm", "left_mag_avg_rpm", "right_mag_min_rpm", "right_mag_avg_rpm",
        "omega_used_radps", "omega_encoder_est_radps",
        "x_m", "y_m", "theta_rad", "theta_deg", "path_m", "left_dist_m", "right_dist_m",
        "imu_raw_yaw_deg", "heading_source",
    ]
    required += [f"cmd_m{i}" for i in range(4)] + [f"rpm_m{i}" for i in range(4)]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Is this a log from "
                          f"keyboard_drive_imu_fused_odometry.py?")

    for c in required:
        if c != "heading_source":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # pwm_m0..3 are optional -- older logs (before PWM logging was added
    # back) won't have them. Detect and coerce if present, skip otherwise.
    has_pwm = all(f"pwm_m{i}" in df.columns for i in range(4))
    if has_pwm:
        for i in range(4):
            df[f"pwm_m{i}"] = pd.to_numeric(df[f"pwm_m{i}"], errors="coerce")

    if "imu_connected" in df.columns:
        df["imu_connected_bool"] = to_bool(df["imu_connected"])
    if "imu_fresh" in df.columns:
        df["imu_fresh_bool"] = to_bool(df["imu_fresh"])

    df = df.dropna(subset=["time_s"]).copy()
    df["t_s"] = df["time_s"] - df["time_s"].min()
    df.attrs["has_pwm"] = has_pwm

    for i in range(4):
        df[f"rpm_abs_m{i}"] = df[f"rpm_m{i}"].abs()
    rpm_cols = [f"rpm_abs_m{i}" for i in range(4)]
    df["mean_abs_rpm"] = df[rpm_cols].mean(axis=1)
    df["rpm_spread_abs"] = df[rpm_cols].max(axis=1) - df[rpm_cols].min(axis=1)

    df["using_imu"] = df["heading_source"].astype(str).str.upper().eq("IMU")

    # Reconstruct "what heading/path would encoder-only odometry have given"
    # by integrating omega_encoder_est_radps and v_hall_raw independently,
    # purely for comparison against the fused result.
    dt = df["t_s"].diff().fillna(0.0).to_numpy().copy()
    dt[(dt <= 0) | (dt > 0.5)] = 0.0
    omega_enc = arr(df["omega_encoder_est_radps"])
    theta_enc = np.concatenate([[0.0], np.cumsum(omega_enc[1:] * dt[1:])])
    df["theta_encoder_only_deg"] = np.degrees(theta_enc)

    return df


def make_outdir(csv_path, out_arg):
    out = Path(out_arg) if out_arg else Path(str(csv_path).replace(".csv", "_plots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_xy(df, out):
    x, y = arr(df["x_m"]), arr(df["y_m"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, marker=".", markersize=2, label="Fused path (v_corrected + IMU theta)")
    ax.scatter([x[0]], [y[0]], marker="o", s=80, label="Start")
    ax.scatter([x[-1]], [y[-1]], marker="x", s=80, label="End")
    ax.axis("equal")
    ax.set_title("Estimated XY path (IMU-fused, slip-corrected)")
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
    axes[1].plot(t, arr(df["theta_deg"]), label="fused theta")
    axes[1].set_ylabel("heading deg")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, arr(df["path_m"]))
    axes[2].set_ylabel("path m (slip-corrected)")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, arr(df["v_corrected_mps"]), label="v_corrected m/s")
    axes[3].plot(t, arr(df["omega_used_radps"]), label="omega_used rad/s")
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
    axes[1].plot(t, arr(df["v_corrected_mps"]), label="v_corrected m/s")
    axes[1].plot(t, arr(df["omega_used_radps"]), label="omega_used rad/s")
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Commands and estimated motion", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "02_commands_and_estimated_speed.png", dpi=160)
    plt.close(fig)


def plot_slip_correction(df, out):
    """v_hall_raw vs v_corrected, and the slip factor k that separates them."""
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t, arr(df["v_hall_raw_mps"]), label="v_hall_raw (uncorrected)")
    axes[0].plot(t, arr(df["v_corrected_mps"]), label="v_corrected (slip-adjusted)")
    axes[0].set_ylabel("m/s")
    axes[0].set_title("Forward speed: raw Hall estimate vs slip-corrected")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["slip_k"]), label="slip_k (smoothed)")
    axes[1].plot(t, arr(df["slip_k_raw"]), alpha=0.4, label="slip_k_raw (instantaneous)")
    axes[1].axhline(1.0, linestyle="--", linewidth=0.8, color="gray")
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("slip factor k")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_slip_correction.png", dpi=160)
    plt.close(fig)


def plot_fr_slip(df, out):
    """Front/rear intra-side slip ratios -- should sit near 0 on smooth, even ground."""
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t, arr(df["left_mag_min_rpm"]), label="left min(F,R) rpm -- used")
    axes[0].plot(t, arr(df["left_mag_avg_rpm"]), alpha=0.5, linestyle="--", label="left avg(F,R) rpm -- old method")
    axes[0].plot(t, arr(df["right_mag_min_rpm"]), label="right min(F,R) rpm -- used")
    axes[0].plot(t, arr(df["right_mag_avg_rpm"]), alpha=0.5, linestyle="--", label="right avg(F,R) rpm -- old method")
    axes[0].set_ylabel("|RPM|")
    axes[0].set_title("Per-side wheel magnitude: min (slip-robust) vs average (old)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["fr_slip_ratio_left"]), label="left fr_slip_ratio")
    axes[1].plot(t, arr(df["fr_slip_ratio_right"]), label="right fr_slip_ratio")
    axes[1].set_ylim(-0.02, max(0.3, float(np.nanmax(arr(df["fr_slip_ratio_left"]).tolist() +
                                                       arr(df["fr_slip_ratio_right"]).tolist())) * 1.1))
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("front/rear slip ratio (0=matched)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Front/rear intra-side slip", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "04_front_rear_slip.png", dpi=160)
    plt.close(fig)


def plot_heading_comparison(df, out):
    """Fused (IMU) heading vs raw IMU yaw vs a reconstructed encoder-only heading."""
    t = arr(df["t_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t, arr(df["theta_deg"]), label="fused theta (used)")
    axes[0].plot(t, arr(df["imu_raw_yaw_deg"]), alpha=0.6, label="raw IMU yaw")
    axes[0].plot(t, arr(df["theta_encoder_only_deg"]), alpha=0.6, linestyle="--",
                 label="reconstructed encoder-only heading (comparison)")
    axes[0].set_ylabel("deg")
    axes[0].set_title("Heading: fused vs raw IMU vs encoder-only reconstruction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, arr(df["omega_used_radps"]), label="omega_used (fused)")
    axes[1].plot(t, arr(df["omega_encoder_est_radps"]), alpha=0.6, label="omega_encoder_est")
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("rad/s")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "05_heading_comparison.png", dpi=160)
    plt.close(fig)


def plot_imu_health(df, out):
    """Which heading source was active, and IMU freshness, over the run."""
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.fill_between(t, 0, 1, where=df["using_imu"].to_numpy(), step="mid",
                     alpha=0.4, label="IMU driving heading")
    ax.fill_between(t, 0, 1, where=~df["using_imu"].to_numpy(), step="mid",
                     alpha=0.4, color="red", label="Encoder fallback (IMU stale/missing)")
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Heading source over time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "06_imu_health.png", dpi=160)
    plt.close(fig)


def plot_matching(df, out):
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(4):
        ax.plot(t, arr(df[f"rpm_abs_m{i}"]), label=f"{MOTOR_NAMES[i]} |RPM|")
    ax.plot(t, arr(df["rpm_spread_abs"]), alpha=0.7, label="4-motor spread")
    ax.set_title("Motor RPM magnitude matching")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|RPM|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "07_motor_rpm_magnitude_matching.png", dpi=160)
    plt.close(fig)


def plot_pwm(df, out):
    """PID effort (PWM/'out') per motor -- shows how hard each motor is being
    pushed, not just how fast it's turning. Useful for e.g. checking whether
    a motor is saturating at PWM_MAX, or how the dead-zone kick floor and
    Ki/Kff tuning behave in practice (e.g. after adjusting M1's DZ_KICK_PWM)."""
    t = arr(df["t_s"])
    fig, ax = plt.subplots(figsize=(15, 7))
    for i in range(4):
        ax.plot(t, arr(df[f"pwm_m{i}"]), label=MOTOR_NAMES[i])
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("PWM / PID effort outputs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "08_pwm_outputs.png", dpi=160)
    plt.close(fig)


def summary(df, out):
    row = {
        "duration_s": float(df["t_s"].max() - df["t_s"].min()),
        "final_x_m": float(df["x_m"].iloc[-1]),
        "final_y_m": float(df["y_m"].iloc[-1]),
        "final_theta_deg": float(df["theta_deg"].iloc[-1]),
        "path_length_m": float(df["path_m"].iloc[-1]),
        "max_abs_v_corrected_mps": float(df["v_corrected_mps"].abs().max()),
        "max_abs_v_hall_raw_mps": float(df["v_hall_raw_mps"].abs().max()),
        "max_abs_omega_radps": float(df["omega_used_radps"].abs().max()),
        "median_slip_k": float(df["slip_k"].median()),
        "min_slip_k": float(df["slip_k"].min()),
        "median_fr_slip_ratio_left": float(df["fr_slip_ratio_left"].median()),
        "median_fr_slip_ratio_right": float(df["fr_slip_ratio_right"].median()),
        "max_fr_slip_ratio_left": float(df["fr_slip_ratio_left"].max()),
        "max_fr_slip_ratio_right": float(df["fr_slip_ratio_right"].max()),
        "pct_time_using_imu": float(100.0 * df["using_imu"].mean()),
        "median_rpm_spread": float(df["rpm_spread_abs"].median()),
        "max_rpm_spread": float(df["rpm_spread_abs"].max()),
    }
    pd.DataFrame([row]).to_csv(out / "09_odometry_summary.csv", index=False)
    lines = [
        "IMU-fused, slip-corrected keyboard odometry report",
        "=" * 52,
        f"Duration: {row['duration_s']:.2f} s",
        f"Estimated final x: {row['final_x_m']:.3f} m",
        f"Estimated final y: {row['final_y_m']:.3f} m",
        f"Estimated final heading: {row['final_theta_deg']:.1f} deg",
        f"Estimated path length (slip-corrected): {row['path_length_m']:.3f} m",
        f"Max |v_corrected|: {row['max_abs_v_corrected_mps']:.3f} m/s "
        f"(raw Hall max was {row['max_abs_v_hall_raw_mps']:.3f} m/s)",
        f"Max |omega|: {row['max_abs_omega_radps']:.3f} rad/s",
        "",
        f"Turning-scrub slip factor k: median {row['median_slip_k']:.3f}, min {row['min_slip_k']:.3f}",
        f"Front/rear slip ratio -- left: median {row['median_fr_slip_ratio_left']:.3f}, "
        f"max {row['max_fr_slip_ratio_left']:.3f}",
        f"Front/rear slip ratio -- right: median {row['median_fr_slip_ratio_right']:.3f}, "
        f"max {row['max_fr_slip_ratio_right']:.3f}",
        "",
        f"IMU was driving heading {row['pct_time_using_imu']:.1f}% of the run "
        f"(rest used encoder fallback)",
        f"Median 4-motor RPM spread: {row['median_rpm_spread']:.2f} rpm",
        "",
        "Caveat: v_corrected uses a symmetric-slip approximation (see script",
        "docstring). fr_slip_ratio assumes slip shows up as wheel over-speed,",
        "not a dragging/bound wheel.",
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
    plot_slip_correction(df, out)
    plot_fr_slip(df, out)
    plot_heading_comparison(df, out)
    plot_imu_health(df, out)
    plot_matching(df, out)
    if df.attrs.get("has_pwm"):
        plot_pwm(df, out)
    else:
        print("Note: this log has no pwm_m0..3 columns (older logger version) -- skipping PWM plot.")
    summary(df, out)

    print(f"Saved plots/report to: {out}")
    print("Key files:")
    print(f"  {out / '00_xy_path.png'}")
    print(f"  {out / '01_pose_over_time.png'}")
    print(f"  {out / '03_slip_correction.png'}")
    print(f"  {out / '04_front_rear_slip.png'}")
    print(f"  {out / '05_heading_comparison.png'}")
    print(f"  {out / '06_imu_health.png'}")
    if df.attrs.get("has_pwm"):
        print(f"  {out / '08_pwm_outputs.png'}")
    print(f"  {out / 'diagnostic_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
