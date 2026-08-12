#!/usr/bin/env python3
"""
Plot and summarize M1 open-loop windowed Hall RPM logs.

Use with CSV files from:
  m1_window_rpm_csv_logger.py

Example:
  python3 plot_m1_window_rpm_log.py logs/m1_window_rpm_YYYYMMDD_HHMMSS.csv
  python3 plot_m1_window_rpm_log.py logs/m1_window_rpm_YYYYMMDD_HHMMSS.csv --out m1_window_analysis

Outputs:
  01_pwm_command.png
  02_period_vs_window_rpm.png
  03_window_rpm_moving_only.png
  04_window_pulse_counts.png
  05_pulse_age.png
  06_period_minus_window_residual.png
  07_window_rpm_vs_pwm.png
  08_window_boxplot_by_pwm.png
  09_sample_interval_histogram.png
  summary_by_pwm.csv
  derived_window_log.csv
  diagnostic_report.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERIC_COLS = [
    "pc_time_s",
    "arduino_ms",
    "armed",
    "target_pwm",
    "current_pwm",
    "sign",
    "limit_pwm",
    "watchdog_ms",
    "raw_count",
    "valid_count",
    "reject_count",
    "window_raw_delta",
    "window_valid_delta",
    "window_dt_ms",
    "last_raw_period_us",
    "last_valid_period_us",
    "pulse_age_ms",
    "rpm_period_raw",
    "rpm_period_valid",
    "rpm_window",
    "rpm_window_filt",
    "hall_level",
]


def npv(s: pd.Series) -> np.ndarray:
    """Pandas/Matplotlib compatibility helper."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def ensure_out(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty. Record a new log first.")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = ["target_pwm", "current_pwm", "rpm_period_valid", "rpm_window", "rpm_window_filt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV does not look like a window RPM log. Missing columns: {missing}")
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pc_time_s" in df.columns and df["pc_time_s"].notna().any():
        t0 = df["pc_time_s"].dropna().iloc[0]
        df["t_s"] = df["pc_time_s"] - t0
        df["time_source"] = "pc_time_s"
    elif "arduino_ms" in df.columns and df["arduino_ms"].notna().any():
        t0 = df["arduino_ms"].dropna().iloc[0]
        df["t_s"] = (df["arduino_ms"] - t0) / 1000.0
        df["time_source"] = "arduino_ms"
    else:
        df["t_s"] = np.arange(len(df), dtype=float)
        df["time_source"] = "sample_index"

    df["target_abs_pwm"] = df["target_pwm"].abs()
    df["applied_signed_pwm"] = df["current_pwm"] * df["sign"].fillna(0)
    df["moving_cmd"] = df["target_abs_pwm"] > 0
    df["current_nonzero"] = df["current_pwm"].abs() > 0
    df["pwm_level"] = df["target_abs_pwm"].round().astype("Int64")

    # Phase-relative time, useful for excluding ramp-up transients.
    if "phase" in df.columns:
        first_t_by_phase = df.groupby("phase")["t_s"].transform("min")
        df["phase_t_s"] = df["t_s"] - first_t_by_phase
        df["is_pwm_phase"] = df["phase"].astype(str).str.startswith("pwm_")
    else:
        df["phase_t_s"] = df["t_s"] - df["t_s"].min()
        df["is_pwm_phase"] = df["moving_cmd"]

    df["settled"] = (
        df["is_pwm_phase"]
        & (df["target_abs_pwm"] > 0)
        & (np.abs(df["current_pwm"].abs() - df["target_abs_pwm"]) <= 1)
        & (df["phase_t_s"] >= 1.0)
    )

    df["period_minus_window"] = df["rpm_period_valid"] - df["rpm_window_filt"]
    df["window_dropout"] = df["settled"] & (df["rpm_window_filt"] <= 1.0)
    df["period_dropout"] = df["settled"] & (df["rpm_period_valid"] <= 1.0)

    # PC sample interval; can show serial buffering artifacts.
    df["sample_dt_ms"] = df["t_s"].diff() * 1000.0
    return df


def add_phase_lines(ax, df: pd.DataFrame) -> None:
    if "phase" not in df.columns:
        return
    phase = df["phase"].astype(str)
    change = phase.ne(phase.shift())
    xs = df.loc[change, "t_s"].to_numpy(dtype=float)
    for x in xs:
        if np.isfinite(x):
            ax.axvline(x, linestyle="--", alpha=0.25, linewidth=0.8)


def savefig(out: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(out / name, dpi=160)
    plt.close()


def plot_pwm(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(npv(df["t_s"]), npv(df["target_pwm"]), label="Target signed PWM")
    ax.plot(npv(df["t_s"]), npv(df["applied_signed_pwm"]), label="Applied signed PWM")
    add_phase_lines(ax, df)
    ax.set_title("PWM command and applied PWM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PWM")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "01_pwm_command.png")


def plot_period_vs_window(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(npv(df["t_s"]), npv(df["rpm_period_valid"]), label="Period RPM")
    ax.plot(npv(df["t_s"]), npv(df["rpm_window"]), label="Window RPM")
    ax.plot(npv(df["t_s"]), npv(df["rpm_window_filt"]), label="Filtered window RPM", linewidth=2.0)
    if "window_dropout" in df.columns:
        m = df["window_dropout"].fillna(False)
        if m.any():
            ax.scatter(npv(df.loc[m, "t_s"]), npv(df.loc[m, "rpm_window_filt"]), marker="o", label="Window dropout candidates")
    add_phase_lines(ax, df)
    ax.set_title("Period RPM vs window RPM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "02_period_vs_window_rpm.png")


def plot_moving_only(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    m = df["moving_cmd"].fillna(False) | df["current_nonzero"].fillna(False)
    dd = df.loc[m].copy()
    ax.plot(npv(dd["t_s"]), npv(dd["rpm_period_valid"]), label="Period RPM")
    ax.plot(npv(dd["t_s"]), npv(dd["rpm_window_filt"]), label="Filtered window RPM", linewidth=2.0)
    add_phase_lines(ax, dd)
    ax.set_title("RPM during nonzero PWM commands only")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "03_window_rpm_moving_only.png")


def plot_pulse_counts(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(npv(df["t_s"]), npv(df["window_raw_delta"]), label="Raw pulses/window")
    ax.plot(npv(df["t_s"]), npv(df["window_valid_delta"]), label="Valid pulses/window")
    if "reject_count" in df.columns:
        rej_delta = df["reject_count"].diff().fillna(0).clip(lower=0)
        ax.plot(npv(df["t_s"]), rej_delta.to_numpy(dtype=float), label="Rejected pulses/sample")
    add_phase_lines(ax, df)
    ax.set_title("Hall pulse count per RPM window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pulse count")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "04_window_pulse_counts.png")


def plot_pulse_age(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(npv(df["t_s"]), npv(df["pulse_age_ms"]), label="Pulse age")
    ax.axhline(500, linestyle="--", alpha=0.6, label="500 ms timeout reference")
    add_phase_lines(ax, df)
    ax.set_title("Time since last valid Hall pulse")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pulse age (ms)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "05_pulse_age.png")


def plot_residual(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(npv(df["t_s"]), npv(df["period_minus_window"]), label="Period RPM - filtered window RPM")
    ax.axhline(0, linewidth=0.8)
    add_phase_lines(ax, df)
    ax.set_title("Period estimator residual relative to window estimator")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM difference")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "06_period_minus_window_residual.png")


def plot_rpm_vs_pwm(df: pd.DataFrame, out: Path) -> None:
    settled = df.loc[df["settled"].fillna(False)].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    if not settled.empty:
        ax.scatter(npv(settled["target_abs_pwm"]), npv(settled["rpm_window_filt"]), alpha=0.35, label="Samples")
        med = settled.groupby("target_abs_pwm", dropna=True)["rpm_window_filt"].median().reset_index()
        ax.plot(npv(med["target_abs_pwm"]), npv(med["rpm_window_filt"]), marker="o", linewidth=2.0, label="Median")
    ax.set_title("Open-loop filtered window RPM versus PWM")
    ax.set_xlabel("Target PWM magnitude")
    ax.set_ylabel("Filtered window RPM")
    ax.grid(True, alpha=0.35)
    ax.legend()
    savefig(out, "07_window_rpm_vs_pwm.png")


def plot_boxplot(df: pd.DataFrame, out: Path) -> None:
    settled = df.loc[df["settled"].fillna(False)].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    groups: List[np.ndarray] = []
    labels: List[str] = []
    if not settled.empty:
        for pwm, group in settled.groupby("target_abs_pwm"):
            vals = pd.to_numeric(group["rpm_window_filt"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                groups.append(vals)
                labels.append(str(int(pwm)))
    if groups:
        # labels= works on older Matplotlib; tick_labels= is newer only.
        ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_title("Filtered window RPM stability at each PWM level")
    ax.set_xlabel("Target PWM magnitude")
    ax.set_ylabel("Filtered window RPM")
    ax.grid(True, alpha=0.35)
    savefig(out, "08_window_boxplot_by_pwm.png")


def plot_sample_interval(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = pd.to_numeric(df["sample_dt_ms"], errors="coerce").dropna()
    vals = vals[(vals >= 0) & (vals < 1000)]
    if len(vals):
        ax.hist(vals.to_numpy(dtype=float), bins=50)
    ax.set_title("Telemetry sample interval")
    ax.set_xlabel("Sample interval (ms)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.35)
    savefig(out, "09_sample_interval_histogram.png")


def make_plots(df: pd.DataFrame, out: Path) -> None:
    plot_pwm(df, out)
    plot_period_vs_window(df, out)
    plot_moving_only(df, out)
    plot_pulse_counts(df, out)
    plot_pulse_age(df, out)
    plot_residual(df, out)
    plot_rpm_vs_pwm(df, out)
    plot_boxplot(df, out)
    plot_sample_interval(df, out)


def summarize(df: pd.DataFrame, out: Path, source: Path) -> None:
    steady = df.loc[df["settled"].fillna(False)].copy()
    rows = []

    for pwm, g in steady.groupby("target_abs_pwm"):
        rpm_win = pd.to_numeric(g["rpm_window_filt"], errors="coerce")
        rpm_per = pd.to_numeric(g["rpm_period_valid"], errors="coerce")
        rpm_win_nonzero = rpm_win[rpm_win > 1.0]
        rpm_per_nonzero = rpm_per[rpm_per > 1.0]

        n = len(g)
        if n == 0:
            continue

        def metrics(vals: pd.Series, prefix: str) -> dict:
            vals = vals.dropna()
            if len(vals) == 0:
                return {
                    f"{prefix}_median": np.nan,
                    f"{prefix}_mean": np.nan,
                    f"{prefix}_std": np.nan,
                    f"{prefix}_p2p": np.nan,
                    f"{prefix}_cv_pct": np.nan,
                }
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            p2p = float(vals.max() - vals.min())
            cv = 100.0 * std / mean if abs(mean) > 1e-9 else np.nan
            return {
                f"{prefix}_median": float(vals.median()),
                f"{prefix}_mean": mean,
                f"{prefix}_std": std,
                f"{prefix}_p2p": p2p,
                f"{prefix}_cv_pct": cv,
            }

        row = {
            "pwm": int(pwm),
            "samples": int(n),
            "window_dropout_pct": 100.0 * float((rpm_win <= 1.0).sum()) / n,
            "period_dropout_pct": 100.0 * float((rpm_per <= 1.0).sum()) / n,
            "rejected_pulses_total": int(pd.to_numeric(g["reject_count"], errors="coerce").max() - pd.to_numeric(g["reject_count"], errors="coerce").min()) if "reject_count" in g.columns else 0,
        }
        row.update(metrics(rpm_win_nonzero, "window"))
        row.update(metrics(rpm_per_nonzero, "period"))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = out / "summary_by_pwm.csv"
    summary.to_csv(summary_path, index=False)

    df.to_csv(out / "derived_window_log.csv", index=False)

    report_lines: List[str] = []
    report_lines.append("M1 open-loop window RPM diagnostic report")
    report_lines.append("=" * 48)
    report_lines.append(f"Loaded from CSV: {source.name}")
    report_lines.append(f"Rows analyzed: {len(df)}")
    report_lines.append("")

    if summary.empty:
        report_lines.append("No settled PWM segments found. Check command phases and log duration.")
    else:
        worst_window_cv = summary["window_cv_pct"].max(skipna=True)
        worst_period_cv = summary["period_cv_pct"].max(skipna=True)
        worst_window_dropout = summary["window_dropout_pct"].max(skipna=True)
        worst_period_dropout = summary["period_dropout_pct"].max(skipna=True)
        report_lines.append("PWM levels analyzed: " + ", ".join(str(int(x)) for x in summary["pwm"].tolist()))
        report_lines.append(f"Worst filtered-window RPM CV: {worst_window_cv:.2f}%")
        report_lines.append(f"Worst period RPM CV: {worst_period_cv:.2f}%")
        report_lines.append(f"Worst filtered-window dropout percentage: {worst_window_dropout:.2f}%")
        report_lines.append(f"Worst period dropout percentage: {worst_period_dropout:.2f}%")
        if "reject_count" in df.columns:
            total_rejects = int(pd.to_numeric(df["reject_count"], errors="coerce").max() - pd.to_numeric(df["reject_count"], errors="coerce").min())
            report_lines.append(f"Total rejected pulses: {total_rejects}")
        report_lines.append("")
        report_lines.append("Per-PWM settled summary:")
        for _, r in summary.iterrows():
            report_lines.append(
                f"  PWM {int(r['pwm']):>3}: "
                f"window_median={r['window_median']:.2f} rpm, "
                f"window_mean={r['window_mean']:.2f}, "
                f"window_std={r['window_std']:.2f}, "
                f"window_CV={r['window_cv_pct']:.2f}%, "
                f"window_dropouts={r['window_dropout_pct']:.2f}%, "
                f"period_CV={r['period_cv_pct']:.2f}%, "
                f"period_dropouts={r['period_dropout_pct']:.2f}%"
            )
        report_lines.append("")
        report_lines.append("Interpretation:")
        if worst_window_dropout > 1.0:
            report_lines.append("  - Filtered window RPM still has dropout/zero periods during settled PWM. Check Hall signal or estimator settings.")
        else:
            report_lines.append("  - Filtered window RPM has few/no dropouts during settled PWM.")
        if worst_window_cv < worst_period_cv * 0.65:
            report_lines.append("  - Window RPM is much more stable than period RPM. Use window RPM for PID feedback, not single-pulse RPM.")
        elif worst_window_cv < 12.0:
            report_lines.append("  - Window RPM stability is reasonably good for low-speed open-loop testing.")
        else:
            report_lines.append("  - Window RPM variation is still high. This may be actual speed ripple, Hall quantization, or remaining measurement noise.")
        report_lines.append("  - Do not tune PID until filtered-window RPM looks smooth enough at fixed PWM.")
        report_lines.append("")
        report_lines.append("Most useful plots:")
        report_lines.append("  - 02_period_vs_window_rpm.png")
        report_lines.append("  - 03_window_rpm_moving_only.png")
        report_lines.append("  - 04_window_pulse_counts.png")
        report_lines.append("  - 08_window_boxplot_by_pwm.png")

    (out / "diagnostic_report.txt").write_text("\n".join(report_lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot M1 window RPM logger CSV")
    p.add_argument("csv", help="CSV from m1_window_rpm_csv_logger.py")
    p.add_argument("--out", default=None, help="Output directory; default: <csv_stem>_plots")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    out = Path(args.out) if args.out else csv_path.with_suffix("").parent / f"{csv_path.with_suffix('').name}_plots"
    ensure_out(out)

    try:
        df = load_csv(csv_path)
        df = add_derived(df)
        make_plots(df, out)
        summarize(df, out, csv_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Saved plots/report to: {out}")
    print("Open diagnostic_report.txt first, then 02_period_vs_window_rpm.png and 08_window_boxplot_by_pwm.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
