#!/usr/bin/env python3
"""
Robust plotter/analyzer for M1 open-loop Hall/RPM logs.

Accepts:
  1) CSV from m1_openloop_hall_csv_logger_fixed.py
  2) Header-only/empty CSV from the earlier buggy logger, if the sibling raw log
     exists beside it: <csv_stem>_raw.txt
  3) Raw serial log directly: *_raw.txt

Examples:
  python3 plot_m1_openloop_hall_log_v2.py logs/m1_hall_openloop_20260624_172455.csv
  python3 plot_m1_openloop_hall_log_v2.py logs/m1_hall_openloop_20260624_172455_raw.txt
  python3 plot_m1_openloop_hall_log_v2.py logs/file.csv --out m1_analysis --settle-s 1.0

Outputs:
  01_pwm_command.png
  02_rpm_raw_valid_filtered.png
  03_rpm_zoom_moving_only.png
  04_pulse_counts_delta.png
  05_pulse_quality_rates.png
  06_rejected_pulses.png
  07_pulse_age.png
  08_rpm_residuals.png
  09_rpm_vs_pwm.png
  10_rpm_boxplot_by_pwm.png
  11_sample_interval_histogram.png
  summary_by_pwm.csv
  derived_log.csv
  repaired_from_raw.csv       only if raw recovery was needed
  diagnostic_report.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # reliable headless plotting on Linux/SSH
import matplotlib.pyplot as plt

MOVING_PWM_EPS = 1
DEFAULT_SETTLE_S = 1.0
DEFAULT_SPIKE_RPM = 25.0
DEFAULT_DROPOUT_RPM = 1.0
DEFAULT_STALE_MS = 300.0

DATA_FIELDS = [
    "pc_time_s", "phase", "arduino_ms", "armed", "target_pwm", "current_pwm",
    "sign", "limit_pwm", "watchdog_ms", "raw_count", "valid_count", "reject_count",
    "last_raw_period_us", "last_valid_period_us", "rpm_raw", "rpm_valid", "rpm_filt",
    "pulse_age_ms", "hall_level",
]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_data_tokens(raw_line: str, pc_time_s: float, phase: str) -> Optional[Dict[str, object]]:
    """Parse an Arduino DATA line.

    Correct Arduino format has 18 tokens total:
      DATA + 17 telemetry fields
    """
    if not raw_line.startswith("DATA,"):
        return None

    parts = raw_line.strip().split(",")
    if len(parts) != 18:
        return None

    try:
        return {
            "pc_time_s": float(pc_time_s),
            "phase": phase,
            "arduino_ms": int(parts[1]),
            "armed": int(parts[2]),
            "target_pwm": int(parts[3]),
            "current_pwm": int(parts[4]),
            "sign": int(parts[5]),
            "limit_pwm": int(parts[6]),
            "watchdog_ms": int(parts[7]),
            "raw_count": int(parts[8]),
            "valid_count": int(parts[9]),
            "reject_count": int(parts[10]),
            "last_raw_period_us": int(parts[11]),
            "last_valid_period_us": int(parts[12]),
            "rpm_raw": float(parts[13]),
            "rpm_valid": float(parts[14]),
            "rpm_filt": float(parts[15]),
            "pulse_age_ms": float(parts[16]),
            "hall_level": int(parts[17]),
        }
    except ValueError:
        return None


def load_from_raw_txt(raw_path: Path) -> pd.DataFrame:
    """Load rows from raw serial log.

    The raw file lines are written as:
      pc_time_s,phase,raw_arduino_line
    but raw_arduino_line itself contains commas, so normal read_csv is not safe.
    """
    rows: List[Dict[str, object]] = []
    with raw_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("pc_time_s,"):
                continue
            parts = line.split(",", 2)
            if len(parts) != 3:
                continue
            pc_s, phase, raw_arduino_line = parts
            try:
                pc_time_s = float(pc_s)
            except ValueError:
                continue
            row = parse_data_tokens(raw_arduino_line, pc_time_s, phase)
            if row is not None:
                rows.append(row)

    if not rows:
        raise ValueError(f"No parseable DATA rows found in raw log: {raw_path}")

    return pd.DataFrame(rows, columns=DATA_FIELDS)


def sibling_raw_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_raw.txt")


def load_log(path: Path, out: Path) -> Tuple[pd.DataFrame, str]:
    """Load CSV or recover from raw log if needed."""
    suffix = path.suffix.lower()

    if suffix == ".txt" or path.name.endswith("_raw.txt"):
        df = load_from_raw_txt(path)
        source_note = f"Loaded from raw serial log: {path.name}"
        repaired = out / "repaired_from_raw.csv"
        df.to_csv(repaired, index=False)
        return df, source_note

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    # Earlier buggy logger can produce a header-only CSV. Recover from sibling raw log.
    if df.empty:
        raw = sibling_raw_path(path)
        if raw.exists():
            df = load_from_raw_txt(raw)
            source_note = f"Input CSV had no DATA rows; recovered from sibling raw log: {raw.name}"
            repaired = out / "repaired_from_raw.csv"
            df.to_csv(repaired, index=False)
            return df, source_note
        raise ValueError(
            f"CSV has no data rows: {path}\n"
            f"I also looked for sibling raw log but did not find: {raw}\n"
            "Rerun the logger using m1_openloop_hall_csv_logger_fixed.py."
        )

    return df, f"Loaded from CSV: {path.name}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    variants = {
        "target_pwm": ["target_pwm", "target", "targetSignedPwm", "setpoint_pwm"],
        "current_pwm": ["current_pwm", "current", "applied_pwm", "pwm"],
        "limit_pwm": ["limit_pwm", "limit", "pwm_limit"],
        "watchdog_ms": ["watchdog_ms", "wd", "watchdog", "wd_ms"],
        "last_raw_period_us": ["last_raw_period_us", "last_raw_us"],
        "last_valid_period_us": ["last_valid_period_us", "last_valid_us"],
        "hall_level": ["hall_level", "hall"],
    }
    rename_map = {}
    for canonical, names in variants.items():
        if canonical in df.columns:
            continue
        found = first_existing_col(df, names)
        if found is not None and found != canonical:
            rename_map[found] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def shifted_time(series: pd.Series) -> Optional[pd.Series]:
    s = pd.to_numeric(series, errors="coerce")
    non_nan = s.dropna()
    if non_nan.empty:
        return None
    return s - non_nan.iloc[0]


def build_time_axis(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "pc_time_s" in df.columns:
        t = shifted_time(df["pc_time_s"])
        if t is not None:
            return t, "PC time (s)"

    if "arduino_ms" in df.columns:
        t = shifted_time(pd.to_numeric(df["arduino_ms"], errors="coerce") / 1000.0)
        if t is not None:
            return t, "Arduino time (s)"

    return pd.Series(np.arange(len(df)), index=df.index, dtype=float), "Sample index"


def add_derived_columns(
    df: pd.DataFrame,
    settle_s: float,
    spike_rpm: float,
    dropout_rpm: float,
    stale_ms: float,
) -> pd.DataFrame:
    df = normalize_columns(df)
    if df.empty:
        raise ValueError("No rows to plot. The CSV is empty/header-only.")

    required = ["target_pwm", "current_pwm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    numeric_cols = [
        "pc_time_s", "arduino_ms", "armed", "target_pwm", "current_pwm", "sign",
        "limit_pwm", "watchdog_ms", "raw_count", "valid_count", "reject_count",
        "last_raw_period_us", "last_valid_period_us", "rpm_raw", "rpm_valid", "rpm_filt",
        "pulse_age_ms", "hall_level",
    ]
    to_numeric_inplace(df, numeric_cols)

    t, label = build_time_axis(df)
    df["t_s"] = t
    df.attrs["time_axis_label"] = label

    # RPM fallbacks.
    if "rpm_raw" not in df.columns:
        df["rpm_raw"] = np.nan
    if "rpm_valid" not in df.columns:
        df["rpm_valid"] = df["rpm_raw"]
    if "rpm_filt" not in df.columns:
        df["rpm_filt"] = df["rpm_valid"]

    df["target_pwm"] = df["target_pwm"].fillna(0)
    df["current_pwm"] = df["current_pwm"].fillna(0)
    df["moving_cmd"] = df["target_pwm"].abs() > MOVING_PWM_EPS

    # Segment by phase and target PWM. This is better than target alone because every sweep level has stop intervals.
    phase = df["phase"].astype(str) if "phase" in df.columns else pd.Series("", index=df.index)
    target = df["target_pwm"]
    df["segment_id"] = ((target != target.shift(1)) | (phase != phase.shift(1))).cumsum()
    df["segment_start_s"] = df.groupby("segment_id")["t_s"].transform("min")
    df["t_in_segment_s"] = df["t_s"] - df["segment_start_s"]
    df["settled"] = df["t_in_segment_s"] >= settle_s

    for c in ["raw_count", "valid_count", "reject_count"]:
        if c in df.columns:
            d = df[c].diff().fillna(0)
            d[d < 0] = 0
            df[f"{c}_delta"] = d
        else:
            df[f"{c}_delta"] = np.nan

    dt = df["t_s"].diff()
    dt = dt.where(dt > 0, np.nan)
    df["sample_dt_s"] = dt
    df["raw_pulse_rate_hz"] = df["raw_count_delta"] / dt
    df["valid_pulse_rate_hz"] = df["valid_count_delta"] / dt
    df["reject_rate_hz"] = df["reject_count_delta"] / dt

    denom = df["valid_count_delta"] + df["reject_count_delta"]
    df["reject_fraction"] = np.where(denom > 0, df["reject_count_delta"] / denom, np.nan)

    df["rpm_raw_minus_valid"] = df["rpm_raw"] - df["rpm_valid"]
    df["rpm_valid_minus_filt"] = df["rpm_valid"] - df["rpm_filt"]
    df["rpm_jump"] = df["rpm_valid"].diff().abs()

    stale = pd.Series(False, index=df.index)
    if "pulse_age_ms" in df.columns:
        stale = df["pulse_age_ms"] > stale_ms

    df["spike_flag"] = (df["moving_cmd"] & df["settled"] & (df["rpm_jump"] > spike_rpm)).astype(int)
    df["dropout_flag"] = (
        df["moving_cmd"] & df["settled"] & ((df["rpm_valid"].abs() < dropout_rpm) | stale)
    ).astype(int)

    return df


def as_float_array(obj) -> np.ndarray:
    """Return a 1-D NumPy float array for matplotlib.

    Pandas 2.x no longer supports the multidimensional indexing that some
    matplotlib versions still try on Series objects, so every plotted Series
    is converted explicitly.
    """
    return pd.to_numeric(obj, errors="coerce").to_numpy(dtype=float)


def save_fig(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_pwm_change_lines(ax: plt.Axes, df: pd.DataFrame) -> None:
    changes = df.loc[df["segment_id"] != df["segment_id"].shift(1), "t_s"]
    for x in changes.iloc[1:]:
        if pd.notna(x):
            ax.axvline(x, linestyle="--", linewidth=0.8, alpha=0.35)


def plot_pwm(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["target_pwm"]), label="Target PWM")
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["current_pwm"]), label="Applied/current PWM")
    add_pwm_change_lines(ax, df)
    ax.set_title("PWM command and applied PWM")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("PWM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "01_pwm_command.png")


def plot_rpm(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["rpm_raw"]), label="Raw RPM", linewidth=1.0, alpha=0.55)
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["rpm_valid"]), label="Valid/debounced RPM", linewidth=1.1)
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["rpm_filt"]), label="Filtered RPM", linewidth=1.7)
    spikes = df[df["spike_flag"] == 1]
    if not spikes.empty:
        ax.scatter(as_float_array(spikes["t_s"]), as_float_array(spikes["rpm_valid"]), marker="x", label="Spike candidates")
    drops = df[df["dropout_flag"] == 1]
    if not drops.empty:
        ax.scatter(as_float_array(drops["t_s"]), as_float_array(drops["rpm_valid"]), marker="o", label="Dropout candidates")
    add_pwm_change_lines(ax, df)
    ax.set_title("Hall RPM: raw, valid, and filtered")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "02_rpm_raw_valid_filtered.png")


def plot_rpm_moving_only(df: pd.DataFrame, out: Path) -> None:
    use = df[df["moving_cmd"]].copy()
    if use.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(as_float_array(use["t_s"]), as_float_array(use["rpm_valid"]), label="Valid RPM", linewidth=1.1)
    ax.plot(as_float_array(use["t_s"]), as_float_array(use["rpm_filt"]), label="Filtered RPM", linewidth=1.7)
    add_pwm_change_lines(ax, use)
    ax.set_title("RPM during nonzero PWM commands only")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "03_rpm_zoom_moving_only.png")


def plot_pulse_counts(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["raw_count_delta"]), label="Raw pulses per sample")
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["valid_count_delta"]), label="Valid pulses per sample")
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["reject_count_delta"]), label="Rejected pulses per sample")
    add_pwm_change_lines(ax, df)
    ax.set_title("Hall pulse count deltas")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("Pulses/sample")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "04_pulse_counts_delta.png")


def plot_pulse_rates(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["raw_pulse_rate_hz"]), label="Raw pulse rate")
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["valid_pulse_rate_hz"]), label="Valid pulse rate")
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["reject_rate_hz"]), label="Reject rate")
    add_pwm_change_lines(ax, df)
    ax.set_title("Hall pulse rate estimates")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("Hz")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "05_pulse_quality_rates.png")


def plot_rejected(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["reject_count"]), label="Cumulative rejected pulses")
    add_pwm_change_lines(ax, df)
    ax.set_title("Rejected Hall pulses")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("Rejected pulse count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "06_rejected_pulses.png")


def plot_pulse_age(df: pd.DataFrame, out: Path) -> None:
    if "pulse_age_ms" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["pulse_age_ms"]), label="Pulse age")
    add_pwm_change_lines(ax, df)
    ax.set_title("Time since last valid Hall pulse")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("Pulse age (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "07_pulse_age.png")


def plot_residuals(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["rpm_raw_minus_valid"]), label="Raw - valid")
    ax.plot(as_float_array(df["t_s"]), as_float_array(df["rpm_valid_minus_filt"]), label="Valid - filtered")
    ax.axhline(0, linewidth=1.0)
    add_pwm_change_lines(ax, df)
    ax.set_title("RPM residuals / measurement corrections")
    ax.set_xlabel(df.attrs.get("time_axis_label", "Time (s)"))
    ax.set_ylabel("RPM difference")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "08_rpm_residuals.png")


def plot_rpm_vs_pwm(df: pd.DataFrame, out: Path) -> None:
    use = df[df["moving_cmd"] & df["settled"]].copy()
    if use.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(as_float_array(use["target_pwm"]), as_float_array(use["rpm_filt"]), alpha=0.35, label="Samples")
    med = use.groupby("target_pwm", as_index=False)["rpm_filt"].median().sort_values("target_pwm")
    ax.plot(as_float_array(med["target_pwm"]), as_float_array(med["rpm_filt"]), marker="o", linewidth=2.0, label="Median")
    ax.set_title("Open-loop filtered RPM versus PWM")
    ax.set_xlabel("Target PWM")
    ax.set_ylabel("Filtered RPM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out, "09_rpm_vs_pwm.png")


def plot_boxplot(df: pd.DataFrame, out: Path) -> None:
    use = df[df["moving_cmd"] & df["settled"]].copy()
    if use.empty:
        return
    pwms = sorted([p for p in use["target_pwm"].dropna().unique() if abs(p) > MOVING_PWM_EPS])
    groups = []
    labels = []
    for p in pwms:
        vals = use.loc[use["target_pwm"] == p, "rpm_filt"].dropna().to_numpy()
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(int(p)) if float(p).is_integer() else str(p))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_title("Filtered RPM stability at each PWM level")
    ax.set_xlabel("Target PWM")
    ax.set_ylabel("Filtered RPM")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out, "10_rpm_boxplot_by_pwm.png")


def plot_sample_interval(df: pd.DataFrame, out: Path) -> None:
    dt_ms = (df["sample_dt_s"] * 1000.0).replace([np.inf, -np.inf], np.nan).dropna()
    if dt_ms.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.hist(as_float_array(dt_ms), bins=40)
    ax.set_title("Telemetry sample interval")
    ax.set_xlabel("Sample interval (ms)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    save_fig(fig, out, "11_sample_interval_histogram.png")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    use = df[df["moving_cmd"] & df["settled"]].copy()
    if use.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for pwm, g in use.groupby("target_pwm"):
        rpm = g["rpm_filt"].replace([np.inf, -np.inf], np.nan).dropna()
        duration = float(g["t_s"].max() - g["t_s"].min()) if len(g) > 1 else 0.0
        rpm_mean = float(rpm.mean()) if len(rpm) else np.nan
        rpm_std = float(rpm.std(ddof=1)) if len(rpm) > 1 else 0.0
        rpm_median = float(rpm.median()) if len(rpm) else np.nan
        rpm_min = float(rpm.min()) if len(rpm) else np.nan
        rpm_max = float(rpm.max()) if len(rpm) else np.nan
        rpm_p2p = rpm_max - rpm_min if np.isfinite(rpm_max) and np.isfinite(rpm_min) else np.nan
        cv_pct = 100.0 * rpm_std / abs(rpm_mean) if np.isfinite(rpm_mean) and abs(rpm_mean) > 1e-9 else np.nan

        valid_pulses = float(g["valid_count_delta"].sum(skipna=True))
        rejected_pulses = float(g["reject_count_delta"].sum(skipna=True))
        raw_pulses = float(g["raw_count_delta"].sum(skipna=True))
        reject_pct = 100.0 * rejected_pulses / (valid_pulses + rejected_pulses) if (valid_pulses + rejected_pulses) > 0 else 0.0

        rows.append({
            "target_pwm": pwm,
            "samples": int(len(g)),
            "duration_s": duration,
            "rpm_mean": rpm_mean,
            "rpm_median": rpm_median,
            "rpm_std": rpm_std,
            "rpm_min": rpm_min,
            "rpm_max": rpm_max,
            "rpm_peak_to_peak": rpm_p2p,
            "rpm_cv_percent": cv_pct,
            "current_pwm_mean": float(g["current_pwm"].mean()),
            "raw_pulses": raw_pulses,
            "valid_pulses": valid_pulses,
            "rejected_pulses": rejected_pulses,
            "reject_percent": reject_pct,
            "dropout_percent": float(100.0 * g["dropout_flag"].mean()),
            "spike_count": int(g["spike_flag"].sum()),
            "mean_pulse_age_ms": float(g["pulse_age_ms"].mean()) if "pulse_age_ms" in g.columns else np.nan,
            "max_pulse_age_ms": float(g["pulse_age_ms"].max()) if "pulse_age_ms" in g.columns else np.nan,
        })

    return pd.DataFrame(rows).sort_values("target_pwm")


def diagnostic_text(summary: pd.DataFrame, df: pd.DataFrame, source_note: str) -> str:
    lines: List[str] = []
    lines.append("M1 open-loop Hall/RPM diagnostic report")
    lines.append("=" * 46)
    lines.append(source_note)
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append("")

    if summary.empty:
        lines.append("No settled moving segments were found.")
        lines.append("Increase --duration or reduce --settle-s.")
        return "\n".join(lines)

    total_valid = summary["valid_pulses"].sum()
    total_rej = summary["rejected_pulses"].sum()
    reject_pct = 100.0 * total_rej / (total_valid + total_rej) if (total_valid + total_rej) > 0 else 0.0
    worst_cv = summary["rpm_cv_percent"].replace([np.inf, -np.inf], np.nan).max()
    worst_dropout = summary["dropout_percent"].max()
    total_spikes = int(summary["spike_count"].sum())
    max_age = summary["max_pulse_age_ms"].replace([np.inf, -np.inf], np.nan).max()

    pwm_list = ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in summary["target_pwm"])
    lines.append(f"PWM levels analyzed: {pwm_list}")
    lines.append(f"Total valid pulses: {total_valid:.0f}")
    lines.append(f"Total rejected pulses: {total_rej:.0f} ({reject_pct:.2f}% of valid+rejected)")
    lines.append(f"Worst RPM coefficient of variation: {worst_cv:.2f}%")
    lines.append(f"Worst dropout percentage: {worst_dropout:.2f}%")
    lines.append(f"Spike candidates: {total_spikes}")
    if pd.notna(max_age):
        lines.append(f"Max pulse age: {max_age:.1f} ms")
    lines.append("")

    lines.append("Per-PWM settled summary:")
    for _, r in summary.iterrows():
        lines.append(
            f"  PWM {r['target_pwm']:>5.0f}: "
            f"median={r['rpm_median']:.2f} rpm, mean={r['rpm_mean']:.2f}, "
            f"std={r['rpm_std']:.2f}, p2p={r['rpm_peak_to_peak']:.2f}, "
            f"CV={r['rpm_cv_percent']:.2f}%, rejects={r['reject_percent']:.2f}%, "
            f"dropouts={r['dropout_percent']:.2f}%, spikes={int(r['spike_count'])}"
        )
    lines.append("")

    hall_suspect = False
    lines.append("Interpretation:")
    if total_valid <= 0:
        hall_suspect = True
        lines.append("  - No valid Hall pulses were detected during moving commands. Hall tap/pin/pull-up/ground is suspect.")
    if reject_pct > 2.0:
        hall_suspect = True
        lines.append("  - Rejected Hall pulse percentage is high. Raw Hall signal is noisy or has false edges.")
    if worst_dropout > 1.0:
        hall_suspect = True
        lines.append("  - Dropouts/stale-pulse periods occurred while PWM was nonzero. Hall measurement is unreliable.")
    if total_spikes > 0:
        hall_suspect = True
        lines.append("  - Sudden RPM spike candidates were detected. This usually comes from false pulse timing or RPM calculation artifacts.")
    if pd.notna(worst_cv) and worst_cv > 15.0:
        hall_suspect = True
        lines.append("  - Fixed-PWM RPM variation is high. This may be measurement noise or actual speed instability.")

    if not hall_suspect:
        lines.append("  - Hall/RPM measurement looks reasonably clean in this open-loop log.")
        lines.append("  - If the wheel also sounded smooth, the earlier bad closed-loop plot is more likely PID/control logic than motor hardware.")
    else:
        lines.append("  - Do not tune PID yet. First clean or validate the Hall/RPM measurement.")

    lines.append("")
    lines.append("Most useful plots:")
    lines.append("  - 02_rpm_raw_valid_filtered.png")
    lines.append("  - 04_pulse_counts_delta.png")
    lines.append("  - 06_rejected_pulses.png")
    lines.append("  - 10_rpm_boxplot_by_pwm.png")
    return "\n".join(lines)


def make_plots(df: pd.DataFrame, out: Path) -> None:
    plot_pwm(df, out)
    plot_rpm(df, out)
    plot_rpm_moving_only(df, out)
    plot_pulse_counts(df, out)
    plot_pulse_rates(df, out)
    plot_rejected(df, out)
    plot_pulse_age(df, out)
    plot_residuals(df, out)
    plot_rpm_vs_pwm(df, out)
    plot_boxplot(df, out)
    plot_sample_interval(df, out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust M1 open-loop Hall/RPM plotting and analysis")
    parser.add_argument("log", help="CSV log or *_raw.txt log")
    parser.add_argument("--out", default=None, help="Output folder. Default: <log_stem>_plots")
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S,
                        help="Ignore this many seconds after each PWM/phase change")
    parser.add_argument("--spike-rpm", type=float, default=DEFAULT_SPIKE_RPM,
                        help="Adjacent-sample RPM jump threshold for spike flag")
    parser.add_argument("--dropout-rpm", type=float, default=DEFAULT_DROPOUT_RPM,
                        help="RPM below this while moving is a dropout candidate")
    parser.add_argument("--stale-ms", type=float, default=DEFAULT_STALE_MS,
                        help="Pulse age above this while moving is stale/dropout candidate")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    stem = log_path.stem.replace("_raw", "")
    out = Path(args.out).expanduser().resolve() if args.out else log_path.parent / f"{stem}_plots"
    safe_mkdir(out)

    df, source_note = load_log(log_path, out)
    df = add_derived_columns(df, args.settle_s, args.spike_rpm, args.dropout_rpm, args.stale_ms)

    derived_path = out / "derived_log.csv"
    df.to_csv(derived_path, index=False)

    summary = build_summary(df)
    summary_path = out / "summary_by_pwm.csv"
    summary.to_csv(summary_path, index=False)

    make_plots(df, out)

    report = diagnostic_text(summary, df, source_note)
    report_path = out / "diagnostic_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\nAnalysis complete.")
    print(f"Input: {log_path}")
    print(f"Output folder: {out}")
    print("Generated files:")
    for p in sorted(out.iterdir()):
        print(f"  - {p.name}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
