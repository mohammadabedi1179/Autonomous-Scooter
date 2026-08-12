#!/usr/bin/env python3
"""
Plot and summarize the CSV created by run_all_motors_full_window_diag.py.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MOTOR_NAMES = {0: "Front Left", 1: "Front Right", 2: "Back Left", 3: "Back Right"}


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_csv(path):
    df = pd.read_csv(path)
    required = ["pc_time_s", "motor", "test", "phase", "commanded_setpoint_rpm", "rpm", "error_rpm", "pwm_out", "rpm_abs", "pwm_abs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = to_num(df, ["pc_time_s", "motor", "commanded_setpoint_rpm", "rpm", "error_rpm", "pwm_out", "rpm_abs", "pwm_abs"])
    df = df.dropna(subset=["pc_time_s", "motor"])
    df["t_s"] = df["pc_time_s"] - df["pc_time_s"].min()
    df["sp_abs"] = df["commanded_setpoint_rpm"].abs()
    df["rpm_mag"] = df["rpm_abs"].copy()
    return df


def arr(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def make_outdir(csv_path, out_arg):
    out = Path(out_arg) if out_arg else Path(str(csv_path).replace(".csv", "_plots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def title_for_motor(m):
    return f"M{int(m)} {MOTOR_NAMES.get(int(m), '')}"


def get_axes_grid(n=4, figsize=(15, 10), sharex=False, sharey=False):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=sharex, sharey=sharey)
    return fig, axes.ravel()


def plot_overview(df, out):
    motors = sorted(df["motor"].unique())
    fig, axes = plt.subplots(len(motors), 1, figsize=(15, 3.0 * len(motors)), sharex=True)
    if len(motors) == 1:
        axes = [axes]
    for ax, m in zip(axes, motors):
        d = df[df["motor"] == m]
        ax.plot(arr(d["t_s"]), arr(d["commanded_setpoint_rpm"]), linestyle="--", label="Setpoint")
        ax.plot(arr(d["t_s"]), arr(d["rpm"]), label="Measured RPM")
        ax.set_ylabel("RPM")
        ax.set_title(title_for_motor(m))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("All motors full diagnostic overview", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "00_all_motors_overview.png", dpi=160)
    plt.close(fig)


def plot_step_responses(df, out):
    step = df[df["test"] == "step"].copy()
    if step.empty:
        return
    fig, axes = get_axes_grid(sharey=True)
    for ax, m in zip(axes, sorted(step["motor"].unique())):
        d = step[step["motor"] == m]
        for sp in [30, 50, 70]:
            ds = d[d["phase"] == f"step_{sp}"].copy()
            if ds.empty:
                continue
            t = arr(ds["t_s"])
            t = t - np.nanmin(t)
            ax.plot(t, arr(ds["rpm_mag"]), label=f"{sp} rpm")
            ax.axhline(sp, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_title(title_for_motor(m))
        ax.set_xlabel("Time since command (s)")
        ax.set_ylabel("|RPM|")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Forward step responses by motor", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "01_step_responses_by_motor.png", dpi=160)
    plt.close(fig)


def plot_ramp(df, out):
    ramp = df[df["test"] == "ramp"].copy()
    if ramp.empty:
        return
    fig, axes = get_axes_grid(sharey=True)
    for ax, m in zip(axes, sorted(ramp["motor"].unique())):
        d = ramp[ramp["motor"] == m]
        t = arr(d["t_s"])
        t = t - np.nanmin(t)
        ax.plot(t, arr(d["sp_abs"]), linestyle="--", label="Setpoint magnitude")
        ax.plot(t, arr(d["rpm_mag"]), label="Measured |RPM|")
        ax.set_title(title_for_motor(m))
        ax.set_xlabel("Ramp time (s)")
        ax.set_ylabel("RPM")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Ramp tracking by motor", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "02_ramp_tracking_by_motor.png", dpi=160)
    plt.close(fig)


def plot_reverse(df, out):
    rev = df[df["test"] == "reverse"].copy()
    if rev.empty:
        return
    fig, axes = get_axes_grid(sharey=True)
    for ax, m in zip(axes, sorted(rev["motor"].unique())):
        d = rev[rev["motor"] == m]
        t = arr(d["t_s"])
        t = t - np.nanmin(t)
        ax.plot(t, arr(d["commanded_setpoint_rpm"]), linestyle="--", label="Setpoint")
        ax.plot(t, arr(d["rpm"]), label="Measured RPM")
        ax.plot(t, arr(d["pwm_out"]), alpha=0.45, label="PWM out")
        ax.set_title(title_for_motor(m))
        ax.set_xlabel("Reverse test time (s)")
        ax.set_ylabel("RPM / PWM")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Reverse step by motor; visually confirm physical direction", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "03_reverse_by_motor.png", dpi=160)
    plt.close(fig)


def plot_reversal(df, out):
    revs = df[df["test"] == "reversal"].copy()
    if revs.empty:
        return
    fig, axes = get_axes_grid(sharey=True)
    for ax, m in zip(axes, sorted(revs["motor"].unique())):
        d = revs[revs["motor"] == m]
        t = arr(d["t_s"])
        t = t - np.nanmin(t)
        ax.plot(t, arr(d["commanded_setpoint_rpm"]), linestyle="--", label="Setpoint")
        ax.plot(t, arr(d["rpm"]), label="Measured RPM")
        ax.plot(t, arr(d["pwm_out"]), alpha=0.45, label="PWM out")
        ax.set_title(title_for_motor(m))
        ax.set_xlabel("Reversal test time (s)")
        ax.set_ylabel("RPM / PWM")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Direction reversal by motor; measured RPM may be direction-blind", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "04_reversal_by_motor.png", dpi=160)
    plt.close(fig)


def plot_pwm_vs_rpm(df, out):
    fig, axes = get_axes_grid(sharex=True, sharey=True)
    sc = None
    for ax, m in zip(axes, sorted(df["motor"].unique())):
        d = df[(df["motor"] == m) & (df["sp_abs"] > 5)]
        if d.empty:
            continue
        sc = ax.scatter(arr(d["rpm_mag"]), arr(d["pwm_abs"]), c=arr(d["sp_abs"]), s=12, alpha=0.55)
        ax.set_title(title_for_motor(m))
        ax.set_xlabel("|RPM|")
        ax.set_ylabel("|PWM out|")
        ax.grid(True, alpha=0.3)
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), label="|Setpoint| rpm")
    fig.suptitle("PWM effort versus measured speed", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "05_pwm_vs_rpm_by_motor.png", dpi=160)
    plt.close(fig)


def compute_step_metrics(df):
    rows = []
    step = df[df["test"] == "step"].copy()
    if step.empty:
        return pd.DataFrame()
    for m in sorted(step["motor"].unique()):
        d_m = step[step["motor"] == m]
        for sp in [30, 50, 70]:
            d = d_m[d_m["phase"] == f"step_{sp}"].copy()
            if d.empty:
                continue
            t = arr(d["t_s"])
            t = t - np.nanmin(t)
            rpm = arr(d["rpm_mag"])
            valid = np.isfinite(t) & np.isfinite(rpm)
            t = t[valid]
            rpm = rpm[valid]
            if len(t) < 5:
                continue
            target = float(sp)
            peak = float(np.nanmax(rpm))
            overshoot_pct = max(0.0, (peak - target) / target * 100.0)
            last = rpm[t >= (np.nanmax(t) - 1.0)]
            ss_mean = float(np.nanmean(last)) if len(last) else np.nan
            ss_error = float(abs(ss_mean - target)) if np.isfinite(ss_mean) else np.nan
            idx10 = np.where(rpm >= 0.10 * target)[0]
            idx90 = np.where(rpm >= 0.90 * target)[0]
            t10 = float(t[idx10[0]]) if len(idx10) else np.nan
            t90 = float(t[idx90[0]]) if len(idx90) else np.nan
            rise_ms = (t90 - t10) * 1000.0 if np.isfinite(t10) and np.isfinite(t90) else np.nan
            rows.append({"motor": int(m), "motor_name": MOTOR_NAMES.get(int(m), ""), "step_rpm": sp, "peak_rpm": peak, "overshoot_pct": overshoot_pct, "ss_mean_rpm": ss_mean, "ss_error_rpm": ss_error, "rise_time_ms": rise_ms})
    return pd.DataFrame(rows)


def plot_step_metrics(metrics, out):
    if metrics.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    motors = sorted(metrics["motor"].unique())
    labels = [f"M{m}" for m in motors]
    x = np.arange(len(labels))
    width = 0.22
    fields = [("overshoot_pct", "Overshoot (%)"), ("ss_error_rpm", "SS error (rpm)"), ("rise_time_ms", "Rise time (ms)")]
    for ax, (field, ylabel) in zip(axes, fields):
        for i, sp in enumerate([30, 50, 70]):
            d = metrics[metrics["step_rpm"] == sp].set_index("motor")
            vals = [d.loc[m, field] if m in d.index else np.nan for m in motors]
            ax.bar(x + (i - 1) * width, vals, width, label=f"{sp} rpm")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    fig.suptitle("Step metrics comparison by motor", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "06_step_metrics_by_motor.png", dpi=160)
    plt.close(fig)


def make_summary(df, out):
    rows = []
    for (m, test), d in df.groupby(["motor", "test"]):
        active = d[d["sp_abs"] > 5]
        if active.empty:
            continue
        rows.append({"motor": int(m), "motor_name": MOTOR_NAMES.get(int(m), ""), "test": test, "samples": int(len(active)), "rpm_abs_median": float(active["rpm_mag"].median()), "rpm_abs_mean": float(active["rpm_mag"].mean()), "rpm_abs_std": float(active["rpm_mag"].std()), "pwm_abs_median": float(active["pwm_abs"].median()), "pwm_abs_mean": float(active["pwm_abs"].mean()), "max_abs_error_rpm": float(active["error_rpm"].abs().max())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "07_summary_table.csv", index=False)
    metrics = compute_step_metrics(df)
    if not metrics.empty:
        metrics.to_csv(out / "08_step_metrics_table.csv", index=False)
    return summary, metrics


def write_report(df, summary, metrics, out):
    lines = []
    lines.append("All motors full window-RPM diagnostic report")
    lines.append("=" * 52)
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append("")
    lines.append("Motors present:")
    for m in sorted(df["motor"].unique()):
        lines.append(f"  M{int(m)} {MOTOR_NAMES.get(int(m), '')}")
    lines.append("")
    if not metrics.empty:
        lines.append("Step metrics:")
        for _, r in metrics.iterrows():
            lines.append(f"  M{int(r.motor)} {int(r.step_rpm):2d} rpm: peak={r.peak_rpm:.1f}, overshoot={r.overshoot_pct:.1f}%, ss_error={r.ss_error_rpm:.1f} rpm, rise={r.rise_time_ms:.0f} ms")
        lines.append("")
    lines.append("Interpretation hints:")
    lines.append("  - Overshoot above about 15-20% is still aggressive.")
    lines.append("  - Large steady-state error at 30 rpm means low-speed Ki/Kff may need adjustment.")
    lines.append("  - Reverse/reversal plots cannot prove physical direction if Hall sensing is direction-blind.")
    lines.append("  - For reverse/reversal, trust visual direction confirmation plus absence of runaway.")
    lines.append("")
    lines.append("Most useful plots:")
    lines.append("  - 00_all_motors_overview.png")
    lines.append("  - 01_step_responses_by_motor.png")
    lines.append("  - 02_ramp_tracking_by_motor.png")
    lines.append("  - 03_reverse_by_motor.png")
    lines.append("  - 04_reversal_by_motor.png")
    lines.append("  - 06_step_metrics_by_motor.png")
    (out / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Plot full all-motor diagnostic CSV.")
    p.add_argument("csv")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_csv(csv_path)
    out = make_outdir(csv_path, args.out)
    plot_overview(df, out)
    plot_step_responses(df, out)
    plot_ramp(df, out)
    plot_reverse(df, out)
    plot_reversal(df, out)
    plot_pwm_vs_rpm(df, out)
    summary, metrics = make_summary(df, out)
    plot_step_metrics(metrics, out)
    write_report(df, summary, metrics, out)
    print(f"Saved plots/report to: {out}")
    print("Key files:")
    for name in ["00_all_motors_overview.png", "01_step_responses_by_motor.png", "02_ramp_tracking_by_motor.png", "03_reverse_by_motor.png", "04_reversal_by_motor.png", "06_step_metrics_by_motor.png", "diagnostic_report.txt"]:
        print(f"  {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
