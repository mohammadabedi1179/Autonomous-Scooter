"""
plot_motor_log.py
=================
Reads a CSV produced by motor_individual_test.py and plots:
  • Setpoint vs measured RPM  (per motor)
  • PID tracking error        (per motor)
  • PWM output                (per motor)

Usage
-----
  python3 plot_motor_log.py logs/motor_test_20250101_120000.csv

  If no file is given, the most-recent file in ./logs/ is used.
"""

import sys
import os
import glob
import csv
from collections import defaultdict

# ── find file ──
if len(sys.argv) > 1:
    log_file = sys.argv[1]
else:
    files = sorted(glob.glob("logs/motor_test_*.csv"))
    if not files:
        print("No log files found in ./logs/")
        sys.exit(1)
    log_file = files[-1]

print(f"Reading: {log_file}")

# ── parse CSV ──
data = defaultdict(lambda: {"t": [], "sp": [], "rpm": [], "err": [], "out": []})

with open(log_file) as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        m = int(row["motor"])
        data[m]["t"  ].append(float(row["timestamp_s"  ]))
        data[m]["sp" ].append(float(row["setpoint_rpm" ]))
        data[m]["rpm"].append(float(row["measured_rpm" ]))
        data[m]["err"].append(float(row["error_rpm"    ]))
        data[m]["out"].append(float(row["pwm_out"      ]))

MOTOR_NAMES = [
    "M0 Front Left", "M1 Front Right",
    "M2 Back Left",  "M3 Back Right",
]

# ── plot ──
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Motor PID Performance\n{os.path.basename(log_file)}", fontsize=13)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

    for m in range(4):
        d   = data[m]
        col = colors[m]
        t   = d["t"]

        if not t:
            continue

        # Row 0: setpoint vs RPM
        ax0 = fig.add_subplot(gs[0, m])
        ax0.plot(t, d["sp"],  '--', color='grey', lw=1.2, label='Setpoint')
        ax0.plot(t, d["rpm"], '-',  color=col,    lw=1.5, label='RPM')
        ax0.set_title(MOTOR_NAMES[m], fontsize=9, fontweight='bold')
        ax0.set_ylabel("RPM" if m == 0 else "")
        ax0.legend(fontsize=7, loc='upper right')
        ax0.grid(True, alpha=0.3)

        # Row 1: error
        ax1 = fig.add_subplot(gs[1, m])
        ax1.axhline(0, color='grey', lw=0.8, ls='--')
        ax1.plot(t, d["err"], '-', color=col, lw=1.2)
        ax1.set_ylabel("Error (RPM)" if m == 0 else "")
        ax1.set_title("Tracking Error", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Row 2: PWM output
        ax2 = fig.add_subplot(gs[2, m])
        ax2.axhline(0, color='grey', lw=0.8, ls='--')
        ax2.fill_between(t, d["out"], alpha=0.25, color=col)
        ax2.plot(t, d["out"], '-', color=col, lw=1.2)
        ax2.set_ylim(-210, 210)
        ax2.set_ylabel("PWM out" if m == 0 else "")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("PWM Output", fontsize=8)
        ax2.grid(True, alpha=0.3)

    # ── stats summary ──
    print("\n── PID error statistics per motor ──")
    print(f"{'Motor':<18} {'Mean |err|':>10} {'Max |err|':>10} {'Std err':>8}  "
          f"{'Packets':>7}")
    print("─" * 60)
    for m in range(4):
        e = data[m]["err"]
        if not e:
            print(f"  {MOTOR_NAMES[m]:<16}  (no data)")
            continue
        import statistics
        abs_e = [abs(x) for x in e]
        print(f"  {MOTOR_NAMES[m]:<16}  {sum(abs_e)/len(abs_e):>9.2f}  "
              f"{max(abs_e):>9.2f}  {statistics.stdev(e):>7.2f}  {len(e):>7}")

    out_png = log_file.replace(".csv", "_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out_png}")
    plt.show()

except ImportError:
    # matplotlib not available — print text summary only
    print("\nmatplotlib not installed; printing text summary.\n")
    print(f"{'Motor':<18} {'Mean |err|':>10} {'Max |err|':>10} {'Packets':>7}")
    print("─" * 50)
    import statistics
    for m in range(4):
        e = data[m]["err"]
        if not e:
            print(f"  {MOTOR_NAMES[m]:<16}  (no data)")
            continue
        abs_e = [abs(x) for x in e]
        print(f"  {MOTOR_NAMES[m]:<16}  {sum(abs_e)/len(abs_e):>9.2f}  "
              f"{max(abs_e):>9.2f}  {len(e):>7}")
