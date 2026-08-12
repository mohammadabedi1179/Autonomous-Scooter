"""
plot_diag.py
============
Visualises the CSV produced by motor_diagnostic_test.py.

Panels:
  1  Full timeline — all phases annotated with coloured bands
  2  Step responses — overlaid & time-normalised to the step command,
                      with rise-time and settling markers
  3  Dead-zone probe — setpoint vs peak RPM bar chart
  4  Ramp tracking   — setpoint vs measured RPM + tracking-error band
  5  Direction reversal — full reversal sequence, RPM & PWM

Usage
─────
  python3 plot_diag.py logs/diag_M0_20260623_120000.csv

  If no file is given, the most-recent diag_*.csv in ./logs/ is used.
"""

import sys
import os
import csv
import glob
from collections import defaultdict, OrderedDict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy are required.\n"
          "  pip install matplotlib numpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  Resolve input file
# ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    log_file = sys.argv[1]
else:
    candidates = sorted(glob.glob('logs/diag_M*.csv'))
    if not candidates:
        print("No diag_M*.csv files found in ./logs/")
        sys.exit(1)
    log_file = candidates[-1]

print(f"Reading: {log_file}")

# ─────────────────────────────────────────────────────────────
#  Parse CSV
# ─────────────────────────────────────────────────────────────
rows = []
with open(log_file) as fh:
    for row in csv.DictReader(fh):
        try:
            rows.append({
                't':   float(row['time_s']),
                'ph':  row['phase'],
                'sp':  float(row['setpoint_rpm']),
                'rpm': float(row['rpm']),
                'err': float(row['error_rpm']),
                'pwm': float(row['pwm_out']),
            })
        except (ValueError, KeyError):
            pass

if not rows:
    print("No valid rows found — check CSV format.")
    sys.exit(1)

# Extract motor index from filename
import re
m_match = re.search(r'diag_M(\d)', os.path.basename(log_file))
mi      = int(m_match.group(1)) if m_match else 0
MOTOR_NAMES = {0: 'Front Left', 1: 'Front Right',
               2: 'Back Left',  3: 'Back Right'}
motor_label = f"M{mi} {MOTOR_NAMES.get(mi, '')}"

# Colour palette (per phase group)
PHASE_COLORS = {
    'deadzone':      '#9E9E9E',
    'dz_settle':     '#9E9E9E',
    'step':          '#2196F3',
    'step_pause':    '#90CAF9',
    'step_settle':   '#90CAF9',
    'ramp':          '#4CAF50',
    'ramp_settle':   '#A5D6A7',
    'reverse':       '#FF9800',
    'rev_pause':     '#FFCC80',
    'reversal_fwd':  '#E91E63',
    'reversal_fwd2': '#E91E63',
    'reversal_rev':  '#9C27B0',
    'rev_idle':      '#CE93D8',
}

SETTLE_BAND_PCT = 0.05   # must match test script

# Group rows by phase
by_phase = defaultdict(list)
for r in rows:
    by_phase[r['ph']].append(r)


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def arrays(row_list, *keys):
    """Return numpy arrays for requested keys."""
    return tuple(np.array([r[k] for r in row_list]) for k in keys)


def shade_phases(ax, rows_all, alpha=0.10):
    """Draw translucent phase bands on ax."""
    if not rows_all:
        return
    prev_ph = rows_all[0]['ph']
    seg_start = rows_all[0]['t']
    for r in rows_all[1:]:
        if r['ph'] != prev_ph:
            color = PHASE_COLORS.get(prev_ph, '#EEEEEE')
            ax.axvspan(seg_start, r['t'], color=color, alpha=alpha)
            seg_start = r['t']
            prev_ph   = r['ph']
    color = PHASE_COLORS.get(prev_ph, '#EEEEEE')
    ax.axvspan(seg_start, rows_all[-1]['t'], color=color, alpha=alpha)


def add_settle_band(ax, setpoint, t_arr, label=True):
    """Draw ±5% settling band around setpoint."""
    band = abs(setpoint) * SETTLE_BAND_PCT
    if setpoint == 0 or band == 0:
        return
    ax.axhspan(setpoint - band, setpoint + band, color='#4CAF50', alpha=0.15)
    if label:
        ax.text(t_arr[-1], setpoint + band,
                f'±{SETTLE_BAND_PCT*100:.0f}%', fontsize=6.5,
                va='bottom', ha='right', color='#388E3C')


# ─────────────────────────────────────────────────────────────
#  Figure layout
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 20))
fig.suptitle(f"PID Diagnostic  ·  {motor_label}\n{os.path.basename(log_file)}",
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.30,
                       top=0.96, bottom=0.04)

# ─────────────────────────────────────────────────────────────
#  PANEL 1: Full timeline (spans both columns)
# ─────────────────────────────────────────────────────────────
ax_full = fig.add_subplot(gs[0, :])
t_all, sp_all, rpm_all = arrays(rows, 't', 'sp', 'rpm')

ax_full.plot(t_all, sp_all,  '--', color='#546E7A', lw=1.2,  label='Setpoint', zorder=3)
ax_full.plot(t_all, rpm_all, '-',  color='#1565C0', lw=1.0,  label='RPM',      zorder=4)
ax_full.axhline(0, color='black', lw=0.5, ls=':')

shade_phases(ax_full, rows, alpha=0.12)

ax_full.set_title('Full Test Timeline', fontsize=10, fontweight='bold', pad=4)
ax_full.set_xlabel('Time (s)', fontsize=8)
ax_full.set_ylabel('RPM', fontsize=8)
ax_full.legend(fontsize=8, loc='upper right')
ax_full.grid(True, alpha=0.25)

# Phase legend patches
seen = {}
for r in rows:
    ph = r['ph']
    grp = ph.split('_')[0]  # collapse e.g. 'step_pause' → 'step'
    if grp not in seen:
        seen[grp] = PHASE_COLORS.get(ph, '#EEEEEE')
patches = [mpatches.Patch(color=c, label=g, alpha=0.6) for g, c in seen.items()]
ax_full.legend(handles=patches + [
    plt.Line2D([0],[0], color='#546E7A', ls='--', label='Setpoint'),
    plt.Line2D([0],[0], color='#1565C0',           label='RPM'),
], fontsize=7, loc='lower right', ncol=4)

# ─────────────────────────────────────────────────────────────
#  PANEL 2: Overlaid normalised step responses
# ─────────────────────────────────────────────────────────────
ax_step = fig.add_subplot(gs[1, 0])
step_rows = by_phase.get('step', [])

if step_rows:
    # Split into individual steps by detecting setpoint changes
    segments = []
    seg = []
    for i, r in enumerate(step_rows):
        if seg and abs(r['sp'] - seg[0]['sp']) > 1:
            segments.append(seg)
            seg = [r]
        else:
            seg.append(r)
    if seg:
        segments.append(seg)

    cmap   = plt.cm.Blues(np.linspace(0.35, 0.95, len(segments)))
    for seg, col in zip(segments, cmap):
        if not seg:
            continue
        t0  = seg[0]['t']
        sp  = seg[0]['sp']
        t_s = np.array([r['t'] - t0 for r in seg])
        rpm_s = np.array([abs(r['rpm']) for r in seg])

        ax_step.plot(t_s, rpm_s, lw=1.5, color=col, label=f"{sp:.0f} RPM")
        ax_step.axhline(sp, color=col, ls=':', lw=0.8, alpha=0.6)

        # Mark 90% rise
        for idx, r in enumerate(rpm_s):
            if r >= sp * 0.90:
                ax_step.axvline(t_s[idx], color=col, ls='--', lw=0.7, alpha=0.5)
                break

    ax_step.set_title('Step Responses (time-normalised to command)', fontsize=9, fontweight='bold')
    ax_step.set_xlabel('Time since step (s)', fontsize=8)
    ax_step.set_ylabel('|RPM|', fontsize=8)
    ax_step.legend(fontsize=7, loc='lower right')
    ax_step.grid(True, alpha=0.25)
    ax_step.set_xlim(left=0)
else:
    ax_step.text(0.5, 0.5, 'No step data', ha='center', va='center',
                 transform=ax_step.transAxes, color='grey')
    ax_step.set_title('Step Responses', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────
#  PANEL 3: Tracking error per step level
# ─────────────────────────────────────────────────────────────
ax_err = fig.add_subplot(gs[1, 1])
if step_rows:
    for seg, col in zip(segments, cmap):
        if not seg:
            continue
        t0    = seg[0]['t']
        sp    = seg[0]['sp']
        t_s   = np.array([r['t'] - t0 for r in seg])
        err_s = np.array([r['err']     for r in seg])

        ax_err.plot(t_s, err_s, lw=1.2, color=col, label=f"{sp:.0f} RPM")

    ax_err.axhline(0, color='black', lw=0.8, ls='--')
    ax_err.set_title('Tracking Error per Step Level', fontsize=9, fontweight='bold')
    ax_err.set_xlabel('Time since step (s)', fontsize=8)
    ax_err.set_ylabel('Error (RPM)', fontsize=8)
    ax_err.legend(fontsize=7, loc='upper right')
    ax_err.grid(True, alpha=0.25)
    ax_err.set_xlim(left=0)
else:
    ax_err.text(0.5, 0.5, 'No step data', ha='center', va='center',
                transform=ax_err.transAxes, color='grey')
    ax_err.set_title('Tracking Error per Step Level', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────
#  PANEL 4a: Dead-zone bar chart
# ─────────────────────────────────────────────────────────────
ax_dz = fig.add_subplot(gs[2, 0])
dz_rows = by_phase.get('deadzone', [])

if dz_rows:
    # Compute peak RPM per setpoint step
    dz_data = {}
    for r in dz_rows:
        sp = int(round(r['sp']))
        dz_data[sp] = max(dz_data.get(sp, 0.0), abs(r['rpm']))

    sps    = sorted(dz_data.keys())
    peaks  = [dz_data[s] for s in sps]
    # Colour bars: grey until first meaningful response, then blue
    dz_th  = next((s for s, p in zip(sps, peaks) if p >= 2.0), None)
    colors = ['#1565C0' if s >= (dz_th or 9999) else '#B0BEC5' for s in sps]

    ax_dz.bar(sps, peaks, width=2.2, color=colors, edgecolor='white')
    ax_dz.bar(sps, sps,   width=2.2, color='none',
              edgecolor='#78909C', linestyle='--', linewidth=0.6,
              label='Setpoint level')

    if dz_th:
        ax_dz.axvline(dz_th, color='#F44336', lw=1.2, ls='--',
                      label=f'First response @ {dz_th} RPM')

    ax_dz.set_title('Dead-zone Probe  (setpoint vs peak RPM)', fontsize=9, fontweight='bold')
    ax_dz.set_xlabel('Setpoint (RPM)', fontsize=8)
    ax_dz.set_ylabel('Peak measured RPM', fontsize=8)
    ax_dz.legend(fontsize=7)
    ax_dz.grid(True, axis='y', alpha=0.25)
else:
    ax_dz.text(0.5, 0.5, 'No dead-zone data', ha='center', va='center',
               transform=ax_dz.transAxes, color='grey')
    ax_dz.set_title('Dead-zone Probe', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────
#  PANEL 4b: PWM vs RPM scatter (all data — reveals nonlinearity)
# ─────────────────────────────────────────────────────────────
ax_sc = fig.add_subplot(gs[2, 1])
sp_all2 = np.array([r['sp']  for r in rows])
rpm_all2 = np.array([r['rpm'] for r in rows])
pwm_all2 = np.array([r['pwm'] for r in rows])

# Only plot points where setpoint > 0 (forward motion)
mask = sp_all2 > 2
if mask.any():
    sc = ax_sc.scatter(rpm_all2[mask], pwm_all2[mask],
                       c=sp_all2[mask], cmap='Blues',
                       s=6, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax_sc, label='Setpoint (RPM)', pad=0.02)

ax_sc.set_title('Motor Characteristic  (RPM vs PWM, coloured by SP)', fontsize=9, fontweight='bold')
ax_sc.set_xlabel('Measured RPM', fontsize=8)
ax_sc.set_ylabel('PWM output', fontsize=8)
ax_sc.axhline(0, color='black', lw=0.5, ls=':')
ax_sc.grid(True, alpha=0.25)

# ─────────────────────────────────────────────────────────────
#  PANEL 5: Ramp tracking
# ─────────────────────────────────────────────────────────────
ax_ramp = fig.add_subplot(gs[3, :])
ramp_rows = by_phase.get('ramp', [])

if ramp_rows:
    t_r, sp_r, rpm_r, err_r = arrays(ramp_rows, 't', 'sp', 'rpm', 'err')
    t_offset = t_r[0]
    t_r = t_r - t_offset

    ax_ramp.plot(t_r, sp_r,  '--', color='#546E7A', lw=1.4, label='Setpoint')
    ax_ramp.plot(t_r, rpm_r, '-',  color='#4CAF50', lw=1.2, label='Measured RPM')
    ax_ramp.fill_between(t_r, sp_r, rpm_r,
                         where=(sp_r > 0), alpha=0.2, color='#F44336',
                         label='Tracking error')

    ax_ramp2 = ax_ramp.twinx()
    ax_ramp2.plot(t_r, np.abs(err_r), color='#F44336', lw=0.8, alpha=0.7, ls=':')
    ax_ramp2.set_ylabel('|Error| (RPM)', color='#F44336', fontsize=8)
    ax_ramp2.tick_params(axis='y', labelcolor='#F44336', labelsize=7)
    ax_ramp2.set_ylim(bottom=0)

    ax_ramp.set_title('Ramp Tracking  (shaded area = error)', fontsize=9, fontweight='bold')
    ax_ramp.set_xlabel('Time (s)', fontsize=8)
    ax_ramp.set_ylabel('RPM', fontsize=8)
    ax_ramp.legend(fontsize=7, loc='upper left')
    ax_ramp.grid(True, alpha=0.25)
else:
    ax_ramp.text(0.5, 0.5, 'No ramp data', ha='center', va='center',
                 transform=ax_ramp.transAxes, color='grey')
    ax_ramp.set_title('Ramp Tracking', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────
#  PANEL 6a: Reverse step
# ─────────────────────────────────────────────────────────────
ax_rev = fig.add_subplot(gs[4, 0])
rev_rows = by_phase.get('reverse', [])

if rev_rows:
    t_rv, sp_rv, rpm_rv, pwm_rv = arrays(rev_rows, 't', 'sp', 'rpm', 'pwm')
    t_rv = t_rv - t_rv[0]

    sp_abs  = abs(sp_rv[0])
    rpm_abs = np.abs(rpm_rv)

    ax_rev.axhline(sp_abs, color='#546E7A', ls='--', lw=1.0, label=f'Target {sp_abs:.0f} RPM')
    ax_rev.fill_between(t_rv,
                        sp_abs * (1 - SETTLE_BAND_PCT),
                        sp_abs * (1 + SETTLE_BAND_PCT),
                        color='#4CAF50', alpha=0.15, label=f'±{SETTLE_BAND_PCT*100:.0f}% band')
    ax_rev.plot(t_rv, rpm_abs, color='#FF9800', lw=1.4, label='|RPM|')

    ax_rev2 = ax_rev.twinx()
    ax_rev2.plot(t_rv, np.abs(pwm_rv), color='#795548', lw=0.8, ls=':', alpha=0.7)
    ax_rev2.set_ylabel('|PWM|', color='#795548', fontsize=7)
    ax_rev2.tick_params(axis='y', labelcolor='#795548', labelsize=7)

    ax_rev.set_title('Reverse Step Response', fontsize=9, fontweight='bold')
    ax_rev.set_xlabel('Time since command (s)', fontsize=8)
    ax_rev.set_ylabel('|RPM|', fontsize=8)
    ax_rev.legend(fontsize=7, loc='lower right')
    ax_rev.grid(True, alpha=0.25)
else:
    ax_rev.text(0.5, 0.5, 'No reverse data', ha='center', va='center',
                transform=ax_rev.transAxes, color='grey')
    ax_rev.set_title('Reverse Step Response', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────
#  PANEL 6b: Direction reversal
# ─────────────────────────────────────────────────────────────
ax_rvl = fig.add_subplot(gs[4, 1])
rvl_rows = (by_phase.get('reversal_fwd', []) +
            by_phase.get('reversal_rev', []) +
            by_phase.get('reversal_fwd2', []) +
            by_phase.get('rev_idle', []))
rvl_rows.sort(key=lambda r: r['t'])

if rvl_rows:
    t_rl, sp_rl, rpm_rl, pwm_rl = arrays(rvl_rows, 't', 'sp', 'rpm', 'pwm')
    t_rl = t_rl - t_rl[0]

    ax_rvl.axhline(0, color='black', lw=0.5, ls=':')
    ax_rvl.plot(t_rl, sp_rl,  '--', color='#546E7A', lw=1.2, label='Setpoint')
    ax_rvl.plot(t_rl, rpm_rl, '-',  color='#E91E63', lw=1.3, label='RPM')

    # Mark direction-change transitions
    for i in range(1, len(sp_rl)):
        if sp_rl[i] * sp_rl[i-1] < 0:   # sign flip
            ax_rvl.axvline(t_rl[i], color='#FF5722', lw=1.0, ls='--', alpha=0.7)

    ax_rvl2 = ax_rvl.twinx()
    ax_rvl2.plot(t_rl, pwm_rl, color='#795548', lw=0.7, ls=':', alpha=0.6)
    ax_rvl2.set_ylabel('PWM out', color='#795548', fontsize=7)
    ax_rvl2.tick_params(axis='y', labelcolor='#795548', labelsize=7)

    ax_rvl.set_title('Direction Reversal  (dashed lines = sign change)', fontsize=9, fontweight='bold')
    ax_rvl.set_xlabel('Time (s)', fontsize=8)
    ax_rvl.set_ylabel('RPM', fontsize=8)
    ax_rvl.legend(fontsize=7, loc='upper right')
    ax_rvl.grid(True, alpha=0.25)
else:
    ax_rvl.text(0.5, 0.5, 'No reversal data', ha='center', va='center',
                transform=ax_rvl.transAxes, color='grey')
    ax_rvl.set_title('Direction Reversal', fontsize=9, fontweight='bold')


# ─────────────────────────────────────────────────────────────
#  Console summary
# ─────────────────────────────────────────────────────────────
print(f"\n── Diagnostic summary  ({motor_label}) ──")

if dz_rows:
    dz_th = next((s for s, p in zip(sorted(dz_data.keys()),
                                    [dz_data[s] for s in sorted(dz_data.keys())])
                  if p >= 2.0), None)
    print(f"  Dead-zone threshold : {'~' + str(dz_th) + ' RPM setpoint' if dz_th else 'not detected'}")

if step_rows and segments:
    print(f"\n  {'SP':>6}  {'Rise(ms)':>10}  {'Overshoot%':>11}  {'Settle(s)':>10}  {'SS-err(RPM)':>12}")
    print('  ' + '─' * 56)
    for seg in segments:
        if not seg:
            continue
        sp_v  = seg[0]['sp']
        t0    = seg[0]['t']
        ts_   = [r['t'] - t0 for r in seg]
        ra_   = [abs(r['rpm']) for r in seg]

        # Rise time
        t10 = t90 = None
        for t_, r_ in zip(ts_, ra_):
            if t10 is None and r_ >= sp_v * 0.10: t10 = t_
            if t90 is None and r_ >= sp_v * 0.90: t90 = t_
        rt = (t90 - t10) * 1000 if (t10 and t90) else None

        # Overshoot
        pk  = max(ra_)
        os_ = max(0.0, 100 * (pk - sp_v) / sp_v) if sp_v else 0

        # Settling
        band = sp_v * SETTLE_BAND_PCT
        st = None; ib = None
        for t_, r_ in zip(ts_, ra_):
            if abs(r_ - sp_v) <= band:
                if ib is None: ib = t_
                elif t_ - ib >= 0.35: st = ib; break
            else:
                ib = None

        # SS error
        last = [(t_, r_) for t_, r_ in zip(ts_, ra_) if t_ >= ts_[-1] - 1.0]
        sse = sum(abs(r_ - sp_v) for _, r_ in last) / len(last) if last else None

        def f(v, u='', d=1):
            return f"{v:.{d}f}{u}" if v is not None else 'n/a'

        print(f"  {sp_v:>6.0f}  {f(rt,'ms',0):>10}  {f(os_,'%',1):>11}  "
              f"{f(st,'s',2):>10}  {f(sse,'',2):>12}")

# ─────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────
out_png = log_file.replace('.csv', '_diag_plot.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nPlot saved → {out_png}")
plt.show()
