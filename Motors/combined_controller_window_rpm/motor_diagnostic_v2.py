"""
motor_diagnostic_v2.py
======================
PID diagnostic for combined_controller_v2.ino (adaptive-window RPM firmware).

Changes vs motor_diagnostic_window_safe.py
──────────────────────────────────────────
  • NEW --low-speed flag / TEST 2b: 5/10/15 RPM steps to validate the
    adaptive RPM window fix. Uses an absolute settling band (±2 RPM) at
    low speed instead of 5 % of setpoint (which is sub-resolution at 5 RPM).
  • Step-response metrics now note that rise times reflect the Arduino's
    internal setpoint ramp (40 RPM/s), not a true instantaneous step.
  • Ramp error collection extended down to sp > 1.0 RPM (was 5.0).
  • --all-motors flag runs the selected test sequence on every motor in
    order (M0 → M3) and saves a separate CSV per motor.
  • SETTLE_BAND_RPM_ABS: minimum absolute settling band used when the
    percentage band would be smaller than measurement resolution.

TEST SEQUENCE (flags control which tests run)
─────────────────────────────────────────────
  1.  Dead-zone probe        --deadzone
  2a. Step responses         (default, --no-steps to skip)
  2b. Low-speed steps        --low-speed   ← new
  3.  Ramp tracking          (default, --no-ramp to skip)
  4.  Reverse step           --reverse
  5.  Direction reversal     --reversal
      All of the above       --full

USAGE
─────
  python3 motor_diagnostic_v2.py                        # M1 default
  python3 motor_diagnostic_v2.py --motor 2              # M2 step+ramp
  python3 motor_diagnostic_v2.py --motor 2 --low-speed  # add low-speed test
  python3 motor_diagnostic_v2.py --all-motors           # M0..M3 in sequence
  python3 motor_diagnostic_v2.py --motor 2 --full       # all tests
"""

import serial
import time
import csv
import os
import sys
import argparse
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
PORT  = '/dev/ttyACM0'
BAUD  = 115200
LOG_DIR = 'logs'

MOTOR_NAMES = {0: 'Front Left', 1: 'Front Right',
               2: 'Back Left',  3: 'Back Right'}

# ── Which tests to run by default ───────────────────────────
RUN_DEADZONE   = False
RUN_STEPS      = True
RUN_LOW_SPEED  = False   # enable with --low-speed
RUN_RAMP       = True
RUN_REVERSE    = False
RUN_REVERSAL   = False

# ── Dead-zone probe ──────────────────────────────────────────
DZ_MAX_RPM   = 50
DZ_STEP_RPM  = 5
DZ_HOLD_S    = 0.8
DZ_DETECT_TH = 2.0

# ── Normal step responses ────────────────────────────────────
STEP_LEVELS  = [30, 50, 70]
STEP_HOLD_S  = 6.0
STEP_PAUSE_S = 2.0

# ── Low-speed step responses  ────────────────────────────────
# These target the adaptive-window RPM region (< 20 RPM).
# Hold time is longer to allow the slower window (500–1000 ms) to settle.
# Settling is judged on an absolute band (see SETTLE_BAND_RPM_ABS),
# because 5 % of 5 RPM = 0.25 RPM which is below the 1.33 RPM/pulse
# measurement resolution even with the 1000 ms window.
LOW_SPEED_LEVELS  = [5, 10, 15]
LOW_SPEED_HOLD_S  = 10.0   # longer hold: 1000 ms window needs time to settle
LOW_SPEED_PAUSE_S = 3.0

# ── Settling detection ───────────────────────────────────────
SETTLE_BAND_PCT     = 0.05   # ±5 % of setpoint (normal speeds)
SETTLE_BAND_RPM_ABS = 2.0    # ±2 RPM absolute floor — used when pct band is smaller.
                              # At 5 RPM, 5 % = 0.25 RPM < 1.33 RPM/pulse → use abs.
                              # At 50 RPM, 5 % = 2.5 RPM > 2.0 RPM → use pct.
SETTLE_MIN_S = 0.5

# ── Ramp tracking ────────────────────────────────────────────
RAMP_PEAK_RPM = 70
RAMP_RISE_S   = 10.0
RAMP_HOLD_S   = 3.0
RAMP_FALL_S   = 8.0

# ── Reverse step ─────────────────────────────────────────────
REV_RPM    = 40
REV_HOLD_S = 5.0

# ── Direction reversal ───────────────────────────────────────
REVERSAL_RPM    = 35
REVERSAL_HOLD_S = 4.0

# ── Data sampling ────────────────────────────────────────────
SAMPLE_DT = 0.10   # 100 ms — matches Arduino telemetry period
# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
#  Serial helpers
# ─────────────────────────────────────────────────────────────
def fmt_rpm(v):
    sign = '+' if v >= 0 else '-'
    return f"{sign}{int(abs(v)):03d}"

def make_cmd(rpms):
    return (f"SP#{fmt_rpm(rpms[0])}#{fmt_rpm(rpms[1])}"
            f"#{fmt_rpm(rpms[2])}#{fmt_rpm(rpms[3])}\n")

def send(ser, rpms):
    ser.write(make_cmd(rpms).encode())


# ─────────────────────────────────────────────────────────────
#  Telemetry parser
# ─────────────────────────────────────────────────────────────
class Telemetry:
    def __init__(self):
        self.rpm   = [0.0] * 4
        self.error = [0.0] * 4
        self.pwm   = [0.0] * 4
        self.valid = False

    def update(self, line: str) -> bool:
        if not line or line.startswith('S:') or line.startswith('ERR'):
            return False
        parts = line.split(',')
        if len(parts) != 12:
            return False
        try:
            self.rpm   = [float(parts[i])   for i in range(4)]
            self.error = [float(parts[i+4]) for i in range(4)]
            self.pwm   = [float(parts[i+8]) for i in range(4)]
            self.valid = True
            return True
        except ValueError:
            return False


# ─────────────────────────────────────────────────────────────
#  Metrics calculator
# ─────────────────────────────────────────────────────────────
def step_metrics(setpoint_abs, times, rpms_abs, low_speed=False):
    """
    Compute step-response metrics from (time, |rpm|) samples.

    NOTE: when the Arduino has an internal setpoint ramp (MAX_RPM_RATE = 40 RPM/s),
    'rise time' reflects the ramp + controller response, not a true instantaneous
    step.  At 70 RPM the internal ramp alone takes 1.75 s.

    low_speed=True uses SETTLE_BAND_RPM_ABS instead of the percentage band
    to avoid sub-resolution settling criteria.
    """
    if not times or setpoint_abs <= 0:
        return None

    pct_band = setpoint_abs * SETTLE_BAND_PCT
    band = max(pct_band, SETTLE_BAND_RPM_ABS) if low_speed else pct_band

    # Rise time 10 % → 90 %
    t10 = t90 = None
    for t, r in zip(times, rpms_abs):
        if t10 is None and r >= setpoint_abs * 0.10:
            t10 = t
        if t90 is None and r >= setpoint_abs * 0.90:
            t90 = t
    rise_time_ms = (t90 - t10) * 1000 if (t10 and t90) else None

    # Peak & overshoot
    peak = max(rpms_abs) if rpms_abs else 0.0
    overshoot_pct = max(0.0, 100.0 * (peak - setpoint_abs) / setpoint_abs)

    # Settling time
    settling_time_s = None
    in_band_since   = None
    for t, r in zip(times, rpms_abs):
        if abs(r - setpoint_abs) <= band:
            if in_band_since is None:
                in_band_since = t
            elif t - in_band_since >= SETTLE_MIN_S:
                settling_time_s = in_band_since
                break
        else:
            in_band_since = None

    # Steady-state error (last 1 s of data)
    t_last = times[-1]
    window = [(t, r) for t, r in zip(times, rpms_abs) if t >= t_last - 1.0]
    ss_error_rpm = (sum(abs(r - setpoint_abs) for _, r in window) / len(window)
                    if window else None)
    ss_error_pct = (100 * ss_error_rpm / setpoint_abs
                    if ss_error_rpm is not None else None)

    return {
        'rise_time_ms':    rise_time_ms,
        'overshoot_pct':   overshoot_pct,
        'settling_time_s': settling_time_s,
        'ss_error_rpm':    ss_error_rpm,
        'ss_error_pct':    ss_error_pct,
        'peak_rpm':        peak,
        'settle_band_rpm': band,
    }


def _fmt_metric(v, unit='', decimals=1, none_str='n/a'):
    if v is None:
        return none_str
    return f"{v:.{decimals}f}{unit}"


# ─────────────────────────────────────────────────────────────
#  Core test harness
# ─────────────────────────────────────────────────────────────
class DiagnosticTest:
    def __init__(self, motor_index: int, ser: serial.Serial, writer: csv.DictWriter):
        self.mi     = motor_index
        self.ser    = ser
        self.writer = writer
        self.telem  = Telemetry()
        self.t0     = time.monotonic()

    def _elapsed(self):
        return time.monotonic() - self.t0

    def _drain(self):
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                self.telem.update(line)
            except Exception:
                pass

    def _set(self, rpm_val: float):
        rpms = [0, 0, 0, 0]
        rpms[self.mi] = rpm_val
        send(self.ser, rpms)

    def _log(self, phase: str, setpoint: float):
        self.writer.writerow({
            'time_s':       f"{self._elapsed():.3f}",
            'phase':        phase,
            'setpoint_rpm': f"{setpoint:.1f}",
            'rpm':          f"{self.telem.rpm[self.mi]:.3f}",
            'error_rpm':    f"{self.telem.error[self.mi]:.3f}",
            'pwm_out':      f"{self.telem.pwm[self.mi]:.1f}",
        })

    def _hold(self, setpoint: float, duration: float, phase: str,
              collect: bool = False):
        t_start = time.monotonic()
        t_end   = t_start + duration
        times, rpms_abs = [], []

        while time.monotonic() < t_end:
            self._set(setpoint)
            self._drain()
            self._log(phase, setpoint)
            if collect:
                times.append(time.monotonic() - t_start)
                rpms_abs.append(abs(self.telem.rpm[self.mi]))
            time.sleep(SAMPLE_DT)

        return (times, rpms_abs) if collect else None

    # ── TEST 1: DEAD-ZONE PROBE ──────────────────────────────
    def run_deadzone(self):
        _hdr("TEST 1 · Dead-zone probe",
             f"Steps: 0 → {DZ_MAX_RPM} RPM in {DZ_STEP_RPM} RPM increments  "
             f"({DZ_HOLD_S}s hold each)")

        dead_zone_rpm = None
        max_per_step  = {}

        for sp in range(DZ_STEP_RPM, DZ_MAX_RPM + DZ_STEP_RPM, DZ_STEP_RPM):
            t_end    = time.monotonic() + DZ_HOLD_S
            peak_rpm = 0.0
            while time.monotonic() < t_end:
                self._set(sp)
                self._drain()
                self._log('deadzone', sp)
                peak_rpm = max(peak_rpm, abs(self.telem.rpm[self.mi]))
                time.sleep(SAMPLE_DT)

            max_per_step[sp] = peak_rpm
            marker = ''
            if dead_zone_rpm is None and peak_rpm >= DZ_DETECT_TH:
                dead_zone_rpm = sp
                marker = '  ◄ first response'
            print(f"    SP={sp:3d} RPM  →  peak = {peak_rpm:5.1f} RPM{marker}",
                  flush=True)

        self._hold(0, 1.0, 'dz_settle')

        if dead_zone_rpm:
            print(f"\n  ✓ Motor first responded at ≈ {dead_zone_rpm} RPM")
        else:
            print(f"\n  ✗ WARNING: no response up to {DZ_MAX_RPM} RPM — check wiring")

        return {'threshold_rpm': dead_zone_rpm, 'profile': max_per_step}

    # ── TEST 2a: STEP RESPONSES (normal speed) ───────────────
    def run_steps(self):
        _hdr("TEST 2a · Step responses  (30 / 50 / 70 RPM)",
             f"Hold: {STEP_HOLD_S}s  |  Pause: {STEP_PAUSE_S}s  |  "
             f"NOTE: rise time includes Arduino internal ramp (40 RPM/s)")

        print(f"  Arduino ramps setpoint at 40 RPM/s internally.")
        print(f"  70 RPM step → internal ramp alone takes ~1.75 s.")
        print(f"  Rise time = ramp delay + controller response combined.\n")

        results = {}
        col_w   = [8, 12, 12, 12, 12]
        header  = (f"{'SP(RPM)':>{col_w[0]}}  {'Rise(ms)':>{col_w[1]}}"
                   f"  {'Overshoot%':>{col_w[2]}}  {'Settle(s)':>{col_w[3]}}"
                   f"  {'SS-err(RPM)':>{col_w[4]}}")
        print('  ' + header)
        print('  ' + '─' * (sum(col_w) + 8))

        for sp in STEP_LEVELS:
            self._hold(0, STEP_PAUSE_S, 'step_pause')
            times, rpms_abs = self._hold(sp, STEP_HOLD_S, 'step', collect=True)
            m = step_metrics(sp, times, rpms_abs, low_speed=False)
            results[sp] = m
            if m:
                row = (
                    f"{sp:>{col_w[0]}}  "
                    f"{_fmt_metric(m['rise_time_ms'],    'ms', 0):>{col_w[1]}}  "
                    f"{_fmt_metric(m['overshoot_pct'],    '%', 1):>{col_w[2]}}  "
                    f"{_fmt_metric(m['settling_time_s'],  's', 2):>{col_w[3]}}  "
                    f"{_fmt_metric(m['ss_error_rpm'],      '', 2):>{col_w[4]}}"
                )
                print('  ' + row, flush=True)

        self._hold(0, 1.0, 'step_pause')
        return results

    # ── TEST 2b: LOW-SPEED STEPS ─────────────────────────────
    def run_low_speed_steps(self):
        _hdr("TEST 2b · Low-speed step responses  (5 / 10 / 15 RPM)",
             f"Hold: {LOW_SPEED_HOLD_S}s  |  Settling band: ±{SETTLE_BAND_RPM_ABS} RPM (absolute)")

        print(f"  Validates adaptive RPM window fix.")
        print(f"  At  5 RPM: firmware uses 1000 ms window → 1.33 RPM/pulse resolution")
        print(f"  At 10 RPM: firmware uses  500 ms window → 2.67 RPM/pulse resolution")
        print(f"  At 15 RPM: firmware uses  500 ms window → 2.67 RPM/pulse resolution")
        print(f"  Settling band is ±{SETTLE_BAND_RPM_ABS} RPM absolute (5% would be sub-resolution).\n")

        results = {}
        col_w   = [8, 12, 12, 12, 12]
        header  = (f"{'SP(RPM)':>{col_w[0]}}  {'Rise(ms)':>{col_w[1]}}"
                   f"  {'Overshoot%':>{col_w[2]}}  {'Settle(s)':>{col_w[3]}}"
                   f"  {'SS-err(RPM)':>{col_w[4]}}")
        print('  ' + header)
        print('  ' + '─' * (sum(col_w) + 8))

        for sp in LOW_SPEED_LEVELS:
            self._hold(0, LOW_SPEED_PAUSE_S, 'ls_pause')
            times, rpms_abs = self._hold(sp, LOW_SPEED_HOLD_S, 'low_speed', collect=True)
            m = step_metrics(sp, times, rpms_abs, low_speed=True)
            results[sp] = m
            if m:
                row = (
                    f"{sp:>{col_w[0]}}  "
                    f"{_fmt_metric(m['rise_time_ms'],    'ms', 0):>{col_w[1]}}  "
                    f"{_fmt_metric(m['overshoot_pct'],    '%', 1):>{col_w[2]}}  "
                    f"{_fmt_metric(m['settling_time_s'],  's', 2):>{col_w[3]}}  "
                    f"{_fmt_metric(m['ss_error_rpm'],      '', 2):>{col_w[4]}}"
                )
                print('  ' + row, flush=True)

        self._hold(0, 1.0, 'ls_pause')
        return results

    # ── TEST 3: RAMP TRACKING ────────────────────────────────
    def run_ramp(self):
        _hdr("TEST 3 · Ramp tracking",
             f"0 → {RAMP_PEAK_RPM} RPM over {RAMP_RISE_S}s, "
             f"hold {RAMP_HOLD_S}s, back to 0 over {RAMP_FALL_S}s")

        t_start       = time.monotonic()
        total_dur     = RAMP_RISE_S + RAMP_HOLD_S + RAMP_FALL_S
        t_end         = t_start + total_dur
        tracking_errs = []

        while time.monotonic() < t_end:
            elapsed = time.monotonic() - t_start

            if elapsed <= RAMP_RISE_S:
                sp = RAMP_PEAK_RPM * (elapsed / RAMP_RISE_S)
            elif elapsed <= RAMP_RISE_S + RAMP_HOLD_S:
                sp = RAMP_PEAK_RPM
            else:
                elapsed_fall = elapsed - RAMP_RISE_S - RAMP_HOLD_S
                sp = max(RAMP_PEAK_RPM * (1.0 - elapsed_fall / RAMP_FALL_S), 0.0)

            self._set(sp)
            self._drain()
            self._log('ramp', sp)

            # Collect error for sp > 1 RPM (extended from 5 to capture low-speed region)
            if sp > 1.0:
                tracking_errs.append(abs(self.telem.error[self.mi]))

            time.sleep(SAMPLE_DT)

        self._hold(0, 1.0, 'ramp_settle')

        mean_err = sum(tracking_errs) / len(tracking_errs) if tracking_errs else None
        max_err  = max(tracking_errs)                        if tracking_errs else None

        print(f"\n  Mean tracking error: {_fmt_metric(mean_err, ' RPM', 2)}")
        print(f"  Peak tracking error: {_fmt_metric(max_err,  ' RPM', 2)}")
        print(f"  (High mean = controller is slow for your ramp rate)")

        return {'mean_err_rpm': mean_err, 'peak_err_rpm': max_err}

    # ── TEST 4: REVERSE STEP ─────────────────────────────────
    def run_reverse(self):
        _hdr("TEST 4 · Reverse step response",
             f"Step to −{REV_RPM} RPM, hold {REV_HOLD_S}s")
        print("  Checks: sign handling, no runaway, reverse settling.")

        self._hold(0, STEP_PAUSE_S, 'rev_pause')
        times, rpms_abs = self._hold(-REV_RPM, REV_HOLD_S, 'reverse', collect=True)

        m = step_metrics(REV_RPM, times, rpms_abs, low_speed=False)
        if m:
            print(f"\n  Rise:      {_fmt_metric(m['rise_time_ms'],    'ms', 0)}")
            print(f"  Overshoot: {_fmt_metric(m['overshoot_pct'],    '%',  1)}")
            print(f"  Settle:    {_fmt_metric(m['settling_time_s'],  's',  2)}")
            print(f"  SS-error:  {_fmt_metric(m['ss_error_rpm'], ' RPM',  2)}")
            if m['peak_rpm'] > REV_RPM * 2:
                print(f"\n  ✗ WARNING: peak {m['peak_rpm']:.0f} RPM >> setpoint — "
                      f"possible reverse runaway!")
            elif m['settling_time_s'] is None:
                print(f"\n  ✗ WARNING: never settled in {REV_HOLD_S}s")
            else:
                print(f"\n  ✓ Reverse direction working normally.")
        else:
            print("  ✗ No data collected.")

        self._hold(0, 1.0, 'rev_pause')
        return m

    # ── TEST 5: DIRECTION REVERSAL ───────────────────────────
    def run_reversal(self):
        _hdr("TEST 5 · Direction reversal (no stop through zero)",
             f"+{REVERSAL_RPM} ↔ −{REVERSAL_RPM} RPM  |  "
             f"{REVERSAL_HOLD_S}s per direction  ×2 cycles")
        print("  Exposes integral windup spikes on direction change.")

        self._hold(0, 1.0, 'rev_idle')
        for cycle in range(2):
            print(f"\n  Cycle {cycle+1}: FWD → REV → FWD", flush=True)
            self._hold( REVERSAL_RPM, REVERSAL_HOLD_S, 'reversal_fwd')
            print(f"    → switching to −{REVERSAL_RPM} RPM", flush=True)
            self._hold(-REVERSAL_RPM, REVERSAL_HOLD_S, 'reversal_rev')
            print(f"    → switching to +{REVERSAL_RPM} RPM", flush=True)

        self._hold(0, 1.0, 'rev_idle')
        print("\n  Check plot for overshoot/runaway spike at direction transitions.")


# ─────────────────────────────────────────────────────────────
#  Summary printer
# ─────────────────────────────────────────────────────────────
def _hdr(title, subtitle=''):
    sep = '═' * 64
    print(f"\n{sep}\n  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(sep, flush=True)


def _print_summary(motor_index, summary):
    _hdr(f"DIAGNOSTIC SUMMARY  —  M{motor_index} {MOTOR_NAMES[motor_index]}")

    dz = summary.get('dead_zone')
    if dz:
        th = dz.get('threshold_rpm')
        if th:
            print(f"  Dead-zone threshold : ~{th} RPM")
        else:
            print("  Dead-zone threshold : NO response — check hardware")

    ramp = summary.get('ramp')
    if ramp:
        print(f"\n  Ramp tracking : "
              f"mean {_fmt_metric(ramp.get('mean_err_rpm'), ' RPM', 2)}, "
              f"peak {_fmt_metric(ramp.get('peak_err_rpm'), ' RPM', 2)}")

    for label, key, low_speed in [
        ("Step responses (30/50/70 RPM)", 'steps',      False),
        ("Low-speed steps  (5/10/15 RPM)", 'low_speed', True),
    ]:
        steps = summary.get(key)
        if not steps:
            continue
        print(f"\n  {label}:")
        print(f"  {'SP(RPM)':>8}  {'Rise(ms)':>10}  {'Overshoot%':>11}"
              f"  {'Settle(s)':>10}  {'SS-err(RPM)':>12}  {'SS-err%':>8}")
        print('  ' + '─' * 70)
        for sp, m in steps.items():
            if m:
                print(
                    f"  {sp:>8}  "
                    f"{_fmt_metric(m['rise_time_ms'],    'ms', 0):>10}  "
                    f"{_fmt_metric(m['overshoot_pct'],    '%', 1):>11}  "
                    f"{_fmt_metric(m['settling_time_s'],  's', 2):>10}  "
                    f"{_fmt_metric(m['ss_error_rpm'],      '', 2):>12}  "
                    f"{_fmt_metric(m['ss_error_pct'],     '%', 1):>8}"
                )

    # Tuning hints
    print(f"\n  ── Tuning hints ──")
    steps = summary.get('steps', {})
    ls    = summary.get('low_speed', {})
    all_steps = {**steps, **ls}

    high_os   = [sp for sp, m in all_steps.items() if m and m['overshoot_pct'] > 20]
    no_settle = [sp for sp, m in all_steps.items() if m and m['settling_time_s'] is None]
    high_ss   = [sp for sp, m in all_steps.items()
                 if m and m['ss_error_rpm'] is not None and m['ss_error_rpm'] > 3.0]

    if high_os:
        print(f"  ✗ Overshoot > 20% at {high_os} RPM → reduce Kp or Ki")
    if no_settle:
        print(f"  ✗ Never settled at {no_settle} RPM → increase Ki or check dead-zone")
    if high_ss:
        print(f"  ✗ High SS error at {high_ss} RPM → increase Ki")
    if not high_os and not no_settle and not high_ss:
        print(f"  ✓ All tested levels look well-tuned")

    print('═' * 64)


# ─────────────────────────────────────────────────────────────
#  Single-motor test runner
# ─────────────────────────────────────────────────────────────
def run_motor(mi, ser, args):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(LOG_DIR, f"diag_window_M{mi}_{ts}.csv")

    print(f"\n{'═'*64}")
    print(f"  Motor  : M{mi} — {MOTOR_NAMES[mi]}")
    print(f"  Log    : {csv_path}")
    print(f"{'═'*64}\n")

    FIELDNAMES = ['time_s', 'phase', 'setpoint_rpm', 'rpm', 'error_rpm', 'pwm_out']
    summary = {}

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        test = DiagnosticTest(mi, ser, writer)

        try:
            run_deadzone  = args.full or args.deadzone
            run_low_speed = args.full or args.low_speed
            run_reverse   = args.full or args.reverse
            run_reversal  = args.full or args.reversal

            if run_deadzone:
                summary['dead_zone'] = test.run_deadzone()
            if args.st:
                summary['steps'] = test.run_steps()
            if run_low_speed:
                summary['low_speed'] = test.run_low_speed_steps()
            if args.rp:
                summary['ramp'] = test.run_ramp()
            if run_reverse:
                summary['reverse'] = test.run_reverse()
            if run_reversal:
                test.run_reversal()

        except KeyboardInterrupt:
            print("\n\n  Interrupted — stopping motor safely.")
        finally:
            for _ in range(5):
                send(ser, [0, 0, 0, 0])
                time.sleep(0.05)

    _print_summary(mi, summary)
    print(f"\n  Log → {csv_path}")
    print(f"  Plot: python3 plot_diag.py {csv_path}\n")
    return csv_path


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Motor PID diagnostic — v2')
    p.add_argument('--motor',      type=int, default=1, choices=[0,1,2,3])
    p.add_argument('--all-motors', action='store_true',
                   help='Run selected tests on all four motors in sequence')
    p.add_argument('--port',       default=PORT)
    p.add_argument('--deadzone',   action='store_true')
    p.add_argument('--low-speed',  action='store_true',
                   help='Add TEST 2b: 5/10/15 RPM steps to validate adaptive window')
    p.add_argument('--reverse',    action='store_true')
    p.add_argument('--reversal',   action='store_true')
    p.add_argument('--full',       action='store_true',
                   help='Run all tests including deadzone, low-speed, reverse, reversal')
    p.add_argument('--no-steps',   dest='st', action='store_false')
    p.add_argument('--no-ramp',    dest='rp', action='store_false')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*64}")
    print(f"  Window-RPM Motor Diagnostic  v2")
    print(f"  Port : {args.port} @ {BAUD}")
    print(f"  SAFETY: wheels lifted, no reverse by default")
    print(f"{'═'*64}\n")

    try:
        ser = serial.Serial(args.port, BAUD, timeout=0.15)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}")
        sys.exit(1)

    print("  Waiting for Arduino boot (2 s)...", end='', flush=True)
    time.sleep(2)
    print(" ready.\n")

    motors = list(range(4)) if args.all_motors else [args.motor]
    logs   = []

    for mi in motors:
        if args.all_motors and mi > 0:
            print(f"\n  Pausing 3 s before next motor...", flush=True)
            time.sleep(3)
        logs.append(run_motor(mi, ser, args))

    ser.close()

    if len(logs) > 1:
        print(f"\n{'═'*64}")
        print(f"  All logs saved:")
        for p in logs:
            print(f"    {p}")
        print(f"{'═'*64}\n")


if __name__ == '__main__':
    main()
