"""
motor_diagnostic_test.py
========================
Safe PID diagnostic test for the autonomous scooter using the window-RPM Arduino controller.

Adapted for combined_controller_window_rpm.ino. The Arduino serial protocol is
unchanged, but this script uses safer defaults for the first closed-loop tests:
M1 by default, no reverse/reversal by default, and moderate target RPM levels.

TEST SEQUENCE
─────────────
  1. Dead-zone probe     — small setpoint steps; detects the minimum RPM the
                           motor actually responds to (stiction threshold).
  2. Step responses      — instantaneous setpoint jumps at multiple RPM levels;
                           computes rise time, overshoot, settling time, SS error.
  3. Ramp tracking       — slow linear ramp; measures how well the controller
                           tracks a moving setpoint (velocity lag).
  4. Reverse step        — step to a negative RPM; confirms sign handling works
                           and that reverse direction doesn't runaway.
  5. Direction reversal  — +RPM → −RPM without stopping through zero; exposes
                           integral windup on direction change.

CSV OUTPUT FORMAT
─────────────────
  time_s, phase, setpoint_rpm, rpm, error_rpm, pwm_out

Run plot_diag.py to visualise.

USAGE
─────
  python3 motor_diagnostic_window_safe.py                 # M1 safe test
  python3 motor_diagnostic_window_safe.py --motor 1
  python3 motor_diagnostic_window_safe.py --motor 1 --full
  python3 motor_diagnostic_window_safe.py --motor 1 --reverse

DEPENDENCIES
────────────
  pip install pyserial  (standard library otherwise)
"""

import serial
import time
import csv
import os
import sys
import argparse
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  CONFIG — edit here or pass CLI args
# ══════════════════════════════════════════════════════════════
PORT       = '/dev/ttyACM0'
BAUD       = 115200
LOG_DIR    = 'logs'

MOTOR_NAMES = {
    0: 'Front Left',
    1: 'Front Right',
    2: 'Back Left',
    3: 'Back Right',
}

# ── Which tests to run ──────────────────────────────────────
RUN_DEADZONE  = False   # not useful for first test with closed-loop + feed-forward
RUN_STEPS     = True
RUN_RAMP      = True
RUN_REVERSE   = False   # enable manually only after forward tests are safe
RUN_REVERSAL  = False   # enable manually only after reverse is safe

# ── Dead-zone probe ─────────────────────────────────────────
DZ_MAX_RPM   = 50      # probe up to this setpoint (RPM)
DZ_STEP_RPM  = 5       # setpoint increment per step
DZ_HOLD_S    = 0.8     # hold time at each step (s)
DZ_DETECT_TH = 2.0     # RPM above which we call "motor is moving"

# ── Step responses ───────────────────────────────────────────
STEP_LEVELS  = [30, 50, 70]       # safe first forward RPM levels for window-RPM controller
STEP_HOLD_S  = 6.0                # hold at each level (s) — needs enough time to settle
STEP_PAUSE_S = 2.0                # dwell at 0 between steps

# ── Ramp tracking ────────────────────────────────────────────
RAMP_PEAK_RPM  = 70    # peak RPM at top of ramp
RAMP_RISE_S    = 10.0   # duration of the rising ramp
RAMP_HOLD_S    = 3.0   # hold at peak before descending
RAMP_FALL_S    = 8.0   # duration of the falling ramp

# ── Reverse step ─────────────────────────────────────────────
REV_RPM      = 40      # magnitude of reverse test (RPM)
REV_HOLD_S   = 5.0

# ── Direction reversal (no stop through zero) ────────────────
REVERSAL_RPM    = 35   # ±RPM
REVERSAL_HOLD_S = 4.0  # hold at each direction

# ── Settling detection criteria ──────────────────────────────
SETTLE_BAND_PCT  = 0.05   # ±5 % of setpoint counts as "settled"
SETTLE_MIN_S     = 0.35   # must stay in band for this long to count

# ── Data sampling ────────────────────────────────────────────
SAMPLE_DT = 0.10   # Arduino telemetry is 100 ms in combined_controller_window_rpm.ino
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
#  Arduino sends: rpm0,rpm1,rpm2,rpm3,err0,err1,err2,err3,out0,out1,out2,out3
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
def step_metrics(setpoint_abs, times, rpms_abs):
    """
    Compute standard step-response metrics from a list of (time, |rpm|) samples.
    times[0] is assumed to be t=0 of the step command.

    Returns a dict with keys:
        rise_time_ms     — time from 10 % to 90 % of setpoint (ms)
        overshoot_pct    — peak above setpoint as percentage
        settling_time_s  — first time the signal enters ±SETTLE_BAND_PCT and stays
        ss_error_rpm     — mean |error| in the last second of data
        ss_error_pct     — ss_error as % of setpoint
        peak_rpm         — absolute peak measured RPM
    """
    if not times or setpoint_abs <= 0:
        return None

    band = setpoint_abs * SETTLE_BAND_PCT

    # Rise time (10 % → 90 %)
    t10 = t90 = None
    for t, r in zip(times, rpms_abs):
        if t10 is None and r >= setpoint_abs * 0.10:
            t10 = t
        if t90 is None and r >= setpoint_abs * 0.90:
            t90 = t

    rise_time_ms = (t90 - t10) * 1000 if (t10 is not None and t90 is not None) else None

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
    if window:
        ss_error_rpm = sum(abs(r - setpoint_abs) for _, r in window) / len(window)
    else:
        ss_error_rpm = None

    ss_error_pct = (100 * ss_error_rpm / setpoint_abs) if ss_error_rpm is not None else None

    return {
        'rise_time_ms':    rise_time_ms,
        'overshoot_pct':   overshoot_pct,
        'settling_time_s': settling_time_s,
        'ss_error_rpm':    ss_error_rpm,
        'ss_error_pct':    ss_error_pct,
        'peak_rpm':        peak,
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

    # ── internal helpers ─────────────────────────────────────

    def _elapsed(self) -> float:
        return time.monotonic() - self.t0

    def _drain(self):
        """Read all waiting bytes; update telemetry with any valid line."""
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
            'time_s':      f"{self._elapsed():.3f}",
            'phase':       phase,
            'setpoint_rpm': f"{setpoint:.1f}",
            'rpm':         f"{self.telem.rpm[self.mi]:.3f}",
            'error_rpm':   f"{self.telem.error[self.mi]:.3f}",
            'pwm_out':     f"{self.telem.pwm[self.mi]:.1f}",
        })

    def _hold(self, setpoint: float, duration: float, phase: str,
              collect: bool = False):
        """
        Hold setpoint for `duration` seconds, logging at SAMPLE_DT.
        If collect=True, also returns (times_rel, rpms_abs) for metric calc.
        """
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

        dead_zone_rpm   = None   # first setpoint that produced movement
        max_per_step    = {}

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
            print(f"    SP={sp:3d} RPM  →  peak measured = {peak_rpm:5.1f} RPM{marker}",
                  flush=True)

        self._hold(0, 1.0, 'dz_settle')

        if dead_zone_rpm:
            print(f"\n  ✓ Motor first responded at setpoint ≈ {dead_zone_rpm} RPM")
            print(f"    (anything below this will not produce motion)")
        else:
            print(f"\n  ✗ WARNING: no response detected up to {DZ_MAX_RPM} RPM!")
            print(f"    Check wiring, PWM pin, and direction pin.")

        return {'threshold_rpm': dead_zone_rpm, 'profile': max_per_step}

    # ── TEST 2: STEP RESPONSES ───────────────────────────────
    def run_steps(self):
        _hdr("TEST 2 · Step responses",
             f"Levels: {STEP_LEVELS} RPM  |  Hold: {STEP_HOLD_S}s  |  "
             f"Pause: {STEP_PAUSE_S}s")

        results = {}
        col_w   = [8, 12, 12, 12, 12]

        # Print table header
        header = (f"{'SP(RPM)':>{col_w[0]}}  {'Rise(ms)':>{col_w[1]}}"
                  f"  {'Overshoot%':>{col_w[2]}}  {'Settle(s)':>{col_w[3]}}"
                  f"  {'SS-err(RPM)':>{col_w[4]}}")
        print('\n  ' + header)
        print('  ' + '─' * (sum(col_w) + 8))

        for sp in STEP_LEVELS:
            # Brief pause at zero so the step is clean
            self._hold(0, STEP_PAUSE_S, 'step_pause')

            # Step command — start recording immediately
            times, rpms_abs = self._hold(sp, STEP_HOLD_S, 'step', collect=True)

            m = step_metrics(sp, times, rpms_abs)
            results[sp] = m

            if m:
                row = (
                    f"{sp:>{col_w[0]}}  "
                    f"{_fmt_metric(m['rise_time_ms'],    'ms', 0):>{col_w[1]}}  "
                    f"{_fmt_metric(m['overshoot_pct'],   '%',  1):>{col_w[2]}}  "
                    f"{_fmt_metric(m['settling_time_s'], 's',  2):>{col_w[3]}}  "
                    f"{_fmt_metric(m['ss_error_rpm'],    '',   2):>{col_w[4]}}"
                )
                print('  ' + row, flush=True)

        self._hold(0, 1.0, 'step_pause')
        return results

    # ── TEST 3: RAMP TRACKING ────────────────────────────────
    def run_ramp(self):
        _hdr("TEST 3 · Ramp tracking",
             f"0 → {RAMP_PEAK_RPM} RPM over {RAMP_RISE_S}s, "
             f"hold {RAMP_HOLD_S}s, "
             f"back to 0 over {RAMP_FALL_S}s")

        t_start       = time.monotonic()
        total_dur     = RAMP_RISE_S + RAMP_HOLD_S + RAMP_FALL_S
        t_end         = t_start + total_dur
        tracking_errs = []

        while time.monotonic() < t_end:
            elapsed = time.monotonic() - t_start

            if elapsed <= RAMP_RISE_S:
                # Rising slope
                sp = RAMP_PEAK_RPM * (elapsed / RAMP_RISE_S)
            elif elapsed <= RAMP_RISE_S + RAMP_HOLD_S:
                # Hold at peak
                sp = RAMP_PEAK_RPM
            else:
                # Falling slope
                elapsed_fall = elapsed - RAMP_RISE_S - RAMP_HOLD_S
                sp = RAMP_PEAK_RPM * (1.0 - elapsed_fall / RAMP_FALL_S)
                sp = max(sp, 0.0)

            self._set(sp)
            self._drain()
            self._log('ramp', sp)

            # Only count error while setpoint is meaningfully non-zero
            if sp > 5.0:
                tracking_errs.append(abs(self.telem.error[self.mi]))

            time.sleep(SAMPLE_DT)

        self._hold(0, 1.0, 'ramp_settle')

        mean_err = sum(tracking_errs) / len(tracking_errs) if tracking_errs else None
        max_err  = max(tracking_errs)                        if tracking_errs else None

        print(f"\n  Mean tracking error: {_fmt_metric(mean_err, ' RPM', 2)}")
        print(f"  Peak tracking error: {_fmt_metric(max_err,  ' RPM', 2)}")
        print(f"  (High mean error = controller is slow for your ramp rate)")

        return {'mean_err_rpm': mean_err, 'peak_err_rpm': max_err}

    # ── TEST 4: REVERSE STEP ─────────────────────────────────
    def run_reverse(self):
        _hdr("TEST 4 · Reverse step response",
             f"Step to −{REV_RPM} RPM, hold {REV_HOLD_S}s")
        print("  Checks: sign handling, no runaway, reverse settling.")

        self._hold(0, STEP_PAUSE_S, 'rev_pause')

        times, rpms_abs = self._hold(-REV_RPM, REV_HOLD_S, 'reverse', collect=True)

        m = step_metrics(REV_RPM, times, rpms_abs)
        if m:
            print(f"\n  Rise:      {_fmt_metric(m['rise_time_ms'],    'ms', 0)}")
            print(f"  Overshoot: {_fmt_metric(m['overshoot_pct'],   '%',  1)}")
            print(f"  Settle:    {_fmt_metric(m['settling_time_s'], 's',  2)}")
            print(f"  SS-error:  {_fmt_metric(m['ss_error_rpm'],    ' RPM', 2)}")
            if m['peak_rpm'] > REV_RPM * 2:
                print(f"\n  ✗ WARNING: peak RPM {m['peak_rpm']:.0f} >> setpoint "
                      f"— possible reverse runaway! Check signed-RPM logic.")
            elif m['settling_time_s'] is None:
                print(f"\n  ✗ WARNING: motor never settled in {REV_HOLD_S}s — "
                      f"increase hold time or check reverse stiction.")
            else:
                print(f"\n  ✓ Reverse direction working normally.")
        else:
            print("  ✗ No data collected.")

        self._hold(0, 1.0, 'rev_pause')
        return m

    # ── TEST 5: DIRECTION REVERSAL (no stop) ─────────────────
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
        print("\n  Check plot for any overshoot/runaway spike at direction transitions.")


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────
def _hdr(title: str, subtitle: str = ''):
    sep = '═' * 64
    print(f"\n{sep}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(sep, flush=True)


def _print_summary(motor_index, summary):
    _hdr(f"DIAGNOSTIC SUMMARY  —  M{motor_index} {MOTOR_NAMES[motor_index]}")

    dz = summary.get('dead_zone')
    if dz:
        th = dz.get('threshold_rpm')
        if th:
            print(f"  Dead zone threshold : ~{th} RPM setpoint")
            print(f"    → Min setpoint in your controller should be ≥ {th} RPM")
        else:
            print("  Dead zone threshold : NO response detected — check hardware")

    ramp = summary.get('ramp')
    if ramp:
        print(f"\n  Ramp tracking error : "
              f"mean {_fmt_metric(ramp.get('mean_err_rpm'), ' RPM', 2)}, "
              f"peak {_fmt_metric(ramp.get('peak_err_rpm'), ' RPM', 2)}")

    steps = summary.get('steps')
    if steps:
        print(f"\n  Step response table:")
        print(f"  {'SP(RPM)':>8}  {'Rise(ms)':>10}  {'Overshoot%':>11}"
              f"  {'Settle(s)':>10}  {'SS-err(RPM)':>12}  {'SS-err%':>8}")
        print('  ' + '─' * 70)
        for sp, m in steps.items():
            if m:
                row = (
                    f"  {sp:>8}  "
                    f"{_fmt_metric(m['rise_time_ms'],    'ms', 0):>10}  "
                    f"{_fmt_metric(m['overshoot_pct'],   '%',  1):>11}  "
                    f"{_fmt_metric(m['settling_time_s'], 's',  2):>10}  "
                    f"{_fmt_metric(m['ss_error_rpm'],    '',   2):>12}  "
                    f"{_fmt_metric(m['ss_error_pct'],    '%',  1):>8}"
                )
                print(row)

    # Diagnosis hints
    print(f"\n  ── Tuning hints ──")
    if steps:
        high_os = [sp for sp, m in steps.items()
                   if m and m['overshoot_pct'] > 20]
        no_settle = [sp for sp, m in steps.items()
                     if m and m['settling_time_s'] is None]
        high_ss = [sp for sp, m in steps.items()
                   if m and m['ss_error_rpm'] is not None and m['ss_error_rpm'] > 3.0]

        if high_os:
            print(f"  ✗ Large overshoot at {high_os} RPM → reduce Kp or add Kd")
        if no_settle:
            print(f"  ✗ Never settled at {no_settle} RPM → Ki may be too low "
                  f"or dead-zone interfering")
        if high_ss:
            print(f"  ✗ High SS error at {high_ss} RPM → increase Ki")
        if not high_os and not no_settle and not high_ss:
            print(f"  ✓ Step responses look well-tuned at tested levels")

    print('═' * 64)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Motor PID diagnostic test')
    p.add_argument('--motor', type=int, default=1, choices=[0,1,2,3],
                   help='Motor index (0=FL, 1=FR, 2=BL, 3=BR)')
    p.add_argument('--port', default=PORT)
    p.add_argument('--deadzone', action='store_true',
                   help='Enable dead-zone test. Not recommended for first closed-loop test.')
    p.add_argument('--reverse', action='store_true',
                   help='Enable reverse step test. Use only after forward test is safe.')
    p.add_argument('--reversal', action='store_true',
                   help='Enable direction reversal test. Use only after reverse is safe.')
    p.add_argument('--full', action='store_true',
                   help='Run all tests including deadzone, reverse, and reversal.')
    p.add_argument('--no-steps',     dest='st',  action='store_false')
    p.add_argument('--no-ramp',      dest='rp',  action='store_false')
    return p.parse_args()


def main():
    args = parse_args()
    mi   = args.motor

    os.makedirs(LOG_DIR, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(LOG_DIR, f"diag_window_M{mi}_{ts}.csv")

    print(f"\n{'═'*64}")
    print(f"  Window-RPM Motor Diagnostic Test")
    print(f"  Motor  : M{mi} — {MOTOR_NAMES[mi]}")
    print(f"  Port   : {args.port} @ {BAUD}")
    print(f"  Log    : {csv_path}")
    print(f"{'═'*64}\n")

    FIELDNAMES = ['time_s', 'phase', 'setpoint_rpm', 'rpm', 'error_rpm', 'pwm_out']

    try:
        ser = serial.Serial(args.port, BAUD, timeout=0.15)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}")
        sys.exit(1)

    # Wait for Arduino reset after serial connect
    print("  Waiting for Arduino boot (2 s)...", end='', flush=True)
    time.sleep(2)
    print(" ready.\n")
    print("  SAFETY: wheels lifted, PWM limit in Arduino kept conservative, no reverse by default.\n")

    summary = {}

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        test = DiagnosticTest(mi, ser, writer)

        try:
            run_deadzone = args.full or args.deadzone or RUN_DEADZONE
            run_reverse  = args.full or args.reverse  or RUN_REVERSE
            run_reversal = args.full or args.reversal or RUN_REVERSAL

            if run_deadzone:
                summary['dead_zone'] = test.run_deadzone()

            if args.st and RUN_STEPS:
                summary['steps'] = test.run_steps()

            if args.rp and RUN_RAMP:
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
            ser.close()

    _print_summary(mi, summary)
    print(f"\n  Log saved → {csv_path}")
    print(f"  Run:  python3 plot_diag.py {csv_path}\n")


if __name__ == '__main__':
    main()
