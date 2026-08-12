#!/usr/bin/env python3
"""
Sequential full lifted diagnostic for the 4 scooter motors using the window-RPM Arduino controller.

Tests one motor at a time. It never commands all motors together.
Tests per motor: forward steps, forward ramp, reverse step, and direction reversal.

Arduino telemetry expected:
  rpm0,rpm1,rpm2,rpm3,err0,err1,err2,err3,out0,out1,out2,out3
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Install with: pip install pyserial")
    sys.exit(1)

BAUD = 115200
LOG_DIR = "logs"

MOTOR_NAMES = {
    0: "Front Left",
    1: "Front Right",
    2: "Back Left",
    3: "Back Right",
}

STEP_LEVELS = [30, 50, 70]
STEP_HOLD_S = 6.0
RAMP_PEAK_RPM = 70
RAMP_RISE_S = 10.0
RAMP_HOLD_S = 3.0
RAMP_FALL_S = 8.0
REV_RPM = 40
REV_HOLD_S = 5.0
REVERSAL_RPM = 35
REVERSAL_HOLD_S = 4.0
REVERSAL_CYCLES = 2
SAMPLE_DT = 0.10
STOP_RPM_THRESH = 5.0
STOP_PWM_THRESH = 4.0
STOP_WAIT_TIMEOUT_S = 6.0


def fmt_rpm(v):
    sign = "+" if v >= 0 else "-"
    return f"{sign}{int(abs(round(v))):03d}"


def make_cmd(rpms):
    return f"SP#{fmt_rpm(rpms[0])}#{fmt_rpm(rpms[1])}#{fmt_rpm(rpms[2])}#{fmt_rpm(rpms[3])}\n"


def send(ser, rpms):
    ser.write(make_cmd(rpms).encode("ascii"))


def safe_stop(ser, repeat=8, delay_s=0.05):
    for _ in range(repeat):
        send(ser, [0, 0, 0, 0])
        time.sleep(delay_s)


class Telemetry:
    def __init__(self):
        self.rpm = [0.0] * 4
        self.error = [0.0] * 4
        self.pwm = [0.0] * 4
        self.valid = False
        self.last_line = ""

    def update(self, line):
        line = line.strip()
        self.last_line = line
        if not line or line.startswith("S:") or line.startswith("ERR"):
            return False
        parts = line.split(",")
        if len(parts) != 12:
            return False
        try:
            self.rpm = [float(parts[i]) for i in range(4)]
            self.error = [float(parts[i + 4]) for i in range(4)]
            self.pwm = [float(parts[i + 8]) for i in range(4)]
            self.valid = True
            return True
        except ValueError:
            return False


class FullMotorTester:
    def __init__(self, ser, writer, raw_file, motor_index, t0):
        self.ser = ser
        self.writer = writer
        self.raw_file = raw_file
        self.mi = motor_index
        self.t0 = t0
        self.telem = Telemetry()

    def elapsed(self):
        return time.monotonic() - self.t0

    def drain(self, test_name="drain", phase="drain"):
        got_valid = False
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                self.raw_file.write(f"{self.elapsed():.3f},{self.mi},{test_name},{phase},{line}\n")
                if self.telem.update(line):
                    got_valid = True
            except Exception as exc:
                self.raw_file.write(f"{self.elapsed():.3f},{self.mi},{test_name},{phase},PYTHON_READ_ERROR:{exc}\n")
        self.raw_file.flush()
        return got_valid

    def set_motor(self, sp):
        rpms = [0, 0, 0, 0]
        rpms[self.mi] = sp
        send(self.ser, rpms)

    def log(self, test_name, phase, commanded_sp):
        rpm = self.telem.rpm[self.mi]
        err = self.telem.error[self.mi]
        pwm = self.telem.pwm[self.mi]
        self.writer.writerow({
            "pc_time_s": f"{self.elapsed():.3f}",
            "motor": self.mi,
            "motor_name": MOTOR_NAMES[self.mi],
            "test": test_name,
            "phase": phase,
            "commanded_setpoint_rpm": f"{commanded_sp:.2f}",
            "rpm": f"{rpm:.3f}",
            "error_rpm": f"{err:.3f}",
            "pwm_out": f"{pwm:.3f}",
            "rpm_abs": f"{abs(rpm):.3f}",
            "pwm_abs": f"{abs(pwm):.3f}",
        })

    def hold(self, sp, duration_s, test_name, phase):
        t_end = time.monotonic() + duration_s
        while time.monotonic() < t_end:
            self.set_motor(sp)
            self.drain(test_name, phase)
            self.log(test_name, phase, sp)
            time.sleep(SAMPLE_DT)

    def wait_until_stopped(self, test_name, phase, timeout_s=STOP_WAIT_TIMEOUT_S):
        t_start = time.monotonic()
        while True:
            self.set_motor(0)
            self.drain(test_name, phase)
            self.log(test_name, phase, 0)
            rpm_ok = abs(self.telem.rpm[self.mi]) <= STOP_RPM_THRESH
            pwm_ok = abs(self.telem.pwm[self.mi]) <= STOP_PWM_THRESH
            timed_out = (time.monotonic() - t_start) >= timeout_s
            if (rpm_ok and pwm_ok) or timed_out:
                break
            time.sleep(SAMPLE_DT)
        self.hold(0, 0.5, test_name, phase + "_dwell")

    def run_steps(self):
        print(f"    Test 1/4: forward step responses {STEP_LEVELS} rpm", flush=True)
        self.wait_until_stopped("step", "pre_step_stop")
        for sp in STEP_LEVELS:
            self.wait_until_stopped("step", "step_pause")
            self.hold(sp, STEP_HOLD_S, "step", f"step_{sp}")
        self.wait_until_stopped("step", "post_step_stop")

    def run_ramp(self):
        print("    Test 2/4: forward ramp 0 -> 70 -> 0 rpm", flush=True)
        self.wait_until_stopped("ramp", "pre_ramp_stop")
        t_start = time.monotonic()
        total = RAMP_RISE_S + RAMP_HOLD_S + RAMP_FALL_S
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed > total:
                break
            if elapsed <= RAMP_RISE_S:
                sp = RAMP_PEAK_RPM * elapsed / RAMP_RISE_S
                phase = "ramp_up"
            elif elapsed <= RAMP_RISE_S + RAMP_HOLD_S:
                sp = RAMP_PEAK_RPM
                phase = "ramp_hold"
            else:
                fall_t = elapsed - RAMP_RISE_S - RAMP_HOLD_S
                sp = max(RAMP_PEAK_RPM * (1.0 - fall_t / RAMP_FALL_S), 0.0)
                phase = "ramp_down"
            self.set_motor(sp)
            self.drain("ramp", phase)
            self.log("ramp", phase, sp)
            time.sleep(SAMPLE_DT)
        self.wait_until_stopped("ramp", "post_ramp_stop")

    def run_reverse(self):
        print("    Test 3/4: reverse step -40 rpm", flush=True)
        self.wait_until_stopped("reverse", "pre_reverse_stop")
        self.hold(-REV_RPM, REV_HOLD_S, "reverse", "reverse_step")
        self.wait_until_stopped("reverse", "post_reverse_stop")

    def run_reversal(self):
        print("    Test 4/4: direction reversal +35 <-> -35 rpm", flush=True)
        self.wait_until_stopped("reversal", "pre_reversal_stop")
        sign = 1
        for cycle in range(REVERSAL_CYCLES):
            self.hold(sign * REVERSAL_RPM, REVERSAL_HOLD_S, "reversal", f"cycle_{cycle+1}_pos")
            sign *= -1
            self.hold(sign * REVERSAL_RPM, REVERSAL_HOLD_S, "reversal", f"cycle_{cycle+1}_neg")
            sign *= -1
        self.wait_until_stopped("reversal", "post_reversal_stop")


def parse_motors(s):
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        m = int(item)
        if m not in MOTOR_NAMES:
            raise argparse.ArgumentTypeError("motors must be comma-separated values from 0,1,2,3")
        out.append(m)
    if not out:
        raise argparse.ArgumentTypeError("empty motor list")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run full lifted diagnostics for all scooter motors.")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--motors", type=parse_motors, default=[0, 1, 2, 3], help="Comma-separated motor indices. Default: 0,1,2,3")
    p.add_argument("--out-dir", default=LOG_DIR)
    p.add_argument("--no-steps", action="store_true")
    p.add_argument("--no-ramp", action="store_true")
    p.add_argument("--no-reverse", action="store_true")
    p.add_argument("--no-reversal", action="store_true")
    p.add_argument("--yes", action="store_true", help="Skip interactive safety confirmation.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  FULL 4-MOTOR WINDOW-RPM DIAGNOSTIC")
    print("=" * 72)
    print(f"  Port   : {args.port} @ {BAUD}")
    print(f"  Motors : {args.motors}")
    print("  Tests  :" f" steps={not args.no_steps}," f" ramp={not args.no_ramp}," f" reverse={not args.no_reverse}," f" reversal={not args.no_reversal}")
    print("\n  SAFETY:")
    print("   - Lift all wheels.")
    print("   - The script tests one motor at a time.")
    print("   - Visually confirm reverse direction for each motor.")
    print("   - Press Ctrl+C immediately for hard stop if anything looks wrong.\n")

    if not args.yes:
        ans = input("Type LIFTED to confirm wheels are lifted and free: ").strip()
        if ans != "LIFTED":
            print("Aborted.")
            return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.out_dir) / f"all_motors_full_window_diag_{ts}.csv"
    raw_path = Path(args.out_dir) / f"all_motors_full_window_diag_{ts}_raw.txt"

    fieldnames = ["pc_time_s", "motor", "motor_name", "test", "phase", "commanded_setpoint_rpm", "rpm", "error_rpm", "pwm_out", "rpm_abs", "pwm_abs"]

    try:
        ser = serial.Serial(args.port, BAUD, timeout=0.10)
    except serial.SerialException as exc:
        print(f"ERROR: could not open serial port: {exc}")
        return 1

    print("Waiting for Arduino reset/boot...")
    time.sleep(2.0)
    safe_stop(ser, repeat=10)
    t0 = time.monotonic()

    try:
        with open(csv_path, "w", newline="") as f_csv, open(raw_path, "w") as f_raw:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            f_raw.write("pc_time_s,motor,test,phase,raw_arduino_line\n")
            for mi in args.motors:
                print("\n" + "-" * 72)
                print(f"  Motor M{mi}: {MOTOR_NAMES[mi]}")
                print("-" * 72)
                tester = FullMotorTester(ser, writer, f_raw, mi, t0)
                tester.drain("motor_start", "drain")
                tester.wait_until_stopped("motor_start", "zero_before_motor")
                if not args.no_steps:
                    tester.run_steps()
                if not args.no_ramp:
                    tester.run_ramp()
                if not args.no_reverse:
                    tester.run_reverse()
                    print("    Visual check: did this motor physically rotate backward?")
                if not args.no_reversal:
                    tester.run_reversal()
                    print("    Visual check: did this motor physically switch directions cleanly?")
                tester.wait_until_stopped("motor_end", "zero_after_motor")
                f_csv.flush()
                f_raw.flush()
                print(f"  Finished M{mi}. Log is being saved continuously.")
    except KeyboardInterrupt:
        print("\nInterrupted. Sending safe stop...")
    finally:
        safe_stop(ser, repeat=15)
        ser.close()

    print("\n" + "=" * 72)
    print("DONE")
    print(f"CSV log : {csv_path}")
    print(f"Raw log : {raw_path}")
    print("Plot with:")
    print(f"  python3 plot_all_motors_full_window_diag.py {csv_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
