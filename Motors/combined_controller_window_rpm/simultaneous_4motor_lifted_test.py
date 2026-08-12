#!/usr/bin/env python3
"""
simultaneous_4motor_lifted_test.py
==================================

Safe simultaneous lifted test for the 4 scooter motors using the Arduino
window-RPM controller.

This script commands all four motors at the same time, but only with conservative
setpoints. It logs all telemetry to CSV for later analysis.

Arduino command protocol:
  SP#+030#+030#+030#+030

Arduino telemetry expected:
  rpm0,rpm1,rpm2,rpm3,err0,err1,err2,err3,out0,out1,out2,out3

Default test sequence:
  1) all motors +30 rpm
  2) stop/wait
  3) all motors +50 rpm
  4) stop/wait
  5) all motors forward ramp: 0 -> 50 -> 0 rpm
  6) stop/wait
  7) all motors -30 rpm reverse
  8) final stop

Optional:
  --include-reversal adds a conservative simultaneous +30 -> -30 -> +30 reversal test.
  Do not use it until forward and reverse look good.

SAFETY:
  - Wheels must be lifted and free.
  - Keep one hand near the emergency power switch.
  - Press Ctrl+C to stop. The script sends repeated zero commands on exit.
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

MOTOR_NAMES = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]

SAMPLE_DT = 0.10

FORWARD_1_RPM = 30
FORWARD_2_RPM = 50
REVERSE_RPM = -30
REVERSAL_RPM = 30

HOLD_S = 6.0
STOP_WAIT_S = 4.0

RAMP_PEAK_RPM = 50
RAMP_UP_S = 8.0
RAMP_HOLD_S = 2.0
RAMP_DOWN_S = 6.0

REVERSAL_HOLD_S = 4.0


def fmt_rpm(v):
    sign = "+" if v >= 0 else "-"
    return f"{sign}{int(abs(round(v))):03d}"


def make_cmd(rpms):
    return (
        f"SP#{fmt_rpm(rpms[0])}#{fmt_rpm(rpms[1])}"
        f"#{fmt_rpm(rpms[2])}#{fmt_rpm(rpms[3])}\n"
    )


def send(ser, rpms):
    ser.write(make_cmd(rpms).encode("ascii"))


def safe_stop(ser, repeat=12, delay_s=0.05):
    for _ in range(repeat):
        send(ser, [0, 0, 0, 0])
        time.sleep(delay_s)


class Telemetry:
    def __init__(self):
        self.rpm = [0.0] * 4
        self.err = [0.0] * 4
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
            self.err = [float(parts[i + 4]) for i in range(4)]
            self.pwm = [float(parts[i + 8]) for i in range(4)]
            self.valid = True
            return True
        except ValueError:
            return False


class SimultaneousTester:
    def __init__(self, ser, csv_writer, raw_file):
        self.ser = ser
        self.writer = csv_writer
        self.raw_file = raw_file
        self.tel = Telemetry()
        self.t0 = time.monotonic()

    def elapsed(self):
        return time.monotonic() - self.t0

    def drain(self, phase):
        got = False
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            self.raw_file.write(f"{self.elapsed():.3f},{phase},{line}\n")
            if self.tel.update(line):
                got = True
        self.raw_file.flush()
        return got

    def log(self, phase, cmd):
        row = {
            "time_s": f"{self.elapsed():.3f}",
            "phase": phase,
        }

        for i in range(4):
            row[f"cmd_m{i}"] = f"{cmd[i]:.2f}"
        for i in range(4):
            row[f"rpm_m{i}"] = f"{self.tel.rpm[i]:.3f}"
        for i in range(4):
            row[f"err_m{i}"] = f"{self.tel.err[i]:.3f}"
        for i in range(4):
            row[f"pwm_m{i}"] = f"{self.tel.pwm[i]:.3f}"

        row["rpm_mean_abs"] = f"{sum(abs(x) for x in self.tel.rpm) / 4.0:.3f}"
        row["rpm_spread_abs"] = f"{max(abs(x) for x in self.tel.rpm) - min(abs(x) for x in self.tel.rpm):.3f}"
        row["pwm_mean_abs"] = f"{sum(abs(x) for x in self.tel.pwm) / 4.0:.3f}"
        row["pwm_spread_abs"] = f"{max(abs(x) for x in self.tel.pwm) - min(abs(x) for x in self.tel.pwm):.3f}"

        self.writer.writerow(row)

    def command_for(self, cmd, duration_s, phase):
        t_end = time.monotonic() + duration_s
        while time.monotonic() < t_end:
            send(self.ser, cmd)
            self.drain(phase)
            self.log(phase, cmd)
            time.sleep(SAMPLE_DT)

    def stop_wait(self, phase="stop_wait", duration_s=STOP_WAIT_S):
        self.command_for([0, 0, 0, 0], duration_s, phase)

    def all_same(self, rpm):
        return [rpm, rpm, rpm, rpm]

    def ramp_all(self, peak=RAMP_PEAK_RPM):
        print(f"  Ramp: all motors 0 -> {peak} -> 0 rpm")
        t_start = time.monotonic()
        total = RAMP_UP_S + RAMP_HOLD_S + RAMP_DOWN_S

        while True:
            t = time.monotonic() - t_start
            if t > total:
                break

            if t <= RAMP_UP_S:
                sp = peak * t / RAMP_UP_S
                phase = "ramp_up"
            elif t <= RAMP_UP_S + RAMP_HOLD_S:
                sp = peak
                phase = "ramp_hold"
            else:
                fall_t = t - RAMP_UP_S - RAMP_HOLD_S
                sp = peak * (1.0 - fall_t / RAMP_DOWN_S)
                sp = max(sp, 0.0)
                phase = "ramp_down"

            cmd = self.all_same(sp)
            send(self.ser, cmd)
            self.drain(phase)
            self.log(phase, cmd)
            time.sleep(SAMPLE_DT)

    def run(self, include_reverse=True, include_reversal=False):
        print("  Initial stop...")
        self.stop_wait("initial_stop", 2.0)

        print(f"  Step: all motors +{FORWARD_1_RPM} rpm")
        self.command_for(self.all_same(FORWARD_1_RPM), HOLD_S, "all_forward_30")
        self.stop_wait("stop_after_forward_30")

        print(f"  Step: all motors +{FORWARD_2_RPM} rpm")
        self.command_for(self.all_same(FORWARD_2_RPM), HOLD_S, "all_forward_50")
        self.stop_wait("stop_after_forward_50")

        self.ramp_all(RAMP_PEAK_RPM)
        self.stop_wait("stop_after_ramp")

        if include_reverse:
            print(f"  Reverse: all motors {REVERSE_RPM} rpm")
            print("  VISUAL CHECK: all wheels must rotate in the correct reverse direction.")
            self.command_for(self.all_same(REVERSE_RPM), HOLD_S, "all_reverse_30")
            self.stop_wait("stop_after_reverse_30")

        if include_reversal:
            print("  Direction reversal: +30 -> -30 -> +30")
            print("  WARNING: use only after forward and reverse are confirmed.")
            self.command_for(self.all_same(REVERSAL_RPM), REVERSAL_HOLD_S, "reversal_pos_1")
            self.command_for(self.all_same(-REVERSAL_RPM), REVERSAL_HOLD_S, "reversal_neg_1")
            self.command_for(self.all_same(REVERSAL_RPM), REVERSAL_HOLD_S, "reversal_pos_2")
            self.stop_wait("stop_after_reversal")


def parse_args():
    p = argparse.ArgumentParser(description="Safe simultaneous lifted test for all four motors.")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--out-dir", default=LOG_DIR)
    p.add_argument("--no-reverse", action="store_true",
                   help="Skip simultaneous reverse test.")
    p.add_argument("--include-reversal", action="store_true",
                   help="Add simultaneous direction reversal test. Use only after forward/reverse are safe.")
    p.add_argument("--yes", action="store_true",
                   help="Skip typing LIFTED confirmation.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  SIMULTANEOUS 4-MOTOR LIFTED TEST")
    print("=" * 72)
    print(f"  Port: {args.port} @ {BAUD}")
    print("  Sequence: +30, stop, +50, stop, ramp +0->50->0, stop, -30, stop")
    if args.include_reversal:
        print("  Extra: simultaneous +30 -> -30 -> +30 reversal")
    print("\n  SAFETY:")
    print("   - All wheels must be lifted.")
    print("   - Keep one hand near power cutoff.")
    print("   - Watch for direction mismatch, heating, harsh clicking, or one wheel lagging.")
    print("   - Press Ctrl+C for safe stop.\n")

    if not args.yes:
        ans = input("Type LIFTED to confirm all wheels are lifted and free: ").strip()
        if ans != "LIFTED":
            print("Aborted.")
            return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.out_dir) / f"simultaneous_4motor_lifted_{ts}.csv"
    raw_path = Path(args.out_dir) / f"simultaneous_4motor_lifted_{ts}_raw.txt"

    fieldnames = ["time_s", "phase"]
    fieldnames += [f"cmd_m{i}" for i in range(4)]
    fieldnames += [f"rpm_m{i}" for i in range(4)]
    fieldnames += [f"err_m{i}" for i in range(4)]
    fieldnames += [f"pwm_m{i}" for i in range(4)]
    fieldnames += ["rpm_mean_abs", "rpm_spread_abs", "pwm_mean_abs", "pwm_spread_abs"]

    try:
        ser = serial.Serial(args.port, BAUD, timeout=0.10)
    except serial.SerialException as exc:
        print(f"ERROR: could not open serial port: {exc}")
        return 1

    print("Waiting for Arduino reset/boot...")
    time.sleep(2.0)
    safe_stop(ser, repeat=10)

    try:
        with open(csv_path, "w", newline="") as f_csv, open(raw_path, "w") as f_raw:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            f_raw.write("time_s,phase,raw_arduino_line\n")

            tester = SimultaneousTester(ser, writer, f_raw)
            tester.run(
                include_reverse=not args.no_reverse,
                include_reversal=args.include_reversal,
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Sending safe stop...")
    finally:
        safe_stop(ser, repeat=15)
        ser.close()

    print("\nDONE")
    print(f"CSV log: {csv_path}")
    print(f"Raw log: {raw_path}")
    print("Plot with:")
    print(f"  python3 plot_simultaneous_4motor_lifted_test.py {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
