#!/usr/bin/env python3
"""
Fixed M1 open-loop PWM + Hall CSV logger.

Use with Arduino sketch: m1_openloop_hall_logger.ino

Why this fixed version exists:
  The Arduino DATA line has 18 comma-separated tokens total:
    DATA + 17 telemetry fields
  The earlier Python parser expected 19 tokens, so it could silently save an
  empty/header-only CSV. This version accepts the correct length and prints the
  number of saved DATA rows.

Examples:
  python3 m1_openloop_hall_csv_logger_fixed.py --port /dev/ttyACM0 --pwm 40 --duration 15
  python3 m1_openloop_hall_csv_logger_fixed.py --port /dev/ttyACM0 --sweep 20,30,40,50,60 --duration 10

Outputs:
  logs/m1_hall_openloop_<timestamp>.csv
  logs/m1_hall_openloop_<timestamp>_raw.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import serial

DEFAULT_BAUD = 115200
PYTHON_PWM_LIMIT = 60
SEND_PERIOD_S = 0.10
STOP_HOLD_S = 1.2

DATA_FIELDS = [
    "pc_time_s",
    "phase",
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
    "last_raw_period_us",
    "last_valid_period_us",
    "rpm_raw",
    "rpm_valid",
    "rpm_filt",
    "pulse_age_ms",
    "hall_level",
]

saved_rows = 0


def write_line(ser: serial.Serial, line: str) -> None:
    ser.write((line.strip() + "\n").encode("ascii"))
    ser.flush()


def clamp_pwm(pwm: int) -> int:
    if pwm > PYTHON_PWM_LIMIT:
        print(f"PWM {pwm} > limit {PYTHON_PWM_LIMIT}; clamped.")
        return PYTHON_PWM_LIMIT
    if pwm < -PYTHON_PWM_LIMIT:
        print(f"PWM {pwm} < limit {-PYTHON_PWM_LIMIT}; clamped.")
        return -PYTHON_PWM_LIMIT
    return pwm


def parse_data_line(line: str, pc_time_s: float, phase: str) -> Optional[Dict[str, object]]:
    """Parse Arduino DATA line into a CSV row.

    Arduino prints:
      DATA,ms,armed,target,current,sign,limit,wd,raw_cnt,valid_cnt,reject_cnt,
           last_raw_us,last_valid_us,rpm_raw,rpm_valid,rpm_filt,pulse_age_ms,hall

    That is 18 tokens total: index 0 is DATA, indices 1..17 are fields.
    """
    if not line.startswith("DATA,"):
        return None

    parts = line.split(",")
    if len(parts) != 18:
        return None

    try:
        return {
            "pc_time_s": f"{pc_time_s:.6f}",
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


def drain_serial(
    ser: serial.Serial,
    csv_writer: csv.DictWriter,
    raw_file,
    phase: str,
    t0: float,
    duration_s: float,
    echo: bool = False,
) -> List[str]:
    """Read serial for duration_s; save DATA rows to CSV and all raw lines to txt."""
    global saved_rows

    end = time.time() + duration_s
    lines: List[str] = []

    while time.time() < end:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"SERIAL_READ_ERROR: {exc}")
            break

        if not line:
            continue

        pc_time_s = time.time() - t0
        lines.append(line)
        raw_file.write(f"{pc_time_s:.6f},{phase},{line}\n")
        raw_file.flush()

        row = parse_data_line(line, pc_time_s, phase)
        if row is not None:
            csv_writer.writerow(row)
            saved_rows += 1

        if echo and (not line.startswith("DATA,")):
            print("ARD:", line)

    return lines


def wait_for_ready(ser, csv_writer, raw_file, t0, timeout_s=8.0) -> bool:
    print("Waiting for M1_HALL_OPEN_LOOP_READY...")
    end = time.time() + timeout_s
    while time.time() < end:
        lines = drain_serial(ser, csv_writer, raw_file, "wait_ready", t0, 0.25, echo=True)
        if any("M1_HALL_OPEN_LOOP_READY" in ln for ln in lines):
            return True
        write_line(ser, "HELLO")
    return False


def send_and_expect(ser, csv_writer, raw_file, t0, cmd: str, expected: str, timeout_s=1.0) -> bool:
    write_line(ser, cmd)
    end = time.time() + timeout_s
    seen: List[str] = []
    while time.time() < end:
        lines = drain_serial(ser, csv_writer, raw_file, f"cmd_{cmd.split(',')[0]}", t0, 0.1, echo=True)
        seen.extend(lines)
        if any(expected in ln for ln in lines):
            return True
    print(f"ERROR: command {cmd!r} did not receive expected response {expected!r}")
    print("Last Arduino lines:")
    for ln in seen[-10:]:
        print("  ", ln)
    return False


def safe_stop(ser, csv_writer, raw_file, t0):
    print("\nSafe stop...")
    for _ in range(12):
        write_line(ser, "STOP")
        drain_serial(ser, csv_writer, raw_file, "safe_stop", t0, 0.04, echo=False)
        time.sleep(SEND_PERIOD_S)
    write_line(ser, "DISARM")
    drain_serial(ser, csv_writer, raw_file, "disarm", t0, 0.5, echo=True)


def run_pwm(ser, csv_writer, raw_file, t0, pwm: int, duration_s: float, echo_data: bool = False):
    pwm = clamp_pwm(int(pwm))
    phase = f"pwm_{pwm:+d}"

    print(f"\nRUN: PWM={pwm} for {duration_s:.1f}s")
    print("Keep wheel lifted. Stop if hum/heat/violent ticking occurs.")

    end = time.time() + duration_s
    next_send = 0.0

    while time.time() < end:
        now = time.time()
        if now >= next_send:
            write_line(ser, f"PWM,{pwm:+d}")
            next_send = now + SEND_PERIOD_S

        drain_serial(ser, csv_writer, raw_file, phase, t0, 0.02, echo=echo_data)

    print("Ramping down...")
    stop_end = time.time() + STOP_HOLD_S
    while time.time() < stop_end:
        write_line(ser, "STOP")
        drain_serial(ser, csv_writer, raw_file, f"stop_after_{pwm:+d}", t0, 0.04, echo=False)
        time.sleep(SEND_PERIOD_S)


def parse_sweep(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def summarize_csv(csv_path: str) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("\nWARNING: No DATA rows were saved. Do not plot this CSV.")
        print("Check that the Arduino sketch is m1_openloop_hall_logger.ino and that it prints DATA lines.")
        return

    print("\nQuick summary from CSV:")
    phases = sorted(set(r["phase"] for r in rows if r["phase"].startswith("pwm_")))
    for ph in phases:
        pr = [r for r in rows if r["phase"] == ph]
        if len(pr) > 5:
            t_start = float(pr[0]["pc_time_s"])
            ss = [r for r in pr if float(r["pc_time_s"]) >= t_start + 1.0]
        else:
            ss = pr
        if not ss:
            ss = pr
        rpms = [float(r["rpm_filt"]) for r in ss]
        raws = [float(r["rpm_raw"]) for r in ss]
        rejects0 = int(pr[0]["reject_count"])
        rejects1 = int(pr[-1]["reject_count"])
        val0 = int(pr[0]["valid_count"])
        val1 = int(pr[-1]["valid_count"])
        print(
            f"  {ph:>8}: rpm_filt mean={sum(rpms)/len(rpms):7.2f}, "
            f"min={min(rpms):7.2f}, max={max(rpms):7.2f}, "
            f"raw_max={max(raws):7.2f}, valid_pulses={val1-val0}, rejected={rejects1-rejects0}"
        )


def main() -> int:
    global saved_rows

    ap = argparse.ArgumentParser(description="Fixed M1 open-loop PWM + Hall CSV logger")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--pwm", type=int, default=None)
    ap.add_argument("--sweep", default=None, help="Example: 20,30,40,50,60")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds per PWM level")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--echo-data", action="store_true", help="print DATA lines too; normally too noisy")
    args = ap.parse_args()

    if args.pwm is None and args.sweep is None:
        ap.error("Use --pwm N or --sweep 20,30,40")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"m1_hall_openloop_{ts}.csv")
    raw_path = os.path.join(args.out_dir, f"m1_hall_openloop_{ts}_raw.txt")

    print(f"Opening {args.port} at {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    t0 = time.time()

    with open(csv_path, "w", newline="") as csv_f, open(raw_path, "w", encoding="utf-8") as raw_f:
        writer = csv.DictWriter(csv_f, fieldnames=DATA_FIELDS)
        writer.writeheader()
        raw_f.write("pc_time_s,phase,raw_arduino_line\n")

        try:
            time.sleep(2.0)

            if not wait_for_ready(ser, writer, raw_f, t0):
                print("\nFATAL: I did not see M1_HALL_OPEN_LOOP_READY.")
                print("Upload m1_openloop_hall_logger.ino first.")
                return 2

            if not send_and_expect(ser, writer, raw_f, t0, "HELLO", "OK,HELLO"):
                return 2
            if not send_and_expect(ser, writer, raw_f, t0, f"LIMIT,{PYTHON_PWM_LIMIT}", "OK,LIMIT"):
                return 2
            if not send_and_expect(ser, writer, raw_f, t0, "ARM", "OK,ARMED"):
                return 2
            send_and_expect(ser, writer, raw_f, t0, "RESET_HALL", "OK,RESET_HALL", timeout_s=0.5)

            if args.pwm is not None:
                run_pwm(ser, writer, raw_f, t0, args.pwm, args.duration, echo_data=args.echo_data)

            if args.sweep is not None:
                for pwm in parse_sweep(args.sweep):
                    run_pwm(ser, writer, raw_f, t0, pwm, args.duration, echo_data=args.echo_data)
                    time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            safe_stop(ser, writer, raw_f, t0)
            ser.close()
            print("Done.")

    print(f"\nCSV saved: {csv_path}")
    print(f"Raw serial log saved: {raw_path}")
    print(f"DATA rows saved: {saved_rows}")
    summarize_csv(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
