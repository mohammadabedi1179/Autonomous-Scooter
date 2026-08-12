#!/usr/bin/env python3
"""
M1 open-loop PWM + windowed Hall RPM CSV logger.

Use with Arduino sketch:
  m1_openloop_window_rpm_logger.ino

Purpose:
  - Keep M1 open-loop: no PID.
  - Send safe fixed PWM or sweep commands.
  - Record both RPM estimators:
      rpm_period_valid  : old single-pulse-period RPM
      rpm_window        : new pulse-count-window RPM
      rpm_window_filt   : EMA-filtered window RPM, intended for later PID feedback
  - Save both clean CSV and raw serial text.

Examples:
  python3 m1_window_rpm_csv_logger.py --port /dev/ttyACM0 --pwm 40 --duration 20
  python3 m1_window_rpm_csv_logger.py --port /dev/ttyACM0 --sweep 20,30,40,50,60 --duration 10

Outputs:
  logs/m1_window_rpm_<timestamp>.csv
  logs/m1_window_rpm_<timestamp>_raw.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Install with: pip install pyserial")
    raise

DEFAULT_BAUD = 115200
DEFAULT_PWM_LIMIT = 60
ABSOLUTE_PWM_LIMIT = 80
SEND_PERIOD_S = 0.10
STOP_HOLD_S = 1.2
BETWEEN_LEVELS_S = 0.60

READY_TOKEN = "M1_WINDOW_RPM_READY"

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

saved_rows = 0


def write_line(ser: serial.Serial, line: str) -> None:
    ser.write((line.strip() + "\n").encode("ascii"))
    ser.flush()


def clamp_pwm(pwm: int, limit: int) -> int:
    limit = max(0, min(int(limit), ABSOLUTE_PWM_LIMIT))
    if pwm > limit:
        print(f"PWM {pwm} > limit {limit}; clamped.")
        return limit
    if pwm < -limit:
        print(f"PWM {pwm} < limit {-limit}; clamped.")
        return -limit
    return pwm


def parse_data2_line(line: str, pc_time_s: float, phase: str) -> Optional[Dict[str, object]]:
    """Parse Arduino DATA2 line into a CSV row.

    Arduino prints 22 comma-separated tokens total:
      DATA2 + 21 telemetry fields.
    """
    if not line.startswith("DATA2,"):
        return None

    parts = line.split(",")
    if len(parts) != 22:
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
            "window_raw_delta": int(parts[11]),
            "window_valid_delta": int(parts[12]),
            "window_dt_ms": int(parts[13]),
            "last_raw_period_us": int(parts[14]),
            "last_valid_period_us": int(parts[15]),
            "pulse_age_ms": float(parts[16]),
            "rpm_period_raw": float(parts[17]),
            "rpm_period_valid": float(parts[18]),
            "rpm_window": float(parts[19]),
            "rpm_window_filt": float(parts[20]),
            "hall_level": int(parts[21]),
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
    """Read serial for duration_s; save DATA2 rows to CSV and all raw lines to txt."""
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

        row = parse_data2_line(line, pc_time_s, phase)
        if row is not None:
            csv_writer.writerow(row)
            saved_rows += 1

        if echo and (not line.startswith("DATA2,")):
            print("ARD:", line)

    return lines


def command_and_listen(
    ser: serial.Serial,
    cmd: str,
    csv_writer: csv.DictWriter,
    raw_file,
    phase: str,
    t0: float,
    listen_s: float = 0.25,
    echo: bool = True,
) -> List[str]:
    write_line(ser, cmd)
    return drain_serial(ser, csv_writer, raw_file, phase, t0, listen_s, echo=echo)


def wait_for_ready(ser: serial.Serial, csv_writer: csv.DictWriter, raw_file, t0: float, timeout_s: float = 10.0) -> bool:
    print(f"Waiting for {READY_TOKEN}...")
    end = time.time() + timeout_s
    found = False

    while time.time() < end:
        lines = drain_serial(ser, csv_writer, raw_file, "wait_ready", t0, 0.25, echo=True)
        for line in lines:
            if READY_TOKEN in line:
                found = True
        if found:
            return True

    return False


def send_repeated_pwm(
    ser: serial.Serial,
    pwm: int,
    duration_s: float,
    csv_writer: csv.DictWriter,
    raw_file,
    phase: str,
    t0: float,
    echo: bool,
) -> None:
    print(f"\nRUN: PWM={pwm} for {duration_s:.1f}s")
    print("Keep wheel lifted. Stop if hum/heat/violent ticking occurs.")

    end = time.time() + duration_s
    next_send = 0.0

    while time.time() < end:
        now = time.time()
        if now >= next_send:
            write_line(ser, f"PWM,{pwm}")
            next_send = now + SEND_PERIOD_S
        drain_serial(ser, csv_writer, raw_file, phase, t0, 0.02, echo=echo)


def stop_and_hold(
    ser: serial.Serial,
    csv_writer: csv.DictWriter,
    raw_file,
    phase: str,
    t0: float,
    hold_s: float = STOP_HOLD_S,
    echo: bool = False,
) -> None:
    end = time.time() + hold_s
    next_send = 0.0
    while time.time() < end:
        now = time.time()
        if now >= next_send:
            write_line(ser, "STOP")
            next_send = now + SEND_PERIOD_S
        drain_serial(ser, csv_writer, raw_file, phase, t0, 0.02, echo=echo)


def run_pwm_level(
    ser: serial.Serial,
    pwm: int,
    duration_s: float,
    csv_writer: csv.DictWriter,
    raw_file,
    t0: float,
    echo: bool,
    reset_before_level: bool = True,
) -> None:
    phase = f"pwm_{pwm:+d}"

    if reset_before_level:
        # Reset before each level so period-based RPM does not use a long interval across a stopped gap.
        command_and_listen(ser, "RESET_HALL", csv_writer, raw_file, f"reset_before_{pwm:+d}", t0, 0.20, echo=echo)

    send_repeated_pwm(ser, pwm, duration_s, csv_writer, raw_file, phase, t0, echo=echo)
    print("Ramping down...")
    stop_and_hold(ser, csv_writer, raw_file, f"stop_after_{pwm:+d}", t0, hold_s=STOP_HOLD_S, echo=echo)
    drain_serial(ser, csv_writer, raw_file, f"gap_after_{pwm:+d}", t0, BETWEEN_LEVELS_S, echo=echo)


def parse_sweep(s: str) -> List[int]:
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise argparse.ArgumentTypeError("Sweep must contain at least one PWM value.")
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M1 open-loop PWM + window RPM CSV logger")
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 or /dev/serial/by-id/...")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--pwm", type=int, default=None, help="Single PWM command, e.g. 40")
    p.add_argument("--sweep", type=parse_sweep, default=None, help="Comma-separated PWM values, e.g. 20,30,40,50,60")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds per PWM level")
    p.add_argument("--limit", type=int, default=DEFAULT_PWM_LIMIT, help="Soft PWM limit sent to Arduino and used by Python")
    p.add_argument("--outdir", default="logs", help="Output directory")
    p.add_argument("--echo-data", action="store_true", help="Echo DATA2 lines too; very verbose")
    p.add_argument("--no-reset-per-level", action="store_true", help="Do not RESET_HALL before each PWM level")
    return p


def main() -> int:
    global saved_rows
    args = build_arg_parser().parse_args()

    if args.pwm is None and args.sweep is None:
        print("ERROR: choose --pwm or --sweep")
        return 2
    if args.pwm is not None and args.sweep is not None:
        print("ERROR: choose only one of --pwm or --sweep")
        return 2

    pwm_limit = max(0, min(int(args.limit), ABSOLUTE_PWM_LIMIT))

    if args.pwm is not None:
        levels = [clamp_pwm(args.pwm, pwm_limit)]
    else:
        levels = [clamp_pwm(v, pwm_limit) for v in args.sweep]

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"m1_window_rpm_{ts}.csv")
    raw_path = os.path.join(args.outdir, f"m1_window_rpm_{ts}_raw.txt")

    print(f"Opening {args.port} at {args.baud}...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
    except Exception as exc:
        print(f"ERROR opening serial port: {exc}")
        return 1

    t0 = time.time()
    saved_rows = 0

    try:
        with open(csv_path, "w", newline="") as csv_f, open(raw_path, "w", newline="") as raw_f:
            writer = csv.DictWriter(csv_f, fieldnames=DATA_FIELDS)
            writer.writeheader()
            raw_f.write("pc_time_s,phase,raw_arduino_line\n")

            # Opening Arduino USB serial resets the board.
            time.sleep(2.0)
            ser.reset_input_buffer()

            if not wait_for_ready(ser, writer, raw_f, t0, timeout_s=10.0):
                print(f"ERROR: did not receive {READY_TOKEN}.")
                print("Check that m1_openloop_window_rpm_logger.ino is uploaded and baud=115200.")
                return 1

            command_and_listen(ser, "HELLO", writer, raw_f, "cmd_HELLO", t0, 0.25, echo=True)
            command_and_listen(ser, f"LIMIT,{pwm_limit}", writer, raw_f, "cmd_LIMIT", t0, 0.25, echo=True)
            command_and_listen(ser, "ARM", writer, raw_f, "cmd_ARM", t0, 0.25, echo=True)
            command_and_listen(ser, "RESET_HALL", writer, raw_f, "cmd_RESET_HALL", t0, 0.25, echo=True)

            for pwm in levels:
                run_pwm_level(
                    ser,
                    pwm,
                    args.duration,
                    writer,
                    raw_f,
                    t0,
                    echo=args.echo_data,
                    reset_before_level=(not args.no_reset_per_level),
                )

            print("\nSafe stop...")
            stop_and_hold(ser, writer, raw_f, "final_stop", t0, hold_s=1.0, echo=True)
            command_and_listen(ser, "DISARM", writer, raw_f, "cmd_DISARM", t0, 0.50, echo=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Sending STOP/DISARM...")
        try:
            write_line(ser, "STOP")
            time.sleep(0.2)
            write_line(ser, "DISARM")
        except Exception:
            pass
    finally:
        try:
            ser.close()
        except Exception:
            pass

    print("Done.")
    print(f"Saved DATA2 rows: {saved_rows}")
    print(f"CSV: {csv_path}")
    print(f"RAW: {raw_path}")
    if saved_rows == 0:
        print("WARNING: CSV has no DATA2 rows. Check Arduino sketch version and serial output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
