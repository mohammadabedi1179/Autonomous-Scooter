#!/usr/bin/env python3
"""
Strict safe Python sender for M1 open-loop PWM test.
It refuses to arm unless it sees M1_OPEN_LOOP_READY from the Arduino.
"""

import argparse
import sys
import time
import serial

DEFAULT_BAUD = 115200
PYTHON_PWM_LIMIT = 60
SEND_PERIOD_S = 0.10


def write_line(ser, line):
    ser.write((line.strip() + "\n").encode("ascii"))
    ser.flush()


def read_lines_for(ser, seconds, echo=True):
    lines = []
    end = time.time() + seconds
    while time.time() < end:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"SERIAL_READ_ERROR: {exc}")
            break
        if line:
            lines.append(line)
            if echo:
                print("ARD:", line)
    return lines


def wait_for_ready(ser, timeout_s=6.0):
    print("Waiting for M1_OPEN_LOOP_READY...")
    end = time.time() + timeout_s
    while time.time() < end:
        lines = read_lines_for(ser, 0.25, echo=True)
        if any("M1_OPEN_LOOP_READY" in ln for ln in lines):
            return True
        write_line(ser, "HELLO")
    return False


def send_and_expect(ser, cmd, expected, timeout_s=1.0):
    write_line(ser, cmd)
    end = time.time() + timeout_s
    seen = []
    while time.time() < end:
        lines = read_lines_for(ser, 0.1, echo=True)
        seen.extend(lines)
        if any(expected in ln for ln in lines):
            return True
    print(f"ERROR: command {cmd!r} did not receive expected response {expected!r}")
    return False


def safe_stop(ser):
    print("\nSafe stop...")
    for _ in range(10):
        write_line(ser, "STOP")
        read_lines_for(ser, 0.05, echo=False)
        time.sleep(SEND_PERIOD_S)
    write_line(ser, "DISARM")
    read_lines_for(ser, 0.5, echo=True)


def clamp_pwm(pwm):
    if pwm > PYTHON_PWM_LIMIT:
        print(f"PWM {pwm} > limit {PYTHON_PWM_LIMIT}; clamped.")
        return PYTHON_PWM_LIMIT
    if pwm < -PYTHON_PWM_LIMIT:
        print(f"PWM {pwm} < limit {-PYTHON_PWM_LIMIT}; clamped.")
        return -PYTHON_PWM_LIMIT
    return pwm


def run_pwm(ser, pwm, duration_s):
    pwm = clamp_pwm(int(pwm))
    print(f"\nRUN: PWM={pwm} for {duration_s:.1f}s")
    print("Keep wheel lifted. Stop if hum/heat/violent ticking occurs.")
    end = time.time() + duration_s
    while time.time() < end:
        write_line(ser, f"PWM,{pwm:+d}")
        read_lines_for(ser, 0.02, echo=True)
        time.sleep(SEND_PERIOD_S)
    print("Ramping down...")
    for _ in range(12):
        write_line(ser, "STOP")
        read_lines_for(ser, 0.02, echo=True)
        time.sleep(SEND_PERIOD_S)


def parse_sweep(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--pwm", type=int, default=None)
    ap.add_argument("--sweep", default=None, help="Example: 20,30,40")
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()

    if args.pwm is None and args.sweep is None:
        ap.error("Use --pwm N or --sweep 20,30,40")

    print(f"Opening {args.port} at {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.05)

    try:
        # Opening serial resets Arduino Mega. Give it time.
        time.sleep(2.0)

        if not wait_for_ready(ser):
            print("\nFATAL: I did not see M1_OPEN_LOOP_READY.")
            print("Do NOT continue motor testing.")
            print("Likely causes: wrong port, wrong baud, sketch not uploaded, or another device is on this port.")
            return 2

        if not send_and_expect(ser, "HELLO", "OK,HELLO"):
            return 2
        if not send_and_expect(ser, f"LIMIT,{PYTHON_PWM_LIMIT}", "OK,LIMIT"):
            return 2
        if not send_and_expect(ser, "ARM", "OK,ARMED"):
            return 2

        if args.pwm is not None:
            run_pwm(ser, args.pwm, args.duration)

        if args.sweep is not None:
            for pwm in parse_sweep(args.sweep):
                run_pwm(ser, pwm, args.duration)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        safe_stop(ser)
        ser.close()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
