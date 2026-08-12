#!/usr/bin/env python3
"""
keyboard_drive_test_signed_wheels.py
====================================

Interactive keyboard control for motor-only scooter testing.

IMPORTANT SIGN CONVENTION
-------------------------
Because the left and right wheels have opposite rotation axes:

  Forward vehicle command:
    M0 Front Left  = +RPM
    M1 Front Right = -RPM
    M2 Back Left   = +RPM
    M3 Back Right  = -RPM

  Backward vehicle command:
    M0 Front Left  = -RPM
    M1 Front Right = +RPM
    M2 Back Left   = -RPM
    M3 Back Right  = +RPM

So the command sent to Arduino is:

  SP#<left_front>#<right_front>#<left_back>#<right_back>

Example for +20 rpm forward:
  SP#+020#-020#+020#-020

This script ignores ultrasonic sensor lines and displays motor telemetry only.

Controls
--------
  UP    arrow  -> increase forward vehicle speed by SPEED_STEP rpm
  DOWN  arrow  -> increase backward vehicle speed by SPEED_STEP rpm
  SPACE        -> emergency stop
  ESC / Q      -> quit

Motor layout
------------
  0 = Front Left
  1 = Front Right
  2 = Back  Left
  3 = Back  Right
"""

import serial
import time
import threading
import sys
import os
import termios
import tty
import argparse
import csv
from pathlib import Path
from datetime import datetime

# ======================== CONFIG ========================
PORT       = "/dev/ttyACM0"
BAUD       = 115200

# Keep this low for first ground/lifted keyboard tests.
MAX_SPEED  = 90     # vehicle RPM command cap, absolute
SPEED_STEP = 5      # RPM per key press
CMD_HZ     = 10     # setpoint send rate
LOG_DIR    = "logs"
# ========================================================

MOTOR_NAMES = [
    "Front Left  M0",
    "Front Right M1",
    "Back  Left  M2",
    "Back  Right M3",
]

# For forward vehicle motion: left wheels positive, right wheels negative.
WHEEL_SIGN = [+1, -1, +1, -1]

# ======================== SHARED STATE ========================
state_lock     = threading.Lock()
current_speed  = 0        # vehicle-level command, not individual wheel command
motor_rpm      = [0.0] * 4
motor_err      = [0.0] * 4
motor_out      = [0.0] * 4
running        = True
last_motor_ts  = 0.0
latest_cmd     = [0.0] * 4
# ==============================================================


def fmt(v: float) -> str:
    sign = '+' if v >= 0 else '-'
    return f"{sign}{int(abs(round(v))):03d}"


def wheel_commands_from_vehicle_speed(vehicle_speed: float):
    """
    Convert desired vehicle forward/backward command to individual wheel signs.

    vehicle_speed > 0 means scooter forward:
      left wheels  +
      right wheels -
    """
    return [WHEEL_SIGN[i] * vehicle_speed for i in range(4)]


def make_cmd(vehicle_speed: float) -> bytes:
    rpms = wheel_commands_from_vehicle_speed(vehicle_speed)
    return (
        f"SP#{fmt(rpms[0])}#{fmt(rpms[1])}"
        f"#{fmt(rpms[2])}#{fmt(rpms[3])}\n"
    ).encode("ascii")


def safe_stop(ser: serial.Serial, repeat=8, delay_s=0.05):
    for _ in range(repeat):
        try:
            ser.write(make_cmd(0))
            time.sleep(delay_s)
        except Exception:
            break


def serial_reader(ser: serial.Serial):
    global motor_rpm, motor_err, motor_out
    global last_motor_ts, running

    while running:
        try:
            if ser.in_waiting:
                raw = ser.readline().decode(errors='ignore').strip()
                if not raw:
                    continue

                # Motor-only test: ignore ultrasonic lines completely.
                if raw.startswith("S:"):
                    continue

                if raw == "ERR":
                    continue

                try:
                    vals = list(map(float, raw.split(",")))
                    if len(vals) == 12:
                        with state_lock:
                            motor_rpm[:]  = vals[0:4]
                            motor_err[:]  = vals[4:8]
                            motor_out[:]  = vals[8:12]
                            last_motor_ts = time.time()
                except ValueError:
                    pass

        except serial.SerialException:
            break
        except Exception:
            pass

        time.sleep(0.001)


def keyboard_reader():
    global current_speed, running

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while running:
            ch = os.read(fd, 1)

            if ch == b'\x1b':
                import select
                r, _, _ = select.select([fd], [], [], 0.05)
                if r:
                    ch2 = os.read(fd, 1)
                    r2, _, _ = select.select([fd], [], [], 0.05)
                    if r2:
                        ch3 = os.read(fd, 1)
                        if ch2 == b'[':
                            if ch3 == b'A':          # UP
                                with state_lock:
                                    current_speed = min(current_speed + SPEED_STEP, MAX_SPEED)
                            elif ch3 == b'B':        # DOWN
                                with state_lock:
                                    current_speed = max(current_speed - SPEED_STEP, -MAX_SPEED)
                else:
                    running = False

            elif ch == b' ':
                with state_lock:
                    current_speed = 0

            elif ch in (b'q', b'Q'):
                running = False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


BAR_WIDTH = 24

def speed_bar(speed: float) -> str:
    half = BAR_WIDTH // 2
    filled = int(abs(speed) / MAX_SPEED * half) if MAX_SPEED > 0 else 0
    filled = max(0, min(filled, half))
    if speed >= 0:
        bar = ' ' * half + '█' * filled + '░' * (half - filled)
    else:
        bar = '░' * (half - filled) + '█' * filled + ' ' * half
    return f"[{bar}]"


def draw(vehicle_speed: float):
    now = time.time()

    with state_lock:
        rpm_  = motor_rpm[:]
        err_  = motor_err[:]
        out_  = motor_out[:]
        m_age = now - last_motor_ts if last_motor_ts else -1
        cmd_  = latest_cmd[:]

    direction = "FORWARD " if vehicle_speed > 0 else "BACKWARD" if vehicle_speed < 0 else "STOPPED "
    m_tag = f"{m_age:.1f}s ago" if 0 <= m_age < 60 else "waiting…"

    W = 78
    sep = "─" * W

    lines = [
        f"╔{sep}╗",
        f"║{'MOTOR-ONLY KEYBOARD DRIVE TEST':^{W}}║",
        f"╠{sep}╣",
        f"║  Vehicle command : {vehicle_speed:+5.0f} RPM   {direction}{'':>39}║",
        f"║  Wheel signs     : M0=+  M1=-  M2=+  M3=- for FORWARD{'':>20}║",
        f"║  Arduino command : SP#{fmt(cmd_[0])}#{fmt(cmd_[1])}#{fmt(cmd_[2])}#{fmt(cmd_[3])}{'':>26}║",
        f"║  {speed_bar(vehicle_speed)}  {vehicle_speed/MAX_SPEED*100:+5.1f}% {'':>27}║",
        f"║  ↑ +{SPEED_STEP} RPM   ↓ -{SPEED_STEP} RPM   SPACE stop   ESC/Q quit{'':>21}║",
        f"╠{sep}╣",
        f"║  MOTOR TELEMETRY ONLY  ({m_tag}){'':>{W-32}}║",
        f"║  {'Motor':<18} {'Cmd':>7}  {'RPM':>8}  {'Error':>8}  {'PWM out':>8}{'':>19}║",
    ]

    for i, name in enumerate(MOTOR_NAMES):
        lines.append(
            f"║    {name:<16} {cmd_[i]:>7.1f}  {rpm_[i]:>8.1f}  {err_[i]:>8.1f}  {out_[i]:>8.1f}{'':>19}║"
        )

    lines += [
        f"╠{sep}╣",
        f"║  SAFETY: first test lifted. For ground, start at +5 or +10 only. SPACE=STOP{'':>5}║",
        f"╚{sep}╝",
    ]

    sys.stdout.write('\033[H' + '\n'.join(lines) + '\n')
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description="Motor-only keyboard drive test with left/right sign correction.")
    p.add_argument("--port", default=PORT)
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument("--max-speed", type=int, default=MAX_SPEED)
    p.add_argument("--step", type=int, default=SPEED_STEP)
    p.add_argument("--cmd-hz", type=float, default=CMD_HZ)
    p.add_argument("--log", action="store_true", help="Save telemetry CSV log.")
    p.add_argument("--yes", action="store_true", help="Skip LIFTED confirmation.")
    return p.parse_args()


def main():
    global running, current_speed, latest_cmd
    global MAX_SPEED, SPEED_STEP, CMD_HZ

    args = parse_args()
    MAX_SPEED = args.max_speed
    SPEED_STEP = args.step
    CMD_HZ = args.cmd_hz

    print(f"Connecting to {args.port} at {args.baud} baud …")
    print("Forward sign convention: M0 +, M1 -, M2 +, M3 -")
    print("Ultrasonic lines are ignored by this script.")

    if not args.yes:
        ans = input("Type LIFTED to confirm wheels are lifted for first test: ").strip()
        if ans != "LIFTED":
            print("Aborted.")
            return 2

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()
    safe_stop(ser)

    csv_file = None
    writer = None
    if args.log:
        Path(LOG_DIR).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(LOG_DIR) / f"keyboard_motor_only_signed_{ts}.csv"
        csv_file = open(csv_path, "w", newline="")
        fieldnames = ["time_s", "vehicle_cmd"] + [f"cmd_m{i}" for i in range(4)] + [f"rpm_m{i}" for i in range(4)] + [f"err_m{i}" for i in range(4)] + [f"pwm_m{i}" for i in range(4)]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Logging to {csv_path}")
    else:
        csv_path = None

    t_start = time.time()

    t_serial = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    t_serial.start()

    t_keys = threading.Thread(target=keyboard_reader, daemon=True)
    t_keys.start()

    sys.stdout.write('\033[2J\033[H')
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()

    interval = 1.0 / CMD_HZ
    try:
        while running:
            loop_t = time.time()

            with state_lock:
                vehicle_speed = current_speed

            cmd_rpms = wheel_commands_from_vehicle_speed(vehicle_speed)
            with state_lock:
                latest_cmd = cmd_rpms[:]

            ser.write(make_cmd(vehicle_speed))
            draw(vehicle_speed)

            if writer:
                with state_lock:
                    rpm_ = motor_rpm[:]
                    err_ = motor_err[:]
                    out_ = motor_out[:]
                row = {
                    "time_s": f"{time.time() - t_start:.3f}",
                    "vehicle_cmd": f"{vehicle_speed:.2f}",
                }
                for i in range(4):
                    row[f"cmd_m{i}"] = f"{cmd_rpms[i]:.2f}"
                    row[f"rpm_m{i}"] = f"{rpm_[i]:.3f}"
                    row[f"err_m{i}"] = f"{err_[i]:.3f}"
                    row[f"pwm_m{i}"] = f"{out_[i]:.3f}"
                writer.writerow(row)
                csv_file.flush()

            rem = interval - (time.time() - loop_t)
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        safe_stop(ser, repeat=12)
        sys.stdout.write('\033[?25h\n')
        sys.stdout.flush()
        print("Motors stopped. Disconnected.")
        if csv_file:
            csv_file.close()
            print(f"Log saved: {csv_path}")
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
