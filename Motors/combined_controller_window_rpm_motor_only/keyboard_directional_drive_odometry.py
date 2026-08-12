#!/usr/bin/env python3
"""
keyboard_directional_drive_odometry.py

Directional keyboard control + odometry logging for the 4-wheel scooter.

Geometry:
  wheel diameter = 0.185 m
  wheelbase      = 0.68  m  # logged for reference
  track width    = 0.57  m  # used for differential/skid-steer yaw estimate

Sign convention:
  Forward vehicle motion:
    M0 Front Left  = +
    M1 Front Right = -
    M2 Back Left   = +
    M3 Back Right  = -

Controls:
  UP/DOWN     : increase/decrease linear forward command
  LEFT/RIGHT  : increase/decrease turn command
  C           : center steering command, turn = 0
  R           : reset odometry
  SPACE       : stop
  Q or ESC    : quit safely

Odometry caveat:
  This is approximate. It assumes skid-steer/differential-drive kinematics.
  Hall RPM may be direction-blind, so wheel direction is inferred from command.
"""

import argparse
import csv
import math
import os
import select
import serial
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

PORT = "/dev/ttyACM0"
BAUD = 115200
LOG_DIR = "logs"

WHEEL_DIAMETER_M = 0.185
WHEELBASE_M = 0.68
TRACK_WIDTH_M = 0.57

MAX_WHEEL_RPM = 150.0
LINEAR_STEP_RPM = 10.0
TURN_STEP_RPM = 60.0
CMD_HZ = 20.0

MOTOR_NAMES = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]

state_lock = threading.Lock()
running = True

linear_cmd = 0.0
turn_cmd = 0.0

cmd_m = [0.0, 0.0, 0.0, 0.0]
rpm_m = [0.0, 0.0, 0.0, 0.0]
err_m = [0.0, 0.0, 0.0, 0.0]
pwm_m = [0.0, 0.0, 0.0, 0.0]
last_motor_ts = 0.0

odom_x = 0.0
odom_y = 0.0
odom_theta = 0.0
odom_path = 0.0
odom_left_dist = 0.0
odom_right_dist = 0.0
last_odom_t = None
last_left_sign = 0.0
last_right_sign = 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def fmt(v):
    sign = "+" if v >= 0 else "-"
    return f"{sign}{int(abs(round(v))):03d}"


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def side_rpm_from_linear_turn(linear_rpm, turn_rpm, max_wheel_rpm):
    # Positive side RPM means that side drives the vehicle forward.
    # Positive turn_rpm means turn left: right side faster than left side.
    left = linear_rpm - turn_rpm
    right = linear_rpm + turn_rpm

    max_abs = max(abs(left), abs(right), 1e-9)
    if max_abs > max_wheel_rpm:
        scale = max_wheel_rpm / max_abs
        left *= scale
        right *= scale

    return left, right


def motor_cmd_from_side(left_side_rpm, right_side_rpm):
    # Left forward is motor +. Right forward is motor - because of mounting.
    return [left_side_rpm, -right_side_rpm, left_side_rpm, -right_side_rpm]


def make_cmd(motor_rpms):
    return (
        f"SP#{fmt(motor_rpms[0])}#{fmt(motor_rpms[1])}"
        f"#{fmt(motor_rpms[2])}#{fmt(motor_rpms[3])}\n"
    ).encode("ascii")


def safe_stop(ser, repeat=12, delay=0.05):
    zero = make_cmd([0, 0, 0, 0])
    for _ in range(repeat):
        try:
            ser.write(zero)
            time.sleep(delay)
        except Exception:
            break


def serial_reader(ser):
    global rpm_m, err_m, pwm_m, last_motor_ts, running
    while running:
        try:
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if not line or line.startswith("S:") or line == "ERR":
                    continue
                parts = line.split(",")
                if len(parts) != 12:
                    continue
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue
                with state_lock:
                    rpm_m[:] = vals[0:4]
                    err_m[:] = vals[4:8]
                    pwm_m[:] = vals[8:12]
                    last_motor_ts = time.time()
        except Exception:
            pass
        time.sleep(0.001)


def keyboard_reader(max_wheel_rpm, linear_step, turn_step):
    global running, linear_cmd, turn_cmd
    global odom_x, odom_y, odom_theta, odom_path, odom_left_dist, odom_right_dist

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while running:
            ch = os.read(fd, 1)

            if ch == b"\x1b":
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    running = False
                    break
                ch2 = os.read(fd, 1)
                r2, _, _ = select.select([fd], [], [], 0.05)
                if not r2:
                    continue
                ch3 = os.read(fd, 1)
                if ch2 == b"[":
                    with state_lock:
                        if ch3 == b"A":       # up
                            linear_cmd = clamp(linear_cmd + linear_step, -max_wheel_rpm, max_wheel_rpm)
                        elif ch3 == b"B":     # down
                            linear_cmd = clamp(linear_cmd - linear_step, -max_wheel_rpm, max_wheel_rpm)
                        elif ch3 == b"D":     # left
                            turn_cmd = clamp(turn_cmd + turn_step, -max_wheel_rpm, max_wheel_rpm)
                        elif ch3 == b"C":     # right
                            turn_cmd = clamp(turn_cmd - turn_step, -max_wheel_rpm, max_wheel_rpm)

            elif ch == b" ":
                with state_lock:
                    linear_cmd = 0.0
                    turn_cmd = 0.0

            elif ch in (b"c", b"C"):
                with state_lock:
                    turn_cmd = 0.0

            elif ch in (b"r", b"R"):
                with state_lock:
                    odom_x = odom_y = odom_theta = odom_path = 0.0
                    odom_left_dist = odom_right_dist = 0.0

            elif ch in (b"q", b"Q"):
                running = False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def update_odometry(now, wheel_circ, track_width, left_cmd, right_cmd):
    global odom_x, odom_y, odom_theta, odom_path, odom_left_dist, odom_right_dist
    global last_odom_t, last_left_sign, last_right_sign

    with state_lock:
        r0, r1, r2, r3 = rpm_m[:]

    if last_odom_t is None:
        last_odom_t = now
        return 0.0, 0.0, 0.0, 0.0

    dt = now - last_odom_t
    last_odom_t = now
    if dt <= 0 or dt > 0.5:
        dt = 0.0

    left_mag = 0.5 * (abs(r0) + abs(r2))
    right_mag = 0.5 * (abs(r1) + abs(r3))

    if abs(left_cmd) > 0.5:
        last_left_sign = 1.0 if left_cmd > 0 else -1.0
    elif left_mag < 1.0:
        last_left_sign = 0.0

    if abs(right_cmd) > 0.5:
        last_right_sign = 1.0 if right_cmd > 0 else -1.0
    elif right_mag < 1.0:
        last_right_sign = 0.0

    left_rpm = last_left_sign * left_mag
    right_rpm = last_right_sign * right_mag

    v_l = left_rpm * wheel_circ / 60.0
    v_r = right_rpm * wheel_circ / 60.0

    v = 0.5 * (v_l + v_r)
    omega = (v_r - v_l) / track_width if track_width > 1e-9 else 0.0

    theta_mid = odom_theta + 0.5 * omega * dt
    odom_x += v * math.cos(theta_mid) * dt
    odom_y += v * math.sin(theta_mid) * dt
    odom_theta = wrap_pi(odom_theta + omega * dt)
    odom_path += abs(v) * dt
    odom_left_dist += v_l * dt
    odom_right_dist += v_r * dt

    return left_rpm, right_rpm, v, omega


def draw(left_cmd, right_cmd, left_est, right_est, v, omega, max_wheel_rpm):
    with state_lock:
        lin = linear_cmd
        trn = turn_cmd
        rpm = rpm_m[:]
        pwm = pwm_m[:]
        cmds = cmd_m[:]
        x = odom_x
        y = odom_y
        th = odom_theta
        path = odom_path
        age = time.time() - last_motor_ts if last_motor_ts else -1

    age_s = f"{age:.1f}s" if 0 <= age < 60 else "waiting"
    arduino = f"SP#{fmt(cmds[0])}#{fmt(cmds[1])}#{fmt(cmds[2])}#{fmt(cmds[3])}"

    lines = [
        "DIRECTIONAL KEYBOARD DRIVE + ODOMETRY",
        "-" * 88,
        f"linear={lin:+6.1f} rpm, turn={trn:+6.1f} rpm | left_side={left_cmd:+6.1f}, right_side={right_cmd:+6.1f}",
        f"Arduino command: {arduino}",
        "Controls: UP/DOWN linear | LEFT/RIGHT turn | C center | R reset odom | SPACE stop | Q/ESC quit",
        "-" * 88,
        f"ODOM: x={x:+7.3f} m, y={y:+7.3f} m, theta={math.degrees(th):+7.1f} deg, path={path:7.3f} m",
        f"      v={v:+7.3f} m/s, omega={omega:+7.3f} rad/s, left_est={left_est:+6.1f} rpm, right_est={right_est:+6.1f} rpm, tel={age_s}",
        "-" * 88,
        f"{'Motor':<16} {'Cmd':>8} {'RPM':>8} {'PWM':>8}",
    ]
    for i, name in enumerate(MOTOR_NAMES):
        lines.append(f"{name:<16} {cmds[i]:>8.1f} {rpm[i]:>8.1f} {pwm[i]:>8.1f}")
    lines.append("-" * 88)
    lines.append("SAFETY: First test lifted. On ground use small max-wheel-rpm and avoid sharp turns.")

    sys.stdout.write("\033[H" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=PORT)
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument("--max-wheel-rpm", type=float, default=MAX_WHEEL_RPM)
    p.add_argument("--linear-step", type=float, default=LINEAR_STEP_RPM)
    p.add_argument("--turn-step", type=float, default=TURN_STEP_RPM)
    p.add_argument("--cmd-hz", type=float, default=CMD_HZ)
    p.add_argument("--wheel-diameter", type=float, default=WHEEL_DIAMETER_M)
    p.add_argument("--wheelbase", type=float, default=WHEELBASE_M)
    p.add_argument("--track-width", type=float, default=TRACK_WIDTH_M)
    p.add_argument("--log", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def main():
    global running, cmd_m

    args = parse_args()
    wheel_circ = math.pi * args.wheel_diameter

    print(f"Connecting to {args.port} @ {args.baud}")
    print(f"Geometry: wheel_diameter={args.wheel_diameter:.3f} m, wheelbase={args.wheelbase:.3f} m, track_width={args.track_width:.3f} m")
    print("Forward signs: M0 +, M1 -, M2 +, M3 -")
    print("Positive turn command = left/CCW turn.")

    if not args.yes:
        ans = input("Type LIFTED to confirm wheels are lifted for first directional test: ").strip()
        if ans != "LIFTED":
            print("Aborted.")
            return 2

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as exc:
        print(f"ERROR: {exc}")
        return 1

    time.sleep(2.0)
    ser.reset_input_buffer()
    safe_stop(ser)

    csv_file = None
    writer = None
    csv_path = None
    if args.log:
        Path(LOG_DIR).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(LOG_DIR) / f"keyboard_directional_odometry_{ts}.csv"
        csv_file = open(csv_path, "w", newline="")
        fieldnames = ["time_s", "linear_cmd_rpm", "turn_cmd_rpm", "left_physical_cmd_rpm", "right_physical_cmd_rpm"]
        fieldnames += [f"cmd_m{i}" for i in range(4)]
        fieldnames += [f"rpm_m{i}" for i in range(4)]
        fieldnames += [f"err_m{i}" for i in range(4)]
        fieldnames += [f"pwm_m{i}" for i in range(4)]
        fieldnames += ["left_rpm_est", "right_rpm_est", "v_mps", "omega_radps", "x_m", "y_m", "theta_rad", "theta_deg", "path_m", "left_dist_m", "right_dist_m", "wheel_diameter_m", "wheelbase_m", "track_width_m"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Logging to {csv_path}")

    threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()
    threading.Thread(target=keyboard_reader, args=(args.max_wheel_rpm, args.linear_step, args.turn_step), daemon=True).start()

    sys.stdout.write("\033[2J\033[H\033[?25l")
    sys.stdout.flush()

    t0 = time.time()
    interval = 1.0 / args.cmd_hz if args.cmd_hz > 0 else 0.1

    try:
        while running:
            loop_t = time.time()
            with state_lock:
                lin = linear_cmd
                trn = turn_cmd

            left_cmd, right_cmd = side_rpm_from_linear_turn(lin, trn, args.max_wheel_rpm)
            motor_cmds = motor_cmd_from_side(left_cmd, right_cmd)

            with state_lock:
                cmd_m = motor_cmds[:]

            ser.write(make_cmd(motor_cmds))

            left_est, right_est, v, omega = update_odometry(loop_t, wheel_circ, args.track_width, left_cmd, right_cmd)
            draw(left_cmd, right_cmd, left_est, right_est, v, omega, args.max_wheel_rpm)

            if writer:
                with state_lock:
                    rpm = rpm_m[:]
                    err = err_m[:]
                    pwm = pwm_m[:]
                    x = odom_x
                    y = odom_y
                    th = odom_theta
                    path = odom_path
                    ld = odom_left_dist
                    rd = odom_right_dist

                row = {
                    "time_s": f"{loop_t - t0:.3f}",
                    "linear_cmd_rpm": f"{lin:.3f}",
                    "turn_cmd_rpm": f"{trn:.3f}",
                    "left_physical_cmd_rpm": f"{left_cmd:.3f}",
                    "right_physical_cmd_rpm": f"{right_cmd:.3f}",
                    "left_rpm_est": f"{left_est:.3f}",
                    "right_rpm_est": f"{right_est:.3f}",
                    "v_mps": f"{v:.6f}",
                    "omega_radps": f"{omega:.6f}",
                    "x_m": f"{x:.6f}",
                    "y_m": f"{y:.6f}",
                    "theta_rad": f"{th:.6f}",
                    "theta_deg": f"{math.degrees(th):.3f}",
                    "path_m": f"{path:.6f}",
                    "left_dist_m": f"{ld:.6f}",
                    "right_dist_m": f"{rd:.6f}",
                    "wheel_diameter_m": f"{args.wheel_diameter:.6f}",
                    "wheelbase_m": f"{args.wheelbase:.6f}",
                    "track_width_m": f"{args.track_width:.6f}",
                }
                for i in range(4):
                    row[f"cmd_m{i}"] = f"{motor_cmds[i]:.3f}"
                    row[f"rpm_m{i}"] = f"{rpm[i]:.3f}"
                    row[f"err_m{i}"] = f"{err[i]:.3f}"
                    row[f"pwm_m{i}"] = f"{pwm[i]:.3f}"

                writer.writerow(row)
                csv_file.flush()

            rem = interval - (time.time() - loop_t)
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        safe_stop(ser, repeat=15)
        ser.close()
        if csv_file:
            csv_file.close()
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
        print("Motors stopped. Disconnected.")
        if csv_path:
            print(f"Log saved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
