#!/usr/bin/env python3
"""
motor_monitor.py
================
Real-time monitor + CSV logger for all 4 motors.
Displays CMD sent, RPM measured, PWM output, and error for each motor.
Designed to diagnose low-speed PWM oscillation (the 5 RPM limit-cycle issue).

USAGE
─────
  # Monitor only — send a fixed setpoint to one motor:
  python3 motor_monitor.py --motor 0 --rpm 5

  # Monitor all 4 motors with a fixed setpoint:
  python3 motor_monitor.py --all --rpm 5

  # Log to CSV as well:
  python3 motor_monitor.py --motor 2 --rpm 30 --log

  # Just watch what's happening without sending any command:
  python3 motor_monitor.py --listen

  # Sweep through RPM values to compare PWM behaviour:
  python3 motor_monitor.py --motor 0 --sweep 5 10 15 20 30

SERIAL FORMAT (Arduino → Pi)
─────────────────────────────
  rpm0,rpm1,rpm2,rpm3,err0,err1,err2,err3,out0,out1,out2,out3
"""

import argparse
import csv
import os
import serial
import sys
import time
from datetime import datetime

PORT    = "/dev/ttyACM0"
BAUD    = 115200
LOG_DIR = "logs"

MOTOR_NAMES = ["M0 FL", "M1 FR", "M2 BL", "M3 BR"]

# ─────────────────────────────────────────────────────────────
#  Serial helpers
# ─────────────────────────────────────────────────────────────
def fmt_rpm(v):
    sign = "+" if v >= 0 else "-"
    return f"{sign}{int(abs(round(v))):03d}"

def make_cmd(rpms):
    return (f"SP#{fmt_rpm(rpms[0])}#{fmt_rpm(rpms[1])}"
            f"#{fmt_rpm(rpms[2])}#{fmt_rpm(rpms[3])}\n").encode()

def send_cmd(ser, rpms):
    ser.write(make_cmd(rpms))

def stop(ser, n=8):
    for _ in range(n):
        ser.write(make_cmd([0, 0, 0, 0]))
        time.sleep(0.05)

def parse_line(line):
    """Parse Arduino telemetry line → (rpm[4], err[4], pwm[4]) or None."""
    if not line or line.startswith("S:") or line.startswith("ERR"):
        return None
    parts = line.split(",")
    if len(parts) != 12:
        return None
    try:
        vals = [float(p) for p in parts]
        return vals[0:4], vals[4:8], vals[8:12]
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────────
#  Display
# ─────────────────────────────────────────────────────────────
HEADER = (
    "\033[H\033[2J"   # clear screen
    "━" * 72 + "\n"
    "  MOTOR TELEMETRY MONITOR\n"
    "━" * 72 + "\n"
)

COL  = f"  {'Motor':<8} {'CMD (RPM)':>10} {'RPM meas':>10} {'Error':>10} {'PWM out':>10}  {'PWM/CMD':>8}\n"
SEP  = "  " + "─" * 68 + "\n"

def render(cmd_rpms, rpm, err, pwm, stats, elapsed, sample_count):
    lines = [HEADER, COL, SEP]
    for i in range(4):
        cmd = cmd_rpms[i]
        ratio = f"{pwm[i]/cmd:.2f}" if abs(cmd) > 0.5 else "  —"
        flag = ""
        if abs(cmd) > 0.5:
            if pwm[i] == 0 and abs(rpm[i]) < 1:
                flag = "  ← STALLED"
            elif abs(pwm[i] - stats[i]['pwm_last']) > 5:
                flag = "  ← PWM JUMP"
        lines.append(
            f"  {MOTOR_NAMES[i]:<8} {cmd:>+10.1f} {rpm[i]:>+10.2f} "
            f"{err[i]:>+10.2f} {pwm[i]:>10.1f}  {ratio:>8}{flag}\n"
        )

    lines.append(SEP)
    lines.append(f"  Samples: {sample_count}   Elapsed: {elapsed:.1f}s\n")
    lines.append("━" * 72 + "\n\n")

    # Per-motor statistics block
    lines.append(f"  {'Motor':<8} {'PWM min':>8} {'PWM max':>8} {'PWM mean':>9} "
                 f"{'RPM min':>8} {'RPM max':>8} {'RPM mean':>9} {'#zeros':>7}\n")
    lines.append("  " + "─" * 68 + "\n")
    for i in range(4):
        s = stats[i]
        n = s['n']
        if n == 0:
            continue
        pwm_mean = s['pwm_sum'] / n
        rpm_mean = s['rpm_sum'] / n
        lines.append(
            f"  {MOTOR_NAMES[i]:<8} {s['pwm_min']:>8.1f} {s['pwm_max']:>8.1f} {pwm_mean:>9.2f} "
            f"{s['rpm_min']:>8.2f} {s['rpm_max']:>8.2f} {rpm_mean:>9.2f} {s['pwm_zeros']:>7}\n"
        )

    lines.append("\n  Press Ctrl-C to stop.\n")
    sys.stdout.write("".join(lines))
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────
#  Statistics tracker
# ─────────────────────────────────────────────────────────────
def make_stats():
    return [{'n': 0, 'pwm_sum': 0, 'pwm_min': 1e9, 'pwm_max': -1e9,
             'rpm_sum': 0, 'rpm_min': 1e9, 'rpm_max': -1e9,
             'pwm_zeros': 0, 'pwm_last': 0}
            for _ in range(4)]

def update_stats(stats, rpm, pwm):
    for i in range(4):
        s = stats[i]
        s['n'] += 1
        s['pwm_sum'] += pwm[i]
        s['pwm_min']  = min(s['pwm_min'], pwm[i])
        s['pwm_max']  = max(s['pwm_max'], pwm[i])
        s['rpm_sum'] += abs(rpm[i])
        s['rpm_min']  = min(s['rpm_min'], abs(rpm[i]))
        s['rpm_max']  = max(s['rpm_max'], abs(rpm[i]))
        if pwm[i] == 0:
            s['pwm_zeros'] += 1
        s['pwm_last'] = pwm[i]

# ─────────────────────────────────────────────────────────────
#  Modes
# ─────────────────────────────────────────────────────────────
def run_fixed(ser, cmd_rpms, duration, log_path=None):
    """Hold a fixed command and monitor all motors."""
    stats  = make_stats()
    rpm    = [0.0]*4
    err    = [0.0]*4
    pwm    = [0.0]*4
    t0     = time.time()
    count  = 0

    writer = csv_file = None
    if log_path:
        csv_file = open(log_path, "w", newline="")
        fieldnames = ["time_s"] + \
                     [f"cmd_m{i}" for i in range(4)] + \
                     [f"rpm_m{i}" for i in range(4)] + \
                     [f"err_m{i}" for i in range(4)] + \
                     [f"pwm_m{i}" for i in range(4)]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    try:
        while True:
            now = time.time()
            if duration and (now - t0) > duration:
                break

            send_cmd(ser, cmd_rpms)

            # Drain all waiting lines; use the most recent valid one
            telem = None
            deadline = now + 0.12
            while time.time() < deadline:
                if ser.in_waiting:
                    try:
                        line = ser.readline().decode(errors="ignore").strip()
                        parsed = parse_line(line)
                        if parsed:
                            telem = parsed
                    except Exception:
                        pass
                else:
                    time.sleep(0.005)

            if telem:
                rpm, err, pwm = telem
                update_stats(stats, rpm, pwm)
                count += 1

                if writer:
                    row = {"time_s": f"{now - t0:.3f}"}
                    for i in range(4):
                        row[f"cmd_m{i}"] = f"{cmd_rpms[i]:.1f}"
                        row[f"rpm_m{i}"] = f"{rpm[i]:.3f}"
                        row[f"err_m{i}"] = f"{err[i]:.3f}"
                        row[f"pwm_m{i}"] = f"{pwm[i]:.1f}"
                    writer.writerow(row)
                    csv_file.flush()

            render(cmd_rpms, rpm, err, pwm, stats, now - t0, count)

    finally:
        stop(ser)
        if csv_file:
            csv_file.close()
        if log_path:
            print(f"\nLog saved → {log_path}")


def run_sweep(ser, motor_index, rpm_levels, hold_s, log_path=None):
    """Step through a list of RPM setpoints, hold each for hold_s seconds."""
    cmd_rpms = [0.0, 0.0, 0.0, 0.0]

    writer = csv_file = None
    if log_path:
        csv_file = open(log_path, "w", newline="")
        fieldnames = ["time_s", "sweep_rpm"] + \
                     [f"rpm_m{i}" for i in range(4)] + \
                     [f"err_m{i}" for i in range(4)] + \
                     [f"pwm_m{i}" for i in range(4)]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    t_global = time.time()

    try:
        # Coast down between steps
        stop(ser, n=6)
        time.sleep(1.0)

        for sp in rpm_levels:
            cmd_rpms[motor_index] = sp
            stats = make_stats()
            rpm   = [0.0]*4
            err   = [0.0]*4
            pwm   = [0.0]*4
            t0    = time.time()
            count = 0

            print(f"\n  → Setpoint: {sp:+.1f} RPM  (hold {hold_s}s)")

            while (time.time() - t0) < hold_s:
                now = time.time()
                send_cmd(ser, cmd_rpms)

                telem = None
                deadline = now + 0.12
                while time.time() < deadline:
                    if ser.in_waiting:
                        try:
                            line = ser.readline().decode(errors="ignore").strip()
                            parsed = parse_line(line)
                            if parsed:
                                telem = parsed
                        except Exception:
                            pass
                    else:
                        time.sleep(0.005)

                if telem:
                    rpm, err, pwm = telem
                    update_stats(stats, rpm, pwm)
                    count += 1

                    if writer:
                        row = {"time_s": f"{now - t_global:.3f}",
                               "sweep_rpm": f"{sp:.1f}"}
                        for i in range(4):
                            row[f"rpm_m{i}"] = f"{rpm[i]:.3f}"
                            row[f"err_m{i}"] = f"{err[i]:.3f}"
                            row[f"pwm_m{i}"] = f"{pwm[i]:.1f}"
                        writer.writerow(row)
                        csv_file.flush()

                render(cmd_rpms, rpm, err, pwm, stats, now - t0, count)

            # Print summary for this step
            s  = stats[motor_index]
            n  = s['n'] or 1
            print(f"     PWM: min={s['pwm_min']:.1f}  max={s['pwm_max']:.1f}  "
                  f"mean={s['pwm_sum']/n:.2f}  zeros={s['pwm_zeros']}/{n}")
            print(f"     RPM: min={s['rpm_min']:.2f}  max={s['rpm_max']:.2f}  "
                  f"mean={s['rpm_sum']/n:.2f}")

            stop(ser, n=4)
            time.sleep(1.5)

    finally:
        stop(ser)
        if csv_file:
            csv_file.close()
        if log_path:
            print(f"\nLog saved → {log_path}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Motor telemetry monitor")
    p.add_argument("--port",   default=PORT)
    p.add_argument("--baud",   type=int, default=BAUD)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--motor",  type=int, choices=[0,1,2,3],
                      help="Send to one motor only (others get 0)")
    mode.add_argument("--all",    action="store_true",
                      help="Send same RPM to all 4 motors")
    mode.add_argument("--listen", action="store_true",
                      help="Passive: no commands sent, just display telemetry")

    p.add_argument("--rpm",    type=float, default=0.0,
                   help="Fixed RPM setpoint to send")
    p.add_argument("--sweep",  type=float, nargs="+", metavar="RPM",
                   help="Sweep through these RPM levels (e.g. 5 10 15 20 30)")
    p.add_argument("--hold",   type=float, default=8.0,
                   help="Hold time per sweep step in seconds (default 8)")
    p.add_argument("--duration", type=float, default=0,
                   help="Stop after N seconds (0 = run until Ctrl-C)")
    p.add_argument("--log",    action="store_true",
                   help="Save CSV log to logs/")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Connecting to {args.port} @ {args.baud}...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.15)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Waiting for Arduino boot (2s)...", end="", flush=True)
    time.sleep(2)
    ser.reset_input_buffer()
    print(" ready.")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None
    if args.log:
        suffix = "sweep" if args.sweep else f"fixed_{int(args.rpm)}rpm"
        log_path = os.path.join(LOG_DIR, f"monitor_{suffix}_{ts}.csv")

    sys.stdout.write("\033[?25l")  # hide cursor
    sys.stdout.flush()

    try:
        if args.sweep:
            motor_idx = args.motor if args.motor is not None else 0
            print(f"\nSweep on M{motor_idx} {MOTOR_NAMES[motor_idx]}: {args.sweep} RPM")
            run_sweep(ser, motor_idx, args.sweep, args.hold, log_path)

        else:
            # Build command array
            if args.listen:
                cmd_rpms = [0.0, 0.0, 0.0, 0.0]
            elif args.all:
                cmd_rpms = [args.rpm] * 4
                # Apply sign convention: right-side motors run reversed
                cmd_rpms[1] = -cmd_rpms[1]
                cmd_rpms[3] = -cmd_rpms[3]
            elif args.motor is not None:
                cmd_rpms = [0.0, 0.0, 0.0, 0.0]
                cmd_rpms[args.motor] = args.rpm
            else:
                cmd_rpms = [0.0, 0.0, 0.0, 0.0]

            run_fixed(ser, cmd_rpms, args.duration or None, log_path)

    except KeyboardInterrupt:
        pass
    finally:
        stop(ser)
        ser.close()
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
        print("Stopped.")


if __name__ == "__main__":
    main()
