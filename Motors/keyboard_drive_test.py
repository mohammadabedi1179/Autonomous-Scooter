"""
keyboard_drive_test.py
======================
Interactive keyboard control for combined_controller.ino.
Uses only Python standard library (termios/tty) — no pip install needed.

Controls
--------
  UP    arrow  → +5 RPM  (more forward)
  DOWN  arrow  → -5 RPM  (more backward)
  SPACE        → emergency stop
  ESC / Q      → quit

Motor layout
------------
  0 = Front Left    1 = Front Right
  2 = Back  Left    3 = Back  Right

All four motors receive the same setpoint for straight travel.
"""

import serial
import time
import threading
import sys
import os
import termios
import tty

# ======================== CONFIG ========================
PORT       = "/dev/ttyACM0"
BAUD       = 115200
MAX_SPEED  = 80     # RPM cap (absolute)
SPEED_STEP = 5      # RPM per key press
CMD_HZ     = 10     # setpoint send rate (Hz)
# ========================================================

SENSOR_LABELS = ["Right Front", "Left Front ", "Left Side  ", "Right Side ", "Rear       "]
MOTOR_NAMES   = ["Front Left (0)", "Front Right(1)", "Back  Left (2)", "Back  Right(3)"]

# ======================== SHARED STATE ========================
state_lock     = threading.Lock()
current_speed  = 0
distances      = [-1.0] * 5
motor_rpm      = [0.0] * 4
motor_err      = [0.0] * 4
motor_out      = [0.0] * 4
running        = True
last_sensor_ts = 0.0
last_motor_ts  = 0.0
# ==============================================================


# ─────────────────────────── protocol helpers ───────────────────────────

def fmt(v: float) -> str:
    sign = '+' if v >= 0 else '-'
    return f"{sign}{int(abs(v)):03d}"


def make_cmd(speed: float) -> bytes:
    """All four motors get the same setpoint."""
    s = fmt(speed)
    return f"SP#{s}#{s}#{s}#{s}\n".encode()


# ─────────────────────────── serial reader ───────────────────────────

def serial_reader(ser: serial.Serial):
    global distances, motor_rpm, motor_err, motor_out
    global last_sensor_ts, last_motor_ts, running

    while running:
        try:
            if ser.in_waiting:
                raw = ser.readline().decode(errors='ignore').strip()
                if not raw:
                    continue

                if raw.startswith("S:"):
                    try:
                        vals = list(map(float, raw[2:].split(",")))
                        if len(vals) == 5:
                            with state_lock:
                                distances[:]   = vals
                                last_sensor_ts = time.time()
                    except ValueError:
                        pass

                elif raw != "ERR":
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


# ─────────────────────────── keyboard reader ───────────────────────────
# Reads raw terminal bytes in a background thread.
# Arrow keys arrive as 3-byte sequences: ESC [ A/B

def keyboard_reader():
    """
    Runs with the terminal in raw/cbreak mode.
    Restores settings on exit via the saved fd_settings.
    """
    global current_speed, running

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)          # cbreak: no line buffering, no echo

        while running:
            ch = os.read(fd, 1)

            if ch == b'\x1b':      # ESC — could be arrow key or bare ESC
                # Try to read two more bytes within a short window
                import select
                r, _, _ = select.select([fd], [], [], 0.05)
                if r:
                    ch2 = os.read(fd, 1)
                    r2, _, _ = select.select([fd], [], [], 0.05)
                    if r2:
                        ch3 = os.read(fd, 1)
                        if ch2 == b'[':
                            if ch3 == b'A':          # ↑
                                with state_lock:
                                    current_speed = min(
                                        current_speed + SPEED_STEP, MAX_SPEED)
                            elif ch3 == b'B':        # ↓
                                with state_lock:
                                    current_speed = max(
                                        current_speed - SPEED_STEP, -MAX_SPEED)
                else:
                    # Bare ESC → quit
                    running = False

            elif ch in (b' ',):    # SPACE → stop
                with state_lock:
                    current_speed = 0

            elif ch in (b'q', b'Q'):
                running = False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ─────────────────────────── display ───────────────────────────

BAR_WIDTH = 20

def speed_bar(speed: float) -> str:
    half   = BAR_WIDTH // 2
    filled = int(abs(speed) / MAX_SPEED * half)
    filled = min(filled, half)
    if speed >= 0:
        bar = ' ' * half + '█' * filled + '░' * (half - filled)
    else:
        bar = '░' * (half - filled) + '█' * filled + ' ' * half
    return f"[{bar}]"


def draw(speed: float):
    now = time.time()

    with state_lock:
        dist   = distances[:]
        rpm_   = motor_rpm[:]
        err_   = motor_err[:]
        out_   = motor_out[:]
        s_age  = now - last_sensor_ts if last_sensor_ts else -1
        m_age  = now - last_motor_ts  if last_motor_ts  else -1

    direction = "FORWARD " if speed > 0 else "BACKWARD" if speed < 0 else "STOPPED "
    s_tag = f"{s_age:.1f}s ago" if 0 <= s_age < 60 else "waiting…"
    m_tag = f"{m_age:.1f}s ago" if 0 <= m_age < 60 else "waiting…"

    W = 62   # inner width
    sep = "─" * W

    lines = [
        f"╔{sep}╗",
        f"║{'SCOOTER DRIVE TEST':^{W}}║",
        f"╠{sep}╣",
        f"║  Setpoint : {speed:+5.0f} RPM   {direction}{'':>28}║",
        f"║  {speed_bar(speed)}  {speed/MAX_SPEED*100:+5.1f}% {'':>14}║",
        f"║  ↑ +{SPEED_STEP} RPM   ↓ -{SPEED_STEP} RPM   SPACE stop   ESC/Q quit{'':>5}║",
        f"╠{sep}╣",
        f"║  ULTRASONIC DISTANCES  ({s_tag}){'':>{W-32}}║",
    ]

    for label, d in zip(SENSOR_LABELS, dist):
        if d >= 0:
            bar_fill = max(0, min(20, int(d / 400 * 20)))
            d_str    = f"{d:6.1f} cm"
        else:
            bar_fill = 0
            d_str    = "  ---   "
        bar = '▓' * bar_fill + '░' * (20 - bar_fill)
        lines.append(f"║    {label}: {d_str}  [{bar}]    ║")

    lines += [
        f"╠{sep}╣",
        f"║  MOTOR TELEMETRY  ({m_tag}){'':>{W-27}}║",
        f"║  {'Motor':<16} {'RPM':>7}  {'Error':>7}  {'PWM out':>7}{'':>12}║",
    ]

    for i, name in enumerate(MOTOR_NAMES):
        lines.append(
            f"║    {name:<14} {rpm_[i]:>7.1f}  {err_[i]:>7.1f}  {out_[i]:>7.1f}{'':>12}║"
        )

    lines.append(f"╚{sep}╝")

    # Single write → no flicker
    sys.stdout.write('\033[H' + '\n'.join(lines) + '\n')
    sys.stdout.flush()


# ─────────────────────────── main ───────────────────────────

def main():
    global running, current_speed

    print(f"Connecting to {PORT} at {BAUD} baud …")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    # Threads
    t_serial = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    t_serial.start()

    t_keys = threading.Thread(target=keyboard_reader, daemon=True)
    t_keys.start()

    # Prepare display
    sys.stdout.write('\033[2J\033[H')   # clear screen
    sys.stdout.write('\033[?25l')       # hide cursor
    sys.stdout.flush()

    interval = 1.0 / CMD_HZ
    try:
        while running:
            t0 = time.time()

            with state_lock:
                spd = current_speed

            ser.write(make_cmd(spd))
            draw(spd)

            elapsed = time.time() - t0
            rem = interval - elapsed
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        # Double stop for safety
        try:
            ser.write(make_cmd(0))
            time.sleep(0.05)
            ser.write(make_cmd(0))
        except Exception:
            pass

        sys.stdout.write('\033[?25h\n')  # restore cursor
        sys.stdout.flush()
        print("Motors stopped. Disconnected.")
        ser.close()


if __name__ == "__main__":
    main()
