"""
motor_individual_test.py
========================
Test and log each motor individually.

Controls
--------
  0 / 1 / 2 / 3   select active motor
  UP    arrow      +5 RPM  on selected motor
  DOWN  arrow      -5 RPM  on selected motor
  SPACE            stop selected motor
  A                stop ALL motors
  ESC / Q          quit (all motors stopped first)

Display
-------
  Live panel per motor: setpoint · measured RPM · error · PWM bar · direction

Logging
-------
  CSV written to ./logs/motor_test_YYYYMMDD_HHMMSS.csv
  Columns: timestamp_s, motor, setpoint_rpm, measured_rpm, error_rpm, pwm_out

Serial format expected from combined_controller.ino
----------------------------------------------------
  Motor telemetry (100 ms):
      rpm0,rpm1,rpm2,rpm3, err0,err1,err2,err3, out0,out1,out2,out3
  Ultrasonic (200 ms):
      S:d0,d1,d2,d3,d4
"""

import serial
import time
import threading
import sys
import os
import csv
import select
import termios
import tty
from datetime import datetime

# ======================== CONFIG ========================
PORT       = "/dev/ttyACM0"
BAUD       = 115200
MAX_SPEED  = 80    # RPM cap (absolute value)
SPEED_STEP = 5     # RPM per key press
CMD_HZ     = 10    # setpoints re-sent at this rate
LOG_DIR    = "logs"
# ========================================================

MOTOR_NAMES = [
    "M0  Front Left ",
    "M1  Front Right",
    "M2  Back  Left ",
    "M3  Back  Right",
]
SENSOR_LABELS = ["R.Front", "L.Front", "L.Side ", "R.Side ", "Rear   "]

# ======================== SHARED STATE ========================
state_lock    = threading.Lock()
setpoints     = [0.0, 0.0, 0.0, 0.0]   # commanded RPM per motor
motor_rpm     = [0.0, 0.0, 0.0, 0.0]   # measured RPM
motor_err     = [0.0, 0.0, 0.0, 0.0]   # PID error
motor_out     = [0.0, 0.0, 0.0, 0.0]   # PWM output
distances     = [-1.0] * 5
active_motor  = 0                        # which motor keyboard controls
running       = True
last_motor_ts = 0.0
last_sens_ts  = 0.0
packet_count  = 0                        # total telemetry packets received
# ==============================================================


# ─────────────────────────── serial helpers ───────────────────────────

def fmt(v: float) -> str:
    sign = '+' if v >= 0 else '-'
    return f"{sign}{int(abs(v)):03d}"


def make_cmd(spds) -> bytes:
    """Build 22-char command from list of 4 setpoints."""
    parts = "#".join(fmt(s) for s in spds)
    return f"SP#{parts}\n".encode()


# ─────────────────────────── CSV logger ───────────────────────────

def open_log() -> csv.writer:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"motor_test_{ts}.csv")
    fh  = open(path, 'w', newline='')
    w   = csv.writer(fh)
    w.writerow(["timestamp_s", "motor", "setpoint_rpm",
                "measured_rpm", "error_rpm", "pwm_out"])
    print(f"Logging to: {path}")
    return w, fh, path


def log_packet(writer, t: float, spds, rpms, errs, outs):
    for i in range(4):
        writer.writerow([
            f"{t:.4f}", i,
            f"{spds[i]:.1f}", f"{rpms[i]:.2f}",
            f"{errs[i]:.2f}", f"{outs[i]:.2f}",
        ])


# ─────────────────────────── serial reader ───────────────────────────

def serial_reader(ser: serial.Serial, log_writer, log_fh):
    global motor_rpm, motor_err, motor_out, distances
    global last_motor_ts, last_sens_ts, running, packet_count

    t_start = time.time()

    while running:
        try:
            if ser.in_waiting:
                raw = ser.readline().decode(errors='ignore').strip()
                if not raw:
                    continue

                if raw.startswith("S:"):
                    # ── Ultrasonic ──
                    try:
                        vals = list(map(float, raw[2:].split(",")))
                        if len(vals) == 5:
                            with state_lock:
                                distances[:]  = vals
                                last_sens_ts  = time.time()
                    except ValueError:
                        pass

                elif raw != "ERR":
                    # ── Motor telemetry ──
                    try:
                        vals = list(map(float, raw.split(",")))
                        if len(vals) == 12:
                            now = time.time()
                            with state_lock:
                                motor_rpm[:]  = vals[0:4]
                                motor_err[:]  = vals[4:8]
                                motor_out[:]  = vals[8:12]
                                last_motor_ts = now
                                packet_count += 1
                                # snapshot for logging (under lock)
                                _spds = setpoints[:]
                                _rpms = motor_rpm[:]
                                _errs = motor_err[:]
                                _outs = motor_out[:]

                            log_packet(log_writer,
                                       now - t_start,
                                       _spds, _rpms, _errs, _outs)
                            log_fh.flush()

                    except ValueError:
                        pass

        except serial.SerialException:
            break
        except Exception:
            pass

        time.sleep(0.001)


# ─────────────────────────── keyboard reader ───────────────────────────

def keyboard_reader():
    global setpoints, active_motor, running

    fd          = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while running:
            # Block until a byte arrives (sleep-friendly)
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue

            ch = os.read(fd, 1)

            # ── Arrow / escape sequences ──
            if ch == b'\x1b':
                r2, _, _ = select.select([fd], [], [], 0.05)
                if not r2:
                    running = False          # bare ESC → quit
                    continue
                ch2 = os.read(fd, 1)
                r3, _, _ = select.select([fd], [], [], 0.05)
                if r3:
                    ch3 = os.read(fd, 1)
                    if ch2 == b'[':
                        with state_lock:
                            am = active_motor
                        if ch3 == b'A':      # ↑
                            with state_lock:
                                setpoints[am] = min(
                                    setpoints[am] + SPEED_STEP, MAX_SPEED)
                        elif ch3 == b'B':    # ↓
                            with state_lock:
                                setpoints[am] = max(
                                    setpoints[am] - SPEED_STEP, -MAX_SPEED)

            # ── Motor select ──
            elif ch in (b'0', b'1', b'2', b'3'):
                with state_lock:
                    active_motor = int(ch)

            # ── Stop selected ──
            elif ch == b' ':
                with state_lock:
                    setpoints[active_motor] = 0.0

            # ── Stop ALL ──
            elif ch in (b'a', b'A'):
                with state_lock:
                    setpoints[:] = [0.0, 0.0, 0.0, 0.0]

            # ── Quit ──
            elif ch in (b'q', b'Q'):
                running = False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ─────────────────────────── display ───────────────────────────

PWM_MAX   = 200
BAR_CELLS = 20    # cells in the PWM bar

def pwm_bar(out_val: float) -> str:
    """Half-bar centred: negative = left, positive = right."""
    half   = BAR_CELLS // 2
    filled = int(abs(out_val) / PWM_MAX * half)
    filled = min(filled, half)
    if out_val >= 0:
        bar = ' ' * half + '█' * filled + '░' * (half - filled)
    else:
        bar = '░' * (half - filled) + '█' * filled + ' ' * half
    return f"[{bar}]"


def rpm_bar(rpm_val: float, setpoint: float) -> str:
    """Shows measured RPM as fraction of setpoint."""
    half   = BAR_CELLS // 2
    ref    = abs(setpoint) if abs(setpoint) > 0 else MAX_SPEED
    filled = int(min(abs(rpm_val) / ref, 1.0) * BAR_CELLS)
    bar    = '▓' * filled + '░' * (BAR_CELLS - filled)
    return f"[{bar}]"


def direction_str(sp: float) -> str:
    if sp > 0:  return "▶ FWD"
    if sp < 0:  return "◀ REV"
    return       "■ STP"


def draw(active: int, pkts: int):
    now = time.time()

    with state_lock:
        spds  = setpoints[:]
        rpms  = motor_rpm[:]
        errs  = motor_err[:]
        outs  = motor_out[:]
        dist  = distances[:]
        m_age = now - last_motor_ts if last_motor_ts else -1
        s_age = now - last_sens_ts  if last_sens_ts  else -1
        pkt   = pkts

    W = 66
    sep  = "─" * W
    dsep = "═" * W

    lines = [
        f"╔{dsep}╗",
        f"║{'MOTOR INDIVIDUAL TEST':^{W}}║",
        f"╠{dsep}╣",
        f"║  Active motor: M{active}  {MOTOR_NAMES[active]}   "
        f"Packets: {pkt:>5}  ({m_age:.1f}s ago){'':>3}║",
        f"║  ↑↓ speed  |  0–3 select motor  |  SPC stop  |  A stop all  |  Q quit  ║",
        f"╠{sep}╣",
        f"║  {'Motor':<16} {'Setpt':>6} {'RPM':>7} {'Error':>7} {'PWM':>6}  "
        f"{'RPM bar':<{BAR_CELLS+2}}  Dir  ║",
        f"╟{sep}╢",
    ]

    for i, name in enumerate(MOTOR_NAMES):
        marker = "►" if i == active else " "
        sp  = spds[i]
        rp  = rpms[i]
        er  = errs[i]
        ot  = outs[i]
        bar = rpm_bar(rp, sp)
        drn = direction_str(sp)
        lines.append(
            f"║{marker} {name}  {sp:>+6.0f} {rp:>7.1f} {er:>7.1f} {ot:>6.1f}  {bar}  {drn} ║"
        )

    lines += [
        f"╠{sep}╣",
        f"║  ULTRASONIC  ({s_age:.1f}s ago){'':>{W-22}}║",
    ]
    parts = []
    for label, d in zip(SENSOR_LABELS, dist):
        d_str = f"{d:5.1f}cm" if d >= 0 else " --- "
        parts.append(f"{label}:{d_str}")
    lines.append(f"║  {'   '.join(parts)}  ║")
    lines.append(f"╚{dsep}╝")

    sys.stdout.write('\033[H' + '\n'.join(lines) + '\n')
    sys.stdout.flush()


# ─────────────────────────── main ───────────────────────────

def main():
    global running

    print(f"Opening {PORT} @ {BAUD} baud …")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    log_writer, log_fh, log_path = open_log()

    # Launch threads
    t_serial = threading.Thread(
        target=serial_reader, args=(ser, log_writer, log_fh), daemon=True)
    t_serial.start()

    t_keys = threading.Thread(target=keyboard_reader, daemon=True)
    t_keys.start()

    # Clear screen once
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.write('\033[?25l')   # hide cursor
    sys.stdout.flush()

    interval = 1.0 / CMD_HZ

    try:
        while running:
            t0 = time.time()

            with state_lock:
                spds  = setpoints[:]
                am    = active_motor
                pkts  = packet_count

            ser.write(make_cmd(spds))
            draw(am, pkts)

            elapsed = time.time() - t0
            rem = interval - elapsed
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        pass

    finally:
        running = False

        # Emergency stop — send twice
        try:
            ser.write(make_cmd([0, 0, 0, 0]))
            time.sleep(0.05)
            ser.write(make_cmd([0, 0, 0, 0]))
        except Exception:
            pass

        log_fh.close()

        sys.stdout.write('\033[?25h\n')
        sys.stdout.flush()
        print(f"Stopped. Log saved → {log_path}")
        ser.close()


if __name__ == "__main__":
    main()
