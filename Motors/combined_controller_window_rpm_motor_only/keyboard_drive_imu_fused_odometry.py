#!/usr/bin/env python3
"""
keyboard_drive_imu_fused_odometry.py

Directional keyboard control + IMU-fused odometry for the 4-wheel scooter.

Fusion strategy:
  - STRAIGHT-LINE MOTION (v):  from Hall RPM / wheel encoders.
      v = 0.5 * (v_left_side + v_right_side)
  - ROTATION (theta):          from the TSS-USB-S IMU yaw (Euler ":1" command).
      This replaces the old RPM-difference / track-width omega estimate,
      which was biased by ~2.3x due to unmodeled skid-steer tire scrub.
      The IMU yaw is zeroed at start and unwrapped so it doesn't jump at +-pi.

  x, y are integrated using wheel-derived v and IMU-derived theta:
      x += v * cos(theta) * dt
      y += v * sin(theta) * dt

  The old encoder-based omega estimate is still computed and logged
  purely for comparison / diagnostics (e.g. to sanity check the IMU,
  or to re-derive an effective track width later).

  If the IMU is not connected or its telemetry goes stale mid-run, this
  falls back to the encoder-based omega estimate for that period so a
  run doesn't die because of a dropped IMU packet.

SLIP ESTIMATION (v_corrected):
  Skid-steer turning inherently requires tire scrub, which "wastes" some
  wheel rotation instead of converting it into motion. This means the
  Hall-RPM-implied forward speed is an overestimate whenever the vehicle
  is turning.

  We estimate a slip correction factor by comparing the Hall-kinematic
  omega (from RPM difference / track width) against the IMU-measured
  omega (ground truth rotation):

      k = omega_imu / omega_hall_kinematic

  Assuming slip is roughly symmetric between the left and right sides
  (a first-order approximation -- it will be weaker for very asymmetric
  turns, e.g. one side driven, one side near-stationary), the same
  fraction that suppresses measured rotation also suppresses measured
  forward speed:

      v_corrected = k * v_hall_raw

  k is clamped to [0, 1] (slip can only reduce effective motion in this
  model, never amplify it) and smoothed with a simple EMA to avoid noise
  spikes from dividing two small, noisy numbers. When the Hall-kinematic
  omega is near zero (straight-line driving, or no turn commanded), k
  defaults to 1.0 -- no correction -- since that's exactly the regime
  where your encoder-only odometry already tracked ground truth well.

  v_corrected (not v_hall_raw) is what's actually used for x,y
  integration. Both are logged so you can compare.

INTRA-SIDE (FRONT/REAR) SLIP:
  Separate from turning scrub, front and rear wheels on the SAME side
  should have identical forward-rolling velocity even while turning --
  only the lateral scrub differs front-to-rear. So any front-vs-rear RPM
  mismatch on one side is pure evidence of wheel slip (traction loss),
  untangled from the intentional L/R differential that produces turning.

  Per-side slip ratio (standard driven-wheel slip-ratio definition):
      fr_slip_ratio = (max(|F|,|R|) - min(|F|,|R|)) / max(|F|,|R|)

  A driven wheel losing traction spins FASTER than true ground speed,
  not slower, so the slower of the front/rear pair on a side is used as
  that side's speed estimate instead of averaging them -- averaging lets
  a spinning wheel bias the estimate upward. This is a heuristic (it
  assumes slip shows up as over-speed, not a dragging/bound wheel), but
  it's the dominant failure mode for driven wheels.

  This combines multiplicatively with the turning-scrub slip factor k
  above: v_true ~= k_turn * v_side_from_min(front, rear).

Geometry:
  wheel diameter = 0.185 m
  wheelbase      = 0.68  m  # logged for reference
  track width    = 0.57  m  # used ONLY for the fallback omega estimate

Sign convention (motors):
  Forward vehicle motion:
    M0 Front Left  = +
    M1 Front Right = -
    M2 Back Left   = +
    M3 Back Right  = -

Sign convention (IMU yaw):
  Positive yaw = CCW = left turn, matching positive turn_cmd = left turn.
  If your mounted IMU reports the opposite sign, flip IMU_YAW_SIGN below.

Controls:
  UP/DOWN     : increase/decrease linear forward command
  LEFT/RIGHT  : increase/decrease turn command
  C           : center steering command, turn = 0
  R           : reset odometry (also re-zeros IMU heading reference)
  SPACE       : stop
  Q or ESC    : quit safely
"""

import argparse
import csv
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

PORT = "/dev/ttyACM0"
BAUD = 115200
IMU_BAUD = 115200
IMU_COMMAND = "1"  # Euler angles: pitch, yaw, roll (rad)
IMU_YAW_SIGN = 1.0  # flip to -1.0 if IMU yaw sign is reversed vs turn_cmd
IMU_STALE_SEC = 0.35  # if no fresh IMU sample within this window, fall back
LOG_DIR = "logs"

WHEEL_DIAMETER_M = 0.185
WHEELBASE_M = 0.68
TRACK_WIDTH_M = 0.57  # fallback-only, see module docstring

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

# --- IMU state ---
imu_lock = threading.Lock()
imu_connected = False
imu_raw_yaw = 0.0          # latest raw yaw from IMU, radians, wrapped [-pi, pi]
imu_yaw_unwrapped = 0.0    # continuously unwrapped yaw since thread start
imu_last_raw_yaw = None    # for unwrap tracking
imu_last_ts = 0.0          # time.time() of last good IMU sample
imu_yaw_rate = 0.0         # simple derivative, rad/s, for diagnostics

# --- odometry state ---
odom_x = 0.0
odom_y = 0.0
odom_theta = 0.0            # fused heading actually used for x,y integration
odom_path = 0.0
odom_left_dist = 0.0
odom_right_dist = 0.0
last_odom_t = None
last_left_sign = 0.0
last_right_sign = 0.0
imu_theta_offset = 0.0      # zeroing offset applied to imu_yaw_unwrapped
odom_omega_encoder_est = 0.0  # diagnostics-only, from RPM difference
odom_using_imu = False

# --- slip estimation state ---
# Symmetric-slip model: the same fraction of "wasted" wheel rotation that
# suppresses measured turning (Hall-kinematic omega vs IMU-measured omega)
# is assumed to also suppress forward speed. See module docstring section
# "SLIP ESTIMATION" below for the derivation and its limits.
SLIP_OMEGA_THRESHOLD = 0.03   # rad/s; below this, treat as "not really turning" -> no correction
SLIP_K_EMA_ALPHA = 0.25       # smoothing factor for the slip correction factor (0..1, higher = less smoothing)
odom_slip_k = 1.0             # current (smoothed) slip correction factor, 1.0 = no slip
odom_slip_k_raw = 1.0         # unsmoothed instantaneous slip factor, for diagnostics
odom_v_hall_raw = 0.0         # forward speed straight from Hall RPM, uncorrected
odom_v_corrected = 0.0        # forward speed after slip correction, used for x,y integration

# --- intra-side (front/rear) slip diagnostics ---
odom_fr_slip_ratio_left = 0.0
odom_fr_slip_ratio_right = 0.0
odom_left_mag_avg = 0.0   # old average-based magnitude, kept for comparison/logging
odom_right_mag_avg = 0.0
odom_left_mag_min = 0.0   # min-based magnitude actually used for v_hall_raw
odom_right_mag_min = 0.0


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


def deg_0_360(rad):
    """Radians -> degrees, wrapped to [0, 360) instead of (-180, 180].

    Internal heading math (odom_theta, imu_yaw_unwrapped, etc.) stays in
    signed radians via wrap_pi -- that's fine for cos/sin. This is only
    for human-facing display/logging, so crossing due north/whatever-zero
    reads as e.g. 359 -> 0 -> 1 instead of 179 -> -180 -> -179.
    """
    return math.degrees(rad) % 360.0


# ---------------------------------------------------------------------------
# IMU handling
# ---------------------------------------------------------------------------

def find_tss_port(exclude=None):
    exclude = exclude or set()
    ports = list(serial.tools.list_ports.comports())

    for p in ports:
        if p.device in exclude:
            continue
        text = f"{p.device} {p.description} {p.hwid}".lower()
        if "2476:1010" in text or "vid:pid=2476:1010" in text:
            return p.device

    for p in ports:
        if p.device in exclude:
            continue
        text = f"{p.device} {p.description} {p.hwid}".lower()
        if "yost" in text or "yei" in text or "2476" in text:
            return p.device

    return None


def find_motor_port(exclude=None):
    exclude = exclude or set()
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.device in exclude:
            continue
        if "ttyACM" in p.device or "ttyUSB" in p.device or "usbmodem" in p.device.lower():
            return p.device
    return None


def imu_configure_sensor(ser):
    """Stop any previous streaming and disable wired response headers, if supported.

    If the sensor was left in a streaming state from a prior session (or a
    crashed script), unsolicited data can silently corrupt every subsequent
    request/response read. Cheap to send, safe if unsupported.
    """
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(b":86\n")       # stop streaming, if active
        ser.flush()
        time.sleep(0.05)
        ser.reset_input_buffer()
        ser.write(b":221,0\n")    # disable wired response header, if supported
        ser.flush()
        time.sleep(0.05)
        ser.reset_input_buffer()
    except serial.SerialException:
        pass


def imu_read_cmd(ser, cmd, expected_count):
    ser.reset_input_buffer()
    ser.write(f":{cmd}\n".encode("ascii"))
    raw = ser.readline()
    if not raw:
        return None
    text = raw.decode("ascii", errors="replace").strip()
    try:
        values = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        return None
    if len(values) < expected_count:
        return None
    return values[:expected_count]


def imu_reader(ser):
    """Continuously poll IMU yaw and maintain an unwrapped heading."""
    global imu_connected, imu_raw_yaw, imu_yaw_unwrapped
    global imu_last_raw_yaw, imu_last_ts, imu_yaw_rate, running

    while running:
        try:
            values = imu_read_cmd(ser, IMU_COMMAND, 3)
        except Exception:
            values = None

        now = time.time()
        if values is not None:
            _pitch, yaw, _roll = values
            yaw *= IMU_YAW_SIGN

            with imu_lock:
                if imu_last_raw_yaw is None:
                    imu_last_raw_yaw = yaw
                    dt = None
                else:
                    delta = wrap_pi(yaw - imu_last_raw_yaw)
                    imu_yaw_unwrapped += delta
                    dt = now - imu_last_ts if imu_last_ts else None
                    imu_last_raw_yaw = yaw

                if dt and dt > 0:
                    imu_yaw_rate = wrap_pi(yaw - (imu_raw_yaw if imu_raw_yaw is not None else yaw)) / dt

                imu_raw_yaw = yaw
                imu_last_ts = now
                imu_connected = True

        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Motor serial + kinematics helpers
# ---------------------------------------------------------------------------

def side_rpm_from_linear_turn(linear_rpm, turn_rpm, max_wheel_rpm):
    left = linear_rpm - turn_rpm
    right = linear_rpm + turn_rpm

    max_abs = max(abs(left), abs(right), 1e-9)
    if max_abs > max_wheel_rpm:
        scale = max_wheel_rpm / max_abs
        left *= scale
        right *= scale

    return left, right


def motor_cmd_from_side(left_side_rpm, right_side_rpm):
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


def collect_imu_zero(num_samples, sample_dt=0.02):
    """Average several IMU yaw samples for a heading zero reference.

    A single-sample zero is vulnerable to one noisy read setting a biased
    reference for the whole run. Averaging several samples (with the
    sensor held still, as prompted) is much more robust -- same idea as
    read_tss_quat_heading_v1_csv.py's collect_zero(), simplified since we
    only need Euler yaw here and the sensor is assumed stationary during
    this short window (so plain averaging of the unwrapped yaw is safe,
    no wraparound handling needed).
    """
    print(f"Collecting {num_samples} IMU zero samples -- keep the sensor still...")
    samples = []
    deadline = time.time() + max(num_samples * sample_dt * 3, 2.0)
    while len(samples) < num_samples and time.time() < deadline:
        with imu_lock:
            if imu_connected and imu_last_raw_yaw is not None:
                samples.append(imu_yaw_unwrapped)
        time.sleep(sample_dt)

    if not samples:
        print("WARNING: no IMU samples collected for zeroing; heading will start unzeroed.")
        with imu_lock:
            return imu_yaw_unwrapped

    zero = sum(samples) / len(samples)
    print(f"IMU heading zero reference set from {len(samples)} samples "
          f"(spread {math.degrees(max(samples) - min(samples)):.2f} deg)")
    return zero


def keyboard_reader(max_wheel_rpm, linear_step, turn_step):
    global running, linear_cmd, turn_cmd
    global odom_x, odom_y, odom_theta, odom_path, odom_left_dist, odom_right_dist
    global imu_theta_offset

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
                        if ch3 == b"A":
                            linear_cmd = clamp(linear_cmd + linear_step, -max_wheel_rpm, max_wheel_rpm)
                        elif ch3 == b"B":
                            linear_cmd = clamp(linear_cmd - linear_step, -max_wheel_rpm, max_wheel_rpm)
                        elif ch3 == b"D":
                            turn_cmd = clamp(turn_cmd + turn_step, -max_wheel_rpm, max_wheel_rpm)
                        elif ch3 == b"C":
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
                with imu_lock:
                    imu_theta_offset = imu_yaw_unwrapped

            elif ch in (b"q", b"Q"):
                running = False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def update_odometry(now, wheel_circ, track_width, left_cmd, right_cmd):
    global odom_x, odom_y, odom_theta, odom_path, odom_left_dist, odom_right_dist
    global last_odom_t, last_left_sign, last_right_sign
    global odom_omega_encoder_est, odom_using_imu
    global odom_slip_k, odom_slip_k_raw, odom_v_hall_raw, odom_v_corrected
    global odom_fr_slip_ratio_left, odom_fr_slip_ratio_right
    global odom_left_mag_avg, odom_right_mag_avg, odom_left_mag_min, odom_right_mag_min

    with state_lock:
        r0, r1, r2, r3 = rpm_m[:]

    if last_odom_t is None:
        last_odom_t = now
        return 0.0, 0.0, 0.0, 0.0

    dt = now - last_odom_t
    last_odom_t = now
    if dt <= 0 or dt > 0.5:
        dt = 0.0

    left_front_mag = abs(r0)
    left_rear_mag = abs(r2)
    right_front_mag = abs(r1)
    right_rear_mag = abs(r3)

    left_mag = min(left_front_mag, left_rear_mag)   # slip-robust: slower wheel is more trustworthy
    right_mag = min(right_front_mag, right_rear_mag)
    odom_left_mag_min = left_mag
    odom_right_mag_min = right_mag

    odom_left_mag_avg = 0.5 * (left_front_mag + left_rear_mag)   # old average, kept for comparison
    odom_right_mag_avg = 0.5 * (right_front_mag + right_rear_mag)

    left_max = max(left_front_mag, left_rear_mag, 1e-9)
    right_max = max(right_front_mag, right_rear_mag, 1e-9)
    odom_fr_slip_ratio_left = (left_max - left_mag) / left_max
    odom_fr_slip_ratio_right = (right_max - right_mag) / right_max

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

    # encoder-based omega, kept only as a diagnostic / fallback
    omega_encoder = (v_r - v_l) / track_width if track_width > 1e-9 else 0.0
    odom_omega_encoder_est = omega_encoder

    # --- pick heading source ---
    with imu_lock:
        imu_ok = imu_connected and (time.time() - imu_last_ts) < IMU_STALE_SEC
        theta_imu = wrap_pi(imu_yaw_unwrapped - imu_theta_offset) if imu_ok else None

    if theta_imu is not None:
        theta_new = theta_imu
        theta_mid = 0.5 * (wrap_pi(odom_theta) + wrap_pi(theta_new))
        # guard against wrap discontinuity in the midpoint average
        if abs(wrap_pi(theta_new) - wrap_pi(odom_theta)) > math.pi:
            theta_mid = theta_new
        omega_used = wrap_pi(theta_new - odom_theta) / dt if dt > 0 else 0.0
        odom_theta = theta_new
        odom_using_imu = True
    else:
        # fallback: integrate encoder-based omega like the old script
        omega_used = omega_encoder
        theta_mid = odom_theta + 0.5 * omega_used * dt
        odom_theta = wrap_pi(odom_theta + omega_used * dt)
        odom_using_imu = False

    odom_v_hall_raw = v

    # --- slip factor: only estimable when we have an independent IMU
    #     rotation measurement AND the Hall-kinematic omega is large
    #     enough to divide by safely. Otherwise assume no slip (k=1). ---
    # BUG FIX: when omega_used and omega_encoder disagree in SIGN, the ratio
    # is negative and clamp(...,0,1) forces it to exactly 0 -- completely
    # zeroing out v_corrected even when v_hall_raw shows real motion. Sign
    # disagreement happens often near-straight (both estimates are noise-
    # dominated when true omega is small, e.g. one wheel's friction making
    # Hall data look like turning when it barely is) -- that's a signal the
    # correction isn't reliable right now, not evidence of 100% slip.
    # Fall back to no correction (k=1.0) instead of a spurious hard zero.
    if odom_using_imu and abs(omega_encoder) > SLIP_OMEGA_THRESHOLD:
        same_sign = (omega_used >= 0) == (omega_encoder >= 0)
        if same_sign:
            k_raw = omega_used / omega_encoder  # omega_used == omega_imu in this branch
            k_raw = clamp(k_raw, 0.0, 1.0)
        else:
            k_raw = 1.0
    else:
        k_raw = 1.0
    odom_slip_k_raw = k_raw
    # exponential smoothing to avoid noise spikes from dividing two small numbers
    odom_slip_k = odom_slip_k + SLIP_K_EMA_ALPHA * (k_raw - odom_slip_k)

    v_corrected = odom_slip_k * v
    odom_v_corrected = v_corrected

    odom_x += v_corrected * math.cos(theta_mid) * dt
    odom_y += v_corrected * math.sin(theta_mid) * dt
    odom_path += abs(v_corrected) * dt
    odom_left_dist += v_l * dt
    odom_right_dist += v_r * dt

    return left_rpm, right_rpm, v_corrected, omega_used


def draw(left_cmd, right_cmd, left_est, right_est, v, omega, max_wheel_rpm):
    with state_lock:
        lin = linear_cmd
        trn = turn_cmd
        rpm = rpm_m[:]
        cmds = cmd_m[:]
        x = odom_x
        y = odom_y
        th = odom_theta
        path = odom_path
        age = time.time() - last_motor_ts if last_motor_ts else -1

    with imu_lock:
        imu_ok = imu_connected and (time.time() - imu_last_ts) < IMU_STALE_SEC
        imu_age = time.time() - imu_last_ts if imu_last_ts else -1
        raw_yaw_deg = deg_0_360(imu_raw_yaw) if imu_raw_yaw is not None else float("nan")

    age_s = f"{age:.1f}s" if 0 <= age < 60 else "waiting"
    imu_age_s = f"{imu_age:.2f}s" if 0 <= imu_age < 60 else "waiting"
    arduino = f"SP#{fmt(cmds[0])}#{fmt(cmds[1])}#{fmt(cmds[2])}#{fmt(cmds[3])}"
    heading_source = "IMU" if odom_using_imu else "ENCODER-FALLBACK"

    lines = [
        "DIRECTIONAL KEYBOARD DRIVE + IMU-FUSED ODOMETRY",
        "-" * 92,
        f"linear={lin:+6.1f} rpm, turn={trn:+6.1f} rpm | left_side={left_cmd:+6.1f}, right_side={right_cmd:+6.1f}",
        f"Arduino command: {arduino}",
        "Controls: UP/DOWN linear | LEFT/RIGHT turn | C center | R reset odom+IMU zero | SPACE stop | Q/ESC quit",
        "-" * 92,
        f"ODOM: x={x:+7.3f} m, y={y:+7.3f} m, theta={deg_0_360(th):6.1f} deg [{heading_source}], path={path:7.3f} m",
        f"      v_corrected={v:+7.3f} m/s, v_hall_raw={odom_v_hall_raw:+7.3f} m/s, slip_k={odom_slip_k:5.2f} (raw={odom_slip_k_raw:5.2f})",
        f"      omega_used={omega:+7.3f} rad/s, encoder_omega_est={odom_omega_encoder_est:+7.3f} rad/s",
        f"      FR slip ratio: left={odom_fr_slip_ratio_left:5.2f} right={odom_fr_slip_ratio_right:5.2f}"
        f"  (0=matched, higher=more front/rear traction mismatch)",
        f"IMU:  connected={imu_connected} fresh={imu_ok} raw_yaw={raw_yaw_deg:+7.2f} deg, age={imu_age_s}, tel={age_s}",
        "-" * 92,
        f"{'Motor':<16} {'Cmd':>8} {'RPM':>8}",
    ]
    for i, name in enumerate(MOTOR_NAMES):
        lines.append(f"{name:<16} {cmds[i]:>8.1f} {rpm[i]:>8.1f}")
    lines.append("-" * 92)
    lines.append("SAFETY: First test lifted. On ground use small max-wheel-rpm and avoid sharp turns.")

    sys.stdout.write("\033[H" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=PORT, help="Arduino motor controller serial port")
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument("--imu-port", default=None, help="TSS IMU serial port (auto-detected if omitted)")
    p.add_argument("--imu-baud", type=int, default=IMU_BAUD)
    p.add_argument("--imu", choices=["yes", "no"], default="yes",
                   help="Use the IMU for heading fusion (yes, default) or run encoder-only (no, e.g. IMU not attached)")
    p.add_argument("--imu-zero-samples", type=int, default=30,
                   help="Number of IMU yaw samples averaged at startup for the heading zero reference")
    p.add_argument("--slip-omega-threshold", type=float, default=SLIP_OMEGA_THRESHOLD,
                   help="rad/s; below this Hall-kinematic omega, skip slip correction (treat as straight-line)")
    p.add_argument("--slip-ema-alpha", type=float, default=SLIP_K_EMA_ALPHA,
                   help="Smoothing factor (0-1) for the slip correction factor; higher = less smoothing")
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
    global running, cmd_m, imu_theta_offset
    global SLIP_OMEGA_THRESHOLD, SLIP_K_EMA_ALPHA

    args = parse_args()
    SLIP_OMEGA_THRESHOLD = args.slip_omega_threshold
    SLIP_K_EMA_ALPHA = args.slip_ema_alpha
    wheel_circ = math.pi * args.wheel_diameter

    print(f"Motor port: {args.port} @ {args.baud}")
    print(f"Geometry: wheel_diameter={args.wheel_diameter:.3f} m, wheelbase={args.wheelbase:.3f} m, track_width(fallback only)={args.track_width:.3f} m")
    print("Forward signs: M0 +, M1 -, M2 +, M3 -")
    print("Positive turn command = left/CCW turn. IMU yaw fused as heading when fresh.")

    if not args.yes:
        ans = input("Type LIFTED to confirm wheels are lifted for first directional test: ").strip()
        if ans != "LIFTED":
            print("Aborted.")
            return 2

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as exc:
        print(f"ERROR opening motor port: {exc}")
        return 1

    time.sleep(2.0)
    ser.reset_input_buffer()
    safe_stop(ser)

    ser_imu = None
    if args.imu == "yes":
        imu_port = args.imu_port
        if imu_port is None:
            imu_port = find_tss_port(exclude={args.port})
        if imu_port is None:
            print("WARNING: no IMU port found/specified. Continuing with encoder-only omega fallback.")
        else:
            try:
                ser_imu = serial.Serial(imu_port, args.imu_baud, timeout=0.5, write_timeout=0.5)
                time.sleep(0.5)
                imu_configure_sensor(ser_imu)
                print(f"IMU port: {imu_port} @ {args.imu_baud}")
            except serial.SerialException as exc:
                print(f"WARNING: could not open IMU port {imu_port}: {exc}. Falling back to encoder-only omega.")
                ser_imu = None

    csv_file = None
    writer = None
    csv_path = None
    if args.log:
        Path(LOG_DIR).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(LOG_DIR) / f"keyboard_imu_fused_odometry_{ts}.csv"
        csv_file = open(csv_path, "w", newline="")
        fieldnames = ["time_s", "linear_cmd_rpm", "turn_cmd_rpm", "left_physical_cmd_rpm", "right_physical_cmd_rpm"]
        fieldnames += [f"cmd_m{i}" for i in range(4)]
        fieldnames += [f"rpm_m{i}" for i in range(4)]
        fieldnames += [f"pwm_m{i}" for i in range(4)]
        fieldnames += ["left_rpm_est", "right_rpm_est", "v_hall_raw_mps", "v_corrected_mps",
                       "slip_k", "slip_k_raw", "fr_slip_ratio_left", "fr_slip_ratio_right",
                       "left_mag_min_rpm", "left_mag_avg_rpm", "right_mag_min_rpm", "right_mag_avg_rpm",
                       "omega_used_radps", "omega_encoder_est_radps",
                       "x_m", "y_m", "theta_rad", "theta_deg", "path_m", "left_dist_m", "right_dist_m",
                       "imu_connected", "imu_fresh", "imu_raw_yaw_deg", "heading_source",
                       "wheel_diameter_m", "wheelbase_m", "track_width_m"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Logging to {csv_path}")

    threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()
    if ser_imu is not None:
        threading.Thread(target=imu_reader, args=(ser_imu,), daemon=True).start()
        time.sleep(0.2)  # let the IMU thread get its first sample flowing
        imu_theta_offset = collect_imu_zero(args.imu_zero_samples)
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
                    pwm = pwm_m[:]
                    x = odom_x
                    y = odom_y
                    th = odom_theta
                    path = odom_path
                    ld = odom_left_dist
                    rd = odom_right_dist

                with imu_lock:
                    imu_ok = imu_connected and (time.time() - imu_last_ts) < IMU_STALE_SEC
                    raw_yaw_deg = deg_0_360(imu_raw_yaw) if imu_raw_yaw is not None else float("nan")

                row = {
                    "time_s": f"{loop_t - t0:.3f}",
                    "linear_cmd_rpm": f"{lin:.3f}",
                    "turn_cmd_rpm": f"{trn:.3f}",
                    "left_physical_cmd_rpm": f"{left_cmd:.3f}",
                    "right_physical_cmd_rpm": f"{right_cmd:.3f}",
                    "left_rpm_est": f"{left_est:.3f}",
                    "right_rpm_est": f"{right_est:.3f}",
                    "v_hall_raw_mps": f"{odom_v_hall_raw:.6f}",
                    "v_corrected_mps": f"{v:.6f}",
                    "slip_k": f"{odom_slip_k:.4f}",
                    "slip_k_raw": f"{odom_slip_k_raw:.4f}",
                    "fr_slip_ratio_left": f"{odom_fr_slip_ratio_left:.4f}",
                    "fr_slip_ratio_right": f"{odom_fr_slip_ratio_right:.4f}",
                    "left_mag_min_rpm": f"{odom_left_mag_min:.3f}",
                    "left_mag_avg_rpm": f"{odom_left_mag_avg:.3f}",
                    "right_mag_min_rpm": f"{odom_right_mag_min:.3f}",
                    "right_mag_avg_rpm": f"{odom_right_mag_avg:.3f}",
                    "omega_used_radps": f"{omega:.6f}",
                    "omega_encoder_est_radps": f"{odom_omega_encoder_est:.6f}",
                    "x_m": f"{x:.6f}",
                    "y_m": f"{y:.6f}",
                    "theta_rad": f"{th:.6f}",
                    "theta_deg": f"{deg_0_360(th):.3f}",
                    "path_m": f"{path:.6f}",
                    "left_dist_m": f"{ld:.6f}",
                    "right_dist_m": f"{rd:.6f}",
                    "imu_connected": imu_connected,
                    "imu_fresh": imu_ok,
                    "imu_raw_yaw_deg": f"{raw_yaw_deg:.3f}",
                    "heading_source": "IMU" if odom_using_imu else "ENCODER",
                    "wheel_diameter_m": f"{args.wheel_diameter:.6f}",
                    "wheelbase_m": f"{args.wheelbase:.6f}",
                    "track_width_m": f"{args.track_width:.6f}",
                }
                for i in range(4):
                    row[f"cmd_m{i}"] = f"{motor_cmds[i]:.3f}"
                    row[f"rpm_m{i}"] = f"{rpm[i]:.3f}"
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
        if ser_imu is not None:
            try:
                ser_imu.close()
            except Exception:
                pass
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
