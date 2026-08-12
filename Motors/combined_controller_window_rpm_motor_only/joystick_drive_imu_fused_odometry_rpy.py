#!/usr/bin/env python3
"""
joystick_drive_imu_fused_odometry.py

Joystick/gamepad control + IMU-fused odometry for the 4-wheel scooter.

Same IMU fusion, turning-scrub slip correction, and front/rear intra-side
slip correction as keyboard_drive_imu_fused_odometry.py -- see that
script's history for the full derivations. Only the control input source
changed: continuous analog joystick axes instead of discrete keyboard
step commands.

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

IMU protocol:
  Same binary Kalman-fused, tared-quaternion streaming protocol as
  rpy_node.py (10 Hz, wired response header = 4B timestamp + 1B datalen,
  streaming slot 0 = tared orientation quaternion), decoded via
  quaternion -> rotation matrix -> roll/pitch/yaw. Replaces the older
  ASCII ":1\n" polled-Euler command this script used previously -- that
  request/response round trip added latency and was more prone to stalls
  than the sensor pushing data on its own schedule. No axis-frame
  transform is applied (same as rpy_node.py's direct/natural-frame
  extraction); only yaw is consumed here, so any frame mismatch is
  absorbed by IMU_YAW_SIGN and the startup tare + collect_imu_zero
  rather than a full axis remap.

JOYSTICK CONTROL MAPPING:
  Continuous by default -- analog stick position maps proportionally to
  commanded RPM, same shape as your existing controller code's
  map_axis_to_range(): inside the deadzone -> 0, outside the deadzone ->
  scaled between a MIN and MAX RPM (the jump straight to MIN past the
  deadzone avoids commanding sub-stiction RPMs that just heat the motor
  without moving it, matching what you found with PID anti-windup at
  low RPM).

  Optional quantization (--linear-rpm-step / --turn-rpm-step, default 0
  = continuous/off) rounds the mapped command to the nearest N RPM, like
  the keyboard version's fixed step size. This is NOT recommended for
  normal driving -- it throws away the proportional control a joystick
  gives you -- but it's useful for calibration runs where you want a
  reproducible, steady commanded RPM level (e.g. slip-factor or
  track-width calibration) that's easier to hold steady than a free
  continuous stick position.

Controls:
  Linear axis   : forward/backward speed command (continuous or stepped)
  Angular axis  : turn command (continuous or stepped)
  Reset button  : reset odometry + re-zero IMU heading (optional, off by default)
  Panic button  : force both commands to zero while held (optional, off by default)
  Ctrl+C        : stop safely and quit (same as keyboard version)
"""

import argparse
import csv
import math
import struct
import sys
import threading
import time
import pygame
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

PORT = "/dev/ttyACM0"
BAUD = 115200
IMU_BAUD = 115200
IMU_YAW_SIGN = 1.0  # flip to -1.0 if IMU yaw sign is reversed vs turn_cmd
IMU_STALE_SEC = 0.35  # if no fresh IMU sample within this window, fall back
LOG_DIR = "logs"

WHEEL_DIAMETER_M = 0.185
WHEELBASE_M = 0.68
TRACK_WIDTH_M = 0.57  # fallback-only, see module docstring

MAX_WHEEL_RPM = 300.0
LINEAR_STEP_RPM = 20.0   # default quantization step if --linear-rpm-step is enabled (matches keyboard version's granularity)
TURN_STEP_RPM = 120.0     # default quantization step if --turn-rpm-step is enabled
CMD_HZ = 20.0

# --- joystick mapping defaults (RPM units, NOT m/s or rad/s -- unlike
#     your other controller code, this script commands wheel RPM directly) ---
JOYSTICK_AXIS_MAX = 32767
JOYSTICK_DEADZONE = 10000
JOYSTICK_HZ = 50.0
LINEAR_AXIS = 1
ANGULAR_AXIS = 0
LINEAR_RPM_MIN = 20.0    # jump straight to this once past deadzone -- tune to your motor's stiction threshold
LINEAR_RPM_MAX = 300.0   # tune to your desired top speed; side_rpm_from_linear_turn still clamps to MAX_WHEEL_RPM
TURN_RPM_MIN = 20.0
TURN_RPM_MAX = 200.0

MOTOR_NAMES = ["M0 Front Left", "M1 Front Right", "M2 Back Left", "M3 Back Right"]

state_lock = threading.Lock()
running = True

linear_cmd = 0.0
turn_cmd = 0.0
joystick_linear_raw = 0     # diagnostics: raw axis value that produced linear_cmd
joystick_angular_raw = 0    # diagnostics: raw axis value that produced turn_cmd

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


def map_axis_to_range(axis_value, min_val, max_val, deadzone):
    """Map a raw joystick axis value to +/-[min_val, max_val] RPM, 0 inside deadzone.

    Adapted from joystick_debug.py. axis_value is an integer roughly in
    [-32767, 32767]. Jumping straight to min_val once past the deadzone
    (rather than ramping from 0) avoids commanding sub-stiction RPMs that
    just heat the motor without producing motion -- tune min_val to your
    motor's actual dead-zone threshold.
    """
    if -deadzone <= axis_value <= deadzone:
        return 0.0

    abs_val = abs(axis_value)
    value = min_val + (max_val - min_val) * (abs_val - deadzone) / (JOYSTICK_AXIS_MAX - deadzone)
    value = min(value, max_val)

    return value if axis_value > 0 else -value


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


IMU_PACKET_LEN = 21  # 4B timestamp + 1B datalen + 16B tared quaternion (4 floats)


def imu_checksum(packet_bytes):
    return sum(packet_bytes) % 256


def imu_configure_sensor(ser):
    """Configure the TSS for binary streaming of the tared orientation
    quaternion at 10 Hz, then tare the current pose to zero.

    Identical command sequence to rpy_node.py's setup_streaming() +
    tare_orientation(): stop any previous streaming, Kalman filter fusion
    mode (highest-accuracy orientation), disable oversampling, wired
    response header (4B timestamp + 1B datalen), streaming slot 0 = tared
    orientation quaternion (cmd 0x00), 100 ms interval, start streaming,
    then tare. Cheap to send, safe if the sensor was already configured.
    """
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Stop any previous streaming
        ser.write(bytearray([0xF8, 0x56, 0x56]))

        # Kalman Filter Mode (param=1)
        ser.write(bytearray([0xF7, 0x7B, 0x01, 0x7C]))

        # Disable Oversampling
        over_data = [0x6A, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01]
        ser.write(bytearray([0xF7] + over_data + [imu_checksum(over_data)]))

        # Wired Response Header: Timestamp(4B) + DataLen(1B)
        header_data = [0xDD, 0x00, 0x00, 0x00, 0x42]
        ser.write(bytearray([0xF7] + header_data + [imu_checksum(header_data)]))

        # Streaming Slots: slot 0 = command 0x00 (tared orientation as quaternion)
        slots = [0x50, 0x00] + [0xFF] * 7
        ser.write(bytearray([0xF7] + slots + [imu_checksum(slots)]))

        # Streaming Timing (10 Hz -> 100,000 us interval)
        timing = struct.pack('>III', 100000, 0xFFFFFFFF, 0)
        timing_pkt = [0x52] + list(timing)
        ser.write(bytearray([0xF7] + timing_pkt + [imu_checksum(timing_pkt)]))

        # Start Streaming (with response header prepended)
        ser.write(bytearray([0xF9, 0x55, 0x55]))
        ser.reset_input_buffer()

        # Tare current orientation -> becomes roll=pitch=yaw=0
        ser.write(bytearray([0xF7, 0x60, 0x60]))
        ser.reset_input_buffer()
    except serial.SerialException:
        pass


def quat_to_rotation_matrix(qx, qy, qz, qw):
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5

    if n == 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        (2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)),
        (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)),
    )


def rotation_matrix_to_euler_zyx(R):
    """Standard aerospace/robotics roll-pitch-yaw extraction,
    assuming R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    sy = clamp(-R[2][0], -1.0, 1.0)
    pitch = math.asin(sy)
    roll = math.atan2(R[2][1], R[2][2])
    yaw = math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


def imu_reader(ser):
    """Continuously read tared-quaternion stream packets and maintain an
    unwrapped yaw heading.

    Same binary packet format/parsing as rpy_node.py's read_and_publish()
    (21-byte packets: 4B timestamp + 1B datalen==16 + 4 floats quaternion),
    just consumed here as a plain background-thread poll loop over
    ser.in_waiting instead of a ROS timer callback, and only yaw is kept
    (this script only needs heading, not full roll/pitch/yaw).
    """
    global imu_connected, imu_raw_yaw, imu_yaw_unwrapped
    global imu_last_raw_yaw, imu_last_ts, imu_yaw_rate, running

    rx_buffer = bytearray()

    while running:
        try:
            n = ser.in_waiting
            if n:
                rx_buffer.extend(ser.read(n))
        except Exception:
            time.sleep(0.005)
            continue

        while len(rx_buffer) >= IMU_PACKET_LEN:
            found = False

            for i in range(len(rx_buffer) - IMU_PACKET_LEN + 1):
                # packet check: data length byte must be 16 (quaternion payload)
                if rx_buffer[i + 4] == 16:
                    packet = rx_buffer[i:i + IMU_PACKET_LEN]

                    try:
                        unpacked = struct.unpack('>IBffff', packet)
                    except struct.error:
                        continue

                    found = True
                    del rx_buffer[:i + IMU_PACKET_LEN]

                    # Quaternion in sensor's natural frame (x, y, z, w order
                    # per protocol spec); no axis-frame transform applied,
                    # matching rpy_node.py -- see module docstring.
                    qx, qy, qz, qw = unpacked[2:6]

                    R = quat_to_rotation_matrix(qx, qy, qz, qw)
                    _roll, _pitch, yaw = rotation_matrix_to_euler_zyx(R)
                    yaw *= IMU_YAW_SIGN

                    now = time.time()
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

                    break

            if not found:
                del rx_buffer[0]

        time.sleep(0.002)


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


def joystick_reader(joystick, deadzone, linear_axis, angular_axis,
                     linear_rpm_min, linear_rpm_max, turn_rpm_min, turn_rpm_max,
                     linear_rpm_step, turn_rpm_step, hz,
                     reset_button, panic_button):
    """Continuously poll the joystick and update linear_cmd/turn_cmd.

    Unlike the keyboard version's discrete step-on-keypress model, this
    updates every loop from the current stick position -- proportional,
    continuous control by default. Optional quantization (linear_rpm_step/
    turn_rpm_step > 0) rounds to the nearest N RPM for calibration runs;
    see module docstring "JOYSTICK CONTROL MAPPING".
    """
    global running, linear_cmd, turn_cmd, joystick_linear_raw, joystick_angular_raw
    global odom_x, odom_y, odom_theta, odom_path, odom_left_dist, odom_right_dist
    global imu_theta_offset

    period = 1.0 / hz if hz > 0 else 0.02
    reset_held_prev = False

    while running:
        pygame.event.pump()

        num_axes = joystick.get_numaxes()
        num_buttons = joystick.get_numbuttons()

        lin_raw = int(joystick.get_axis(linear_axis) * JOYSTICK_AXIS_MAX) if linear_axis < num_axes else 0
        ang_raw = int(joystick.get_axis(angular_axis) * JOYSTICK_AXIS_MAX) if angular_axis < num_axes else 0

        # same sign convention as joystick_debug.py: negate raw axis before mapping
        lin_cmd_val = map_axis_to_range(-lin_raw, linear_rpm_min, linear_rpm_max, deadzone)
        turn_cmd_val = map_axis_to_range(-ang_raw, turn_rpm_min, turn_rpm_max, deadzone)

        if linear_rpm_step > 0:
            lin_cmd_val = round(lin_cmd_val / linear_rpm_step) * linear_rpm_step
        if turn_rpm_step > 0:
            turn_cmd_val = round(turn_cmd_val / turn_rpm_step) * turn_rpm_step

        panic = panic_button is not None and panic_button >= 0 and panic_button < num_buttons \
            and bool(joystick.get_button(panic_button))
        if panic:
            lin_cmd_val = 0.0
            turn_cmd_val = 0.0

        with state_lock:
            linear_cmd = lin_cmd_val
            turn_cmd = turn_cmd_val
            joystick_linear_raw = lin_raw
            joystick_angular_raw = ang_raw

        # edge-triggered reset: only fire once per press, not once per loop while held
        reset_held = reset_button is not None and reset_button >= 0 and reset_button < num_buttons \
            and bool(joystick.get_button(reset_button))
        if reset_held and not reset_held_prev:
            with state_lock:
                odom_x = odom_y = odom_theta = odom_path = 0.0
                odom_left_dist = odom_right_dist = 0.0
            with imu_lock:
                imu_theta_offset = imu_yaw_unwrapped
        reset_held_prev = reset_held

        time.sleep(period)


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


# ---------------------------------------------------------------------------
# Live heading/position map window (--show-imu)
# ---------------------------------------------------------------------------
# Runs entirely on the main thread, updated once per control loop iteration.
# Deliberately NOT a separate thread: the joystick_reader thread already
# calls pygame functions, and matplotlib's GUI calls (like pygame's) are not
# safe to make from multiple threads. Keeping this on the main thread avoids
# that entirely -- it just costs a small, bounded amount of time per loop
# for canvas redraw, similar in spirit to the existing terminal draw().

_live_map_state = None  # holds fig/ax/artists once initialized


def init_live_map():
    """Open a matplotlib window for live heading/position display.

    Returns None (with a printed warning) if matplotlib or a display isn't
    available, so callers can just skip live-map updates rather than crash
    a real test run over a visualization nicety.
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")  # fallback below if this backend can't load
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: --show-imu requested but matplotlib/display unavailable ({exc}). "
              f"Continuing without the live map window.")
        return None

    try:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Live heading + position")
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y left (m)")
        ax.set_aspect("equal", adjustable="box")  # "box" respects our explicit
        # set_xlim/set_ylim each frame; the default "datalim" mode would instead
        # override them using the full path history's bounding box, which can
        # push the current position outside the visible window.
        ax.grid(True, alpha=0.3)

        (path_line,) = ax.plot([], [], "-", color="#4C72B0", linewidth=1.5, label="path")
        (pos_dot,) = ax.plot([], [], "o", color="#C44E52", markersize=8)
        (heading_line,) = ax.plot([], [], "-", color="#C44E52", linewidth=2.5)
        heading_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                                va="top", ha="left", fontsize=10,
                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(loc="lower right")
        fig.canvas.draw()
        fig.canvas.flush_events()

        return {
            "plt": plt, "fig": fig, "ax": ax,
            "path_line": path_line, "pos_dot": pos_dot, "heading_line": heading_line,
            "heading_text": heading_text,
            "xs": [], "ys": [],
        }
    except Exception as exc:
        print(f"WARNING: --show-imu window failed to open ({exc}). "
              f"Continuing without the live map window.")
        return None


def update_live_map(state, x, y, theta_rad, heading_source, arrow_len=0.3):
    """Redraw the live map with the current position/heading. Cheap no-op if state is None."""
    if state is None:
        return

    state["xs"].append(x)
    state["ys"].append(y)
    state["path_line"].set_data(state["xs"], state["ys"])
    state["pos_dot"].set_data([x], [y])

    hx = x + arrow_len * math.cos(theta_rad)
    hy = y + arrow_len * math.sin(theta_rad)
    state["heading_line"].set_data([x, hx], [y, hy])

    state["heading_text"].set_text(f"heading: {deg_0_360(theta_rad):6.1f} deg  [{heading_source}]")

    # keep current position comfortably in view, following as the robot moves
    ax = state["ax"]
    pad = max(arrow_len * 2, 0.5)
    xs, ys = state["xs"], state["ys"]
    x_lo, x_hi = min(xs) - pad, max(xs) + pad
    y_lo, y_hi = min(ys) - pad, max(ys) + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    try:
        state["fig"].canvas.draw_idle()
        state["fig"].canvas.flush_events()
    except Exception:
        pass  # window was closed by the user; just stop updating silently


def close_live_map(state):
    if state is None:
        return
    try:
        state["plt"].close(state["fig"])
    except Exception:
        pass


def draw(left_cmd, right_cmd, left_est, right_est, v, omega, max_wheel_rpm):
    with state_lock:
        lin = linear_cmd
        trn = turn_cmd
        lin_raw = joystick_linear_raw
        ang_raw = joystick_angular_raw
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
        "JOYSTICK DRIVE + IMU-FUSED, SLIP-CORRECTED ODOMETRY",
        "-" * 92,
        f"linear={lin:+6.1f} rpm, turn={trn:+6.1f} rpm | raw_axes: lin={lin_raw:+6d} ang={ang_raw:+6d}"
        f" | left_side={left_cmd:+6.1f}, right_side={right_cmd:+6.1f}",
        f"Arduino command: {arduino}",
        "Controls: joystick axes = speed/turn | reset/panic buttons if configured | Ctrl+C quit",
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
    lines.append("SAFETY: First test lifted. On ground use small max-rpm settings and avoid sharp turns.")

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
    p.add_argument("--joystick-index", type=int, default=0, help="Joystick/controller index, default: 0")
    p.add_argument("--joystick-hz", type=float, default=JOYSTICK_HZ, help="Joystick poll rate, default: 50")
    p.add_argument("--deadzone", type=int, default=JOYSTICK_DEADZONE)
    p.add_argument("--linear-axis", type=int, default=LINEAR_AXIS, help="Axis used for forward/backward")
    p.add_argument("--angular-axis", type=int, default=ANGULAR_AXIS, help="Axis used for left/right turn")
    p.add_argument("--linear-rpm-min", type=float, default=LINEAR_RPM_MIN,
                   help="RPM commanded as soon as the stick leaves the deadzone (tune to motor stiction)")
    p.add_argument("--linear-rpm-max", type=float, default=LINEAR_RPM_MAX,
                   help="RPM commanded at full stick deflection")
    p.add_argument("--turn-rpm-min", type=float, default=TURN_RPM_MIN)
    p.add_argument("--turn-rpm-max", type=float, default=TURN_RPM_MAX)
    p.add_argument("--linear-rpm-step", type=float, default=0.0,
                   help="Quantize linear command to nearest N RPM; 0 = continuous (default, recommended for driving)")
    p.add_argument("--turn-rpm-step", type=float, default=0.0,
                   help="Quantize turn command to nearest N RPM; 0 = continuous (default, recommended for driving)")
    p.add_argument("--reset-button", type=int, default=-1,
                   help="Button index that resets odometry + re-zeros IMU heading; -1 = disabled (default)")
    p.add_argument("--panic-button", type=int, default=-1,
                   help="Button index that forces both commands to zero while held; -1 = disabled (default)")
    p.add_argument("--cmd-hz", type=float, default=CMD_HZ)
    p.add_argument("--wheel-diameter", type=float, default=WHEEL_DIAMETER_M)
    p.add_argument("--wheelbase", type=float, default=WHEELBASE_M)
    p.add_argument("--track-width", type=float, default=TRACK_WIDTH_M)
    p.add_argument("--log", action="store_true")
    p.add_argument("--show-imu", action="store_true",
                   help="Open a live window showing real-time heading + position during the test")
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

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("ERROR: No joystick/controller detected.")
        return 1
    if args.joystick_index >= count:
        print(f"ERROR: joystick index {args.joystick_index} not available. Detected {count} controller(s).")
        return 1
    joystick = pygame.joystick.Joystick(args.joystick_index)
    joystick.init()
    print(f"Joystick: {joystick.get_name()} | axes={joystick.get_numaxes()} "
          f"buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")
    print(f"Linear axis={args.linear_axis} range=[{args.linear_rpm_min:.1f},{args.linear_rpm_max:.1f}] rpm"
          f"{'  (quantized to ' + str(args.linear_rpm_step) + ' rpm)' if args.linear_rpm_step > 0 else '  (continuous)'}")
    print(f"Angular axis={args.angular_axis} range=[{args.turn_rpm_min:.1f},{args.turn_rpm_max:.1f}] rpm"
          f"{'  (quantized to ' + str(args.turn_rpm_step) + ' rpm)' if args.turn_rpm_step > 0 else '  (continuous)'}")

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
        csv_path = Path(LOG_DIR) / f"joystick_imu_fused_odometry_{ts}.csv"
        csv_file = open(csv_path, "w", newline="")
        fieldnames = ["time_s", "linear_cmd_rpm", "turn_cmd_rpm", "left_physical_cmd_rpm", "right_physical_cmd_rpm",
                      "joystick_linear_raw", "joystick_angular_raw"]
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
    threading.Thread(target=joystick_reader, args=(
        joystick, args.deadzone, args.linear_axis, args.angular_axis,
        args.linear_rpm_min, args.linear_rpm_max, args.turn_rpm_min, args.turn_rpm_max,
        args.linear_rpm_step, args.turn_rpm_step, args.joystick_hz,
        args.reset_button, args.panic_button,
    ), daemon=True).start()

    sys.stdout.write("\033[2J\033[H\033[?25l")
    sys.stdout.flush()

    live_map = init_live_map() if args.show_imu else None

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

            if live_map is not None:
                with state_lock:
                    x_now, y_now, th_now = odom_x, odom_y, odom_theta
                heading_src = "IMU" if odom_using_imu else "ENCODER-FALLBACK"
                update_live_map(live_map, x_now, y_now, th_now, heading_src)

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
                    lin_raw = joystick_linear_raw
                    ang_raw = joystick_angular_raw

                with imu_lock:
                    imu_ok = imu_connected and (time.time() - imu_last_ts) < IMU_STALE_SEC
                    raw_yaw_deg = deg_0_360(imu_raw_yaw) if imu_raw_yaw is not None else float("nan")

                row = {
                    "time_s": f"{loop_t - t0:.3f}",
                    "linear_cmd_rpm": f"{lin:.3f}",
                    "turn_cmd_rpm": f"{trn:.3f}",
                    "left_physical_cmd_rpm": f"{left_cmd:.3f}",
                    "right_physical_cmd_rpm": f"{right_cmd:.3f}",
                    "joystick_linear_raw": lin_raw,
                    "joystick_angular_raw": ang_raw,
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
        close_live_map(live_map)
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
        print("Motors stopped. Disconnected.")
        if csv_path:
            print(f"Log saved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
