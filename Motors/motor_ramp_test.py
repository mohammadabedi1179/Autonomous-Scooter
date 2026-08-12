"""
Gradual ramp-up motor test for controller_PID.ino

Instead of jumping straight to a target RPM, this ramps the setpoint
up step by step, holds at the target, then ramps back down to zero.
Gentler on the motor/gearbox and easier to watch behave on first test.
"""

import serial
import time

# ============== CONFIG - EDIT THESE ==============
PORT = '/dev/ttyACM0'      # Windows: e.g. 'COM5'
BAUD = 115200

MOTOR_INDEX = 3             # which motor to test: 0, 1, 2, or 3
TARGET_RPM = 60              # final RPM to ramp up to
RAMP_STEP = 5                 # RPM increase per step
STEP_DELAY = 0.3              # seconds between ramp steps
HOLD_TIME = 3.0               # seconds to hold at target RPM
# ===================================================


def fmt(rpm):
    sign = '+' if rpm >= 0 else '-'
    return f"{sign}{int(abs(rpm)):03d}"


def make_cmd(rpms):
    """rpms = list of 4 RPM values for motors 0..3"""
    return f"SP#{fmt(rpms[0])}#{fmt(rpms[1])}#{fmt(rpms[2])}#{fmt(rpms[3])}\n"


def send(ser, rpms):
    ser.write(make_cmd(rpms).encode())


def print_telemetry(ser):
    """Print any telemetry lines the Arduino has sent (rpm,error,out)."""
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print("  >", line)


def ramp(ser, motor_index, start_rpm, end_rpm, step, delay):
    """Ramp one motor's setpoint from start_rpm to end_rpm, others held at 0."""
    rpms = [0, 0, 0, 0]
    direction = 1 if end_rpm >= start_rpm else -1
    current = start_rpm

    while (direction == 1 and current < end_rpm) or (direction == -1 and current > end_rpm):
        current += direction * step
        if (direction == 1 and current > end_rpm) or (direction == -1 and current < end_rpm):
            current = end_rpm

        rpms[motor_index] = current
        send(ser, rpms)
        print(f"setpoint -> {current} RPM", flush=True)
        time.sleep(delay)
        print("  checking telemetry...", flush=True)
        print_telemetry(ser)
        print("  telemetry check done", flush=True)

    return rpms


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)  # allow Arduino to reset after serial connect
    print(f"Connected to {PORT}. Testing motor {MOTOR_INDEX}.\n")

    try:
        print("--- Ramping up ---")
        rpms = ramp(ser, MOTOR_INDEX, 0, TARGET_RPM, RAMP_STEP, STEP_DELAY)

        print(f"\n--- Holding at {TARGET_RPM} RPM for {HOLD_TIME}s ---")
        t_end = time.time() + HOLD_TIME
        while time.time() < t_end:
            send(ser, rpms)  # keep sending so the watchdog/timeout doesn't kick in
            print_telemetry(ser)
            time.sleep(0.2)

        print("\n--- Ramping down ---")
        ramp(ser, MOTOR_INDEX, TARGET_RPM, 0, RAMP_STEP, STEP_DELAY)

        print("\nDone. Motor should be stopped.")

    except KeyboardInterrupt:
        print("\nInterrupted - stopping motor.")

    finally:
        send(ser, [0, 0, 0, 0])  # always stop on exit
        time.sleep(0.1)
        ser.close()


if __name__ == "__main__":
    main()
