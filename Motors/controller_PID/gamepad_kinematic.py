import pygame
import serial
import time
import math

# SERIAL CONFIG
PORT = '/dev/ttyACM0'
BAUD = 115200
SEND_HZ = 20
SEND_INTERVAL = 1.0 / SEND_HZ
STOP_CMD = 'SP#+000#-000#+000#-000\n'

# ROBOT PARAMETERS
wheel_radius = 0.19 / 2
track_width  = 0.56
V_MIN = 0.2
V_MAX = 0.6
OMEGA_MIN = 0.7
OMEGA_MAX = 1.5
DEADZONE = 10000
AXIS_MAX = 32767

# INIT SERIAL
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)

# INIT JOYSTICK
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No joystick connected")
    exit()
joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Controller:", joystick.get_name())

def map_axis_to_range(axis_value, min_val, max_val):
    if -DEADZONE <= axis_value <= DEADZONE:
        return 0.0
    abs_val = abs(axis_value)
    value = min_val + (max_val - min_val) * (abs_val - DEADZONE) / (AXIS_MAX - DEADZONE)
    value = min(value, max_val)
    return value if axis_value > 0 else -value

def skid_steer_drive(v, omega):
    v_r = v + (omega * track_width / 2.0)
    v_l = v - (omega * track_width / 2.0)
    rpm_r = v_r / (2 * math.pi * wheel_radius) * 60.0
    rpm_l = v_l / (2 * math.pi * wheel_radius) * 60.0
    return rpm_r, rpm_l

def rpm_to_cmd(rpm_l, rpm_r):
    base_sign_l = 1
    base_sign_r = -1
    final_sign_l = base_sign_l * (1 if rpm_l >= 0 else -1)
    final_sign_r = base_sign_r * (1 if rpm_r >= 0 else -1)
    rpm_l_abs = int(abs(rpm_l))
    rpm_r_abs = int(abs(rpm_r))
    sign_l_char = '+' if final_sign_l >= 0 else '-'
    sign_r_char = '+' if final_sign_r >= 0 else '-'
    return f'SP#{sign_l_char}{rpm_l_abs:03d}#{sign_r_char}{rpm_r_abs:03d}#{sign_l_char}{rpm_l_abs:03d}#{sign_r_char}{rpm_r_abs:03d}\n'

last_send_time = 0

try:
    while True:
        pygame.event.pump()
        axis_linear  = int(joystick.get_axis(1) * AXIS_MAX)
        axis_angular = int(joystick.get_axis(0) * AXIS_MAX)

        v = map_axis_to_range(-axis_linear, V_MIN, V_MAX)
        omega = map_axis_to_range(-axis_angular, OMEGA_MIN, OMEGA_MAX)

        if v == 0.0 and omega == 0.0:
            current_cmd = STOP_CMD
        else:
            rpm_r, rpm_l = skid_steer_drive(v, omega)
            current_cmd = rpm_to_cmd(rpm_l, rpm_r)

        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            ser.write(current_cmd.encode())
            last_send_time = now

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nStopping...")
    ser.write(STOP_CMD.encode())
    ser.close()
    pygame.quit()
