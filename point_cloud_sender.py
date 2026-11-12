import numpy as np
import cv2 as cv
import time
from picamera2 import Picamera2
import json
from libcamera import Transform
import sys
import socket
import pickle
import struct

# --- Network Configuration ---
# Use the IP address of your laptop. You can get this from your laptop's network settings.
# For example, on macOS, run `ifconfig | grep inet`
# On Windows, run `ipconfig`
# On Linux, run `hostname -I` (or similar)
LAPTOP_IP = '10.42.0.236'# <-- **CHANGE THIS TO YOUR LAPTOP'S IP ADDRESS**-->
PORT = 8000

# --- Load Calibration Parameters ---
try:
    with open(sys.path[0]+'/Calibration/stereo_calibration_params.json', 'r') as f:
        calibration_params = json.load(f)
    print("Calibration parameters loaded successfully.")
except FileNotFoundError:
    print("Error: 'stereo_calibration_params.json' not found. Please run the calibration script first.")
    exit()

M1 = np.array(calibration_params['M1'])
D1 = np.array(calibration_params['D1'])
M2 = np.array(calibration_params['M2'])
D2 = np.array(calibration_params['D2'])
R1 = np.array(calibration_params['R1'])
P1 = np.array(calibration_params['P1'])
R2 = np.array(calibration_params['R2'])
P2 = np.array(calibration_params['P2'])
Q = np.array(calibration_params['Q'])

# --- picamera2 setup for both cameras ---
picam2_left = Picamera2(0)
picam2_right = Picamera2(1)
transform = Transform(vflip=1)

preview_config_left = picam2_left.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}, transform=transform)
picam2_left.configure(preview_config_left)

preview_config_right = picam2_right.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}, transform=transform)
picam2_right.configure(preview_config_right)

picam2_left.start()
picam2_right.start()

time.sleep(2)

# --- Compute rectification maps ---
size = (640, 480)
mapL1, mapL2 = cv.initUndistortRectifyMap(M1, D1, R1, P1, size, cv.CV_32FC1)
mapR1, mapR2 = cv.initUndistortRectifyMap(M2, D2, R2, P2, size, cv.CV_32FC1)

# --- Create StereoSGBM object for disparity calculation ---
window_size = 3
min_disparity = 0
num_disparities = 64
block_size = 5

stereo_matcher = cv.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8*3*window_size**2,
    P2=32*3*window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# --- Network Socket Setup ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((LAPTOP_IP, PORT))
print(f"Connected to laptop at {LAPTOP_IP}:{PORT}")

while True:
    try:
        # Capture frames
        frame_left = picam2_left.capture_array()
        frame_right = picam2_right.capture_array()

        # Pre-process frames
        frame_left = cv.flip(frame_left, 1)
        frame_right = cv.flip(frame_right, 1)
        gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)
        rectified_left = cv.remap(gray_left, mapL1, mapL2, cv.INTER_LINEAR)
        rectified_right = cv.remap(gray_right, mapR1, mapR2, cv.INTER_LINEAR)

        # Compute Disparity and Point Cloud
        disparity_map = stereo_matcher.compute(rectified_left, rectified_right)
        points = cv.reprojectImageTo3D(disparity_map, Q)
        colors = cv.cvtColor(rectified_left, cv.COLOR_GRAY2BGR).reshape(-1, 3)

        # Filter out invalid points
        valid_points_mask = disparity_map > disparity_map.min()
        output_points = points[valid_points_mask]
        output_colors = colors[valid_points_mask.flatten()]

        # Serialize and send the data
        data_to_send = {'points': output_points, 'colors': output_colors}
        data = pickle.dumps(data_to_send)
        message_size = struct.pack("L", len(data))
        
        client_socket.sendall(message_size + data)
        print(f"Sent a point cloud with {len(output_points)} points.")

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Cleanup
picam2_left.stop()
picam2_right.stop()
client_socket.close()