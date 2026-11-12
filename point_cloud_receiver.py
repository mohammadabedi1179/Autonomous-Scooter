import threading
import pyvista as pv
import numpy as np
import socket
import pickle
import struct
import sys
import time

LAPTOP_LISTEN_IP = '0.0.0.0'
PORT = 8000

# --- Shared data ---
latest_points = None
latest_colors = None
lock = threading.Lock()

def receive_pointcloud():
    global latest_points, latest_colors
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((LAPTOP_LISTEN_IP, PORT))
    server_socket.listen(1)
    print(f"Listening for connection from Raspberry Pi at {LAPTOP_LISTEN_IP}:{PORT}...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    data_buffer = b""
    payload_size = struct.calcsize("!I")

    while True:
        try:
            while len(data_buffer) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data_buffer += packet

            packed_msg_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            msg_size = struct.unpack("!I", packed_msg_size)[0]

            while len(data_buffer) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data_buffer += packet

            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            data_dict = pickle.loads(frame_data)
            points = data_dict['points']
            colors = data_dict['colors']

            with lock:
                latest_points = points
                latest_colors = colors

        except Exception as e:
            print(f"Receiver error: {e}", file=sys.stderr)
            break

# Start networking in a background thread
threading.Thread(target=receive_pointcloud, daemon=True).start()

# --- PyVista ---
plotter = pv.Plotter(window_size=[1280, 720])
plotter.show_grid()
plotter.add_text("Waiting for point cloud data...", font_size=20, name='status_text')
plotter.show(auto_close=False, interactive_update=True)

first_cloud = True
while True:
    with lock:
        if latest_points is not None and latest_points.size > 0:
            # Remove old
            print(f"Received {latest_points.shape[0]} points")
            print(f"X range: {latest_points[:,0].min()} → {latest_points[:,0].max()}")
            print(f"Y range: {latest_points[:,1].min()} → {latest_points[:,1].max()}")
            print(f"Z range: {latest_points[:,2].min()} → {latest_points[:,2].max()}")
            if plotter.actors.get('live_point_cloud'):
                plotter.remove_actor('live_point_cloud')
            if plotter.actors.get('status_text'):
                plotter.remove_actor('status_text')

            mesh = pv.PolyData(latest_points)
            mesh.point_data['colors'] = latest_colors.astype(np.float32) / 255.0
            plotter.add_mesh(mesh, scalars='colors', rgb=True, point_size=3, name='live_point_cloud')

            if first_cloud:
                plotter.reset_camera()
                first_cloud = False

    plotter.update()
    time.sleep(0.01)