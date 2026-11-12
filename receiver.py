import socket
import struct
import cv2
import numpy as np

HOST = '0.0.0.0'
PORT = 8000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print('Waiting for connection...')

conn, addr = server_socket.accept()
print('Connected by', addr)

data = b""
payload_size = struct.calcsize("!I")

def recv_frame():
    global data
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("!I", packed_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    return cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

try:
    while True:
        frame_left = recv_frame()
        frame_right = recv_frame()

        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
