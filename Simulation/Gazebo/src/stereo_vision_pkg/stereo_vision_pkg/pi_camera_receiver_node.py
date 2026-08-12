#!/usr/bin/env python3
"""
pi_camera_receiver_node.py
==========================
ROS2 node that receives left/right frames from the Raspberry Pi,
applies stereo rectification, and publishes rectified images.

Supports two transport modes (set via 'transport' parameter):

  'tcp' (default):
    Receives length-prefixed PNG/JPEG frames over a TCP socket.
    Pi sends mono (grayscale Y-plane from YUV420) frames.

  'gstreamer':
    Receives RTP JPEG streams via UDP.
    Requires gstreamer1.0-plugins-good + OpenCV built with GStreamer.

Calibration file used:  stereo_calibration_params.json
  image_size  : [1640, 1232]  ← native 2×2 binned IMX219 mode
  baseline_mm : 60.231 mm
  rms_stereo  : 0.757         ← acceptable; recalibrate if disparity is poor

Image format
------------
Frames arrive as mono8 (grayscale). The streamer captures YUV420 and
sends only the Y-plane — no colour processing, no demosaic artifacts.

Design decision — single rectification point
---------------------------------------------
Rectification is applied HERE and nowhere else.
stereo_disparity_node must NOT remap the images it receives on image_rect.

Topics published
----------------
  /follower_robot/depth_cam/left/image_rect   (sensor_msgs/Image, mono8, rectified)
  /follower_robot/depth_cam/right/image_rect  (sensor_msgs/Image, mono8, rectified)
  /follower_robot/depth_cam/left/image_raw    (debug only, if publish_raw_debug=True)
  /follower_robot/depth_cam/right/image_raw   (debug only, if publish_raw_debug=True)
"""

import json
import os
import socket
import struct
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


LEFT_RECT_TOPIC  = "/follower_robot/depth_cam/left/image_rect"
RIGHT_RECT_TOPIC = "/follower_robot/depth_cam/right/image_rect"
LEFT_RAW_TOPIC   = "/follower_robot/depth_cam/left/image_raw"
RIGHT_RAW_TOPIC  = "/follower_robot/depth_cam/right/image_raw"


class PiCameraReceiverNode(Node):

    def __init__(self):
        super().__init__("pi_camera_receiver", namespace="follower_robot")
        self.bridge = CvBridge()

        # ── Parameters ────────────────────────────────────────────────────
        # Transport mode
        self.declare_parameter("transport",   "tcp")     # 'tcp' or 'gstreamer'
                                                          # Use 'gstreamer' only after:
                                                          #   sudo apt install gstreamer1.0-plugins-good
                                                          #   pip install python3-opencv (apt, not pip)

        # GStreamer parameters (used when transport='gstreamer')
        self.declare_parameter("port_left",   5000)   # UDP port for left stream
        self.declare_parameter("port_right",  5001)   # UDP port for right stream
        self.declare_parameter("gst_max_buf", 2)      # appsink buffer — keep small for low latency

        # TCP parameters (used when transport='tcp', legacy)
        self.declare_parameter("host",        "0.0.0.0")
        self.declare_parameter("port",        8000)
        self.declare_parameter("left_frame_id",  "left_camera")
        self.declare_parameter("right_frame_id", "right_camera")
        self.declare_parameter("encoding",     "mono8")   # mono8: grayscale from streamer
        self.declare_parameter(
            "calibration_file",
            "/home/mohammadabedi/Documents/Autonomous Scooter/Calibration/"
            "calibration_data_real_1640_1232/stereo_calibration_params.json",
        )
        self.declare_parameter("publish_raw_debug", False)

        self.transport     = str(self.get_parameter("transport").value).strip().lower()
        self.port_left     = int(self.get_parameter("port_left").value)
        self.port_right    = int(self.get_parameter("port_right").value)
        self.gst_max_buf   = int(self.get_parameter("gst_max_buf").value)
        self.host          = str(self.get_parameter("host").value)
        self.port          = int(self.get_parameter("port").value)
        self.left_frame_id    = str(self.get_parameter("left_frame_id").value)
        self.right_frame_id   = str(self.get_parameter("right_frame_id").value)
        self.encoding         = str(self.get_parameter("encoding").value)
        self.calibration_file = str(self.get_parameter("calibration_file").value)
        self.publish_raw_debug = bool(self.get_parameter("publish_raw_debug").value)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.left_rect_pub  = self.create_publisher(Image, LEFT_RECT_TOPIC,  image_qos)
        self.right_rect_pub = self.create_publisher(Image, RIGHT_RECT_TOPIC, image_qos)
        self.left_raw_pub   = None
        self.right_raw_pub  = None
        if self.publish_raw_debug:
            self.left_raw_pub  = self.create_publisher(Image, LEFT_RAW_TOPIC,  image_qos)
            self.right_raw_pub = self.create_publisher(Image, RIGHT_RAW_TOPIC, image_qos)

        self._buf: bytes = b""
        self._running = True
        self._image_size: Optional[Tuple[int, int]] = None      # (width, height)
        self._maps: Optional[Tuple[
            np.ndarray, np.ndarray,   # left  map1, map2
            np.ndarray, np.ndarray,   # right map1, map2
        ]] = None

        self._load_calibration_and_build_maps()

        # Launch the correct transport thread
        if self.transport == "gstreamer":
            self._thread = threading.Thread(target=self._gstreamer_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._tcp_server, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"Pi camera receiver ready  [transport={self.transport}]\n"
            f"  left  rect → {LEFT_RECT_TOPIC}\n"
            f"  right rect → {RIGHT_RECT_TOPIC}"
        )

    # ── Calibration ───────────────────────────────────────────────────────

    def _load_calibration_and_build_maps(self) -> None:
        path = self.calibration_file.strip()

        if not path:
            raise RuntimeError(
                "calibration_file parameter is empty. "
                "Rectification is required for stereo depth; refusing to start."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"calibration_file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext != ".json":
            raise ValueError(
                f"Expected a .json calibration file, got '{ext}'. "
                "Convert to JSON or adapt _load_calibration_and_build_maps."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ── Extract matrices ───────────────────────────────────────────────
        # The JSON uses the naming convention produced by stereoCalibrate:
        #   M1/D1  → left  camera intrinsics / distortion coefficients
        #   M2/D2  → right camera intrinsics / distortion coefficients
        #   R1/R2  → rectification rotation matrices (from stereoRectify)
        #   P1/P2  → rectified projection matrices   (from stereoRectify)
        # image_size is [1640, 1232] — the resolution at which calibration was done,
        # which is also the stream resolution, so NO scaling is applied.
        required = ["M1", "D1", "M2", "D2", "R1", "R2", "P1", "P2", "image_size"]
        missing = [k for k in required if k not in data]
        if missing:
            raise RuntimeError(
                f"Calibration JSON is missing required keys: {missing}"
            )

        K1 = np.array(data["M1"], dtype=np.float64)
        D1 = np.array(data["D1"], dtype=np.float64)
        K2 = np.array(data["M2"], dtype=np.float64)
        D2 = np.array(data["D2"], dtype=np.float64)
        R1 = np.array(data["R1"], dtype=np.float64)
        R2 = np.array(data["R2"], dtype=np.float64)
        P1 = np.array(data["P1"], dtype=np.float64)
        P2 = np.array(data["P2"], dtype=np.float64)

        image_size_raw = np.array(data["image_size"], dtype=np.int32).reshape(-1)
        if image_size_raw.size != 2:
            raise RuntimeError("image_size must contain exactly [width, height].")

        width, height = int(image_size_raw[0]), int(image_size_raw[1])
        self._image_size = (width, height)

        # ── Build undistort-rectify maps ───────────────────────────────────
        # CV_16SC2 is the recommended fixed-point format — faster than CV_32FC1
        # and indistinguishable in quality at 1640×1232.
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, (width, height), cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, (width, height), cv2.CV_16SC2
        )
        self._maps = (left_map1, left_map2, right_map1, right_map2)

        self.get_logger().info(
            f"Stereo calibration loaded — rectification maps built for {width}×{height}.\n"
            f"  Calibration RMS  : {data.get('rms_stereo', 'n/a')}\n"
            f"  Baseline         : {data.get('baseline_mm', 'n/a')} mm\n"
            f"  Rectified focal  : {P1[0][0]:.2f} px"
        )

    # ── Rectification ─────────────────────────────────────────────────────

    def _rectify_pair(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply stereo rectification. Inputs are (H, W) uint8 mono frames."""
        if self._maps is None or self._image_size is None:
            raise RuntimeError(
                "Rectification maps are not available — calibration was not loaded."
            )

        expected_w, expected_h = self._image_size
        for name, frame in (("Left", left), ("Right", right)):
            # mono frame is 2D: (H, W)
            h = frame.shape[0]
            w = frame.shape[1]
            if (w, h) != (expected_w, expected_h):
                raise ValueError(
                    f"{name} frame is {w}×{h} but calibration expects "
                    f"{expected_w}×{expected_h}."
                )

        left_map1, left_map2, right_map1, right_map2 = self._maps
        left_rect  = cv2.remap(left,  left_map1,  left_map2,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, right_map1, right_map2, cv2.INTER_LINEAR)

        # Do NOT flip here. SGBM must receive images in the calibration
        # orientation so its left→right search direction is correct.
        # The flip for display is applied in _publish_pair after SGBM.
        return left_rect, right_rect

    # ── GStreamer transport ────────────────────────────────────────────────────

    def _make_gst_capture(self, port: int) -> cv2.VideoCapture:
        """
        Open a GStreamer VideoCapture that receives RTP JPEG from the Pi.

        Pipeline:
          udpsrc       : listen on UDP port
          rtpjpegdepay : strip RTP header
          jpegdec      : decode JPEG → raw
          videoconvert : convert to BGR for OpenCV
          appsink      : hand frame to Python

        drop=true + max-buffers=2 keeps latency minimal — old frames are
        discarded rather than queued, so the robot always sees the freshest image.
        """
        pipeline = (
            f"udpsrc port={port} "
            f"caps=\"application/x-rtp,media=application,clock-rate=90000,"
            f"encoding-name=X-GST,payload=96\" ! "
            f"rtpgstdepay ! pngdec ! videoconvert ! "
            f"video/x-raw,format=GRAY8 ! "
            f"appsink drop=true max-buffers={self.gst_max_buf} sync=false"
        )
        self.get_logger().info(f"Opening GStreamer pipeline on UDP:{port}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(
                f"GStreamer VideoCapture failed on port {port}.\n"
                f"Check:\n"
                f"  1. gstreamer1.0-plugins-good is installed on this machine\n"
                f"  2. OpenCV was built with GStreamer support:\n"
                f"       python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep GStreamer\n"
                f"  3. Pi streamer (camera_streamer_gstreamer.py) is running\n"
                f"  4. No firewall blocking UDP port {port}"
            )
        return cap

    def _gstreamer_loop(self) -> None:
        """Capture loop for GStreamer transport. Replaces _tcp_server."""
        self.get_logger().info(
            f"GStreamer transport: left=UDP:{self.port_left}  "
            f"right=UDP:{self.port_right}"
        )

        cap_left = cap_right = None
        retry_delay = 3.0

        while self._running:
            # Open captures (retry on failure)
            try:
                if cap_left is None:
                    cap_left  = self._make_gst_capture(self.port_left)
                if cap_right is None:
                    cap_right = self._make_gst_capture(self.port_right)
            except RuntimeError as e:
                self.get_logger().error(str(e))
                self.get_logger().info(f"Retrying in {retry_delay:.0f}s …")
                time.sleep(retry_delay)
                continue

            # Capture loop
            consecutive_failures = 0
            self.get_logger().info("GStreamer captures open — receiving frames …")

            while self._running:
                ok_l, frame_left  = cap_left.read()
                ok_r, frame_right = cap_right.read()

                if not ok_l or not ok_r:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        self.get_logger().warn(
                            "GStreamer: too many read failures — "
                            "is the Pi streamer still running? Reopening …"
                        )
                        break
                    time.sleep(0.033)
                    continue

                consecutive_failures = 0
                try:
                    if frame_left.ndim == 3:
                        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

                    if frame_right.ndim == 3:
                        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
                    self._publish_pair(frame_left, frame_right)
                except (ValueError, RuntimeError) as e:
                    self.get_logger().error(f"Publish error: {e}")

            # Release on failure and retry
            if cap_left is not None:
                cap_left.release()
                cap_left = None
            if cap_right is not None:
                cap_right.release()
                cap_right = None

            if self._running:
                self.get_logger().info(f"Reopening GStreamer in {retry_delay:.0f}s …")
                time.sleep(retry_delay)

    # ── TCP transport (legacy) ────────────────────────────────────────────────



    def _recv_exactly(self, conn: socket.socket, n: int) -> bytes:
        while len(self._buf) < n:
            # 131072 (128KB) chunks handle large 1640×1232 PNG frames efficiently
            chunk = conn.recv(131072)
            if not chunk:
                raise ConnectionError("Pi disconnected")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _recv_frame(self, conn: socket.socket) -> np.ndarray:
        raw_len  = self._recv_exactly(conn, 4)
        msg_size = struct.unpack("!I", raw_len)[0]
        raw_data = self._recv_exactly(conn, msg_size)
        # IMREAD_GRAYSCALE: decodes JPEG/PNG as single-channel uint8
        frame = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError("cv2.imdecode failed — corrupt or empty frame")
        return frame

    def _tcp_server(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        self.get_logger().info(f"Waiting for Pi to connect on {self.host}:{self.port} …")

        while self._running:
            srv.settimeout(1.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self.get_logger().info(f"Pi connected from {addr}")
            self._buf = b""

            try:
                while self._running:
                    frame_left  = self._recv_frame(conn)
                    frame_right = self._recv_frame(conn)
                    self._publish_pair(frame_left, frame_right)
            except (ConnectionError, ValueError, RuntimeError) as e:
                self.get_logger().warn(f"Connection stopped: {e}")
            finally:
                conn.close()
                self.get_logger().info("Pi disconnected — waiting for reconnect …")

        srv.close()

    # ── Publish ───────────────────────────────────────────────────────────

    def _to_msg(self, frame: np.ndarray, stamp, frame_id: str) -> Image:
        # frame is (H, W) uint8 for mono8, or (H, W, 3) for bgr8
        # cv_bridge needs explicit encoding for 2D arrays
        enc = self.encoding  # 'mono8'
        msg = self.bridge.cv2_to_imgmsg(frame, encoding=enc)
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        return msg

    def _publish_pair(self, left: np.ndarray, right: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()

        try:
            if self.publish_raw_debug and self.left_raw_pub and self.right_raw_pub:
                self.left_raw_pub.publish(self._to_msg(left,  stamp, self.left_frame_id))
                self.right_raw_pub.publish(self._to_msg(right, stamp, self.right_frame_id))

            left_rect, right_rect = self._rectify_pair(left, right)

            self.left_rect_pub.publish(self._to_msg(left_rect,  stamp, self.left_frame_id))
            self.right_rect_pub.publish(self._to_msg(right_rect, stamp, self.right_frame_id))

        except Exception as e:
            self.get_logger().error(
                f"Publish failed: {e}\n"
                f"  left shape={left.shape} dtype={left.dtype}\n"
                f"  right shape={right.shape} dtype={right.dtype}\n"
                f"  encoding={self.encoding}"
            )

    # ── Shutdown ──────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiCameraReceiverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
