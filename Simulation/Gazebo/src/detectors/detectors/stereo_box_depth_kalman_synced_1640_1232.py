#!/usr/bin/env python3
"""
StereoBoxDepthFromDisparity with IQR-over-time + Kalman Filter smoothing (synced)

Changes vs stereo_box_depth_from_disparity_IQR_EMA_synced.py:
- Replaced EMA smoothing with a 4-state Kalman Filter [x, vx, y, vy]
- KF transition matrix updated with real dt each step (handles variable timing)
- Velocity (vx, vy, speed_ms) added to output JSON
- IQR filter on disparity buffer kept for measurement robustness
- EMA-related parameters (ema_alpha, ema_update_every) removed
- KF tuning parameters exposed as ROS2 parameters (kf_q_pos, kf_q_vel, kf_r_xy)
- Proper ApproximateTimeSynchronizer on (Detection2DArray, DisparityImage)
- Faster ROI stats: mask + cv2.mean (avoids fancy-index copies)
- Adaptive minimum-valid pixels (ratio + floor), better for small/far boxes
- Safer per-object keying when det.id is missing (class + quantized center)
- Optional processing rate limit (max_process_hz)
- print_json parameter so you can validate then silence logs (runtime)
"""
import json
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

# === Camera Parameters (fallback if not loaded from param) ===
# Match focal_length / hack_baseline set in depth_camera.xacro for the 1640x1232 cameras.
FOCAL_PX   = 1238.0750785633165
BASELINE_M = 0.06017234719525317

# --- Camera geometry ---
IMAGE_WIDTH_PX = 1640.0
CX_PX = 785.6245269775391

# --- Camera mounting relative to base_link ---
CAMERA_X_M = 0.4
CAMERA_Y_M = 0.0

DEBUG = False


# ---------------------------------------------------------------------------
# Kalman Filter factory
# ---------------------------------------------------------------------------

def make_kalman(x0: float, y0: float,
                q_pos: float, q_vel: float,
                r_xy: float,
                dt_init: float = 0.1) -> cv2.KalmanFilter:
    """
    Constant-velocity Kalman filter in the base_link XY plane.

    State vector : [x, vx, y, vy]   (metres, m/s)
    Measurement  : [x, y]            (metres)

    Args:
        x0, y0   : initial position (metres, base_link frame)
        q_pos    : process noise on position states
        q_vel    : process noise on velocity states  (tune higher for fast obstacles)
        r_xy     : measurement noise variance (metres²)  (tune higher for far/noisy objects)
        dt_init  : initial time step guess (will be overwritten each step)
    """
    kf = cv2.KalmanFilter(4, 2)   # 4 states, 2 measurements

    # Transition  (updated with real dt each step)
    kf.transitionMatrix = np.array([
        [1.0, dt_init, 0.0,     0.0],
        [0.0, 1.0,     0.0,     0.0],
        [0.0, 0.0,     1.0, dt_init],
        [0.0, 0.0,     0.0,     1.0],
    ], dtype=np.float32)

    # Measurement: picks x and y from state
    kf.measurementMatrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    # Process noise Q
    kf.processNoiseCov = np.diag(
        [q_pos, q_vel, q_pos, q_vel]
    ).astype(np.float32)

    # Measurement noise R
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_xy

    # Initial covariance (generous — we're uncertain at birth)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    # Initial state
    kf.statePost = np.array(
        [[x0], [0.0], [y0], [0.0]], dtype=np.float32
    )

    return kf


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class StereoBoxDepthKalman_Synced(Node):

    def __init__(self):
        super().__init__(
            'stereo_box_depth_kalman_synced',
            namespace='follower_robot'
        )
        self.bridge = CvBridge()

        # ---- ROS2 parameters ------------------------------------------------
        # Topics
        self.declare_parameter('detections_topic', '/follower_robot/depth_cam/left/detections')
        self.declare_parameter('disparity_topic',  '/follower_robot/depth_cam/disparity')
        self.declare_parameter('out_topic',        '/follower_robot/obstacles_depth')

        # Synchroniser
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('sync_slop',  0.12)

        # Detection filters
        self.declare_parameter('conf_thresh',    0.35)
        self.declare_parameter('min_disparity',  3.0)
        self.declare_parameter('depth_min',      0.3)
        self.declare_parameter('depth_max',      50.0)

        # ROI / validity
        self.declare_parameter('roi_shrink',      0.2)
        self.declare_parameter('min_valid_ratio', 0.05)
        self.declare_parameter('min_valid_floor', 10)

        # IQR disparity buffer (still used for robust raw measurement)
        self.declare_parameter('buffer_len', 20)

        # Kalman filter tuning
        # q_pos  : process noise on position  — raise if measurement noise is high
        # q_vel  : process noise on velocity  — raise if obstacles accelerate quickly
        # r_xy   : measurement noise (metres²)— raise for far/small/noisy objects
        self.declare_parameter('kf_q_pos', 0.05)
        self.declare_parameter('kf_q_vel', 0.50)
        self.declare_parameter('kf_r_xy',  0.30)

        # Object lifetime
        self.declare_parameter('object_expiration_s', 5.0)

        # Fallback keying when det.id is missing
        self.declare_parameter('key_bin_px', 40)

        # Performance & logging
        self.declare_parameter('max_process_hz', 0.0)
        self.declare_parameter('class_allowlist', [])
        self.declare_parameter('print_json', True)

        # ---- Load params ----------------------------------------------------
        self.detections_topic = self.get_parameter('detections_topic').value
        self.disparity_topic  = self.get_parameter('disparity_topic').value
        self.out_topic        = self.get_parameter('out_topic').value

        self.sync_queue = int(self.get_parameter('sync_queue').value)
        self.sync_slop  = float(self.get_parameter('sync_slop').value)

        self.conf_thresh   = float(self.get_parameter('conf_thresh').value)
        self.min_disparity = float(self.get_parameter('min_disparity').value)
        self.depth_min     = float(self.get_parameter('depth_min').value)
        self.depth_max     = float(self.get_parameter('depth_max').value)

        self.roi_shrink      = float(self.get_parameter('roi_shrink').value)
        self.min_valid_ratio = float(self.get_parameter('min_valid_ratio').value)
        self.min_valid_floor = int(self.get_parameter('min_valid_floor').value)

        self.buffer_len = max(3, int(self.get_parameter('buffer_len').value))

        self.kf_q_pos = float(self.get_parameter('kf_q_pos').value)
        self.kf_q_vel = float(self.get_parameter('kf_q_vel').value)
        self.kf_r_xy  = float(self.get_parameter('kf_r_xy').value)

        self.object_expiration_s = float(self.get_parameter('object_expiration_s').value)
        self.key_bin_px = max(10, int(self.get_parameter('key_bin_px').value))

        self.max_process_hz = float(self.get_parameter('max_process_hz').value)
        self.allowlist: List[str] = [
            str(x) for x in self.get_parameter('class_allowlist').value
        ]
        self.print_json = bool(self.get_parameter('print_json').value)

        self._last_process_t = 0.0

        # Camera intrinsics (used for bearing calculation)
        self.fx = FOCAL_PX
        self.cx = CX_PX

        # Per-object Kalman state store
        # key -> {kf, buffer, last_seen, last_update_t, update_count}
        self.objects: Dict[str, Dict[str, Any]] = {}

        # ---- Logging --------------------------------------------------------
        self.get_logger().info(f'Listening for detections on : {self.detections_topic}')
        self.get_logger().info(f'Listening for disparity on  : {self.disparity_topic}')
        self.get_logger().info(f'Publishing depths on        : {self.out_topic}')
        self.get_logger().info(
            f'KF params — q_pos={self.kf_q_pos} q_vel={self.kf_q_vel} '
            f'r_xy={self.kf_r_xy}'
        )
        self.get_logger().info(
            f'Sync slop={self.sync_slop}s queue={self.sync_queue}'
        )

        # ---- Subscriptions / publishers -------------------------------------
        self.det_sub  = Subscriber(
            self, Detection2DArray, self.detections_topic, qos_profile=10
        )
        self.disp_sub = Subscriber(
            self, DisparityImage, self.disparity_topic, qos_profile=10
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.disp_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_cb)

        self.depth_pub = self.create_publisher(String, self.out_topic, 10)
        self.get_logger().info('StereoBoxDepthKalman_Synced node started.')

    # -------------------------------------------------------------------------
    # Main callback
    # -------------------------------------------------------------------------

    def synced_cb(self, det_msg: Detection2DArray, disp_msg: DisparityImage):
        # Optional rate limit
        if self.max_process_hz and self.max_process_hz > 0.0:
            now = time.time()
            if (now - self._last_process_t) < (1.0 / self.max_process_hz):
                return
            self._last_process_t = now

        # Convert disparity image once per callback
        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'[synced_cb] disparity convert failed: {e}')
            return

        current_time = time.time()
        self._expire_objects(current_time)

        results = []

        for det in det_msg.detections:
            bbox    = self._bbox(det)
            cls_id  = self._cls(det)
            conf    = self._conf(det)
            track_id = det.id if det.id else None

            if bbox is None or cls_id is None:
                continue

            cls_str = str(cls_id)
            if self.allowlist and cls_str not in self.allowlist:
                continue
            if conf is not None and conf < self.conf_thresh:
                continue

            # ---- Raw disparity measurement ----------------------------------
            raw_disp = self._mean_disparity_in_bbox(disp, bbox)
            if raw_disp is None or raw_disp < self.min_disparity:
                continue

            # ---- Convert disparity → depth → base_link XY ------------------
            raw_depth = FOCAL_PX * BASELINE_M / raw_disp
            if not (self.depth_min <= raw_depth <= self.depth_max):
                continue

            cx_box  = 0.5 * (bbox[0] + bbox[2])
            bearing = math.atan((cx_box - self.cx) / self.fx)

            x_cam_rel = float(raw_depth)
            y_cam_rel = float(raw_depth) * math.tan(bearing)

            x_meas = CAMERA_X_M + x_cam_rel
            y_meas = CAMERA_Y_M - y_cam_rel   # camera-right → base-left negative

            # ---- Kalman filter update ---------------------------------------
            key   = self._object_key(track_id, cls_str, bbox)
            state = self.objects.get(key)

            if state is None:
                # First detection: initialise KF at measured position
                state = {
                    'kf':            make_kalman(
                                         x_meas, y_meas,
                                         self.kf_q_pos, self.kf_q_vel,
                                         self.kf_r_xy
                                     ),
                    'buffer':        deque(maxlen=self.buffer_len),
                    'last_seen':     current_time,
                    'last_update_t': current_time,
                    'update_count':  0,
                }
                self.objects[key] = state

            state['last_seen'] = current_time

            # -- IQR on raw disparity buffer (robust measurement pre-filter) --
            state['buffer'].append(float(raw_disp))
            iqr_disp = self._iqr_filter(list(state['buffer']))

            # If IQR gives a value, recompute position from it
            if iqr_disp is not None and iqr_disp >= self.min_disparity:
                iqr_depth = FOCAL_PX * BASELINE_M / iqr_disp
                if self.depth_min <= iqr_depth <= self.depth_max:
                    x_meas = CAMERA_X_M + iqr_depth
                    y_meas = CAMERA_Y_M - iqr_depth * math.tan(bearing)

            # -- Compute real dt for this object ------------------------------
            dt = current_time - state['last_update_t']
            dt = float(np.clip(dt, 0.01, 1.0))   # clamp to [10 ms, 1 s]
            state['last_update_t'] = current_time
            state['update_count'] += 1

            # -- Update KF transition with real dt ----------------------------
            kf = state['kf']
            kf.transitionMatrix[0, 1] = dt   # x  += vx * dt
            kf.transitionMatrix[2, 3] = dt   # y  += vy * dt

            # Predict (extrapolate state forward)
            kf.predict()

            # Correct (fuse measurement)
            measurement = np.array([[x_meas], [y_meas]], dtype=np.float32)
            kf.correct(measurement)

            # -- Read smoothed state ------------------------------------------
            sx    = float(kf.statePost[0])   # smoothed x  (m, base_link forward)
            vx    = float(kf.statePost[1])   # velocity x  (m/s)
            sy    = float(kf.statePost[2])   # smoothed y  (m, base_link lateral)
            vy    = float(kf.statePost[3])   # velocity y  (m/s)
            speed = math.sqrt(vx ** 2 + vy ** 2)

            # Depth from smoothed forward position
            smooth_depth = sx - CAMERA_X_M
            if not (self.depth_min <= smooth_depth <= self.depth_max):
                continue

            results.append({
                'id':           track_id,
                'class':        cls_id,
                'depth_m':      round(smooth_depth, 3),
                'raw_depth_m':  round(raw_depth, 3),
                'x_m':          round(sx, 3),
                'y_m':          round(sy, 3),
                'bearing_rad':  round(float(bearing), 4),
                # --- Kalman velocity outputs ---
                'vx_ms':        round(vx, 3),   # forward velocity (m/s)
                'vy_ms':        round(vy, 3),   # lateral velocity (m/s, +left)
                'speed_ms':     round(speed, 3),
                # --- Debug ---
                'raw_disp':     round(float(raw_disp), 3),
            })

        if results:
            out_msg = String()
            out_msg.data = json.dumps(results, separators=(',', ':'))
            self.depth_pub.publish(out_msg)
            if self.print_json:
                self.get_logger().info(f'object depths: {out_msg.data}')
        else:
            if DEBUG:
                self.get_logger().info('[synced_cb] No valid objects this frame')

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _expire_objects(self, now_s: float):
        expired = [
            k for k, v in self.objects.items()
            if now_s - float(v['last_seen']) > self.object_expiration_s
        ]
        for k in expired:
            del self.objects[k]

    def _mean_disparity_in_bbox(
        self, disp: np.ndarray, bbox
    ) -> Optional[float]:
        h, w = disp.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w,     x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h,     y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # Centre-crop (removes noisy edges of the bounding box)
        if self.roi_shrink > 0.0:
            mx = int(self.roi_shrink * (x2 - x1))
            my = int(self.roi_shrink * (y2 - y1))
            x1 += mx; x2 -= mx
            y1 += my; y2 -= my
            if x2 <= x1 or y2 <= y1:
                return None

        roi = disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid   = np.isfinite(roi) & (roi > 0.0)
        n_valid = int(valid.sum())

        min_valid = max(
            self.min_valid_floor,
            int(self.min_valid_ratio * roi.size)
        )
        if n_valid < min_valid:
            return None

        mask     = (valid.astype(np.uint8) * 255)
        mean_val = cv2.mean(roi, mask=mask)[0]
        return float(mean_val)

    @staticmethod
    def _iqr_filter(values) -> Optional[float]:
        if len(values) < 3:
            return None
        arr  = np.array(values, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr  = q3 - q1
        if iqr == 0:
            return float(arr.mean())
        lower    = q1 - 1.5 * iqr
        upper    = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
        if filtered.size == 0:
            return None
        return float(filtered.mean())

    def _object_key(self, track_id, cls_str: str, bbox) -> str:
        """
        Prefer tracker ID for stable velocity history.
        Fall back to quantised bbox-centre when tracking is unavailable.
        """
        if track_id is not None:
            return str(track_id)
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        b  = self.key_bin_px
        qx = int(cx // b)
        qy = int(cy // b)
        return f'{cls_str}:{qx}:{qy}'

    @staticmethod
    def _bbox(det) -> Optional[Tuple[float, float, float, float]]:
        if det.bbox is None:
            return None
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        bw = det.bbox.size_x
        bh = det.bbox.size_y
        return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)

    @staticmethod
    def _cls(det):
        return det.results[0].hypothesis.class_id if det.results else None

    @staticmethod
    def _conf(det) -> Optional[float]:
        if det.results and det.results[0].hypothesis.score is not None:
            return float(det.results[0].hypothesis.score)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = StereoBoxDepthKalman_Synced()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
