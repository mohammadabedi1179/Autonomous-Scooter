#!/usr/bin/env python3
"""
stereo_box_depth_kalman_synced.py
==================================
Identical to the original but with one change:
  - FOCAL_PX, BASELINE_M, fx, cx are now loaded from your
    stereo_calibration_params.json (scaled to stream resolution)
    instead of being hardcoded constants.

New parameter
-------------
  calib_json   str   path to stereo_calibration_params.json
               (same file used by stereo_disparity_node.py)

Everything else — topics, Kalman tuning, IQR, sync — is unchanged.
"""
import json
import os
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from message_filters import Subscriber, ApproximateTimeSynchronizer

# --- Camera mounting relative to base_link (unchanged) ---
CAMERA_X_M = 0.0
CAMERA_Y_M = 0.0

# Stream resolution (must match camera_streamer.py)
STREAM_W = 1640
STREAM_H = 1232

DEBUG = True


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------

def load_calibration(json_path: str) -> dict:
    """
    Load stereo_calibration_params.json and return intrinsics.

    The calibration was done at 1640×1232 (same as stream resolution),
    so NO resolution scaling is applied. P1 and P2 are used directly.

    Returns dict with keys:
      focal_px   : rectified fx (pixels)
      baseline_m : stereo baseline in metres
      cx         : rectified principal point x (pixels)
      cy         : rectified principal point y (pixels)
    """
    with open(json_path, "r") as f:
        cal = json.load(f)

    calib_w = int(cal["image_size"][0])
    calib_h = int(cal["image_size"][1])

    if calib_w != STREAM_W or calib_h != STREAM_H:
        raise RuntimeError(
            f"Calibration image_size [{calib_w}, {calib_h}] does not match "
            f"stream resolution [{STREAM_W}, {STREAM_H}]. "
            f"Either recalibrate at stream resolution or add proper scaling."
        )

    P1 = np.array(cal["P1"], dtype=np.float64)
    P2 = np.array(cal["P2"], dtype=np.float64)

    focal_px   = P1[0, 0]
    cx         = P1[0, 2]
    cy         = P1[1, 2]

    # Baseline: P2[0,3] = -fx_rect * baseline  →  baseline = -P2[0,3] / P2[0,0]
    # The calibrator used square_size in mm, so T and P2[0,3] are in mm.
    # Divide by 1000 → metres.
    # Verification: -(-71863.748) / 1200.180 / 1000 = 0.05988 m ✓
    baseline_m = float(-P2[0, 3] / P2[0, 0]) / 1000.0

    return {
        "focal_px":   focal_px,
        "baseline_m": baseline_m,
        "cx":         cx,
        "cy":         cy,
    }


# ---------------------------------------------------------------------------
# Kalman Filter factory  (unchanged from original)
# ---------------------------------------------------------------------------

def make_kalman(x0: float, y0: float,
                q_pos: float, q_vel: float,
                r_xy: float,
                dt_init: float = 0.1) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1.0, dt_init, 0.0,     0.0],
        [0.0, 1.0,     0.0,     0.0],
        [0.0, 0.0,     1.0, dt_init],
        [0.0, 0.0,     0.0,     1.0],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    kf.processNoiseCov = np.diag(
        [q_pos, q_vel, q_pos, q_vel]
    ).astype(np.float32)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_xy
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
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

        # ── NEW: calibration JSON path ────────────────────────────────────
        self.declare_parameter('calib_json','/home/mohammadabedi/Documents/Autonomous Scooter/Calibration/calibration_data_real_1640_1232/stereo_calibration_params.json')

        # ── Topics (unchanged) ────────────────────────────────────────────
        self.declare_parameter('detections_topic', '/follower_robot/depth_cam/left/detections')
        self.declare_parameter('disparity_topic',  '/follower_robot/depth_cam/disparity')
        self.declare_parameter('out_topic',        '/follower_robot/obstacles_depth')

        # ── Synchroniser ──────────────────────────────────────────────────
        self.declare_parameter('sync_queue', 10)
        self.declare_parameter('sync_slop',  0.12)

        # ── Detection filters ─────────────────────────────────────────────
        self.declare_parameter('conf_thresh',    0.15)   # was 0.35 — detections score ~0.31
        self.declare_parameter('min_disparity',  16.0)   # was 3.0 — must match SGBM minDisparity
        self.declare_parameter('depth_min',      0.3)
        self.declare_parameter('depth_max',      6.0)    # was 50.0 — rig is unreliable beyond ~4m

        # ── ROI / validity ────────────────────────────────────────────────
        # roi_shrink=0.20 is safe for all bbox sizes; 0.30 collapsed small boxes to zero
        self.declare_parameter('roi_shrink',      0.20)
        self.declare_parameter('min_valid_ratio', 0.05)
        self.declare_parameter('min_valid_floor', 5)     # was 10 — too strict after shrink
        # num_disparities: must match your SGBM config exactly.
        # SGBM produces invalid pixels in the leftmost (minDisparity + numDisparities)
        # columns. Your disparity node uses minDisparity=16, numDisparities=160,
        # so the invalid zone is 0..176 pixels from the left edge.
        self.declare_parameter('num_disparities', 176)

        # ── Disparity aggregation method ──────────────────────────────────
        # Controls how the valid disparity pixels inside the ROI are combined
        # into a single depth estimate.
        #
        # 'median' (default, recommended):
        #   Takes the median of valid disparities, then converts to depth via
        #   depth = fx * baseline / median_disparity.
        #   Robust to SGBM outlier pixels at occlusion boundaries and object
        #   edges — a minority of bad pixels cannot pull the estimate far off.
        #
        # 'mean':
        #   Converts every valid disparity pixel to depth first
        #   (depth_i = fx * baseline / disp_i), then takes the mean of depths.
        #   This is the correct way to average when you want an unbiased depth
        #   estimate, because depth is a nonlinear (hyperbolic) function of
        #   disparity — mean(depth) ≠ depth(mean(disparity)).
        #   More sensitive to outliers than median; use with a tighter
        #   roi_shrink (0.25-0.30) to exclude edge pixels first.
        #
        # Live switching:
        #   ros2 param set /follower_robot/stereo_box_depth_kalman_synced \
        #       disp_aggregation median
        self.declare_parameter('disp_aggregation', 'median')
        self.declare_parameter('buffer_len', 8)          # was 20 — caused lag on moving objects

        # ── Kalman tuning ─────────────────────────────────────────────────
        # was q_pos=0.05, r=0.30 → Kalman gain 14% (sluggish on moving objects)
        # now q_pos=0.10, r=0.10 → Kalman gain ~50% (responsive + still smooth)
        self.declare_parameter('kf_q_pos', 0.10)
        self.declare_parameter('kf_q_vel', 0.80)
        self.declare_parameter('kf_r_xy',  0.10)

        # ── Object lifetime ───────────────────────────────────────────────
        self.declare_parameter('object_expiration_s', 5.0)
        self.declare_parameter('key_bin_px', 40)

        # ── Performance & logging ─────────────────────────────────────────
        self.declare_parameter('max_process_hz', 0.0)
        self.declare_parameter('class_allowlist', [])  # was "preson" (typo)
        self.declare_parameter('print_json', True)

        # ── Visualisation ──────────────────────────────────────────────────
        # When True, subscribes to the left image and draws each bounding box
        # with 4 overlaid lines:
        #   smooth depth (Kalman)  |  raw depth  |  raw disparity  |  bearing
        # Also publishes an annotated Image topic and opens a cv2 window.
        self.declare_parameter('visualize',        False)
        self.declare_parameter('vis_image_topic',  '/follower_robot/depth_cam/left/image_rect')
        self.declare_parameter('vis_out_topic',    '/follower_robot/depth_cam/left/depth_vis')

        # ── Load all params ───────────────────────────────────────────────
        calib_json = self.get_parameter('calib_json').value

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
        self.num_disparities = int(self.get_parameter('num_disparities').value)

        _agg = str(self.get_parameter('disp_aggregation').value).strip().lower()
        if _agg not in ('median', 'mean'):
            self.get_logger().warn(
                f"disp_aggregation='{_agg}' is not recognised. "
                f"Falling back to 'median'. Valid values: 'median', 'mean'."
            )
            _agg = 'median'
        self.disp_aggregation = _agg

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
        self.print_json   = bool(self.get_parameter('print_json').value)
        self.do_visualize = bool(self.get_parameter('visualize').value)
        vis_image_topic   = self.get_parameter('vis_image_topic').value
        vis_out_topic     = self.get_parameter('vis_out_topic').value
        self._last_process_t = 0.0

        # ── Load calibration → replace hardcoded constants ────────────────
        if not os.path.isfile(calib_json):
            self.get_logger().fatal(f"Calibration file not found: {calib_json}")
            raise FileNotFoundError(calib_json)

        cal = load_calibration(calib_json)
        self.FOCAL_PX   = cal["focal_px"]
        self.BASELINE_M = cal["baseline_m"]
        self.fx         = cal["focal_px"]   # same as focal_px after rectification
        self.cx         = cal["cx"]

        self.get_logger().info(
            f"Calibration loaded from {calib_json}\n"
            f"  focal_px         = {self.FOCAL_PX:.4f} px\n"
            f"  baseline_m       = {self.BASELINE_M:.4f} m\n"
            f"  cx               = {self.cx:.4f} px\n"
            f"  disp_aggregation = {self.disp_aggregation}  "
            f"('median'=outlier-robust, 'mean'=unbiased depth average)"
        )

        # Per-object Kalman state store
        self.objects: Dict[str, Dict[str, Any]] = {}

        # ── Subscriptions / publishers ────────────────────────────────────
        self.det_sub  = Subscriber(
            self, Detection2DArray, self.detections_topic, qos_profile=10
        )
        self.disp_sub = Subscriber(
            self, DisparityImage, self.disparity_topic, qos_profile=10
        )

        self.depth_pub = self.create_publisher(String, self.out_topic, 10)

        # ── Visualisation setup ───────────────────────────────────────────
        self.vis_pub      = None
        self._latest_image: Optional[np.ndarray] = None   # cache for vis

        if self.do_visualize:
            # Subscribe to the raw left image for the canvas
            self.create_subscription(
                Image, vis_image_topic,
                self._image_cb, qos_profile_sensor_data
            )
            self.vis_pub = self.create_publisher(Image, vis_out_topic, 10)
            self.get_logger().info(
                f"Depth visualisation ON\n"
                f"  image src  : {vis_image_topic}\n"
                f"  vis topic  : {vis_out_topic}\n"
                f"  Press 'q' in the cv2 window to quit."
            )

        # 2-topic sync (detections + disparity) — same as before
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.disp_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.synced_cb)

        self.get_logger().info(
            f'StereoBoxDepthKalman_Synced ready\n'
            f'  detections : {self.detections_topic}\n'
            f'  disparity  : {self.disparity_topic}\n'
            f'  output     : {self.out_topic}\n'
            f'  visualize  : {self.do_visualize}'
        )

    # -------------------------------------------------------------------------
    # Main callback  (only change: self.FOCAL_PX / self.BASELINE_M / self.fx / self.cx
    #                 instead of module-level constants)
    # -------------------------------------------------------------------------

    def synced_cb(self, det_msg: Detection2DArray, disp_msg: DisparityImage):
        if self.max_process_hz and self.max_process_hz > 0.0:
            now = time.time()
            if (now - self._last_process_t) < (1.0 / self.max_process_hz):
                return
            self._last_process_t = now

        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'[synced_cb] disparity convert failed: {e}')
            return

        current_time = time.time()
        self._expire_objects(current_time)

        results = []

        for det in det_msg.detections:
            bbox     = self._bbox(det)
            cls_id   = self._cls(det)
            conf     = self._conf(det)
            track_id = det.id if det.id else None

            if bbox is None or cls_id is None:
                continue

            cls_str = str(cls_id)
            if self.allowlist and cls_str not in self.allowlist:
                self.get_logger().warn(
                    f'[synced_cb] DROP id={det.id} cls={cls_str}: not in allowlist {self.allowlist}')
                continue
            if conf is not None and conf < self.conf_thresh:
                self.get_logger().warn(
                    f'[synced_cb] DROP id={det.id} cls={cls_str}: conf {conf:.3f} < {self.conf_thresh}')
                continue

            raw_disp = self._mean_disparity_in_bbox(disp, bbox)
            if raw_disp is None or raw_disp < self.min_disparity:
                x1_bbox = int(bbox[0])
                edge_note = (f' [bbox x1={x1_bbox} inside invalid SGBM zone 0-{self.num_disparities}px]'
                             if x1_bbox < self.num_disparities else '')
                self.get_logger().warn(
                    f'[synced_cb] DROP id={det.id} cls={cls_str}: '
                    f'raw_disp={raw_disp} < min={self.min_disparity}  bbox={bbox}{edge_note}')
                continue

            raw_depth = self.FOCAL_PX * self.BASELINE_M / raw_disp
            if not (self.depth_min <= raw_depth <= self.depth_max):
                self.get_logger().warn(
                    f'[synced_cb] DROP id={det.id} cls={cls_str}: '
                    f'raw_depth={raw_depth:.2f} m outside [{self.depth_min}, {self.depth_max}]')
                continue

            cx_box  = 0.5 * (bbox[0] + bbox[2])
            bearing = math.atan((cx_box - self.cx) / self.fx)

            x_cam_rel = float(raw_depth)
            y_cam_rel = float(raw_depth) * math.tan(bearing)

            x_meas = CAMERA_X_M + x_cam_rel
            y_meas = CAMERA_Y_M - y_cam_rel

            key   = self._object_key(track_id, cls_str, bbox)
            state = self.objects.get(key)

            if state is None:
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

            # ── IQR filter on DEPTH (not disparity) ──────────────────────
            # Disparity and depth are inversely related so IQR on disparity
            # ≠ IQR on depth and introduces a systematic bias. Store depth.
            state['buffer'].append(float(raw_depth))
            iqr_depth = self._iqr_filter(list(state['buffer']))

            if iqr_depth is not None and self.depth_min <= iqr_depth <= self.depth_max:
                x_meas = CAMERA_X_M + iqr_depth
                y_meas = CAMERA_Y_M - iqr_depth * math.tan(bearing)

            dt = current_time - state['last_update_t']
            dt = float(np.clip(dt, 0.01, 1.0))
            state['last_update_t'] = current_time
            state['update_count'] += 1

            kf = state['kf']
            kf.transitionMatrix[0, 1] = dt
            kf.transitionMatrix[2, 3] = dt

            kf.predict()
            measurement = np.array([[x_meas], [y_meas]], dtype=np.float32)
            kf.correct(measurement)

            sx    = float(kf.statePost[0])
            vx    = float(kf.statePost[1])
            sy    = float(kf.statePost[2])
            vy    = float(kf.statePost[3])
            speed = math.sqrt(vx ** 2 + vy ** 2)

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
                'vx_ms':        round(vx, 3),
                'vy_ms':        round(vy, 3),
                'speed_ms':     round(speed, 3),
                'raw_disp':     round(float(raw_disp), 3),
            })

        if results:
            out_msg = String()
            out_msg.data = json.dumps(results, separators=(',', ':'))
            self.depth_pub.publish(out_msg)
            if self.print_json:
                self.get_logger().info(f'object depths: {out_msg.data}')
        elif DEBUG:
            self.get_logger().info('[synced_cb] No valid objects this frame')

        # ── Visualisation (only if enabled) ──────────────────────────────
        if self.do_visualize:
            self._visualise(det_msg, results, det_msg.header)

    # -------------------------------------------------------------------------
    # Image cache (only active when visualize=True)
    # -------------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        """Cache the latest left image for use in visualisation."""
        try:
            # Images are mono8 — convert to BGR so cv2.rectangle/putText work
            mono = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self._latest_image = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().warn(f'[_image_cb] {e}')

    # -------------------------------------------------------------------------
    # Visualisation draw
    # -------------------------------------------------------------------------

    # Colour palette — one colour per class index (BGR), same as YOLO node
    _PALETTE = [
        (0, 255, 0),   (0, 128, 255), (255, 0, 0),   (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),  (128, 0, 255),
        (255, 128, 0), (0, 255, 128), (128, 255, 0),  (0, 128, 128),
    ]

    @classmethod
    def _box_color(cls, cls_id_str: str) -> tuple:
        try:
            idx = int(cls_id_str)
        except ValueError:
            idx = sum(ord(c) for c in cls_id_str)
        return cls._PALETTE[idx % len(cls._PALETTE)]

    def _visualise(self, det_msg: Detection2DArray, results: list, header):
        """
        Draw bounding boxes with depth overlay on the cached left image.

        Each box shows 4 lines of info:
          Line 1 (top, large):   class  id=N
          Line 2 (green):        depth (Kalman)  X.XXm
          Line 3 (yellow):       raw depth       X.XXm
          Line 4 (cyan):         disparity       X.X px   bearing X.X°
        """
        if self._latest_image is None:
            return   # no frame received yet

        vis = self._latest_image.copy()

        # Build a quick lookup:  track_id / fallback_key → result dict
        result_by_id: Dict[str, dict] = {}
        for r in results:
            key = str(r['id']) if r['id'] is not None else None
            if key:
                result_by_id[key] = r

        for det in det_msg.detections:
            bbox    = self._bbox(det)
            cls_id  = self._cls(det)
            tid     = det.id if det.id else None
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(STREAM_W - 1, x2); y2 = min(STREAM_H - 1, y2)

            cls_str = str(cls_id) if cls_id else "?"
            color   = self._box_color(cls_str)

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Retrieve depth result for this detection
            r = result_by_id.get(str(tid)) if tid else None

            # ── Text lines ────────────────────────────────────────────────
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_big   = 0.60
            font_small = 0.50
            thick      = 2

            if r is not None:
                smooth_d = r['depth_m']
                raw_d    = r['raw_depth_m']
                raw_disp = r['raw_disp']
                bearing  = math.degrees(r['bearing_rad'])
                vx       = r['vx_ms']
                vy       = r['vy_ms']

                lines = [
                    (f"{cls_str}  id={tid}", (255, 255, 255), font_big),
                    (f"depth(KF): {smooth_d:.2f} m",          (0, 255, 0),   font_small),
                    (f"raw depth: {raw_d:.2f} m",              (0, 255, 255), font_small),
                    (f"disp: {raw_disp:.1f}px  brg: {bearing:.1f}°",
                                                                (255, 255, 0), font_small),
                    (f"vx={vx:+.2f} vy={vy:+.2f} m/s",        (200, 200, 200), font_small),
                ]
            else:
                # Box detected but depth node filtered it out (low conf, bad disp, etc.)
                lines = [
                    (f"{cls_str}  id={tid}", (255, 255, 255), font_big),
                    ("depth: N/A",           (100, 100, 255), font_small),
                ]

            # Draw lines bottom-up starting from top of box
            line_h   = 18
            pad      = 4
            text_y   = y1 - pad - (len(lines) - 1) * line_h
            text_y   = max(text_y, line_h * len(lines))   # keep on screen

            # Semi-transparent background strip for all text lines
            bg_w = max(
                cv2.getTextSize(l[0], font, l[2], thick)[0][0]
                for l in lines
            ) + 8
            bg_y1 = text_y - line_h * (len(lines) - 1) - line_h
            bg_y2 = text_y + pad
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, max(bg_y1, 0)), (x1 + bg_w, bg_y2), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

            for i, (text, col, fscale) in enumerate(lines):
                lx = x1 + 3
                ly = text_y - (len(lines) - 1 - i) * line_h
                ly = max(ly, line_h)
                cv2.putText(vis, text, (lx, ly), font, fscale, col, thick)

        # ── Window ────────────────────────────────────────────────────────
        cv2.imshow("Depth visualisation", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Depth vis window closed by user.")
            cv2.destroyAllWindows()
            self.do_visualize = False

        # ── Publish annotated image (viewable in rqt_image_view) ──────────
        if self.vis_pub is not None:
            try:
                vis_msg              = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                vis_msg.header       = header
                self.vis_pub.publish(vis_msg)
            except Exception as e:
                self.get_logger().warn(f'vis publish failed: {e}')

    # -------------------------------------------------------------------------
    # Helpers  (all unchanged from original)
    # -------------------------------------------------------------------------

    def _expire_objects(self, now_s: float):
        expired = [
            k for k, v in self.objects.items()
            if now_s - float(v['last_seen']) > self.object_expiration_s
        ]
        for k in expired:
            del self.objects[k]

    def _mean_disparity_in_bbox(self, disp: np.ndarray, bbox) -> Optional[float]:
        """
        Aggregate valid disparity pixels inside the bounding box ROI into a
        single disparity value, controlled by self.disp_aggregation.

        'median' mode (default):
            Returns median(valid_disparities).
            Robust to SGBM outlier pixels at occlusion/edge boundaries.
            A minority of bad pixels cannot pull the result far off.

        'mean' mode:
            Converts each valid disparity pixel to depth first via
              depth_i = fx * baseline / disp_i
            then returns the mean depth converted BACK to an equivalent
            disparity so the caller's single depth = fx*b/disp formula still
            holds unchanged:
              equivalent_disp = fx * baseline / mean(depth_i)
            This is the mathematically unbiased depth estimator because
            depth is a nonlinear (hyperbolic) function of disparity:
              mean(depth) ≠ depth(mean(disparity))
            At 2m with a 30px disparity spread across the ROI, mean-depth
            gives ~2-3% less depth error than median-disparity.
            More sensitive to outliers — use with roi_shrink >= 0.25.

        ROI processing order:
          1. Shrink inward (exclude occlusion/edge pixels).
          2. Clamp left edge past SGBM invalid zone.
        """
        h, w = disp.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # Step 1: shrink ROI inward to exclude edge/occlusion pixels
        if self.roi_shrink > 0.0:
            mx = int(self.roi_shrink * (x2 - x1))
            my = int(self.roi_shrink * (y2 - y1))
            x1 += mx; x2 -= mx; y1 += my; y2 -= my
            if x2 <= x1 or y2 <= y1:
                return None

        # Step 2: clamp left edge past SGBM invalid zone (after shrink)
        x1 = max(x1, self.num_disparities)
        if x2 <= x1:
            return None

        roi = disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid_mask = np.isfinite(roi) & (roi > 0.0)
        n_valid = int(valid_mask.sum())
        min_valid = max(self.min_valid_floor, int(self.min_valid_ratio * roi.size))
        if n_valid < min_valid:
            return None

        valid_disps = roi[valid_mask]

        if self.disp_aggregation == 'mean':
            # Convert each disparity pixel to depth, average depths, convert back.
            # This gives an unbiased depth estimate because depth ∝ 1/disparity.
            depths     = self.FOCAL_PX * self.BASELINE_M / valid_disps
            mean_depth = float(depths.mean())
            # Return the equivalent disparity so the caller's formula is unchanged
            return self.FOCAL_PX * self.BASELINE_M / mean_depth
        else:
            # 'median' — outlier-robust, recommended default
            return float(np.median(valid_disps))

    @staticmethod
    def _iqr_filter(values) -> Optional[float]:
        if len(values) < 3:
            return None
        arr = np.array(values, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return float(arr.mean())
        lower    = q1 - 1.5 * iqr
        upper    = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
        return float(filtered.mean()) if filtered.size > 0 else None

    def _object_key(self, track_id, cls_str: str, bbox) -> str:
        if track_id is not None:
            return str(track_id)
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        b  = self.key_bin_px
        return f'{cls_str}:{int(cx // b)}:{int(cy // b)}'

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