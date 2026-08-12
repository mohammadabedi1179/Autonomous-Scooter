#!/usr/bin/env python3
"""
goal_marker_apriltag_depth_node.py
==================================
Detects the printed AprilTag goal marker in the LEFT rectified mono8 image and
uses the already-computed disparity image to estimate goal depth.

Important design decision
-------------------------
This node does NOT compute stereo disparity. It only:
  1. subscribes to the rectified LEFT image published by pi_camera_receiver_node,
  2. detects the AprilTag goal marker in that mono8 image,
  3. samples the disparity image already published by stereo_disparity_node,
  4. publishes one GoalMarkerState containing visibility, depth, bearing, cx, cy.

Expected upstream topics
------------------------
  /follower_robot/depth_cam/left/image_rect     sensor_msgs/Image, mono8, rectified
  /follower_robot/depth_cam/disparity           stereo_msgs/DisparityImage

Default target markers
----------------------
  AprilTag family: tag36h11
  IDs:             0,1,2,3,4,5

ID 0 is the large front-face marker. IDs 1-5 can be used for the other box faces.
The node accepts multiple tag IDs and publishes one GoalMarkerState for the best
visible marker, normally the largest detected tag in the image.

Relative velocity
------------------
A constant-velocity Kalman filter (state=[x,vx,y,vy]) estimates the goal
marker's relative velocity in the camera frame, using the same filter design
as the obstacle-speed Kalman filter in stereo_box_depth_kalman_synced.py.
Published on separate Float32 topics: vx_topic, vy_topic, speed_topic.
"""

import math
import time
from collections import deque
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import Float32

from detectors_msgs.msg import GoalMarkerState


# Fallback values only. Prefer values inside stereo_msgs/DisparityImage.
FALLBACK_FOCAL_PX = 1420.0       # rough 1640-width value; override if needed
FALLBACK_BASELINE_M = 0.060231   # from your receiver comments/calibration metadata

# Camera mounting offset relative to base_link — same convention as
# CAMERA_X_M / CAMERA_Y_M in stereo_box_depth_kalman_synced.py.
CAMERA_X_M = 0.0
CAMERA_Y_M = 0.0


def make_kalman(x0: float, y0: float,
                 q_pos: float, q_vel: float,
                 r_xy: float,
                 dt_init: float = 0.1) -> cv2.KalmanFilter:
    """Constant-velocity 2D Kalman filter — identical design to the one used
    for obstacle speed estimation in stereo_box_depth_kalman_synced.py.
    State: [x, vx, y, vy].  Measurement: [x, y]."""
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


class GoalMarkerAprilTagDepth(Node):
    def __init__(self):
        super().__init__("goal_marker_depth", namespace="follower_robot")
        self.bridge = CvBridge()

        # ── Topics ────────────────────────────────────────────────────────
        self.declare_parameter(
            "left_topic", "/follower_robot/depth_cam/left/image_rect"
        )
        self.declare_parameter(
            "disparity_topic", "/follower_robot/depth_cam/disparity"
        )
        self.declare_parameter(
            "state_topic", "/follower_robot/depth_cam/goal_marker_state"
        )
        self.declare_parameter(
            "depth_topic", "/follower_robot/depth_cam/goal_marker_depth"
        )
        self.declare_parameter(
            "depth_raw_topic", "/follower_robot/depth_cam/goal_marker_depth_raw"
        )
        self.declare_parameter(
            "debug_image_topic", "/follower_robot/depth_cam/goal_marker_debug"
        )
        self.declare_parameter("publish_debug_image", True)

        # ── Synchronization ────────────────────────────────────────────────
        self.declare_parameter("slop", 0.05)
        self.declare_parameter("queue_size", 10)

        # ── AprilTag detection ─────────────────────────────────────────────
        self.declare_parameter("tag_family", "tag36h11")
        # Backward-compatible single ID parameter.
        self.declare_parameter("tag_id", 0)
        # Multi-face goal marker IDs. Use comma-separated IDs, e.g. "0,1,2,3,4,5".
        # If this is empty, the node falls back to tag_id.
        self.declare_parameter("tag_ids", "0,1,2,3,4,5")
        self.declare_parameter("det_downscale", 1.0)  # keep 1.0 for 7–10 m range
        self.declare_parameter("min_tag_area_px", 250.0)
        self.declare_parameter("min_tag_side_px", 20.0)
        self.declare_parameter("max_lost", 5)
        self.declare_parameter("track_margin", 0.75)

        # ── Depth extraction from disparity ────────────────────────────────
        self.declare_parameter("roi_expand", 0.15)     # expand bbox by 15%
        self.declare_parameter("min_valid_pix", 40)
        self.declare_parameter("min_disparity", 0.3)
        self.declare_parameter("depth_min", 0.2)
        self.declare_parameter("depth_max", 30.0)
        self.declare_parameter("depth_window", 5)
        self.declare_parameter("depth_stat", "median")  # median or mean

        # Use disparity_msg.f and disparity_msg.T when available.
        # These are only fallbacks.
        self.declare_parameter("fallback_focal_px", FALLBACK_FOCAL_PX)
        self.declare_parameter("fallback_baseline_m", FALLBACK_BASELINE_M)

        # Bearing intrinsics. If <= 0, use disparity_msg.f for fx and image center for cx.
        self.declare_parameter("fx_left", -1.0)
        self.declare_parameter("cx_left", -1.0)

        # Runtime log
        self.declare_parameter("print_debug", True)

        # ── Kalman filter for goal-marker relative velocity (same design as
        #    the obstacle-speed Kalman filter in stereo_box_depth_kalman_synced.py) ──
        self.declare_parameter("kf_q_pos", 0.10)
        self.declare_parameter("kf_q_vel", 0.80)
        self.declare_parameter("kf_r_xy", 0.10)
        self.declare_parameter("kf_buffer_len", 8)   # IQR buffer length on raw depth
        self.declare_parameter("vx_topic", "/follower_robot/depth_cam/goal_marker_vx")
        self.declare_parameter("vy_topic", "/follower_robot/depth_cam/goal_marker_vy")
        self.declare_parameter("speed_topic", "/follower_robot/depth_cam/goal_marker_speed")

        # Read parameters
        self.left_topic = str(self.get_parameter("left_topic").value)
        self.disparity_topic = str(self.get_parameter("disparity_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        depth_raw_topic = str(self.get_parameter("depth_raw_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)

        slop = float(self.get_parameter("slop").value)
        queue_size = int(self.get_parameter("queue_size").value)

        self.tag_family_name = str(self.get_parameter("tag_family").value).strip()
        self.tag_id = int(self.get_parameter("tag_id").value)
        tag_ids_text = str(self.get_parameter("tag_ids").value).strip()
        self.allowed_tag_ids = self._parse_tag_ids(tag_ids_text, self.tag_id)
        self.det_downscale = float(self.get_parameter("det_downscale").value)
        self.det_downscale = min(1.0, max(0.2, self.det_downscale))
        self.min_tag_area_px = float(self.get_parameter("min_tag_area_px").value)
        self.min_tag_side_px = float(self.get_parameter("min_tag_side_px").value)
        self.max_lost = max(0, int(self.get_parameter("max_lost").value))
        self.track_margin = max(0.0, float(self.get_parameter("track_margin").value))

        self.roi_expand = max(0.0, float(self.get_parameter("roi_expand").value))
        self.min_valid_pix = max(1, int(self.get_parameter("min_valid_pix").value))
        self.min_disparity = float(self.get_parameter("min_disparity").value)
        self.depth_min = float(self.get_parameter("depth_min").value)
        self.depth_max = float(self.get_parameter("depth_max").value)
        self.depth_stat = str(self.get_parameter("depth_stat").value).strip().lower()
        if self.depth_stat not in ("median", "mean"):
            self.get_logger().warn(
                f"depth_stat='{self.depth_stat}' is invalid; using 'median'."
            )
            self.depth_stat = "median"

        depth_window = max(1, int(self.get_parameter("depth_window").value))
        self.depth_history = deque(maxlen=depth_window)

        self.fallback_focal_px = float(self.get_parameter("fallback_focal_px").value)
        self.fallback_baseline_m = float(self.get_parameter("fallback_baseline_m").value)
        self.fx_left_param = float(self.get_parameter("fx_left").value)
        self.cx_left_param = float(self.get_parameter("cx_left").value)
        self.print_debug = bool(self.get_parameter("print_debug").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)

        self.kf_q_pos = float(self.get_parameter("kf_q_pos").value)
        self.kf_q_vel = float(self.get_parameter("kf_q_vel").value)
        self.kf_r_xy = float(self.get_parameter("kf_r_xy").value)
        self.kf_buffer_len = max(3, int(self.get_parameter("kf_buffer_len").value))
        vx_topic = str(self.get_parameter("vx_topic").value)
        vy_topic = str(self.get_parameter("vy_topic").value)
        speed_topic = str(self.get_parameter("speed_topic").value)

        # Tracker state: bbox in full-resolution left-image coordinates
        self._track_bbox: Optional[Tuple[int, int, int, int]] = None
        self._track_id: Optional[int] = None
        self._last_depth_tag_id: Optional[int] = None
        self._lost_count = 0

        # Goal-marker relative-velocity Kalman filter state (single target —
        # no per-object dict needed since there's only one goal marker).
        self._goal_kf: Optional[cv2.KalmanFilter] = None
        self._kf_buffer = deque(maxlen=self.kf_buffer_len)
        self._kf_last_update_t: Optional[float] = None

        # AprilTag detector
        self._aruco_dict = self._make_apriltag_dictionary(self.tag_family_name)
        self._detector = self._make_apriltag_detector(self._aruco_dict)

        # Subscribers
        self.left_sub = Subscriber(self, Image, self.left_topic, qos_profile=10)
        self.disp_sub = Subscriber(self, DisparityImage, self.disparity_topic, qos_profile=10)
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.disp_sub],
            queue_size=queue_size,
            slop=slop,
        )
        self.sync.registerCallback(self.cb_pair)

        # Publishers
        self.state_pub = self.create_publisher(GoalMarkerState, state_topic, 10)
        self.depth_pub = self.create_publisher(Float32, depth_topic, 10)
        self.depth_raw_pub = self.create_publisher(Float32, depth_raw_topic, 10)
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, debug_image_topic, 10)

        self.vx_pub = self.create_publisher(Float32, vx_topic, 10)
        self.vy_pub = self.create_publisher(Float32, vy_topic, 10)
        self.speed_pub = self.create_publisher(Float32, speed_topic, 10)

        self.get_logger().info(
            "GoalMarkerAprilTagDepth ready\n"
            f"  LEFT image      : {self.left_topic}\n"
            f"  disparity       : {self.disparity_topic}\n"
            f"  tag family / ID : {self.tag_family_name} / {self.tag_id}\n"
            f"  det_downscale   : {self.det_downscale}\n"
            f"  depth statistic : {self.depth_stat}\n"
            f"  KF (vel est.)   : q_pos={self.kf_q_pos} q_vel={self.kf_q_vel} "
            f"r_xy={self.kf_r_xy} buffer_len={self.kf_buffer_len}\n"
            f"  vx/vy/speed     : {vx_topic} | {vy_topic} | {speed_topic}"
        )

    # ── Detector setup ────────────────────────────────────────────────────

    @staticmethod
    def _parse_tag_ids(text: str, fallback_id: int) -> Set[int]:
        if not text:
            return {int(fallback_id)}
        ids: Set[int] = set()
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                raise ValueError(
                    f"Invalid tag_ids entry '{part}'. Use comma-separated integers, e.g. '0,1,2,3'."
                )
        return ids if ids else {int(fallback_id)}

    def _make_apriltag_dictionary(self, name: str):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is not available. Install/use an OpenCV build with aruco support."
            )

        key = name.strip().lower().replace("dict_", "")
        mapping = {
            "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "apriltag_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "16h5": cv2.aruco.DICT_APRILTAG_16h5,
        }
        if key not in mapping:
            raise ValueError(
                f"Unsupported tag_family='{name}'. Use tag36h11, tag36h10, tag25h9, or tag16h5."
            )
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(mapping[key])
        if hasattr(cv2.aruco, "Dictionary_get"):
            return cv2.aruco.Dictionary_get(mapping[key])
        raise RuntimeError(
            "This OpenCV aruco module has neither getPredefinedDictionary nor Dictionary_get."
        )

    def _make_apriltag_detector(self, aruco_dict):
        # OpenCV changed the ArUco API across versions:
        #   newer: cv2.aruco.DetectorParameters() + cv2.aruco.ArucoDetector
        #   older: cv2.aruco.DetectorParameters_create() + cv2.aruco.detectMarkers
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        elif hasattr(cv2.aruco, "DetectorParameters_create"):
            params = cv2.aruco.DetectorParameters_create()
        else:
            raise RuntimeError(
                "This OpenCV aruco module has no DetectorParameters API. "
                "Use an OpenCV build with aruco/AprilTag support."
            )

        # Useful for long-range printed tags.
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(params, "cornerRefinementWinSize"):
            params.cornerRefinementWinSize = 5
        if hasattr(params, "cornerRefinementMaxIterations"):
            params.cornerRefinementMaxIterations = 30
        if hasattr(params, "cornerRefinementMinAccuracy"):
            params.cornerRefinementMinAccuracy = 0.05

        # Newer OpenCV API
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(aruco_dict, params)

        # Older API fallback: store params and call cv2.aruco.detectMarkers later
        self._legacy_params = params
        return None

    # ── Main callback ─────────────────────────────────────────────────────

    def cb_pair(self, left_msg: Image, disp_msg: DisparityImage):
        state = self._empty_state(left_msg)

        try:
            gray = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge LEFT mono8 conversion failed: {e}")
            self.state_pub.publish(state)
            return

        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge disparity conversion failed: {e}")
            self.state_pub.publish(state)
            return

        detection = self._detect_goal_tag_tracked(gray)
        if detection is None:
            self.depth_history.clear()
            self._reset_goal_kf(publish_zero=True)
            self.state_pub.publish(state)
            if self.publish_debug_image:
                self._publish_debug(gray, None, None, left_msg.header)
            return

        corners, bbox, detected_id = detection
        cx_px, cy_px = self._corners_center(corners)

        disparity = self._disparity_in_bbox(disp, bbox)
        if disparity is None or disparity < self.min_disparity:
            self.state_pub.publish(state)
            if self.print_debug:
                self.get_logger().info(
                    f"Tag detected but disparity is invalid. bbox={bbox}, disparity={disparity}"
                )
            if self.publish_debug_image:
                self._publish_debug(gray, corners, detected_id, left_msg.header)
            return

        focal_px, baseline_m = self._get_depth_calibration(disp_msg)
        if focal_px <= 0.0 or baseline_m <= 0.0:
            self.get_logger().warn(
                f"Invalid depth calibration: f={focal_px}, baseline={baseline_m}"
            )
            self.state_pub.publish(state)
            return

        depth_raw = (focal_px * baseline_m) / float(disparity)
        if not (self.depth_min <= depth_raw <= self.depth_max):
            self.state_pub.publish(state)
            if self.print_debug:
                self.get_logger().info(
                    f"Depth out of range: depth={depth_raw:.3f} m, disp={disparity:.3f} px"
                )
            return

        if self._last_depth_tag_id is not None and self._last_depth_tag_id != detected_id:
            self.depth_history.clear()
            self._reset_goal_kf(publish_zero=False)
        self._last_depth_tag_id = detected_id

        self.depth_history.append(float(depth_raw))
        depth_smooth = float(sum(self.depth_history) / len(self.depth_history))

        fx_for_bearing = self.fx_left_param if self.fx_left_param > 0.0 else focal_px
        cx_for_bearing = (
            self.cx_left_param if self.cx_left_param >= 0.0 else (gray.shape[1] - 1) / 2.0
        )
        bearing_rad = math.atan2((cx_px - cx_for_bearing), fx_for_bearing)

        # ── Kalman filter: relative velocity of the goal marker ───────────
        # Same constant-velocity design as the obstacle-speed Kalman filter
        # in stereo_box_depth_kalman_synced.py: state=[x,vx,y,vy], x=forward
        # depth, y=lateral (ROS convention: +y=left, since bearing is
        # positive-right), IQR-filtered depth measurement, dt from wall clock.
        x_cam_rel = float(depth_raw)
        y_cam_rel = float(depth_raw) * math.tan(bearing_rad)
        x_meas = CAMERA_X_M + x_cam_rel
        y_meas = CAMERA_Y_M - y_cam_rel

        self._kf_buffer.append(float(depth_raw))
        iqr_depth = self._iqr_filter(list(self._kf_buffer))
        if iqr_depth is not None and self.depth_min <= iqr_depth <= self.depth_max:
            x_meas = CAMERA_X_M + iqr_depth
            y_meas = CAMERA_Y_M - iqr_depth * math.tan(bearing_rad)

        now_t = time.time()
        if self._goal_kf is None:
            self._goal_kf = make_kalman(x_meas, y_meas, self.kf_q_pos, self.kf_q_vel, self.kf_r_xy)
            self._kf_last_update_t = now_t

        dt = float(np.clip(now_t - self._kf_last_update_t, 0.01, 1.0))
        self._kf_last_update_t = now_t

        kf = self._goal_kf
        kf.transitionMatrix[0, 1] = dt
        kf.transitionMatrix[2, 3] = dt
        kf.predict()
        kf.correct(np.array([[x_meas], [y_meas]], dtype=np.float32))

        vx_rel = float(kf.statePost[1])
        vy_rel = float(kf.statePost[3])
        speed_rel = math.sqrt(vx_rel ** 2 + vy_rel ** 2)

        vx_msg_out = Float32(); vx_msg_out.data = vx_rel
        vy_msg_out = Float32(); vy_msg_out.data = vy_rel
        speed_msg_out = Float32(); speed_msg_out.data = speed_rel
        self.vx_pub.publish(vx_msg_out)
        self.vy_pub.publish(vy_msg_out)
        self.speed_pub.publish(speed_msg_out)

        # Legacy depth topics
        raw_msg = Float32()
        raw_msg.data = float(depth_raw)
        self.depth_raw_pub.publish(raw_msg)

        smooth_msg = Float32()
        smooth_msg.data = float(depth_smooth)
        self.depth_pub.publish(smooth_msg)

        # State topic
        state.visible = True
        state.depth_m = float(depth_smooth)
        state.bearing_rad = float(bearing_rad)
        state.cx = float(cx_px)
        state.cy = float(cy_px)
        self.state_pub.publish(state)

        if self.publish_debug_image:
            self._publish_debug(gray, corners, detected_id, left_msg.header)

        if self.print_debug:
            x, y, w, h = bbox
            self.get_logger().info(
                f"Goal tag ID {detected_id}: depth={depth_smooth:.3f} m "
                f"raw={depth_raw:.3f} m disp={disparity:.3f} px "
                f"bearing={bearing_rad:.3f} rad center=({cx_px:.1f},{cy_px:.1f}) "
                f"bbox=({x},{y},{w},{h}) f={focal_px:.2f} T={baseline_m:.5f} "
                f"vx={vx_rel:+.3f} vy={vy_rel:+.3f} speed={speed_rel:.3f} m/s"
            )

    def _empty_state(self, left_msg: Image) -> GoalMarkerState:
        state = GoalMarkerState()
        state.header = left_msg.header
        state.visible = False
        state.depth_m = 0.0
        state.bearing_rad = 0.0
        state.cx = -1.0
        state.cy = -1.0
        return state

    # ── Detection ─────────────────────────────────────────────────────────

    def _detect_goal_tag_tracked(
        self, gray: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
        h, w = gray.shape[:2]

        # Try local ROI around last known tag first.
        if self._track_bbox is not None and self._lost_count < self.max_lost:
            x, y, bw, bh = self._track_bbox
            mx = int(max(10, bw * self.track_margin))
            my = int(max(10, bh * self.track_margin))
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + bw + mx)
            y2 = min(h, y + bh + my)

            crop = gray[y1:y2, x1:x2]
            detected = self._detect_goal_tag(crop)
            if detected is not None:
                corners, bbox, detected_id = detected
                corners[:, 0] += x1
                corners[:, 1] += y1
                bx, by, bw2, bh2 = bbox
                bbox_full = (x1 + bx, y1 + by, bw2, bh2)
                self._track_bbox = bbox_full
                self._track_id = detected_id
                self._lost_count = 0
                return corners, bbox_full, detected_id

            self._lost_count += 1

        # Full-frame search.
        detected = self._detect_goal_tag(gray)
        if detected is not None:
            corners, bbox, detected_id = detected
            self._track_bbox = bbox
            self._track_id = detected_id
            self._lost_count = 0
            return corners, bbox, detected_id

        self._track_bbox = None
        self._track_id = None
        self._lost_count = 0
        return None

    def _detect_goal_tag(
        self, gray: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
        if gray.size == 0:
            return None

        scale = self.det_downscale
        if scale < 1.0:
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        if self._detector is not None:
            corners_list, ids, _ = self._detector.detectMarkers(small)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(
                small, self._aruco_dict, parameters=self._legacy_params
            )

        if ids is None or len(corners_list) == 0:
            return None

        ids_flat = ids.reshape(-1)
        candidate_indices: List[int] = [
            i for i, marker_id in enumerate(ids_flat) if int(marker_id) in self.allowed_tag_ids
        ]
        if not candidate_indices:
            return None

        best = None
        best_id: Optional[int] = None
        best_area = -1.0
        for i in candidate_indices:
            c = corners_list[i].reshape(4, 2).astype(np.float32)
            if scale < 1.0:
                c /= scale

            area = abs(float(cv2.contourArea(c)))
            if area < self.min_tag_area_px:
                continue

            side_lengths = [
                float(np.linalg.norm(c[(j + 1) % 4] - c[j])) for j in range(4)
            ]
            if min(side_lengths) < self.min_tag_side_px:
                continue

            if area > best_area:
                best_area = area
                best = c
                best_id = int(ids_flat[i])

        if best is None:
            return None

        bbox = self._bbox_from_corners(best)
        return best, bbox, int(best_id)

    @staticmethod
    def _corners_center(corners: np.ndarray) -> Tuple[float, float]:
        center = np.mean(corners.reshape(4, 2), axis=0)
        return float(center[0]), float(center[1])

    @staticmethod
    def _bbox_from_corners(corners: np.ndarray) -> Tuple[int, int, int, int]:
        pts = corners.reshape(4, 2)
        x_min = int(math.floor(float(np.min(pts[:, 0]))))
        y_min = int(math.floor(float(np.min(pts[:, 1]))))
        x_max = int(math.ceil(float(np.max(pts[:, 0]))))
        y_max = int(math.ceil(float(np.max(pts[:, 1]))))
        return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

    # ── Depth from disparity ──────────────────────────────────────────────

    def _disparity_in_bbox(
        self, disp: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[float]:
        x, y, w, h = bbox
        h_img, w_img = disp.shape[:2]

        expand_x = int(round(w * self.roi_expand))
        expand_y = int(round(h * self.roi_expand))

        x1 = max(0, min(w_img - 1, x - expand_x))
        y1 = max(0, min(h_img - 1, y - expand_y))
        x2 = max(0, min(w_img, x + w + expand_x))
        y2 = max(0, min(h_img, y + h + expand_y))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid = np.isfinite(roi) & (roi >= self.min_disparity)
        vals = roi[valid]
        if vals.size < self.min_valid_pix:
            return None

        # Avoid outliers from edges, occlusion, and bad SGBM patches.
        if self.depth_stat == "median":
            return float(np.median(vals))

        # Mean with light trimming.
        lo, hi = np.percentile(vals, [10.0, 90.0])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        if trimmed.size < self.min_valid_pix:
            return float(np.mean(vals))
        return float(np.mean(trimmed))

    def _get_depth_calibration(self, disp_msg: DisparityImage) -> Tuple[float, float]:
        f = float(getattr(disp_msg, "f", 0.0))
        t = float(getattr(disp_msg, "t", 0.0))

        # In stereo_msgs/DisparityImage, f and T should be filled by the disparity node.
        # Some implementations leave them zero; then we use fallback params.
        focal_px = f if f > 0.0 else self.fallback_focal_px
        baseline_m = abs(t) if abs(t) > 1e-6 else self.fallback_baseline_m
        return focal_px, baseline_m

    # ── Relative-velocity Kalman filter helpers ─────────────────────────────

    def _reset_goal_kf(self, publish_zero: bool = True) -> None:
        """Reset the relative-velocity Kalman filter (goal lost, or the
        tracked tag ID switched to a different face of the marker)."""
        self._goal_kf = None
        self._kf_buffer.clear()
        self._kf_last_update_t = None
        if publish_zero:
            zero = Float32()
            zero.data = 0.0
            self.vx_pub.publish(zero)
            self.vy_pub.publish(zero)
            self.speed_pub.publish(zero)

    @staticmethod
    def _iqr_filter(values) -> Optional[float]:
        """Identical to the IQR filter in stereo_box_depth_kalman_synced.py:
        rejects outliers in the recent depth buffer before it becomes the
        Kalman filter's measurement."""
        if len(values) < 3:
            return None
        arr = np.array(values, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return float(arr.mean())
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower) & (arr <= upper)]
        return float(filtered.mean()) if filtered.size > 0 else None

    # ── Debug output ──────────────────────────────────────────────────────

    def _publish_debug(
        self,
        gray: np.ndarray,
        corners: Optional[np.ndarray],
        detected_id: Optional[int],
        header,
    ) -> None:
        if self.debug_pub is None:
            return
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            pts = corners.reshape(4, 2).astype(np.int32)
            cv2.polylines(debug, [pts], True, (0, 255, 0), 3)
            cx, cy = self._corners_center(corners)
            cv2.circle(debug, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"tag {detected_id}",
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        msg.header = header
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GoalMarkerAprilTagDepth()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
