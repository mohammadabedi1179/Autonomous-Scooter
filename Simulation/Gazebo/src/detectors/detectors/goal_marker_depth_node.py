#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from detectors_msgs.msg import GoalMarkerState

# Stereo depth params
FOCAL_PX = 555.78
BASELINE_M = 0.0717

# Left camera intrinsics (for bearing only)
FX_LEFT = 555.09075
CX_LEFT = 319.05936

# Disparity / ROI tuning
MIN_VALID_PIX = 50          # min valid disparity pixels in ROI
MIN_DISPARITY = 0.5         # px, below this depth is unreliable
DEPTH_MIN = 0.2             # m
DEPTH_MAX = 50.0            # m

# Set PRINT=True while validating depth estimates. Set to False for silent operation.
PRINT = False


def _camera_depth_bearing_to_base_geometry(
    camera_forward_depth_m: float,
    camera_bearing_rad: float,
    camera_x_offset_m: float,
    camera_y_offset_m: float,
) -> Tuple[float, float, float, float]:
    """Convert stereo camera-frame depth/bearing into base_link planar geometry.

    The stereo formula returns forward depth from the LEFT camera optical center.
    For robot navigation we usually need the distance and bearing from base_link.

    Convention used here:
      x: forward from base_link
      y: lateral, with the same bearing sign convention already published by
         the old node. If your camera-bearing sign is opposite to your base_link
         convention, set parameter bearing_sign:=-1.0.

    Returns:
      (base_range_m, base_bearing_rad, x_base_m, y_base_m)
    """
    x_cam = float(camera_forward_depth_m)
    y_cam = float(camera_forward_depth_m) * math.tan(float(camera_bearing_rad))

    x_base = x_cam + float(camera_x_offset_m)
    y_base = y_cam + float(camera_y_offset_m)

    base_range = math.hypot(x_base, y_base)
    base_bearing = math.atan2(y_base, x_base)
    return float(base_range), float(base_bearing), float(x_base), float(y_base)


class GoalMarkerDepth(Node):
    def __init__(self):
        super().__init__("goal_marker_depth", namespace="follower_robot")

        # --- Parameters ---
        self.declare_parameter("left_topic", "/follower_robot/depth_cam/left/image_raw")
        self.declare_parameter("disparity_topic", "/follower_robot/depth_cam/disparity")
        self.declare_parameter("slop", 0.01)
        self.declare_parameter("min_area", 80.0)
        self.declare_parameter("ema_alpha", 0.4)      # EMA smoothing factor (0 < alpha ≤ 1)

        self.declare_parameter("det_downscale", 1.0)     # 0.5 = detect on half-res
        self.declare_parameter("track_margin", 0.5)      # expand last bbox by 50% each side
        self.declare_parameter("max_lost", 5)            # frames to keep tracking before full search

        # Camera pose relative to base_link, in the robot/base frame.
        # From your xacro chain the left camera is approximately:
        #   x ≈ 0.3563 m forward, y ≈ 0.036 m lateral.
        # If you want the optical child frame instead, x is closer to 0.3663 m.
        self.declare_parameter("camera_x_offset_m", 0.3563)
        self.declare_parameter("camera_y_offset_m", 0.0360)

        # Keep +1.0 to preserve the original node's bearing sign.
        # If base_link uses +y-left and you discover the image bearing is reversed,
        # run with: -p bearing_sign:=-1.0
        self.declare_parameter("bearing_sign", 1.0)

        left_topic = self.get_parameter("left_topic").get_parameter_value().string_value
        disparity_topic = (
            self.get_parameter("disparity_topic").get_parameter_value().string_value
        )
        slop = self.get_parameter("slop").get_parameter_value().double_value

        self.min_area = self.get_parameter("min_area").get_parameter_value().double_value

        # EMA alpha: higher = faster response to new readings, less smoothing.
        # Equivalent "window" feel: alpha ≈ 2 / (N + 1).  Default 0.4 ≈ window of ~4.
        ema_alpha = float(
            self.get_parameter("ema_alpha").get_parameter_value().double_value
        )
        if not (0.0 < ema_alpha <= 1.0):
            self.get_logger().warn(
                f"ema_alpha={ema_alpha} out of (0, 1]; using 0.4."
            )
            ema_alpha = 0.4
        self.ema_alpha = ema_alpha
        self._ema_depth: Optional[float] = None   # EMA of corrected base_link distance

        self.det_downscale = float(
            self.get_parameter("det_downscale").get_parameter_value().double_value
        )
        if not (0.1 <= self.det_downscale <= 1.0):
            self.get_logger().warn(
                f"det_downscale={self.det_downscale} out of [0.1, 1.0]; using 0.5."
            )
            self.det_downscale = 0.5

        self.track_margin = float(
            self.get_parameter("track_margin").get_parameter_value().double_value
        )
        self.max_lost = int(
            self.get_parameter("max_lost").get_parameter_value().integer_value
        )
        self.max_lost = max(0, self.max_lost)

        self.camera_x_offset_m = float(
            self.get_parameter("camera_x_offset_m").get_parameter_value().double_value
        )
        self.camera_y_offset_m = float(
            self.get_parameter("camera_y_offset_m").get_parameter_value().double_value
        )
        self.bearing_sign = float(
            self.get_parameter("bearing_sign").get_parameter_value().double_value
        )
        if self.bearing_sign == 0.0 or not math.isfinite(self.bearing_sign):
            self.get_logger().warn("bearing_sign must be non-zero and finite; using +1.0")
            self.bearing_sign = 1.0
        self.bearing_sign = 1.0 if self.bearing_sign > 0.0 else -1.0

        self.bridge = CvBridge()

        # Precompute HSV thresholds & morphology kernel (avoid per-frame allocations)
        self._lower1 = np.array([0, 80, 80], dtype=np.uint8)
        self._upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self._lower2 = np.array([170, 80, 80], dtype=np.uint8)
        self._upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self._kernel = np.ones((10, 10), np.uint8)

        # Simple tracker state (bbox in full-res image coords)
        self._track_bbox: Optional[Tuple[int, int, int, int]] = None
        self._lost_count: int = 0

        # --- Subscribers: LEFT image + disparity (approx sync) ---
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST, depth=1)
        self.left_sub = Subscriber(self, Image, left_topic, qos_profile=sensor_qos)
        self.disp_sub = Subscriber(self, DisparityImage, disparity_topic, qos_profile=sensor_qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.disp_sub],
            queue_size=2,
            slop=slop,
        )
        self.sync.registerCallback(self.cb_pair)

        # --- Publishers ---
        self.depth_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_depth", 10
        )
        self.depth_raw_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_depth_raw", 10
        )

        # Debug-only camera-frame depths. The legacy topics above now publish
        # base_link-corrected distance so the RL env can consume them directly.
        self.camera_depth_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_camera_depth", 10
        )
        self.camera_depth_raw_pub = self.create_publisher(
            Float32, "/follower_robot/depth_cam/goal_marker_camera_depth_raw", 10
        )

        self.state_pub = self.create_publisher(
            GoalMarkerState,
            "/follower_robot/depth_cam/goal_marker_state",
            10,
        )

        self.get_logger().info(
            f"GoalMarkerDepth ready: red goal detection on LEFT + depth from disparity ROI. "
            f"Publishing base_link-corrected goal distance on legacy depth topics/state "
            f"(camera offset x={self.camera_x_offset_m:.4f} m, "
            f"y={self.camera_y_offset_m:.4f} m, bearing_sign={self.bearing_sign:+.0f}; "
            f"EMA alpha={self.ema_alpha}, ROI shrink=25%, median disparity, "
            f"det_downscale={self.det_downscale})"
        )

    # ---- Synced callback: LEFT image + disparity ----
    def cb_pair(self, left_msg: Image, disp_msg: DisparityImage):
        # Convert inputs
        now = self.get_clock().now()
        age = (now - Time.from_msg(disp_msg.header.stamp)).nanoseconds * 1e-9
        self.get_logger().info(f"disp_age={age*1e3:.0f}ms")
        try:
            left_bgr = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge LEFT conversion failed: {e}")
            return

        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge disparity conversion failed: {e}")
            return

        # Prepare default state (NOT visible)
        state = GoalMarkerState()
        state.header = left_msg.header
        state.visible = False
        state.depth_m = 0.0
        state.bearing_rad = 0.0
        state.cx = -1.0
        state.cy = -1.0

        # Detect bbox in LEFT
        bbox = self._detect_red_bbox_tracked(left_bgr)
        if bbox is None:
            self._ema_depth = None   # reset filter so next detection starts fresh
            self.state_pub.publish(state)
            return

        # Mean disparity within bbox ROI
        disparity = self._mean_disparity_in_bbox(disp, bbox)
        if disparity is None or disparity < MIN_DISPARITY:
            self._ema_depth = None   # reset filter; next valid reading starts fresh
            self.state_pub.publish(state)
            if PRINT:
                self.get_logger().info("Goal detected but disparity invalid/too small.")
            return

        camera_depth_raw_m = (FOCAL_PX * BASELINE_M) / float(disparity)
        if not (DEPTH_MIN <= camera_depth_raw_m <= DEPTH_MAX):
            self._ema_depth = None   # reset filter on out-of-range reading
            self.state_pub.publish(state)
            if PRINT:
                self.get_logger().info(
                    f"Camera depth out of range: {camera_depth_raw_m:.3f} m "
                    f"(disp={disparity:.3f}px)"
                )
            return

        # Center pixel (LEFT)
        x, y, w, h = bbox
        cx_px = x + w / 2.0
        cy_px = y + h / 2.0

        # Raw camera bearing from image center, with configurable sign.
        camera_bearing_rad = self.bearing_sign * math.atan2((cx_px - CX_LEFT), FX_LEFT)

        # Convert camera-frame forward depth + bearing into base_link planar geometry.
        # The corrected distance is what the RL/navigation stack usually needs.
        base_depth_raw_m, base_bearing_rad, x_base_m, y_base_m = (
            _camera_depth_bearing_to_base_geometry(
                camera_forward_depth_m=camera_depth_raw_m,
                camera_bearing_rad=camera_bearing_rad,
                camera_x_offset_m=self.camera_x_offset_m,
                camera_y_offset_m=self.camera_y_offset_m,
            )
        )

        if not (DEPTH_MIN <= base_depth_raw_m <= DEPTH_MAX):
            self._ema_depth = None
            self.state_pub.publish(state)
            if PRINT:
                self.get_logger().info(
                    f"Base-link corrected depth out of range: {base_depth_raw_m:.3f} m "
                    f"(camera_depth={camera_depth_raw_m:.3f} m, "
                    f"bearing={camera_bearing_rad:.3f} rad)"
                )
            return

        # Smooth the corrected base_link distance. This keeps the published state
        # internally consistent: depth_m and bearing_rad are both base_link-frame.
        if self._ema_depth is None:
            self._ema_depth = float(base_depth_raw_m)
        else:
            self._ema_depth = (
                self.ema_alpha * float(base_depth_raw_m)
                + (1.0 - self.ema_alpha) * self._ema_depth
            )
        base_depth_smooth_m = self._ema_depth

        # Publish legacy numeric depths as BASE_LINK-corrected distances.
        msg_smooth = Float32()
        msg_smooth.data = float(base_depth_smooth_m)
        self.depth_pub.publish(msg_smooth)

        msg_raw = Float32()
        msg_raw.data = float(base_depth_raw_m)
        self.depth_raw_pub.publish(msg_raw)

        # Publish camera-frame debug depths separately.
        msg_camera_smooth = Float32()
        # There is no separate camera EMA; this debug topic intentionally exposes
        # the instantaneous stereo camera forward depth.
        msg_camera_smooth.data = float(camera_depth_raw_m)
        self.camera_depth_pub.publish(msg_camera_smooth)

        msg_camera_raw = Float32()
        msg_camera_raw.data = float(camera_depth_raw_m)
        self.camera_depth_raw_pub.publish(msg_camera_raw)

        # Publish state in base_link frame.
        state.visible = True
        state.depth_m = float(base_depth_smooth_m)
        state.bearing_rad = float(base_bearing_rad)
        state.cx = float(cx_px)
        state.cy = float(cy_px)
        self.state_pub.publish(state)

        if PRINT:
            self.get_logger().info(
                f"Goal depth base_link: smooth={base_depth_smooth_m:.3f} m | "
                f"raw={base_depth_raw_m:.3f} m | bearing_base={base_bearing_rad:.3f} rad | "
                f"camera_depth={camera_depth_raw_m:.3f} m | "
                f"bearing_cam={camera_bearing_rad:.3f} rad | "
                f"x_base={x_base_m:.3f} y_base={y_base_m:.3f} | "
                f"disp={disparity:.2f}px | bbox=({x},{y},{w},{h})"
            )

    # ---- Tracking-aware detection (fast path most frames) ----
    def _detect_red_bbox_tracked(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]

        # Try local search around last bbox
        if self._track_bbox is not None and self._lost_count < self.max_lost:
            x, y, bw, bh = self._track_bbox
            mx = int(max(5, bw * self.track_margin))
            my = int(max(5, bh * self.track_margin))

            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + bw + mx)
            y2 = min(h, y + bh + my)

            crop = bgr[y1:y2, x1:x2]
            local = self._detect_red_bbox(crop)
            if local is not None:
                lx, ly, lw, lh = local
                bbox = (x1 + lx, y1 + ly, lw, lh)
                self._track_bbox = bbox
                self._lost_count = 0
                return bbox

            self._lost_count += 1

        # Full-frame search
        bbox = self._detect_red_bbox(bgr)
        if bbox is not None:
            self._track_bbox = bbox
            self._lost_count = 0
            return bbox

        # Lost
        self._track_bbox = None
        self._lost_count = 0
        return None

    # ---- Red detection on an image (or crop) ----
    def _detect_red_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if bgr.size == 0:
            return None

        # Downscale for speed
        scale = self.det_downscale
        if scale < 1.0:
            small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self._lower1, self._upper1)
        mask2 = cv2.inRange(hsv, self._lower2, self._upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphology to clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        min_area_scaled = self.min_area * (scale * scale)
        if area < min_area_scaled:
            return None

        x, y, w, h = cv2.boundingRect(cnt)

        # Scale bbox back to full-res coordinates (of the input 'bgr')
        if scale < 1.0:
            inv = 1.0 / scale
            x = int(round(x * inv))
            y = int(round(y * inv))
            w = int(round(w * inv))
            h = int(round(h * inv))

        return (x, y, w, h)

    # ---- Median disparity inside a shrunk ROI (edges excluded) ----
    def _mean_disparity_in_bbox(
        self, disp: np.ndarray, rect: Tuple[int, int, int, int]
    ) -> Optional[float]:
        x, y, w, h = rect
        h_img, w_img = disp.shape[:2]

        # FIX 2: Shrink the ROI by 25% on every side to exclude noisy
        # foreground/background boundary pixels at the object edge.
        pad_x = max(1, w // 4)
        pad_y = max(1, h // 4)
        x = x + pad_x
        y = y + pad_y
        w = max(1, w - 2 * pad_x)
        h = max(1, h - 2 * pad_y)

        x1 = max(0, min(w_img - 1, int(x)))
        y1 = max(0, min(h_img - 1, int(y)))
        x2 = max(0, min(w_img, int(x + w)))
        y2 = max(0, min(h_img, int(y + h)))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = disp[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid = np.isfinite(roi) & (roi > 0.0)
        n_valid = int(valid.sum())
        if n_valid < MIN_VALID_PIX:
            return None

        # FIX 1: Use median instead of mean so that motion-blur outliers
        # and disparity spikes at object boundaries cannot skew the estimate.
        return float(np.median(roi[valid]))


def main():
    rclpy.init()
    node = GoalMarkerDepth()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
