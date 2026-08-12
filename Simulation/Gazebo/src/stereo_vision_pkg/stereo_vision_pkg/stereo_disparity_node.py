#!/usr/bin/env python3
"""
stereo_disparity_node.py
========================
ROS2 node that computes a dense disparity map from rectified stereo images.

Calibration facts (stereo_calibration_params.json)
---------------------------------------------------
  image_size     : [1640, 1232]   ← IMX219 native binned mode
  P1[0,0]        : 1250.741 px  ← rectified focal length
  P2[0,3]        : -75343.356   ← = -fx * baseline_mm
  baseline       : 75343.356 / 1250.741 = 60.239 mm = 0.060239 m
  rms_stereo     : 0.219        ← excellent

Design — single rectification point
-------------------------------------
Images arriving on image_rect are ALREADY rectified by pi_camera_receiver_node.
This node does NOT remap them again.

Topic wiring
------------
  Inputs:
    /follower_robot/depth_cam/left/image_rect   (sensor_msgs/Image, mono8)
    /follower_robot/depth_cam/right/image_rect  (sensor_msgs/Image, mono8)
  Output:
    /follower_robot/depth_cam/disparity         (stereo_msgs/DisparityImage)

Disparity ↔ depth for THIS rig  (fx=1250.741px, baseline=60.239mm)
------------------------------------------------------------------
  disparity =  fx * baseline / depth
    0.4 m  →  ~188 px      2.0 m  →  ~37 px
    0.5 m  →  ~150 px      3.0 m  →  ~25 px
    1.0 m  →   ~75 px      5.0 m  →  ~15 px
  → minDisparity=16, numDisparities=176 covers 0.42m … 4.7m

Parameters
----------
  calib_json       path to stereo_calibration_params.json
  min_disp         StereoSGBM minDisparity                (default 16)
  num_disp         numDisparities — must be divisible by 16 (default 176)
  block_size       blockSize — odd, 3-11                  (default 7)
  p1_scale         P1 = p1_scale × 3 × block²            (default 8)
  p2_scale         P2 = p2_scale × 3 × block²            (default 32)
  disp12           disp12MaxDiff (-1=off)                  (default 1)
  uniqueness       uniquenessRatio                         (default 5)
  speckle_win      speckleWindowSize                       (default 150)
  speckle_rng      speckleRange                            (default 16)
  wls_lambda       WLS filter lambda (0=disabled)          (default 8000)
  wls_sigma        WLS filter sigma colour                 (default 1.5)
  clahe_enable     CLAHE preprocessing on/off              (default True)
  clahe_clip       CLAHE clip limit                        (default 3.0)
  clahe_tile       CLAHE tile size px                      (default 8)
  disp_aggregation depth aggregation method: median|mean   (default median)
  max_hz           rate limit in Hz (0 = unlimited)        (default 0)
"""

import json
import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rcl_interfaces.msg import SetParametersResult


class StereoDisparityNode(Node):

    def __init__(self):
        super().__init__("stereo_disparity_node", namespace="follower_robot")
        self.bridge = CvBridge()

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter(
            "calib_json",
            "/home/mohammadabedi/Documents/Autonomous Scooter/Calibration/"
            "calibration_data_real_1640_1232/stereo_calibration_params.json",
        )
        self.declare_parameter("left_topic",  "/follower_robot/depth_cam/left/image_rect")
        self.declare_parameter("right_topic", "/follower_robot/depth_cam/right/image_rect")
        self.declare_parameter("out_topic",   "/follower_robot/depth_cam/disparity")
        self.declare_parameter("sync_slop",   0.12)
        self.declare_parameter("sync_queue",  10)

        # ── SGBM parameters for 1640×1232 ───────────────────────────────────
        # Calibration: fx=1250.741px  baseline=60.239mm  rms=0.219
        # Disparity ↔ depth:
        #   0.4m→188px  0.5m→150px  1.0m→75px  2.0m→37px  4.0m→19px
        # minDisparity=16 → ignores objects beyond 4.7m
        # numDisparities=176 → covers 16..192px → 0.39m to 4.7m
        self.declare_parameter("min_disp",    16)
        self.declare_parameter("num_disp",    176)  # must be divisible by 16
        # block_size=7: good balance at 1640×1232 — larger blocks average noise
        # but soften depth edges. 7 works well with lossless PNG input.
        self.declare_parameter("block_size",  7)

        # P1/P2 scale with block_size²
        # P1 = 8  × 3 × 7² = 1176
        # P2 = 32 × 3 × 7² = 4704
        self.declare_parameter("p1_scale",    8)
        self.declare_parameter("p2_scale",    32)

        self.declare_parameter("disp12",      1)
        self.declare_parameter("uniqueness",  5)

        # speckle_win=150: standard for 1640×1232
        # speckle_rng=16: unchanged — SGBM output is always ×16 int16
        self.declare_parameter("speckle_win", 150)
        self.declare_parameter("speckle_rng", 16)

        # WLS filter — enabled with rms=0.219 (excellent calibration)
        self.declare_parameter("wls_lambda",  8000.0)
        self.declare_parameter("wls_sigma",   1.5)

        # CLAHE: tile=8 covers ~205×154px at 11640p — similar physical
        # scene area as tile=8 at 1640p (80×60px per tile)
        self.declare_parameter("clahe_enable", True)
        self.declare_parameter("clahe_clip",   3.0)
        self.declare_parameter("clahe_tile",   8)

        # Disparity aggregation method for depth estimation
        # 'median' : robust to outliers (recommended)
        # 'mean'   : unbiased depth average (better for clean ROIs)
        self.declare_parameter("disp_aggregation", "median")

        self.declare_parameter("max_hz",      10.0)

        # ── Internal SGBM downscale ───────────────────────────────────────
        # Scales the image before SGBM then converts disparity back to full
        # resolution. SGBM is O(W×H×numDisp) — 0.5 scale = ~4× speedup.
        # Depth accuracy is IDENTICAL because scale cancels in:
        #   depth = (fx × baseline) / disparity
        # minDisparity and numDisparities are automatically halved internally.
        # 1.0 = disabled (full resolution)   0.5 = 4× faster   0.75 = 2× faster
        self.declare_parameter("disp_scale",  0.5)

        # ── Read parameter values ──────────────────────────────────────────
        calib_json  = self.get_parameter("calib_json").get_parameter_value().string_value
        left_topic  = self.get_parameter("left_topic").get_parameter_value().string_value
        right_topic = self.get_parameter("right_topic").get_parameter_value().string_value
        out_topic   = self.get_parameter("out_topic").get_parameter_value().string_value
        sync_slop   = float(self.get_parameter("sync_slop").get_parameter_value().double_value)
        sync_queue  = int(self.get_parameter("sync_queue").get_parameter_value().integer_value)

        self._min_disp    = int(self.get_parameter("min_disp").value)
        self._num_disp    = int(self.get_parameter("num_disp").value)
        self._block_size  = int(self.get_parameter("block_size").value)
        self._p1_scale    = int(self.get_parameter("p1_scale").value)
        self._p2_scale    = int(self.get_parameter("p2_scale").value)
        self._disp12      = int(self.get_parameter("disp12").value)
        self._uniqueness  = int(self.get_parameter("uniqueness").value)
        self._speckle_win = int(self.get_parameter("speckle_win").value)
        self._speckle_rng = int(self.get_parameter("speckle_rng").value)
        self._wls_lambda  = float(self.get_parameter("wls_lambda").value)
        self._wls_sigma   = float(self.get_parameter("wls_sigma").value)
        self._max_hz      = float(self.get_parameter("max_hz").get_parameter_value().double_value)
        self._min_period  = (1.0 / self._max_hz) if self._max_hz > 0 else 0.0
        self._last_t      = 0.0
        self._disp_scale  = float(np.clip(
            self.get_parameter("disp_scale").value, 0.25, 1.0
        ))

        self._clahe_enable = bool(self.get_parameter("clahe_enable").value)
        self._clahe_clip   = float(self.get_parameter("clahe_clip").value)
        self._clahe_tile   = int(self.get_parameter("clahe_tile").value)
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip,
            tileGridSize=(self._clahe_tile, self._clahe_tile),
        )

        _agg = str(self.get_parameter("disp_aggregation").value).strip().lower()
        if _agg not in ("median", "mean"):
            self.get_logger().warn(
                f"disp_aggregation='{_agg}' not recognised — using 'median'."
            )
            _agg = "median"
        self.disp_aggregation = _agg

        # ── Load calibration (focal length + baseline only) ────────────────
        # Images are ALREADY rectified — no remap maps needed here.
        # Calibration: 11640×1232, fx=1221.045px, baseline=60.231mm
        self.get_logger().info(f"Loading calibration from: {calib_json}")
        if not os.path.isfile(calib_json):
            self.get_logger().fatal(f"Calibration file not found: {calib_json}")
            raise FileNotFoundError(calib_json)

        with open(calib_json, "r") as f:
            cal = json.load(f)

        # Calibration done at 1640×1232 — no resolution scaling needed.
        P1 = np.array(cal["P1"], dtype=np.float64)
        P2 = np.array(cal["P2"], dtype=np.float64)

        # Rectified focal length (same for left and right after stereoRectify).
        self._focal_px = float(P1[0, 0])

        # Baseline: P2[0,3] = -fx * baseline_px.
        # baseline_px is in the same unit as the calibration (mm here).
        # Divide by 1000 to convert mm → metres.
        # Verification: -(-71863.748) / 1200.180 / 1000 ≈ 0.05988 m ✓
        # (matches baseline_mm = 59.877 in the JSON)
        self._baseline = float(-P2[0, 3] / P2[0, 0]) / 1000.0  # mm → metres

        # cx from the rectified projection matrix.
        self._cx = float(P1[0, 2])

        self.get_logger().info(
            f"Calibration loaded (image_size: {cal.get('image_size')})\n"
            f"  focal_px  = {self._focal_px:.3f} px\n"
            f"  baseline  = {self._baseline * 1000:.3f} mm  ({self._baseline:.6f} m)\n"
            f"  cx        = {self._cx:.3f} px\n"
            f"  rms       = {cal.get('rms_stereo', 'n/a')}"
        )

        # ── Validate num_disp ──────────────────────────────────────────────
        if self._num_disp % 16 != 0:
            corrected = ((self._num_disp // 16) + 1) * 16
            self.get_logger().warn(
                f"num_disp={self._num_disp} is not divisible by 16. "
                f"Correcting to {corrected}."
            )
            self._num_disp = corrected

        # ── Build StereoSGBM and WLS filter ───────────────────────────────
        self._stereo = self._make_sgbm()

        self._use_wls = False
        try:
            self._wls          = cv2.ximgproc.createDisparityWLSFilter(self._stereo)
            self._stereo_right = cv2.ximgproc.createRightMatcher(self._stereo)
            self._wls.setLambda(self._wls_lambda)
            self._wls.setSigmaColor(self._wls_sigma)
        except AttributeError:
            self.get_logger().warn(
                "opencv-contrib not installed — WLS filter disabled. "
                "Install python3-opencv-contrib for better edge quality."
            )
            self._use_wls = False

        # Depth range covered by these settings
        _f_b = self._focal_px * self._baseline  # fx * baseline (m)
        _near = _f_b / (self._min_disp + self._num_disp) if (self._min_disp + self._num_disp) > 0 else float("inf")
        _far  = _f_b / self._min_disp if self._min_disp > 0 else float("inf")
        self.get_logger().info(
            f"SGBM: block={self._block_size}  min_disp={self._min_disp}  "
            f"num_disp={self._num_disp}  "
            f"→ depth range {_near:.2f} m … {_far:.2f} m\n"
            f"      P1={self._p1_scale * 3 * self._block_size**2}  "
            f"P2={self._p2_scale * 3 * self._block_size**2}  "
            f"disp12={self._disp12}  uniqueness={self._uniqueness}\n"
            f"      speckle_win={self._speckle_win}  speckle_rng={self._speckle_rng}\n"
            f"WLS:  lambda={self._wls_lambda}  sigma={self._wls_sigma}  "
            f"enabled={self._use_wls}\n"
            f"CLAHE: enabled={self._clahe_enable}  clip={self._clahe_clip}  "
            f"tile=({self._clahe_tile},{self._clahe_tile})"
        )

        # ── Subscriptions / publisher ─────────────────────────────────────
        self._left_sub  = Subscriber(self, Image, left_topic,  qos_profile=10)
        self._right_sub = Subscriber(self, Image, right_topic, qos_profile=10)
        self._sync = ApproximateTimeSynchronizer(
            [self._left_sub, self._right_sub],
            queue_size=sync_queue,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._cb)
        self._disp_pub = self.create_publisher(DisparityImage, out_topic, 10)

        self.get_logger().info(
            f"Stereo disparity node ready.\n"
            f"  {left_topic}\n"
            f"  {right_topic}\n"
            f"  → {out_topic}"
        )

        self.add_on_set_parameters_callback(self._parameter_callback)

    # ── SGBM factory ──────────────────────────────────────────────────────

    def _make_sgbm(self) -> cv2.StereoSGBM:
        p1 = self._p1_scale * 3 * self._block_size ** 2
        p2 = self._p2_scale * 3 * self._block_size ** 2
        return cv2.StereoSGBM_create(
            minDisparity     = self._min_disp,
            numDisparities   = self._num_disp,
            blockSize        = self._block_size,
            P1               = p1,
            P2               = p2,
            disp12MaxDiff    = self._disp12,
            uniquenessRatio  = self._uniqueness,
            speckleWindowSize= self._speckle_win,
            speckleRange     = self._speckle_rng,
            mode             = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    # ── Main callback ─────────────────────────────────────────────────────

    def _cb(self, left_msg: Image, right_msg: Image) -> None:
        # ── Rate limit ────────────────────────────────────────────────────
        if self._min_period > 0:
            now = time.time()
            if now - self._last_t < self._min_period:
                return
            self._last_t = now

        # ── ROS → OpenCV ──────────────────────────────────────────────────
        try:
            left_gray  = self.bridge.imgmsg_to_cv2(left_msg,  desired_encoding="mono8")
            right_gray = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        orig_h, orig_w = left_gray.shape[:2]

        # ── Optional downscale before SGBM ───────────────────────────────
        # SGBM is O(W × H × numDisparities). Scaling to 0.5 gives ~4× speedup
        # with ZERO depth accuracy loss because scale cancels in the formula:
        #   depth = (fx × baseline) / disparity
        # When image is scaled by s: fx→fx*s, disparity→disp*s → depth unchanged.
        #
        # IMPORTANT: minDisparity and numDisparities must also scale with the
        # image so SGBM searches the same depth range in the smaller image.
        # Failing to scale them is the #1 cause of sparse/wrong disparity.
        scale  = self._disp_scale   # 0.25-1.0 — set via disp_scale parameter
        scaled = scale < 0.999

        if scaled:
            sw = int(orig_w * scale)
            sh = int(orig_h * scale)
            left_gray  = cv2.resize(left_gray,  (sw, sh), interpolation=cv2.INTER_AREA)
            right_gray = cv2.resize(right_gray, (sw, sh), interpolation=cv2.INTER_AREA)

            # Scale SGBM disparity range proportionally.
            # At 0.5× resolution a 1m object produces half the pixel disparity.
            min_d_s = max(0,  int(self._min_disp * scale / 16) * 16)
            num_d_s = max(16, int(self._num_disp * scale / 16) * 16)

            stereo = cv2.StereoSGBM_create(
                minDisparity      = min_d_s,
                numDisparities    = num_d_s,
                blockSize         = self._block_size,
                P1                = self._p1_scale * 3 * self._block_size ** 2,
                P2                = self._p2_scale * 3 * self._block_size ** 2,
                disp12MaxDiff     = self._disp12,
                uniquenessRatio   = self._uniqueness,
                speckleWindowSize = self._speckle_win,
                speckleRange      = self._speckle_rng,
                mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
            if self._use_wls:
                stereo_right = cv2.ximgproc.createRightMatcher(stereo)
                wls          = cv2.ximgproc.createDisparityWLSFilter(stereo)
                wls.setLambda(self._wls_lambda)
                wls.setSigmaColor(self._wls_sigma)
            else:
                stereo_right = wls = None
        else:
            min_d_s      = self._min_disp
            num_d_s      = self._num_disp
            stereo       = self._stereo
            stereo_right = getattr(self, '_stereo_right', None)
            wls          = getattr(self, '_wls', None)

        # ── CLAHE preprocessing ───────────────────────────────────────────
        if self._clahe_enable:
            left_gray  = self._clahe.apply(left_gray)
            right_gray = self._clahe.apply(right_gray)

        # ── Compute disparity ─────────────────────────────────────────────
        if self._use_wls and wls is not None:
            try:
                disp_left  = stereo.compute(left_gray, right_gray)
                disp_right = stereo_right.compute(right_gray, left_gray)
                disp_raw   = wls.filter(
                    disp_left, left_gray, disparity_map_right=disp_right
                )
            except Exception as e:
                self.get_logger().warn(f"WLS failed ({e}), falling back to plain SGBM.")
                self._use_wls = False
                disp_raw = stereo.compute(left_gray, right_gray)
        else:
            disp_raw = stereo.compute(left_gray, right_gray)

        # ── Convert to float32 disparity in full-resolution pixel units ───
        # Step 1: int16×16 fixed-point → real disparity in SCALED pixels
        disp_f32 = disp_raw.astype(np.float32) / 16.0

        if scaled:
            # Step 2: scaled pixels → full-resolution pixels
            # (37px at 0.5 scale = 74px at full resolution)
            disp_f32 = disp_f32 / scale
            # Step 3: spatially upscale the map — cv2.resize fills spatial
            # gaps via interpolation but does NOT change the values
            disp_f32 = cv2.resize(disp_f32, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)

        # ── Mark invalid pixels as NaN ────────────────────────────────────
        disp_f32[disp_f32 < 0] = float("nan")

        # ── Publish DisparityImage ────────────────────────────────────────
        disp_img_msg        = self.bridge.cv2_to_imgmsg(disp_f32, encoding="32FC1")
        disp_img_msg.header = left_msg.header

        out               = DisparityImage()
        out.header        = left_msg.header
        out.image         = disp_img_msg
        out.f             = float(self._focal_px)
        out.t             = float(self._baseline)
        # Always report full-resolution disparity range so downstream
        # consumers interpret the published float32 values correctly.
        out.min_disparity = float(self._min_disp)
        out.max_disparity = float(self._min_disp + self._num_disp)
        out.delta_d       = float(1.0 / 16.0)

        self._disp_pub.publish(out)



    # ── Parameter callback (live tuning) ──────────────────────────────────

    def _parameter_callback(self, params) -> SetParametersResult:
        """Allow live parameter updates via ros2 param set without restarting."""
        _SGBM_PARAMS = {
            "min_disp", "num_disp", "block_size",
            "p1_scale", "p2_scale", "disp12", "uniqueness",
            "speckle_win", "speckle_rng",
        }
        _CLAHE_PARAMS = {"clahe_enable", "clahe_clip", "clahe_tile"}
        recreate_matcher = False
        recreate_clahe   = False

        for param in params:
            attr = "_" + param.name
            if not hasattr(self, attr):
                self.get_logger().warn(f"Unknown parameter ignored: {param.name}")
                continue
            setattr(self, attr, param.value)
            if param.name in _SGBM_PARAMS:
                recreate_matcher = True
            if param.name in _CLAHE_PARAMS:
                recreate_clahe = True
            if param.name == "disp_aggregation":
                val = str(param.value).strip().lower()
                if val in ("median", "mean"):
                    self.disp_aggregation = val
                    self.get_logger().info(f"disp_aggregation set to '{val}'")
                else:
                    self.get_logger().warn(
                        f"disp_aggregation='{val}' not recognised — keeping '{self.disp_aggregation}'"
                    )
            if param.name == "disp_scale":
                new_scale = float(np.clip(param.value, 0.25, 1.0))
                self._disp_scale = new_scale
                self.get_logger().info(f"disp_scale set to {new_scale:.2f}")

        # Validate num_disp divisibility
        if self._num_disp % 16 != 0:
            self._num_disp = ((self._num_disp // 16) + 1) * 16

        if recreate_matcher:
            self._stereo = self._make_sgbm()
            if self._use_wls:
                self._stereo_right = cv2.ximgproc.createRightMatcher(self._stereo)
                self._wls          = cv2.ximgproc.createDisparityWLSFilter(self._stereo)
                self._wls.setLambda(self._wls_lambda)
                self._wls.setSigmaColor(self._wls_sigma)
            self.get_logger().info(
                f"SGBM recreated: block={self._block_size} "
                f"num_disp={self._num_disp} speckle_rng={self._speckle_rng}"
            )
        else:
            # WLS-only update
            if self._use_wls:
                self._wls.setLambda(self._wls_lambda)
                self._wls.setSigmaColor(self._wls_sigma)

        if recreate_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self._clahe_clip,
                tileGridSize=(self._clahe_tile, self._clahe_tile),
            )
            self.get_logger().info(
                f"CLAHE updated: enable={self._clahe_enable} "
                f"clip={self._clahe_clip} tile={self._clahe_tile}"
            )

        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = StereoDisparityNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()