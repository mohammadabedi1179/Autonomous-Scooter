#!/usr/bin/env python3
"""
stereo_disp_4panel_viewer.py
=============================
ROS2 diagnostic node that subscribes to the SAME rectified left/right image
topics as stereo_disparity_node.py and displays a real-time 2x2 panel:

  ┌───────────────────────────┬───────────────────────────┐
  │ 1. Raw left image         │ 2. Raw disparity           │
  │    (as received)          │    (no CLAHE, no WLS)      │
  ├───────────────────────────┼───────────────────────────┤
  │ 3. CLAHE disparity        │ 4. WLS + CLAHE disparity   │
  │    (no WLS)               │    (matches main node)     │
  └───────────────────────────┴───────────────────────────┘

Each quadrant is rotated 180° purely for on-screen viewing (matches the
physical camera mount orientation) — this does not touch the underlying
topics or data, only what's drawn/saved in the panel.

Press 's'  -> saves the current panel as PNG into ./4section_disp_map/
              (created next to this script).
Press 'q' or Esc -> quits.

All SGBM / CLAHE / WLS parameters and the downscale->CLAHE->SGBM->WLS
processing order mirror stereo_disparity_node.py exactly, so panels 2-4
are a fair, apples-to-apples comparison of what each stage contributes.

Requirements: opencv-contrib-python (cv2.ximgproc) for the WLS panel.
If it's missing, panel 4 falls back to panel 3 with a warning label.

Run (example):
    ros2 run <pkg> stereo_disp_4panel_viewer.py
    # or directly:
    python3 stereo_disp_4panel_viewer.py --ros-args -p wls_lambda:=12000.0
"""

import os
import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class StereoDisp4PanelViewer(Node):

    def __init__(self):
        super().__init__("stereo_disp_4panel_viewer", namespace="follower_robot")
        self.bridge = CvBridge()

        # ── Parameters (mirror stereo_disparity_node.py) ───────────────────
        self.declare_parameter("left_topic",  "/follower_robot/depth_cam/left/image_rect")
        self.declare_parameter("right_topic", "/follower_robot/depth_cam/right/image_rect")
        self.declare_parameter("sync_slop",   0.12)
        self.declare_parameter("sync_queue",  10)

        self.declare_parameter("min_disp",    16)
        self.declare_parameter("num_disp",    176)   # must be divisible by 16
        self.declare_parameter("block_size",  7)
        self.declare_parameter("p1_scale",    8)
        self.declare_parameter("p2_scale",    32)
        self.declare_parameter("disp12",      1)
        self.declare_parameter("uniqueness",  5)
        self.declare_parameter("speckle_win", 150)
        self.declare_parameter("speckle_rng", 16)

        self.declare_parameter("wls_lambda",  8000.0)
        self.declare_parameter("wls_sigma",   1.5)

        self.declare_parameter("clahe_clip",  3.0)
        self.declare_parameter("clahe_tile",  8)

        # Downscale before SGBM — identical trick to the main node: cancels
        # out of depth = fx*baseline/disp, pure speedup, no accuracy loss.
        self.declare_parameter("disp_scale",  0.5)

        # Display / save
        self.declare_parameter("panel_scale", 0.45)   # shrink each quadrant for on-screen viewing
        self.declare_parameter("window_name", "Stereo Disparity - 4 Panel")
        self.declare_parameter("save_dir",    "4section_disp_map")

        self._left_topic  = str(self.get_parameter("left_topic").value)
        self._right_topic = str(self.get_parameter("right_topic").value)
        sync_slop  = float(self.get_parameter("sync_slop").value)
        sync_queue = int(self.get_parameter("sync_queue").value)

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
        self._clahe_clip  = float(self.get_parameter("clahe_clip").value)
        self._clahe_tile  = int(self.get_parameter("clahe_tile").value)
        self._disp_scale  = float(np.clip(self.get_parameter("disp_scale").value, 0.25, 1.0))
        self._panel_scale = float(self.get_parameter("panel_scale").value)
        self._window_name = str(self.get_parameter("window_name").value)

        if self._num_disp % 16 != 0:
            corrected = ((self._num_disp // 16) + 1) * 16
            self.get_logger().warn(f"num_disp={self._num_disp} not div. by 16 -> {corrected}")
            self._num_disp = corrected

        # ── Save directory (next to this script) ────────────────────────────
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._save_dir = os.path.join(script_dir, str(self.get_parameter("save_dir").value))
        os.makedirs(self._save_dir, exist_ok=True)
        self.get_logger().info(f"Save directory: {self._save_dir}")

        # ── CLAHE ────────────────────────────────────────────────────────
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip,
            tileGridSize=(self._clahe_tile, self._clahe_tile),
        )

        # ── SGBM + WLS (full-res instances; scaled instances rebuilt per
        #    frame when disp_scale < 1.0, identical to main node) ─────────
        self._stereo = self._make_sgbm(self._min_disp, self._num_disp)
        self._use_wls = True
        try:
            self._stereo_right = cv2.ximgproc.createRightMatcher(self._stereo)
            self._wls = cv2.ximgproc.createDisparityWLSFilter(self._stereo)
            self._wls.setLambda(self._wls_lambda)
            self._wls.setSigmaColor(self._wls_sigma)
        except AttributeError:
            self.get_logger().warn(
                "opencv-contrib not installed -- WLS panel disabled "
                "(falls back to CLAHE-only). Install opencv-contrib-python "
                "for the full 4-way comparison."
            )
            self._use_wls = False
            self._stereo_right = None
            self._wls = None

        # ── Subscriptions ────────────────────────────────────────────────
        self._left_sub  = Subscriber(self, Image, self._left_topic,  qos_profile=10)
        self._right_sub = Subscriber(self, Image, self._right_topic, qos_profile=10)
        self._sync = ApproximateTimeSynchronizer(
            [self._left_sub, self._right_sub], queue_size=sync_queue, slop=sync_slop
        )
        self._sync.registerCallback(self._cb)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"4-panel viewer ready.\n"
            f"  left : {self._left_topic}\n"
            f"  right: {self._right_topic}\n"
            f"  Press 's' in the window to save, 'q'/Esc to quit."
        )

    # ── SGBM factory (identical formula to main node) ──────────────────────
    def _make_sgbm(self, min_d, num_d):
        p1 = self._p1_scale * 3 * self._block_size ** 2
        p2 = self._p2_scale * 3 * self._block_size ** 2
        return cv2.StereoSGBM_create(
            minDisparity      = min_d,
            numDisparities    = num_d,
            blockSize         = self._block_size,
            P1                = p1,
            P2                = p2,
            disp12MaxDiff     = self._disp12,
            uniquenessRatio   = self._uniqueness,
            speckleWindowSize = self._speckle_win,
            speckleRange      = self._speckle_rng,
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    # ── SGBM/WLS raw output -> full-res float32 disparity, NaN = invalid ───
    def _to_full_res(self, disp_raw_i16, scale, orig_w, orig_h):
        scaled = scale < 0.999
        d = disp_raw_i16.astype(np.float32) / 16.0
        if scaled:
            d = d / scale
            d = cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        d[d < 0] = np.nan
        return d

    # ── colorize a full-res disparity map onto a common min/max scale ──────
    def _colorize(self, disp_f32):
        valid = ~np.isnan(disp_f32)
        rng = float(self._num_disp) if self._num_disp > 0 else 1.0
        filled = np.nan_to_num(disp_f32, nan=float(self._min_disp))
        clipped = np.clip((filled - self._min_disp) / rng * 255.0, 0, 255)
        norm = clipped.astype(np.uint8)
        norm[~valid] = 0
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color[~valid] = (0, 0, 0)
        return color

    def _save_panel(self, panel):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = f"panel_{ts}.png"
        fpath = os.path.join(self._save_dir, fname)
        cv2.imwrite(fpath, panel)
        self.get_logger().info(f"Saved -> {fpath}")

    # ── main callback ───────────────────────────────────────────────────
    def _cb(self, left_msg: Image, right_msg: Image) -> None:
        try:
            left_gray  = self.bridge.imgmsg_to_cv2(left_msg,  desired_encoding="mono8")
            right_gray = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        orig_h, orig_w = left_gray.shape[:2]
        raw_left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)

        # ── optional downscale before SGBM (matches main node exactly) ────
        scale = self._disp_scale
        scaled = scale < 0.999
        if scaled:
            sw, sh = int(orig_w * scale), int(orig_h * scale)
            left_s  = cv2.resize(left_gray,  (sw, sh), interpolation=cv2.INTER_AREA)
            right_s = cv2.resize(right_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            min_d_s = max(0,  int(self._min_disp * scale / 16) * 16)
            num_d_s = max(16, int(self._num_disp * scale / 16) * 16)
            stereo = self._make_sgbm(min_d_s, num_d_s)
            if self._use_wls:
                stereo_right = cv2.ximgproc.createRightMatcher(stereo)
                wls = cv2.ximgproc.createDisparityWLSFilter(stereo)
                wls.setLambda(self._wls_lambda)
                wls.setSigmaColor(self._wls_sigma)
            else:
                stereo_right = wls = None
        else:
            left_s, right_s = left_gray, right_gray
            stereo, stereo_right, wls = self._stereo, self._stereo_right, self._wls

        # ── Panel 2: raw disparity — no CLAHE, no WLS ──────────────────────
        disp_raw_i16 = stereo.compute(left_s, right_s)
        disp_2 = self._to_full_res(disp_raw_i16, scale, orig_w, orig_h)

        # ── CLAHE preprocessing (shared by panels 3 & 4, same order as
        #    main node: downscale first, THEN CLAHE) ───────────────────────
        left_c  = self._clahe.apply(left_s)
        right_c = self._clahe.apply(right_s)

        # ── Panel 3: CLAHE, no WLS ──────────────────────────────────────
        disp_left_c_i16 = stereo.compute(left_c, right_c)
        disp_3 = self._to_full_res(disp_left_c_i16, scale, orig_w, orig_h)

        # ── Panel 4: WLS + CLAHE (reuses panel-3's left disparity) ────────
        if self._use_wls and wls is not None:
            try:
                disp_right_c_i16 = stereo_right.compute(right_c, left_c)
                disp_wls_i16 = wls.filter(
                    disp_left_c_i16, left_c, disparity_map_right=disp_right_c_i16
                )
                disp_4 = self._to_full_res(disp_wls_i16, scale, orig_w, orig_h)
            except Exception as e:
                self.get_logger().warn(f"WLS failed ({e}); showing CLAHE-only.")
                disp_4 = disp_3
        else:
            disp_4 = disp_3

        # ── compose panel (rotated 180° for viewing only — camera mount
        #    convention; does not affect the underlying data/topics) ───────
        q1 = cv2.rotate(raw_left_bgr,           cv2.ROTATE_180)
        q2 = cv2.rotate(self._colorize(disp_2), cv2.ROTATE_180)
        q3 = cv2.rotate(self._colorize(disp_3), cv2.ROTATE_180)
        q4 = cv2.rotate(self._colorize(disp_4), cv2.ROTATE_180)

        pw = max(1, int(orig_w * self._panel_scale))
        ph = max(1, int(orig_h * self._panel_scale))
        q1r = cv2.resize(q1, (pw, ph))
        q2r = cv2.resize(q2, (pw, ph))
        q3r = cv2.resize(q3, (pw, ph))
        q4r = cv2.resize(q4, (pw, ph))

        panel = np.vstack([np.hstack([q1r, q2r]), np.hstack([q3r, q4r])])

        cv2.imshow(self._window_name, panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord('S')):
            self._save_panel(panel)
        elif key in (ord('q'), ord('Q'), 27):  # 27 = Esc
            self.get_logger().info("Quit key pressed -- shutting down.")
            cv2.destroyAllWindows()
            if rclpy.ok():
                rclpy.shutdown()


def main():
    rclpy.init()
    node = StereoDisp4PanelViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()