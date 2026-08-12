#!/usr/bin/env python3
"""
goal_marker_visualizer.py
==========================
Real-time viewer for goal_marker_apriltag_depth_node.py's output.
No changes required to the goal marker node itself.

Design note — how the box/label stay legible after rotation
------------------------------------------------------------------------
The goal marker node's debug image bakes the tag's box + "tag N" label into
the pixels, drawn upright in the camera's native (upside-down, on this rig)
orientation. This viewer rotates the whole frame 180° for correct on-screen
viewing — but that also flips the already-baked text, making it unreadable.

Fix, without touching the producer node: GoalMarkerState already carries the
tag's center (cx, cy) in the ORIGINAL (un-rotated) frame. After rotating the
whole frame, that point maps to (W-1-cx, H-1-cy). A small patch cropped from
the ROTATED frame around that mirrored point is exactly the original patch
rotated 180° (since the whole image is one big 180° rotation, any sub-patch
of it equals the corresponding source patch, rotated). Rotating that patch
another 180° cancels the flip locally, restoring the upright box + readable
"tag N" text — right at the correct (mirrored) location in the final image.
Only that small patch is touched; the rest of the frame stays correctly
rotated.

Expected upstream topics (defaults match goal_marker_apriltag_depth_node.py)
------------------------------------------------------------------------------
  /follower_robot/depth_cam/goal_marker_debug   sensor_msgs/Image, bgr8
  /follower_robot/depth_cam/goal_marker_state   detectors_msgs/GoalMarkerState
  /follower_robot/depth_cam/goal_marker_vx      std_msgs/Float32
  /follower_robot/depth_cam/goal_marker_vy      std_msgs/Float32
  /follower_robot/depth_cam/goal_marker_speed   std_msgs/Float32

Keyboard controls (focus must be on the OpenCV window)
--------------------------------------------------------
  s   -> save a snapshot (current HUD-annotated frame) into ./goal_marker_captures/
  r   -> start recording video into ./goal_marker_recordings/
  1   -> stop recording
  q / Esc -> quit

Note: state/vx/vy/speed are cached from their own subscriptions (Float32 has
no header, so they can't be time-synced with message_filters) and drawn onto
whichever debug-image frame triggers the redraw. In practice all of these are
published back-to-back inside the same upstream callback, so drift between
them is at most one control cycle — not visually meaningful.
"""

import os
import math
import datetime

import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

from detectors_msgs.msg import GoalMarkerState


class GoalMarkerVisualizer(Node):

    def __init__(self):
        super().__init__("goal_marker_visualizer", namespace="follower_robot")
        self.bridge = CvBridge()

        # ── Topics ───────────────────────────────────────────────────────
        self.declare_parameter("debug_image_topic", "/follower_robot/depth_cam/goal_marker_debug")
        self.declare_parameter("state_topic",       "/follower_robot/depth_cam/goal_marker_state")
        self.declare_parameter("vx_topic",           "/follower_robot/depth_cam/goal_marker_vx")
        self.declare_parameter("vy_topic",           "/follower_robot/depth_cam/goal_marker_vy")
        self.declare_parameter("speed_topic",        "/follower_robot/depth_cam/goal_marker_speed")

        # Half-size (px) of the square patch around the tag's (cx,cy) that
        # gets locally double-rotated to keep the box+label legible. Bump
        # this up if the tag/label get cut off at close range (bigger
        # on-screen tag size).
        self.declare_parameter("tag_patch_half_px", 160)

        # ── Display / output ─────────────────────────────────────────────
        self.declare_parameter("window_name",   "Goal Marker Viewer")
        self.declare_parameter("display_scale", 0.65)   # shrink for on-screen viewing only
        self.declare_parameter("save_dir",      "goal_marker_captures")
        self.declare_parameter("record_dir",    "goal_marker_recordings")
        self.declare_parameter("record_fps",    15.0)

        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        state_topic       = str(self.get_parameter("state_topic").value)
        vx_topic          = str(self.get_parameter("vx_topic").value)
        vy_topic          = str(self.get_parameter("vy_topic").value)
        speed_topic       = str(self.get_parameter("speed_topic").value)

        self._tag_patch_half_px = int(self.get_parameter("tag_patch_half_px").value)

        self._window_name   = str(self.get_parameter("window_name").value)
        self._display_scale = float(self.get_parameter("display_scale").value)
        self._record_fps    = float(self.get_parameter("record_fps").value)

        # ── Output directories (next to this script) ────────────────────
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._save_dir = os.path.join(script_dir, str(self.get_parameter("save_dir").value))
        self._record_dir = os.path.join(script_dir, str(self.get_parameter("record_dir").value))
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._record_dir, exist_ok=True)

        # ── Cached latest telemetry (updated by their own callbacks) ────
        self._state_visible = False
        self._depth_m       = 0.0
        self._bearing_rad   = 0.0
        self._cx             = -1.0
        self._cy             = -1.0
        self._vx_ms          = 0.0
        self._vy_ms          = 0.0
        self._speed_ms       = 0.0

        # ── Recording / snapshot state ───────────────────────────────────
        self._video_writer = None
        self._record_path  = None
        self._last_full_frame = None
        self._frame_count  = 0

        # ── Subscriptions ────────────────────────────────────────────────
        self.create_subscription(Image, debug_image_topic, self._image_cb, 10)
        self.create_subscription(GoalMarkerState, state_topic, self._state_cb, 10)
        self.create_subscription(Float32, vx_topic, self._vx_cb, 10)
        self.create_subscription(Float32, vy_topic, self._vy_cb, 10)
        self.create_subscription(Float32, speed_topic, self._speed_cb, 10)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "GoalMarkerVisualizer ready.\n"
            f"  debug image : {debug_image_topic}\n"
            f"  state       : {state_topic}\n"
            f"  vx/vy/speed : {vx_topic} | {vy_topic} | {speed_topic}\n"
            f"  tag_patch_half_px : {self._tag_patch_half_px}\n"
            f"  captures -> {self._save_dir}\n"
            f"  recordings -> {self._record_dir}\n"
            "  Keys: 's' save snapshot | 'r' start recording | '1' stop recording | 'q'/Esc quit"
        )

    # ── Telemetry cache callbacks ───────────────────────────────────────
    def _state_cb(self, msg: GoalMarkerState) -> None:
        self._state_visible = bool(msg.visible)
        self._depth_m       = float(msg.depth_m)
        self._bearing_rad   = float(msg.bearing_rad)
        self._cx             = float(msg.cx)
        self._cy             = float(msg.cy)

    def _vx_cb(self, msg: Float32) -> None:
        self._vx_ms = float(msg.data)

    def _vy_cb(self, msg: Float32) -> None:
        self._vy_ms = float(msg.data)

    def _speed_cb(self, msg: Float32) -> None:
        self._speed_ms = float(msg.data)

    # ── HUD drawing ──────────────────────────────────────────────────────
    def _draw_hud(self, img) -> None:
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        line_h = 24
        pad = 8

        if self._state_visible:
            lines = [
                ("GOAL MARKER", (0, 255, 255)),
                ("Visible : YES", (0, 255, 0)),
                (f"Depth   : {self._depth_m:.2f} m", (0, 255, 0)),
                (f"Bearing : {math.degrees(self._bearing_rad):+.1f} deg "
                 f"({self._bearing_rad:+.3f} rad)", (0, 255, 0)),
                (f"Center  : ({self._cx:.1f}, {self._cy:.1f}) px", (200, 200, 200)),
                (f"Speed   : {self._speed_ms:.3f} m/s", (255, 255, 0)),
                (f"vx,vy   : {self._vx_ms:+.3f}, {self._vy_ms:+.3f} m/s", (255, 255, 0)),
            ]
        else:
            lines = [
                ("GOAL MARKER", (0, 255, 255)),
                ("Visible : NO", (0, 0, 255)),
            ]

        box_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t, _ in lines) + 2 * pad
        box_h = line_h * len(lines) + 2 * pad

        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        for i, (text, color) in enumerate(lines):
            y = 10 + pad + line_h * (i + 1) - 6
            cv2.putText(img, text, (10 + pad, y), font, scale, color, thick, cv2.LINE_AA)

        if not self._state_visible:
            banner = "NO GOAL MARKER DETECTED"
            (tw, th), _ = cv2.getTextSize(banner, font, 1.0, 2)
            bx = max(0, (w - tw) // 2)
            by = h // 2
            cv2.putText(img, banner, (bx, by), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_rec_indicator(self, img) -> None:
        h, w = img.shape[:2]
        blink_on = (self._frame_count // 8) % 2 == 0
        if blink_on:
            cv2.circle(img, (w - 90, 26), 8, (0, 0, 255), -1)
        cv2.putText(img, "REC", (w - 70, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2, cv2.LINE_AA)

    def _unrotate_tag_patch(self, img) -> None:
        """Crop a small patch around the tag's mirrored (cx,cy) and rotate
        it another 180° in place. Since the whole frame is already one big
        180° rotation, this cancels the flip locally — restoring the
        upstream box + 'tag N' label to their original, legible, upright
        appearance, at the correct (mirrored) position in the final image.
        Only this patch is touched; the rest of the frame is untouched."""
        if not self._state_visible or self._cx < 0 or self._cy < 0:
            return

        h, w = img.shape[:2]
        cx1 = (w - 1) - self._cx
        cy1 = (h - 1) - self._cy

        half = self._tag_patch_half_px
        x1 = int(max(0, cx1 - half))
        y1 = int(max(0, cy1 - half))
        x2 = int(min(w, cx1 + half))
        y2 = int(min(h, cy1 + half))
        if x2 <= x1 or y2 <= y1:
            return

        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return
        img[y1:y2, x1:x2] = cv2.rotate(patch, cv2.ROTATE_180)

    # ── Main image callback ─────────────────────────────────────────────
    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Rotated 180° for viewing only (matches camera mount orientation) —
        # does not affect the underlying topic/data, only what's drawn/saved
        # here. The tag box/label baked into these pixels rotates with it;
        # _unrotate_tag_patch() below locally cancels that so the label text
        # stays legible.
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        self._frame_count += 1
        vis = frame.copy()
        self._unrotate_tag_patch(vis)
        self._draw_hud(vis)

        # What gets saved/recorded — HUD only, no REC indicator baked in.
        self._last_full_frame = vis

        if self._video_writer is not None:
            self._video_writer.write(vis)

        display_frame = vis.copy()
        if self._video_writer is not None:
            self._draw_rec_indicator(display_frame)

        if self._display_scale < 0.999:
            display_frame = cv2.resize(
                display_frame, None,
                fx=self._display_scale, fy=self._display_scale,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(self._window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)

    # ── Keyboard handling ───────────────────────────────────────────────
    def _handle_key(self, key: int) -> None:
        if key in (ord('s'), ord('S')):
            self._save_snapshot()
        elif key in (ord('r'), ord('R')):
            self._start_recording()
        elif key == ord('1'):
            self._stop_recording()
        elif key in (ord('q'), ord('Q'), 27):  # 27 = Esc
            self.get_logger().info("Quit key pressed -- shutting down.")
            self._stop_recording()
            cv2.destroyAllWindows()
            if rclpy.ok():
                rclpy.shutdown()

    def _save_snapshot(self) -> None:
        if self._last_full_frame is None:
            self.get_logger().warn("No frame received yet -- cannot save.")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = f"goal_marker_{ts}.png"
        fpath = os.path.join(self._save_dir, fname)
        cv2.imwrite(fpath, self._last_full_frame)
        self.get_logger().info(f"Snapshot saved -> {fpath}")

    def _start_recording(self) -> None:
        if self._video_writer is not None:
            self.get_logger().info("Already recording.")
            return
        if self._last_full_frame is None:
            self.get_logger().warn("No frame received yet -- cannot start recording.")
            return
        h, w = self._last_full_frame.shape[:2]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"goal_marker_{ts}.mp4"
        fpath = os.path.join(self._record_dir, fname)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(fpath, fourcc, self._record_fps, (w, h))
        if not writer.isOpened():
            self.get_logger().error(f"Failed to open video writer for {fpath}")
            return
        self._video_writer = writer
        self._record_path = fpath
        self.get_logger().info(f"Recording started -> {fpath} @ {self._record_fps:.1f} fps")

    def _stop_recording(self) -> None:
        if self._video_writer is None:
            return
        self._video_writer.release()
        self.get_logger().info(f"Recording stopped -> {self._record_path}")
        self._video_writer = None
        self._record_path = None


def main():
    rclpy.init()
    node = GoalMarkerVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_recording()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
