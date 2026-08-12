#!/usr/bin/env python3
"""
obstacle_depth_speed_visualizer.py
====================================
Real-time viewer for stereo_box_depth_kalman_synced.py's output (obstacle
depth + Kalman-filtered relative velocity), paired with
yolo_stereo_detector_disparity_light.py's detections for bounding boxes.

Design note — why this draws its own boxes/text instead of using the main
node's built-in visualize:=true output
------------------------------------------------------------------------------
stereo_box_depth_kalman_synced.py already has a visualize option that draws
boxes+text and publishes an annotated image. But like the goal marker debug
image, that text is baked into the pixels in the camera's native (upside-down,
on this rig) orientation. Rotating that image 180° for correct on-screen
viewing would make the baked text unreadable — the same problem already
fixed for goal_marker_visualizer.py. So this node instead:
  1. subscribes to the RAW left rectified image and rotates it itself,
  2. subscribes to the YOLO Detection2DArray for bbox/class/track-id
     (the depth/speed JSON output has no bbox — only per-object state),
  3. subscribes to the obstacles_depth JSON string for depth/speed per ID,
  4. matches detections <-> depth results by track ID (same key logic as
     stereo_box_depth_kalman_synced.py's own _visualise: match by track_id
     only — falls back to "depth: N/A" for un-tracked detections, matching
     existing behaviour),
  5. draws box + text fresh, AFTER rotation, so everything stays legible.

Expected upstream topics (defaults match both upstream files)
------------------------------------------------------------------------------
  /follower_robot/depth_cam/left/image_rect   sensor_msgs/Image, mono8
  /follower_robot/depth_cam/left/detections   vision_msgs/Detection2DArray
  /follower_robot/obstacles_depth             std_msgs/String (JSON list)

Keyboard controls (focus must be on the OpenCV window)
--------------------------------------------------------
  s   -> save a snapshot into ./obstacle_captures/
  r   -> start recording video into ./obstacle_recordings/
  1   -> stop recording
  q / Esc -> quit

Note: Detection2DArray and the depth/speed String have no shared time-sync
mechanism the String has no header, so it can't go through message_filters
either. Both are cached from their own callbacks and drawn onto whichever
image frame triggers the redraw; in practice all three are published within
one upstream control cycle of each other.
"""

import os
import math
import json
import time
import datetime

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge


class ObstacleDepthSpeedVisualizer(Node):

    # Same palette as stereo_box_depth_kalman_synced.py, for visual continuity.
    _PALETTE = [
        (0, 255, 0),   (0, 128, 255), (255, 0, 0),   (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),  (128, 0, 255),
        (255, 128, 0), (0, 255, 128), (128, 255, 0),  (0, 128, 128),
    ]

    def __init__(self):
        super().__init__("obstacle_depth_speed_visualizer", namespace="follower_robot")
        self.bridge = CvBridge()

        # ── Topics ───────────────────────────────────────────────────────
        self.declare_parameter("left_image_topic", "/follower_robot/depth_cam/left/image_rect")
        self.declare_parameter("detections_topic", "/follower_robot/depth_cam/left/detections")
        self.declare_parameter("depth_topic",       "/follower_robot/obstacles_depth")

        # ── Behaviour ────────────────────────────────────────────────────
        self.declare_parameter("rotate_180", True)          # matches camera mount, viewer-only
        self.declare_parameter("detections_max_age_s", 1.0)  # stop drawing stale boxes

        # ── Display / output ─────────────────────────────────────────────
        self.declare_parameter("window_name",   "Obstacle Depth & Speed Viewer")
        self.declare_parameter("display_scale", 0.65)
        self.declare_parameter("save_dir",      "obstacle_captures")
        self.declare_parameter("record_dir",    "obstacle_recordings")
        self.declare_parameter("record_fps",    5.0)

        left_image_topic = str(self.get_parameter("left_image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        depth_topic       = str(self.get_parameter("depth_topic").value)

        self._rotate_180    = bool(self.get_parameter("rotate_180").value)
        self._det_max_age_s = float(self.get_parameter("detections_max_age_s").value)

        self._window_name   = str(self.get_parameter("window_name").value)
        self._display_scale = float(self.get_parameter("display_scale").value)
        self._record_fps    = float(self.get_parameter("record_fps").value)

        # ── Output directories (next to this script) ────────────────────
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._save_dir = os.path.join(script_dir, str(self.get_parameter("save_dir").value))
        self._record_dir = os.path.join(script_dir, str(self.get_parameter("record_dir").value))
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._record_dir, exist_ok=True)

        # ── Cached latest telemetry ─────────────────────────────────────
        self._latest_detections: Detection2DArray = None
        self._last_det_time = 0.0
        self._results_by_id = {}   # str(track_id) -> result dict from the JSON

        # ── Recording / snapshot state ───────────────────────────────────
        self._video_writer = None
        self._record_path  = None
        self._last_full_frame = None
        self._frame_count  = 0

        # ── Subscriptions ────────────────────────────────────────────────
        self.create_subscription(Image, left_image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, detections_topic, self._det_cb, 10)
        self.create_subscription(String, depth_topic, self._depth_cb, 10)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "ObstacleDepthSpeedVisualizer ready.\n"
            f"  left image  : {left_image_topic}\n"
            f"  detections  : {detections_topic}\n"
            f"  depth/speed : {depth_topic}\n"
            f"  rotate_180  : {self._rotate_180}\n"
            f"  captures -> {self._save_dir}\n"
            f"  recordings -> {self._record_dir}\n"
            "  Keys: 's' save snapshot | 'r' start recording | '1' stop recording | 'q'/Esc quit"
        )

    # ── Telemetry cache callbacks ───────────────────────────────────────
    def _det_cb(self, msg: Detection2DArray) -> None:
        self._latest_detections = msg
        self._last_det_time = time.time()

    def _depth_cb(self, msg: String) -> None:
        try:
            results = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse obstacles_depth JSON: {e}")
            return
        by_id = {}
        for r in results:
            rid = r.get("id")
            if rid is not None:
                by_id[str(rid)] = r
        self._results_by_id = by_id

    # ── Detection helpers (mirror stereo_box_depth_kalman_synced.py) ────
    @staticmethod
    def _bbox_from_det(det):
        if det.bbox is None:
            return None
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        bw = det.bbox.size_x
        bh = det.bbox.size_y
        return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)

    @classmethod
    def _box_color(cls, cls_id_str: str) -> tuple:
        try:
            idx = int(cls_id_str)
        except ValueError:
            idx = sum(ord(c) for c in cls_id_str)
        return cls._PALETTE[idx % len(cls._PALETTE)]

    # ── Drawing ──────────────────────────────────────────────────────────
    def _draw_obstacle_text(self, vis, x1, y1, cls_str, tid, result) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_big, font_small, thick = 0.60, 0.50, 2
        tid_str = tid if tid else "—"

        if result is not None:
            bearing_deg = math.degrees(float(result.get("bearing_rad", 0.0)))
            lines = [
                (f"{cls_str}  id={tid_str}", (255, 255, 255), font_big),
                (f"depth(KF): {float(result.get('depth_m', 0.0)):.2f} m", (0, 255, 0), font_small),
                (f"raw depth: {float(result.get('raw_depth_m', 0.0)):.2f} m", (0, 255, 255), font_small),
                (f"disp: {float(result.get('raw_disp', 0.0)):.1f}px  brg: {bearing_deg:.1f} deg",
                 (255, 255, 0), font_small),
                (f"speed: {float(result.get('speed_ms', 0.0)):.2f} m/s  "
                 f"vx={float(result.get('vx_ms', 0.0)):+.2f} vy={float(result.get('vy_ms', 0.0)):+.2f}",
                 (200, 200, 200), font_small),
            ]
        else:
            lines = [
                (f"{cls_str}  id={tid_str}", (255, 255, 255), font_big),
                ("depth: N/A", (100, 100, 255), font_small),
            ]

        line_h = 18
        pad = 4
        text_y = y1 - pad - (len(lines) - 1) * line_h
        text_y = max(text_y, line_h * len(lines))

        bg_w = max(cv2.getTextSize(l[0], font, l[2], thick)[0][0] for l in lines) + 8
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

    def _draw_hud(self, img, n_drawn: int, dets_fresh: bool) -> None:
        status = f"tracked: {n_drawn}" if dets_fresh else "tracked: 0 (stale/no detections)"
        text = f"OBSTACLE DEPTH & SPEED | {status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
        cv2.rectangle(img, (10, 10), (10 + tw + 16, 10 + th + 16), (20, 20, 20), -1)
        cv2.putText(img, text, (18, 10 + th + 6), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_rec_indicator(self, img) -> None:
        h, w = img.shape[:2]
        blink_on = (self._frame_count // 8) % 2 == 0
        if blink_on:
            cv2.circle(img, (w - 90, 26), 8, (0, 0, 255), -1)
        cv2.putText(img, "REC", (w - 70, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # ── Main image callback ─────────────────────────────────────────────
    def _image_cb(self, msg: Image) -> None:
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self._rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        self._frame_count += 1
        vis = frame.copy()
        h, w = vis.shape[:2]

        now = time.time()
        dets_fresh = (
            self._latest_detections is not None
            and (now - self._last_det_time) <= self._det_max_age_s
        )

        n_drawn = 0
        if dets_fresh:
            for det in self._latest_detections.detections:
                bbox = self._bbox_from_det(det)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox

                if self._rotate_180:
                    rx1, ry1, rx2, ry2 = w - x2, h - y2, w - x1, h - y1
                else:
                    rx1, ry1, rx2, ry2 = x1, y1, x2, y2

                rx1 = max(0, min(w - 1, int(rx1))); rx2 = max(0, min(w, int(rx2)))
                ry1 = max(0, min(h - 1, int(ry1))); ry2 = max(0, min(h, int(ry2)))
                if rx2 <= rx1 or ry2 <= ry1:
                    continue

                cls_str = det.results[0].hypothesis.class_id if det.results else "?"
                tid = det.id if det.id else None
                color = self._box_color(cls_str)

                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 2)

                result = self._results_by_id.get(str(tid)) if tid else None
                self._draw_obstacle_text(vis, rx1, ry1, cls_str, tid, result)
                n_drawn += 1

        self._draw_hud(vis, n_drawn, dets_fresh)

        # What gets saved/recorded — HUD+boxes, no REC indicator baked in.
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
        fname = f"obstacles_{ts}.png"
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
        fname = f"obstacles_{ts}.mp4"
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
    node = ObstacleDepthSpeedVisualizer()
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
