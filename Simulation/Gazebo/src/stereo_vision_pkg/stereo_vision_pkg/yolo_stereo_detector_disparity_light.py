#!/usr/bin/env python3
import time
from typing import Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge

from ultralytics import YOLO

# Set True only when validating; set False for silent operation.
PRINT = False

# Colour palette — one colour per class index (BGR)
_PALETTE = [
    (0, 255, 0),    (0, 128, 255),  (255, 0, 0),    (0, 0, 255),
    (255, 255, 0),  (0, 255, 255),  (255, 0, 255),  (128, 0, 255),
    (255, 128, 0),  (0, 255, 128),  (128, 255, 0),  (0, 128, 128),
]

def _box_color(cls_id: int) -> tuple:
    return _PALETTE[cls_id % len(_PALETTE)]


class YoloStereoDetector(Node):
    """
    Lightweight YOLO detector for stereo pipelines.

    Key changes vs your original:
      - NO temp file / disk I/O (runs YOLO directly on numpy image)
      - Optional rate-limit (max_hz)
      - Optional left-only mode (default True) since depth-from-disparity typically needs only left boxes
      - Uses sensor-data QoS to avoid backlog latency
      - Optional class filtering + confidence threshold
      - Visualization removed (keeps matching with downstream depth node)
    """

    def __init__(self):
        super().__init__("yolo_stereo_detector", namespace="follower_robot")
        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter("model_name", "yolo11n.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_th", 0.15)        # 0.25 was too low for safety-critical
        self.declare_parameter("max_hz", 10.0)  # 0 => no rate limit
        self.declare_parameter("use_left", True)
        self.declare_parameter("use_right", False)  # off by default (saves ~50% compute)
        self.declare_parameter("tracker_config", "botsort.yaml")  # or bytetrack.yaml
        self.declare_parameter("track", True)  # True => model.track, False => model.predict

        # Filter: set to empty list to allow all
        # Example: ["person", "car"]
        self.declare_parameter("allowed_classes", ["person"])

        # Visualisation — draws boxes + labels on the left frame and shows
        # a cv2 window AND publishes an annotated Image topic.
        # Enable at launch:  --ros-args -p visualize:=true
        self.declare_parameter("visualize", False)
        self.declare_parameter("vis_topic", "/follower_robot/depth_cam/left/detection_vis")

        # Topics (keep your defaults to preserve matching)
        self.declare_parameter("left_topic", "/follower_robot/depth_cam/left/image_rect")
        self.declare_parameter("right_topic", "/follower_robot/depth_cam/right/image_rect")
        self.declare_parameter("left_out", "/follower_robot/depth_cam/left/detections")
        self.declare_parameter("right_out", "/follower_robot/depth_cam/right/detections")

        model_name = self.get_parameter("model_name").get_parameter_value().string_value
        imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        self.conf_th = float(self.get_parameter("conf_th").get_parameter_value().double_value)
        self.max_hz = float(self.get_parameter("max_hz").get_parameter_value().double_value)
        self.use_left = bool(self.get_parameter("use_left").get_parameter_value().bool_value)
        self.use_right = bool(self.get_parameter("use_right").get_parameter_value().bool_value)
        self.tracker_config = self.get_parameter("tracker_config").get_parameter_value().string_value
        self.do_track = bool(self.get_parameter("track").get_parameter_value().bool_value)

        allowed = self.get_parameter("allowed_classes").get_parameter_value().string_array_value
        self.allowed_classes = set(allowed) if allowed else None

        self.do_visualize = bool(self.get_parameter("visualize").get_parameter_value().bool_value)
        vis_topic = self.get_parameter("vis_topic").get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

        # Ultralytics name map can be dict or list depending on version/model
        names = getattr(self.model.model, "names", None)
        if isinstance(names, dict):
            self.class_names = names
        elif isinstance(names, (list, tuple)):
            self.class_names = {i: n for i, n in enumerate(names)}
        else:
            self.class_names = {}

        self.imgsz = imgsz

        # ---- Subscriptions / pubs ----
        left_topic = self.get_parameter("left_topic").get_parameter_value().string_value
        right_topic = self.get_parameter("right_topic").get_parameter_value().string_value
        left_out = self.get_parameter("left_out").get_parameter_value().string_value
        right_out = self.get_parameter("right_out").get_parameter_value().string_value

        if self.use_left:
            self.create_subscription(Image, left_topic, lambda m: self._cb(m, "left"), qos_profile_sensor_data)

        if self.use_right:
            self.create_subscription(Image, right_topic, lambda m: self._cb(m, "right"), qos_profile_sensor_data)

        self.left_pub = self.create_publisher(Detection2DArray, left_out, 10)
        self.right_pub = self.create_publisher(Detection2DArray, right_out, 10)

        # Visualisation publisher (annotated image)
        self.vis_pub = None
        if self.do_visualize:
            self.vis_pub = self.create_publisher(Image, vis_topic, 10)
            self.get_logger().info(
                f"Visualisation ON — window + topic: {vis_topic}\n"
                f"  Press 'q' in the cv2 window to quit."
            )

        # Rate limiting per side
        self._last_run = {"left": 0.0, "right": 0.0}
        self._min_period = (1.0 / self.max_hz) if self.max_hz and self.max_hz > 0 else 0.0

        self.get_logger().info(
            f"YOLO detector ready (left={self.use_left}, right={self.use_right}, "
            f"track={self.do_track}, imgsz={self.imgsz}, conf_th={self.conf_th}, "
            f"max_hz={self.max_hz}, visualize={self.do_visualize})"
        )

    def _cb(self, msg: Image, side: str):
        # Rate limit (keeps CPU stable; good when image is ~10 Hz)
        now = time.time()
        if self._min_period > 0.0 and (now - self._last_run[side]) < self._min_period:
            return
        self._last_run[side] = now

        # ROS → OpenCV
        # Images are mono8 (grayscale Y-plane from camera_streamer_fixed.py).
        # YOLO requires BGR — convert after decoding.
        try:
            mono   = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv_img = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().error(f"[cv_bridge] failed on {side}: {e}")
            return

        if cv_img is None or not isinstance(cv_img, np.ndarray):
            self.get_logger().error(f"[{side}] invalid image from cv_bridge")
            return

        if cv_img.ndim != 3 or cv_img.shape[2] != 3:
            self.get_logger().error(f"[{side}] unexpected image shape {cv_img.shape}")
            return

        # ── Resize for YOLO inference ─────────────────────────────────────
        # Stream resolution may be 1640×1232 (wide FOV) but YOLO runs fastest
        # at 640×480. Resize here so YOLO doesn't have to do it internally,
        # then scale bboxes back to full-stream coordinates afterwards.
        orig_h, orig_w = cv_img.shape[:2]
        yolo_w = int(self.get_parameter("imgsz").value)
        yolo_h = int(yolo_w * orig_h / orig_w)   # preserve aspect ratio
        if orig_w != yolo_w or orig_h != yolo_h:
            frame = np.ascontiguousarray(
                cv2.resize(cv_img, (yolo_w, yolo_h), interpolation=cv2.INTER_AREA)
            )
            scale_x = orig_w / yolo_w
            scale_y = orig_h / yolo_h
        else:
            frame   = np.ascontiguousarray(cv_img)
            scale_x = 1.0
            scale_y = 1.0

        # Inference
        try:
            t0 = time.time()
            if self.do_track:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    imgsz=self.imgsz,
                    conf=self.conf_th,
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.conf_th,
                    verbose=False,
                )
            dt = time.time() - t0
            if PRINT:
                self.get_logger().info(f"YOLO {side}: {dt*1000.0:.1f} ms")
        except Exception as e:
            self.get_logger().error(f"[YOLO {side}] inference error: {e}")
            return

        out = Detection2DArray()
        out.header = msg.header

        if not results:
            self._publish(side, out)
            return

        res0 = results[0]
        boxes = getattr(res0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            self._publish(side, out)
            return

        # Parse detections
        for box in boxes:
            # xyxy
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]

            # Scale bbox coordinates back to full-stream resolution.
            # YOLO ran on a resized frame — multiply by scale factors
            # so the downstream depth node samples the correct disparity region.
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue

            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, str(cls_id))

            if self.allowed_classes is not None and cls_name not in self.allowed_classes:
                continue

            track_id: Optional[int] = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id.cpu().numpy())
                except Exception:
                    track_id = None

            det = Detection2D()
            det.header = msg.header

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = float(cx)
            det.bbox.center.position.y = float(cy)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_name)
            hyp.hypothesis.score = float(conf)

            # Important for your downstream: keep det.id as string
            det.id = str(track_id) if track_id is not None else ""
            det.results.append(hyp)

            out.detections.append(det)

        # ── Visualisation ──────────────────────────────────────────────────
        if self.do_visualize and side == "left":
            vis = cv_img.copy()   # full-resolution BGR image for display
            for det in out.detections:
                cx_d = det.bbox.center.position.x
                cy_d = det.bbox.center.position.y
                bw   = det.bbox.size_x
                bh   = det.bbox.size_y
                x1v  = int(cx_d - bw / 2)
                y1v  = int(cy_d - bh / 2)
                x2v  = int(cx_d + bw / 2)
                y2v  = int(cy_d + bh / 2)

                cls_name = det.results[0].hypothesis.class_id if det.results else "?"
                conf_val = det.results[0].hypothesis.score    if det.results else 0.0
                tid      = det.id if det.id else "—"

                # look up class index for colour
                cls_idx = next(
                    (k for k, v in self.class_names.items() if v == cls_name), 0
                )
                color = _box_color(cls_idx)

                # bounding box
                cv2.rectangle(vis, (x1v, y1v), (x2v, y2v), color, 2)

                # label: "person  id=3  0.87"
                label = f"{cls_name}  id={tid}  {conf_val:.2f}"
                (tw, th), bl = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                # filled background rectangle for readability
                cv2.rectangle(
                    vis,
                    (x1v, max(y1v - th - bl - 4, 0)),
                    (x1v + tw + 4, max(y1v, th + bl + 4)),
                    color, -1
                )
                cv2.putText(
                    vis, label,
                    (x1v + 2, max(y1v - bl - 2, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                )

            # fps overlay
            fps_text = f"YOLO  {dt * 1000:.0f} ms  |  {len(out.detections)} det"
            cv2.putText(
                vis, fps_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            # cv2 window (requires a display; disable on headless machines)
            cv2.imshow("YOLO detections — left", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Visualisation window closed by user.")
                cv2.destroyAllWindows()
                self.do_visualize = False

            # also publish annotated image so rqt_image_view works too
            if self.vis_pub is not None:
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                    vis_msg.header = msg.header
                    self.vis_pub.publish(vis_msg)
                except Exception as e:
                    self.get_logger().warn(f"vis publish failed: {e}")

        self._publish(side, out)

    def _publish(self, side: str, msg: Detection2DArray):
        if side == "left":
            self.left_pub.publish(msg)
        else:
            self.right_pub.publish(msg)


def main():
    rclpy.init()
    node = YoloStereoDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()