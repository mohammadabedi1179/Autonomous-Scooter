#!/usr/bin/env python3
"""
Obstacle depth accuracy logger for ROS 2 / Gazebo.

Compares /follower_robot/obstacles_depth JSON output against Gazebo /model_states.
It logs one row per ground-truth obstacle per timer tick, plus optional false-positive rows.

Coordinate conventions assumed from stereo_box_depth_kalman_synced.py:
  - Detection JSON x_m, y_m are in base_link frame: +x forward, +y left.
  - Detection JSON depth_m and raw_depth_m are camera-forward depths, not base-link range.
  - Detection bearing_rad is image/camera bearing: positive to image-right.

Ground-truth comparisons logged:
  - Base-link XY errors: compare detection x_m/y_m to GT local x/y.
  - Base-link range/bearing errors: compare hypot(x_m,y_m), atan2(y_m,x_m).
  - Camera-forward depth errors: compare depth_m/raw_depth_m to GT x_cam = gt_x_base - camera_x_offset_m.
  - Camera/image bearing error: compare bearing_rad to atan2(camera_y_offset_m - gt_y_base, gt_x_base - camera_x_offset_m).
"""

import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def local_from_world(rx: float, ry: float, ryaw: float, wx: float, wy: float) -> Tuple[float, float]:
    dx = wx - rx
    dy = wy - ry
    lx = math.cos(ryaw) * dx + math.sin(ryaw) * dy
    ly = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
    return lx, ly


def vel_local_from_world(ryaw: float, vx: float, vy: float) -> Tuple[float, float]:
    lx = math.cos(ryaw) * vx + math.sin(ryaw) * vy
    ly = -math.sin(ryaw) * vx + math.cos(ryaw) * vy
    return lx, ly


def parse_bool_param(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ObstacleDepthAccuracyLogger(Node):
    def __init__(self):
        super().__init__("obstacle_depth_accuracy_logger")

        # Topics / model names
        self.declare_parameter("obstacles_topic", "/follower_robot/obstacles_depth")
        self.declare_parameter("model_states_topic", "/model_states")
        self.declare_parameter("model_states_qos_reliability", "best_effort")  # best_effort or reliable
        self.declare_parameter("robot_model_name", "my_robot")
        self.declare_parameter("obstacle_model_prefix", "person_standing_")
        self.declare_parameter("obstacle_count", 10)
        self.declare_parameter("obstacle_model_names", "")  # comma-separated override

        # Geometry / matching
        self.declare_parameter("camera_x_offset_m", 0.4)
        self.declare_parameter("camera_y_offset_m", 0.0)
        self.declare_parameter("match_max_dist_m", 0.75)
        self.declare_parameter("max_gt_range_m", 10.0)
        self.declare_parameter("min_gt_forward_m", 0.05)
        self.declare_parameter("fov_half_angle_rad", 0.70)  # observable-zone estimate, tune to your camera
        self.declare_parameter("detection_stale_sec", 0.35)

        # Logging
        self.declare_parameter("sample_hz", 5.0)
        self.declare_parameter("csv_path", "/tmp/obstacle_depth_accuracy.csv")
        self.declare_parameter("log_false_positives", True)
        self.declare_parameter("print_summary_every_s", 3.0)

        self.obstacles_topic = str(self.get_parameter("obstacles_topic").value)
        self.model_states_topic = str(self.get_parameter("model_states_topic").value)
        self.robot_model_name = str(self.get_parameter("robot_model_name").value)
        self.obstacle_model_prefix = str(self.get_parameter("obstacle_model_prefix").value)
        self.obstacle_count = int(self.get_parameter("obstacle_count").value)
        self.camera_x = float(self.get_parameter("camera_x_offset_m").value)
        self.camera_y = float(self.get_parameter("camera_y_offset_m").value)
        self.match_max_dist = float(self.get_parameter("match_max_dist_m").value)
        self.max_gt_range = float(self.get_parameter("max_gt_range_m").value)
        self.min_gt_forward = float(self.get_parameter("min_gt_forward_m").value)
        self.fov_half_angle = float(self.get_parameter("fov_half_angle_rad").value)
        self.detection_stale_sec = float(self.get_parameter("detection_stale_sec").value)
        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.csv_path = str(self.get_parameter("csv_path").value)
        self.log_false_positives = bool(self.get_parameter("log_false_positives").value)
        self.print_summary_every_s = float(self.get_parameter("print_summary_every_s").value)

        names_override = str(self.get_parameter("obstacle_model_names").value).strip()
        if names_override:
            self.obstacle_names = [x.strip() for x in names_override.split(",") if x.strip()]
        else:
            self.obstacle_names = [f"{self.obstacle_model_prefix}{i}" for i in range(self.obstacle_count)]

        reliability = str(self.get_parameter("model_states_qos_reliability").value).strip().lower()
        rel_policy = ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
        model_qos = QoSProfile(reliability=rel_policy, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_detection_time: Optional[float] = None
        self.latest_detection_msg_time: Optional[float] = None
        self.model_states: Optional[ModelStates] = None
        self.model_states_time: Optional[float] = None

        self.count_model_states = 0
        self.count_detection_msgs = 0
        self.rows_written = 0
        self.rows_gt = 0
        self.rows_detected = 0
        self.rows_false_positive = 0
        self.last_summary_t = time.time()

        self.create_subscription(String, self.obstacles_topic, self._obstacles_cb, 10)
        self.create_subscription(ModelStates, self.model_states_topic, self._model_states_cb, model_qos)

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self._fieldnames())
        self.writer.writeheader()
        self.csv_file.flush()

        period = 1.0 / max(0.1, self.sample_hz)
        self.timer = self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            "Obstacle depth accuracy logger started. "
            f"CSV={self.csv_path} | robot={self.robot_model_name} | obstacles={self.obstacle_names} | "
            f"camera_offset=({self.camera_x:.3f},{self.camera_y:.3f}) | match_max={self.match_max_dist:.2f} m | "
            f"observable: range<={self.max_gt_range:.1f} m, |bearing_img|<={self.fov_half_angle:.2f} rad"
        )

    @staticmethod
    def _fieldnames() -> List[str]:
        return [
            "t_wall_s", "row_type", "gt_name", "gt_index", "gt_observable", "detected", "matched_detection_index",
            "match_dist_xy_m", "num_detections", "detections_stale", "model_states_age_s", "detections_age_s",
            "robot_x_m", "robot_y_m", "robot_yaw_rad", "robot_speed_xy_mps", "robot_yaw_rate_radps",
            "gt_world_x_m", "gt_world_y_m", "gt_world_z_m", "gt_speed_xy_mps",
            "gt_x_base_m", "gt_y_base_m", "gt_range_base_m", "gt_bearing_base_rad",
            "gt_x_cam_m", "gt_y_cam_right_m", "gt_bearing_img_rad",
            "gt_rel_vx_base_mps", "gt_rel_vy_base_mps", "gt_rel_speed_base_mps",
            "det_id", "det_class", "det_depth_m", "det_raw_depth_m", "det_x_base_m", "det_y_base_m", "det_range_base_m",
            "det_bearing_img_rad", "det_bearing_base_rad", "det_vx_mps", "det_vy_mps", "det_speed_mps", "det_raw_disp_px",
            "err_x_base_m", "err_y_base_m", "err_xy_norm_m", "err_range_base_m", "abs_err_range_base_m",
            "err_depth_cam_m", "abs_err_depth_cam_m", "err_raw_depth_cam_m", "abs_err_raw_depth_cam_m",
            "err_bearing_img_rad", "abs_err_bearing_img_rad", "err_bearing_base_rad", "abs_err_bearing_base_rad",
            "err_vx_base_mps", "err_vy_base_mps", "err_speed_base_mps",
            "motion_class",
        ]

    def _obstacles_cb(self, msg: String) -> None:
        now = time.time()
        self.count_detection_msgs += 1
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_detections = [d for d in data if isinstance(d, dict)]
            else:
                self.latest_detections = []
        except Exception as e:
            self.get_logger().warn(f"Failed to parse obstacle JSON: {e}")
            self.latest_detections = []
        self.latest_detection_time = now
        self.latest_detection_msg_time = now

    def _model_states_cb(self, msg: ModelStates) -> None:
        self.model_states = msg
        self.model_states_time = time.time()
        self.count_model_states += 1

    def _model_index(self, name: str) -> Optional[int]:
        if self.model_states is None:
            return None
        try:
            return self.model_states.name.index(name)
        except ValueError:
            return None

    def _extract_robot(self) -> Optional[Dict[str, float]]:
        if self.model_states is None:
            return None
        i = self._model_index(self.robot_model_name)
        if i is None:
            return None
        p = self.model_states.pose[i].position
        q = self.model_states.pose[i].orientation
        tw = self.model_states.twist[i]
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        # Gazebo twist is world-frame for ModelStates in most gazebo_ros versions.
        vx = float(tw.linear.x)
        vy = float(tw.linear.y)
        return {
            "x": float(p.x), "y": float(p.y), "z": float(p.z), "yaw": yaw,
            "vx": vx, "vy": vy, "speed_xy": math.hypot(vx, vy),
            "wz": float(tw.angular.z),
        }

    def _extract_gt_obstacles(self, robot: Dict[str, float]) -> List[Dict[str, Any]]:
        out = []
        if self.model_states is None:
            return out
        rx, ry, ryaw = robot["x"], robot["y"], robot["yaw"]
        rvx, rvy = robot["vx"], robot["vy"]

        for idx, name in enumerate(self.obstacle_names):
            j = self._model_index(name)
            if j is None:
                continue
            p = self.model_states.pose[j].position
            tw = self.model_states.twist[j]
            wx, wy, wz = float(p.x), float(p.y), float(p.z)
            vxw, vyw = float(tw.linear.x), float(tw.linear.y)
            x_base, y_base = local_from_world(rx, ry, ryaw, wx, wy)
            range_base = math.hypot(x_base, y_base)
            bearing_base = math.atan2(y_base, x_base)

            x_cam = x_base - self.camera_x
            y_cam_right = self.camera_y - y_base
            bearing_img = math.atan2(y_cam_right, x_cam) if abs(x_cam) > 1e-9 else math.copysign(math.pi / 2, y_cam_right)

            rel_vx_w = vxw - rvx
            rel_vy_w = vyw - rvy
            rel_vx_b, rel_vy_b = vel_local_from_world(ryaw, rel_vx_w, rel_vy_w)

            observable = (
                x_cam > self.min_gt_forward
                and range_base <= self.max_gt_range
                and abs(bearing_img) <= self.fov_half_angle
            )
            out.append({
                "name": name, "index": idx,
                "world_x": wx, "world_y": wy, "world_z": wz,
                "vx_w": vxw, "vy_w": vyw, "speed_xy": math.hypot(vxw, vyw),
                "x_base": x_base, "y_base": y_base, "range_base": range_base,
                "bearing_base": bearing_base, "x_cam": x_cam, "y_cam_right": y_cam_right,
                "bearing_img": bearing_img, "rel_vx_base": rel_vx_b, "rel_vy_base": rel_vy_b,
                "rel_speed_base": math.hypot(rel_vx_b, rel_vy_b), "observable": observable,
            })
        return out

    def _clean_detection(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            x = float(d.get("x_m"))
            y = float(d.get("y_m"))
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            depth = self._safe_float(d.get("depth_m"))
            raw_depth = self._safe_float(d.get("raw_depth_m"))
            bearing_img = self._safe_float(d.get("bearing_rad"))
            vx = self._safe_float(d.get("vx_ms"))
            vy = self._safe_float(d.get("vy_ms"))
            speed = self._safe_float(d.get("speed_ms"))
            raw_disp = self._safe_float(d.get("raw_disp"))
            return {
                "id": d.get("id", ""), "class": d.get("class", ""),
                "x": x, "y": y, "range_base": math.hypot(x, y),
                "depth": depth, "raw_depth": raw_depth, "bearing_img": bearing_img,
                "bearing_base": math.atan2(y, x),
                "vx": vx, "vy": vy, "speed": speed, "raw_disp": raw_disp,
                "raw": d,
            }
        except Exception:
            return None

    @staticmethod
    def _safe_float(v: Any) -> float:
        try:
            x = float(v)
            return x if math.isfinite(x) else float("nan")
        except Exception:
            return float("nan")

    def _greedy_match(self, gt_list: List[Dict[str, Any]], detections: List[Dict[str, Any]]) -> Tuple[Dict[int, int], List[int]]:
        candidates = []
        for gi, gt in enumerate(gt_list):
            for di, det in enumerate(detections):
                dx = det["x"] - gt["x_base"]
                dy = det["y"] - gt["y_base"]
                dist = math.hypot(dx, dy)
                if dist <= self.match_max_dist:
                    candidates.append((dist, gi, di))
        candidates.sort(key=lambda t: t[0])
        used_g = set()
        used_d = set()
        matches: Dict[int, int] = {}
        for _, gi, di in candidates:
            if gi in used_g or di in used_d:
                continue
            matches[gi] = di
            used_g.add(gi)
            used_d.add(di)
        unmatched_d = [di for di in range(len(detections)) if di not in used_d]
        return matches, unmatched_d

    def _motion_class(self, robot: Dict[str, float], gt: Optional[Dict[str, Any]]) -> str:
        speed = robot.get("speed_xy", 0.0)
        yaw = abs(robot.get("wz", 0.0))
        if speed < 0.05:
            s = "tiny_speed"
        elif speed < 0.25:
            s = "moderate_speed"
        else:
            s = "substantial_speed"
        if yaw >= 0.15:
            s += "_with_rotation"
        return s

    def _timer_cb(self) -> None:
        now = time.time()
        if self.model_states is None:
            if now - self.last_summary_t > self.print_summary_every_s:
                self.get_logger().warn("No ModelStates received yet; no rows written.")
                self.last_summary_t = now
            return

        robot = self._extract_robot()
        if robot is None:
            if now - self.last_summary_t > self.print_summary_every_s:
                self.get_logger().warn(f"Robot model '{self.robot_model_name}' not found in /model_states.")
                self.last_summary_t = now
            return

        gt_list = self._extract_gt_obstacles(robot)
        det_age = float("inf") if self.latest_detection_time is None else now - self.latest_detection_time
        detections_stale = det_age > self.detection_stale_sec
        detections = [] if detections_stale else [x for x in (self._clean_detection(d) for d in self.latest_detections) if x is not None]

        matches, unmatched_d = self._greedy_match(gt_list, detections)

        model_age = float("nan") if self.model_states_time is None else now - self.model_states_time
        common = {
            "t_wall_s": f"{now:.6f}",
            "num_detections": len(detections),
            "detections_stale": int(detections_stale),
            "model_states_age_s": model_age,
            "detections_age_s": det_age if math.isfinite(det_age) else "",
            "robot_x_m": robot["x"], "robot_y_m": robot["y"], "robot_yaw_rad": robot["yaw"],
            "robot_speed_xy_mps": robot["speed_xy"], "robot_yaw_rate_radps": robot["wz"],
        }

        for gi, gt in enumerate(gt_list):
            det = detections[matches[gi]] if gi in matches else None
            row = self._make_gt_row(common, robot, gt, det, matches.get(gi, None))
            self.writer.writerow(row)
            self.rows_written += 1
            self.rows_gt += 1
            if det is not None:
                self.rows_detected += 1

        if self.log_false_positives:
            for di in unmatched_d:
                det = detections[di]
                row = self._make_false_positive_row(common, robot, det, di)
                self.writer.writerow(row)
                self.rows_written += 1
                self.rows_false_positive += 1

        self.csv_file.flush()

        if now - self.last_summary_t > self.print_summary_every_s:
            gt_obs = sum(1 for g in gt_list if g["observable"])
            det_rate = 100.0 * self.rows_detected / max(1, self.rows_gt)
            self.get_logger().info(
                f"rows={self.rows_written} gt_rows={self.rows_gt} matched={self.rows_detected} "
                f"FP={self.rows_false_positive} cumulative_match_rate={det_rate:.1f}% | "
                f"current_gt={len(gt_list)} observable={gt_obs} detections={len(detections)} stale={detections_stale}"
            )
            self.last_summary_t = now

    def _make_gt_row(self, common: Dict[str, Any], robot: Dict[str, float], gt: Dict[str, Any], det: Optional[Dict[str, Any]], di: Optional[int]) -> Dict[str, Any]:
        row = {k: "" for k in self._fieldnames()}
        row.update(common)
        row.update({
            "row_type": "gt",
            "gt_name": gt["name"], "gt_index": gt["index"], "gt_observable": int(gt["observable"]),
            "detected": int(det is not None), "matched_detection_index": "" if di is None else di,
            "gt_world_x_m": gt["world_x"], "gt_world_y_m": gt["world_y"], "gt_world_z_m": gt["world_z"],
            "gt_speed_xy_mps": gt["speed_xy"],
            "gt_x_base_m": gt["x_base"], "gt_y_base_m": gt["y_base"], "gt_range_base_m": gt["range_base"],
            "gt_bearing_base_rad": gt["bearing_base"],
            "gt_x_cam_m": gt["x_cam"], "gt_y_cam_right_m": gt["y_cam_right"], "gt_bearing_img_rad": gt["bearing_img"],
            "gt_rel_vx_base_mps": gt["rel_vx_base"], "gt_rel_vy_base_mps": gt["rel_vy_base"], "gt_rel_speed_base_mps": gt["rel_speed_base"],
            "motion_class": self._motion_class(robot, gt),
        })
        if det is not None:
            self._fill_det_and_errors(row, gt, det)
        return row

    def _make_false_positive_row(self, common: Dict[str, Any], robot: Dict[str, float], det: Dict[str, Any], di: int) -> Dict[str, Any]:
        row = {k: "" for k in self._fieldnames()}
        row.update(common)
        row.update({
            "row_type": "false_positive",
            "gt_observable": "", "detected": 1, "matched_detection_index": di,
            "motion_class": self._motion_class(robot, None),
        })
        self._fill_det_only(row, det)
        return row

    def _fill_det_only(self, row: Dict[str, Any], det: Dict[str, Any]) -> None:
        row.update({
            "det_id": det["id"], "det_class": det["class"],
            "det_depth_m": det["depth"], "det_raw_depth_m": det["raw_depth"],
            "det_x_base_m": det["x"], "det_y_base_m": det["y"], "det_range_base_m": det["range_base"],
            "det_bearing_img_rad": det["bearing_img"], "det_bearing_base_rad": det["bearing_base"],
            "det_vx_mps": det["vx"], "det_vy_mps": det["vy"], "det_speed_mps": det["speed"], "det_raw_disp_px": det["raw_disp"],
        })

    def _fill_det_and_errors(self, row: Dict[str, Any], gt: Dict[str, Any], det: Dict[str, Any]) -> None:
        self._fill_det_only(row, det)
        err_x = det["x"] - gt["x_base"]
        err_y = det["y"] - gt["y_base"]
        err_xy = math.hypot(err_x, err_y)
        match_dist = err_xy
        err_range = det["range_base"] - gt["range_base"]
        err_depth = det["depth"] - gt["x_cam"] if math.isfinite(det["depth"]) else float("nan")
        err_raw_depth = det["raw_depth"] - gt["x_cam"] if math.isfinite(det["raw_depth"]) else float("nan")
        err_bearing_img = wrap_pi(det["bearing_img"] - gt["bearing_img"]) if math.isfinite(det["bearing_img"]) else float("nan")
        err_bearing_base = wrap_pi(det["bearing_base"] - gt["bearing_base"])
        err_vx = det["vx"] - gt["rel_vx_base"] if math.isfinite(det["vx"]) else float("nan")
        err_vy = det["vy"] - gt["rel_vy_base"] if math.isfinite(det["vy"]) else float("nan")
        row.update({
            "match_dist_xy_m": match_dist,
            "err_x_base_m": err_x, "err_y_base_m": err_y, "err_xy_norm_m": err_xy,
            "err_range_base_m": err_range, "abs_err_range_base_m": abs(err_range),
            "err_depth_cam_m": err_depth, "abs_err_depth_cam_m": abs(err_depth) if math.isfinite(err_depth) else "",
            "err_raw_depth_cam_m": err_raw_depth, "abs_err_raw_depth_cam_m": abs(err_raw_depth) if math.isfinite(err_raw_depth) else "",
            "err_bearing_img_rad": err_bearing_img, "abs_err_bearing_img_rad": abs(err_bearing_img) if math.isfinite(err_bearing_img) else "",
            "err_bearing_base_rad": err_bearing_base, "abs_err_bearing_base_rad": abs(err_bearing_base),
            "err_vx_base_mps": err_vx, "err_vy_base_mps": err_vy,
            "err_speed_base_mps": math.hypot(err_vx, err_vy) if math.isfinite(err_vx) and math.isfinite(err_vy) else "",
        })

    def destroy_node(self) -> bool:
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDepthAccuracyLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
