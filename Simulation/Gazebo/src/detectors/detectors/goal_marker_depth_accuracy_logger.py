#!/usr/bin/env python3
"""
ROS 2 logger for validating goal-marker depth detection against Gazebo /model_states.

This node subscribes to:
  1) detectors_msgs/msg/GoalMarkerState
       default: /follower_robot/depth_cam/goal_marker_state
  2) std_msgs/msg/Float32 raw depth
       default: /follower_robot/depth_cam/goal_marker_depth_raw
  3) gazebo_msgs/msg/ModelStates ground truth
       default: /model_states

It writes one CSV row whenever a GoalMarkerState message arrives and a fresh ModelStates
message is available.

Why this version exists:
  Gazebo /model_states can be BEST_EFFORT in many ROS 2 setups. A RELIABLE subscription may
  not connect to a BEST_EFFORT publisher. Therefore this node uses BEST_EFFORT for
  /model_states by default.
"""

import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from std_msgs.msg import Float32
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Vector3
from detectors_msgs.msg import GoalMarkerState


NAN = float("nan")


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def vector_norm_2d(v: Vector3) -> float:
    return math.hypot(float(v.x), float(v.y))


def vector_norm_3d(v: Vector3) -> float:
    return math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2)


def quat_to_yaw(q) -> float:
    """Return yaw angle from geometry_msgs/Quaternion, assuming ENU/Z-up convention."""
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2)
    return math.atan2(siny_cosp, cosy_cosp)


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm3(v: Tuple[float, float, float]) -> float:
    return math.sqrt(dot(v, v))


def make_qos(reliability_name: str, depth: int = 10) -> QoSProfile:
    name = str(reliability_name).strip().lower()
    if name in ("best_effort", "besteffort", "best-effort"):
        reliability = ReliabilityPolicy.BEST_EFFORT
    elif name in ("reliable",):
        reliability = ReliabilityPolicy.RELIABLE
    else:
        reliability = ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(
        reliability=reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class GoalMarkerDepthAccuracyLogger(Node):
    def __init__(self) -> None:
        super().__init__("goal_marker_depth_accuracy_logger")

        # Topics from your current goal_marker_depth node.
        self.declare_parameter("goal_state_topic", "/follower_robot/depth_cam/goal_marker_state")
        self.declare_parameter("raw_depth_topic", "/follower_robot/depth_cam/goal_marker_depth_raw")

        # Gazebo Classic commonly uses /model_states.
        self.declare_parameter("model_states_topic", "/model_states")

        # Model names inside gazebo_msgs/msg/ModelStates.name.
        self.declare_parameter("robot_model_name", "my_robot")
        self.declare_parameter("goal_model_name", "goal_marker")

        # QoS. BEST_EFFORT for /model_states is the safer default with Gazebo.
        self.declare_parameter("model_states_qos_reliability", "best_effort")
        self.declare_parameter("detector_qos_reliability", "reliable")

        # CSV output.
        self.declare_parameter("csv_path", "~/goal_marker_depth_accuracy_log.csv")
        self.declare_parameter("flush_every_n", 1)

        # Camera origin relative to robot base frame, in meters.
        # x: forward, y: left, z: up. Only yaw rotation is applied here.
        self.declare_parameter("camera_x_offset_m", 0.0)
        self.declare_parameter("camera_y_offset_m", 0.0)
        self.declare_parameter("camera_z_offset_m", 0.0)

        # Classification thresholds for later analysis.
        self.declare_parameter("tiny_speed_threshold_mps", 0.05)
        self.declare_parameter("substantial_speed_threshold_mps", 0.25)
        self.declare_parameter("rotational_speed_threshold_radps", 0.20)
        self.declare_parameter("model_state_max_age_s", 1.0)
        self.declare_parameter("diagnostic_period_s", 2.0)

        self.goal_state_topic = str(self.get_parameter("goal_state_topic").value)
        self.raw_depth_topic = str(self.get_parameter("raw_depth_topic").value)
        self.model_states_topic = str(self.get_parameter("model_states_topic").value)
        self.robot_model_name = str(self.get_parameter("robot_model_name").value)
        self.goal_model_name = str(self.get_parameter("goal_model_name").value)
        self.flush_every_n = max(1, int(self.get_parameter("flush_every_n").value))

        self.camera_x_offset_m = float(self.get_parameter("camera_x_offset_m").value)
        self.camera_y_offset_m = float(self.get_parameter("camera_y_offset_m").value)
        self.camera_z_offset_m = float(self.get_parameter("camera_z_offset_m").value)

        self.tiny_speed_threshold_mps = float(self.get_parameter("tiny_speed_threshold_mps").value)
        self.substantial_speed_threshold_mps = float(self.get_parameter("substantial_speed_threshold_mps").value)
        self.rotational_speed_threshold_radps = float(self.get_parameter("rotational_speed_threshold_radps").value)
        self.model_state_max_age_s = float(self.get_parameter("model_state_max_age_s").value)
        self.diagnostic_period_s = float(self.get_parameter("diagnostic_period_s").value)

        self._latest_model_states: Optional[ModelStates] = None
        self._latest_model_states_time: Optional[Time] = None
        self._latest_raw_depth_m: Optional[float] = None
        self._name_to_index: Dict[str, int] = {}

        self._rows_written = 0
        self._goal_state_count = 0
        self._raw_depth_count = 0
        self._model_states_count = 0
        self._missing_models_warned = False
        self._first_model_states_logged = False
        self._stale_warn_count = 0

        csv_path = Path(os.path.expanduser(str(self.get_parameter("csv_path").value)))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.csv_header())
        self._writer.writeheader()
        self._csv_file.flush()

        detector_qos = make_qos(str(self.get_parameter("detector_qos_reliability").value), depth=10)
        model_qos = make_qos(str(self.get_parameter("model_states_qos_reliability").value), depth=10)

        self.create_subscription(GoalMarkerState, self.goal_state_topic, self.on_goal_state, detector_qos)
        self.create_subscription(Float32, self.raw_depth_topic, self.on_raw_depth, detector_qos)
        self.create_subscription(ModelStates, self.model_states_topic, self.on_model_states, model_qos)

        self.create_timer(self.diagnostic_period_s, self.on_diagnostic_timer)

        self.get_logger().info(
            "Goal-marker depth accuracy logger started. "
            f"CSV: {csv_path} | robot='{self.robot_model_name}' | goal='{self.goal_model_name}' | "
            f"model_states='{self.model_states_topic}' | model_states_qos='{self.get_parameter('model_states_qos_reliability').value}'"
        )

    @staticmethod
    def csv_header() -> List[str]:
        return [
            "ros_time_s",
            "goal_state_stamp_s",
            "model_state_age_s",
            "visible",
            "motion_class",
            "detected_depth_smooth_m",
            "detected_depth_raw_m",
            "detected_bearing_rad",
            "detected_cx_px",
            "detected_cy_px",
            "gt_range_3d_m",
            "gt_range_2d_m",
            "gt_forward_depth_m",
            "gt_lateral_m",
            "gt_vertical_m",
            "gt_bearing_rad",
            "depth_error_forward_smooth_m",
            "depth_abs_error_forward_smooth_m",
            "depth_error_forward_raw_m",
            "depth_abs_error_forward_raw_m",
            "range_error_3d_smooth_m",
            "range_abs_error_3d_smooth_m",
            "robot_x_m",
            "robot_y_m",
            "robot_z_m",
            "goal_x_m",
            "goal_y_m",
            "goal_z_m",
            "camera_x_m",
            "camera_y_m",
            "camera_z_m",
            "robot_yaw_rad",
            "goal_yaw_rad",
            "robot_speed_xy_mps",
            "robot_speed_3d_mps",
            "goal_speed_xy_mps",
            "goal_speed_3d_mps",
            "relative_speed_xy_mps",
            "relative_speed_3d_mps",
            "relative_speed_los_mps",
            "closing_speed_los_mps",
            "robot_yaw_rate_radps",
            "goal_yaw_rate_radps",
        ]

    def on_raw_depth(self, msg: Float32) -> None:
        self._raw_depth_count += 1
        self._latest_raw_depth_m = float(msg.data)

    def on_model_states(self, msg: ModelStates) -> None:
        self._model_states_count += 1
        self._latest_model_states = msg
        self._latest_model_states_time = self.get_clock().now()
        self._name_to_index = {name: i for i, name in enumerate(msg.name)}

        if not self._first_model_states_logged:
            available = ", ".join(msg.name[:20])
            if len(msg.name) > 20:
                available += ", ..."
            self.get_logger().info(f"First ModelStates received. Available names: [{available}]")
            self._first_model_states_logged = True

    def on_goal_state(self, msg: GoalMarkerState) -> None:
        self._goal_state_count += 1
        now = self.get_clock().now()

        if self._latest_model_states is None or self._latest_model_states_time is None:
            # Do not spam. The diagnostic timer will show the counters.
            return

        model_age_s = (now - self._latest_model_states_time).nanoseconds * 1e-9
        if model_age_s > self.model_state_max_age_s:
            self._stale_warn_count += 1
            if self._stale_warn_count <= 3 or self._stale_warn_count % 50 == 0:
                self.get_logger().warn(
                    f"Latest ModelStates is stale: {model_age_s:.3f} s. Row skipped. "
                    f"Increase model_state_max_age_s only if your /model_states rate is low."
                )
            return

        entities = self._extract_robot_and_goal(self._latest_model_states)
        if entities is None:
            return

        robot_pose, robot_twist, goal_pose, goal_twist = entities
        row = self._make_row(msg, robot_pose, robot_twist, goal_pose, goal_twist, now, model_age_s)
        self._writer.writerow(row)
        self._rows_written += 1

        if self._rows_written % self.flush_every_n == 0:
            self._csv_file.flush()

    def on_diagnostic_timer(self) -> None:
        if self._rows_written == 0:
            self.get_logger().warn(
                "No CSV rows written yet. "
                f"counts: goal_state={self._goal_state_count}, raw_depth={self._raw_depth_count}, "
                f"model_states={self._model_states_count}. "
                "Need both goal_state and model_states callbacks."
            )
        else:
            self.get_logger().info(
                f"CSV rows written: {self._rows_written} | "
                f"counts: goal_state={self._goal_state_count}, raw_depth={self._raw_depth_count}, "
                f"model_states={self._model_states_count}"
            )

    def _extract_robot_and_goal(self, model_states: ModelStates) -> Optional[Tuple[Pose, Twist, Pose, Twist]]:
        robot_idx = self._name_to_index.get(self.robot_model_name)
        goal_idx = self._name_to_index.get(self.goal_model_name)

        if robot_idx is None or goal_idx is None:
            if not self._missing_models_warned:
                available = ", ".join(model_states.name[:20])
                if len(model_states.name) > 20:
                    available += ", ..."
                self.get_logger().error(
                    "Could not find robot/goal names in ModelStates. "
                    f"robot='{self.robot_model_name}' found={robot_idx is not None}, "
                    f"goal='{self.goal_model_name}' found={goal_idx is not None}. "
                    f"Available names: [{available}]"
                )
                self._missing_models_warned = True
            return None

        try:
            return (
                model_states.pose[robot_idx],
                model_states.twist[robot_idx],
                model_states.pose[goal_idx],
                model_states.twist[goal_idx],
            )
        except IndexError:
            self.get_logger().error("ModelStates arrays are inconsistent; row skipped.")
            return None

    def _make_row(
        self,
        msg: GoalMarkerState,
        robot_pose: Pose,
        robot_twist: Twist,
        goal_pose: Pose,
        goal_twist: Twist,
        now: Time,
        model_age_s: float,
    ) -> Dict[str, float]:
        visible = bool(msg.visible)
        robot_yaw = quat_to_yaw(robot_pose.orientation)
        goal_yaw = quat_to_yaw(goal_pose.orientation)

        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)

        cam_x = float(robot_pose.position.x) + cy * self.camera_x_offset_m - sy * self.camera_y_offset_m
        cam_y = float(robot_pose.position.y) + sy * self.camera_x_offset_m + cy * self.camera_y_offset_m
        cam_z = float(robot_pose.position.z) + self.camera_z_offset_m

        dx = float(goal_pose.position.x) - cam_x
        dy = float(goal_pose.position.y) - cam_y
        dz = float(goal_pose.position.z) - cam_z

        gt_range_2d = math.hypot(dx, dy)
        gt_range_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Projection of the goal position into the robot/camera forward frame.
        # Positive forward depth means the goal is in front of robot +X.
        gt_forward_depth = dx * cy + dy * sy
        gt_lateral = -dx * sy + dy * cy
        gt_vertical = dz
        gt_bearing = math.atan2(gt_lateral, gt_forward_depth) if gt_range_2d > 1e-9 else NAN

        robot_speed_xy = vector_norm_2d(robot_twist.linear)
        robot_speed_3d = vector_norm_3d(robot_twist.linear)
        goal_speed_xy = vector_norm_2d(goal_twist.linear)
        goal_speed_3d = vector_norm_3d(goal_twist.linear)

        rel_v = (
            float(goal_twist.linear.x) - float(robot_twist.linear.x),
            float(goal_twist.linear.y) - float(robot_twist.linear.y),
            float(goal_twist.linear.z) - float(robot_twist.linear.z),
        )
        relative_speed_3d = norm3(rel_v)
        relative_speed_xy = math.hypot(rel_v[0], rel_v[1])

        if gt_range_3d > 1e-9:
            los_unit = (dx / gt_range_3d, dy / gt_range_3d, dz / gt_range_3d)
            relative_speed_los = dot(rel_v, los_unit)
        else:
            relative_speed_los = NAN
        closing_speed_los = -relative_speed_los if math.isfinite(relative_speed_los) else NAN

        robot_yaw_rate = float(robot_twist.angular.z)
        goal_yaw_rate = float(goal_twist.angular.z)
        motion_class = self._classify_motion(robot_speed_xy, robot_yaw_rate)

        detected_depth_smooth = float(msg.depth_m) if visible else NAN
        detected_depth_raw = self._latest_raw_depth_m if visible and self._latest_raw_depth_m is not None else NAN
        detected_bearing = float(msg.bearing_rad) if visible else NAN
        detected_cx = float(msg.cx) if visible else NAN
        detected_cy = float(msg.cy) if visible else NAN

        smooth_forward_error = (
            detected_depth_smooth - gt_forward_depth
            if visible and math.isfinite(detected_depth_smooth)
            else NAN
        )
        raw_forward_error = (
            detected_depth_raw - gt_forward_depth
            if visible and math.isfinite(detected_depth_raw)
            else NAN
        )
        smooth_range_error = (
            detected_depth_smooth - gt_range_3d
            if visible and math.isfinite(detected_depth_smooth)
            else NAN
        )

        return {
            "ros_time_s": now.nanoseconds * 1e-9,
            "goal_state_stamp_s": stamp_to_sec(msg.header.stamp),
            "model_state_age_s": model_age_s,
            "visible": int(visible),
            "motion_class": motion_class,
            "detected_depth_smooth_m": detected_depth_smooth,
            "detected_depth_raw_m": detected_depth_raw,
            "detected_bearing_rad": detected_bearing,
            "detected_cx_px": detected_cx,
            "detected_cy_px": detected_cy,
            "gt_range_3d_m": gt_range_3d,
            "gt_range_2d_m": gt_range_2d,
            "gt_forward_depth_m": gt_forward_depth,
            "gt_lateral_m": gt_lateral,
            "gt_vertical_m": gt_vertical,
            "gt_bearing_rad": gt_bearing,
            "depth_error_forward_smooth_m": smooth_forward_error,
            "depth_abs_error_forward_smooth_m": abs(smooth_forward_error) if math.isfinite(smooth_forward_error) else NAN,
            "depth_error_forward_raw_m": raw_forward_error,
            "depth_abs_error_forward_raw_m": abs(raw_forward_error) if math.isfinite(raw_forward_error) else NAN,
            "range_error_3d_smooth_m": smooth_range_error,
            "range_abs_error_3d_smooth_m": abs(smooth_range_error) if math.isfinite(smooth_range_error) else NAN,
            "robot_x_m": float(robot_pose.position.x),
            "robot_y_m": float(robot_pose.position.y),
            "robot_z_m": float(robot_pose.position.z),
            "goal_x_m": float(goal_pose.position.x),
            "goal_y_m": float(goal_pose.position.y),
            "goal_z_m": float(goal_pose.position.z),
            "camera_x_m": cam_x,
            "camera_y_m": cam_y,
            "camera_z_m": cam_z,
            "robot_yaw_rad": robot_yaw,
            "goal_yaw_rad": goal_yaw,
            "robot_speed_xy_mps": robot_speed_xy,
            "robot_speed_3d_mps": robot_speed_3d,
            "goal_speed_xy_mps": goal_speed_xy,
            "goal_speed_3d_mps": goal_speed_3d,
            "relative_speed_xy_mps": relative_speed_xy,
            "relative_speed_3d_mps": relative_speed_3d,
            "relative_speed_los_mps": relative_speed_los,
            "closing_speed_los_mps": closing_speed_los,
            "robot_yaw_rate_radps": robot_yaw_rate,
            "goal_yaw_rate_radps": goal_yaw_rate,
        }

    def _classify_motion(self, robot_speed_xy: float, robot_yaw_rate: float) -> str:
        rotating = abs(robot_yaw_rate) >= self.rotational_speed_threshold_radps

        if robot_speed_xy < self.tiny_speed_threshold_mps:
            base = "tiny_speed"
        elif robot_speed_xy >= self.substantial_speed_threshold_mps:
            base = "substantial_speed"
        else:
            base = "moderate_speed"

        if rotating:
            return f"{base}_with_rotation"
        return base

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_csv_file") and not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()
        finally:
            return super().destroy_node()


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node = GoalMarkerDepthAccuracyLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
