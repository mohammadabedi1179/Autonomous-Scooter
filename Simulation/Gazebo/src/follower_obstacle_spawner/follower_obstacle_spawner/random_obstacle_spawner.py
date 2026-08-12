#!/usr/bin/env python3
"""
Curriculum-Compatible Random Obstacle Spawner (On-Demand Spawning)

Key design: Only spawns obstacles that are actually needed.
No pre-spawning, no parking. When the curriculum increases obstacle
count, new ones are spawned. When it decreases, excess are deleted.

Services:
    /reset_random_obstacles  (std_srvs/Trigger)
        Reset positions for a new episode.

Topic (subscriber):
    /obstacle_config  (std_msgs/String)
        JSON: {"num_walking": 4, "num_standing": 1}
        When count increases, new obstacles are spawned on next reset.
        When count decreases, excess are deleted immediately.

Behaviour:
  * Walking (moving) persons:
      - Spawned in working zone on first reset.
      - NEVER teleported (they wander on their own via Gazebo animation).
      - Only newly added ones (from curriculum) get spawned on next reset.

  * Standing (static) persons:
      - Spawned in robot→goal sector on first reset.
      - Re-teleported to new sector positions every episode.
"""

import math
import random
import json
import subprocess
from typing import List, Optional, Tuple, Set

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates


class RandomObstacleSpawner(Node):

    # Maximum capacity (names reserved, NOT pre-spawned)
    MAX_WALKING  = 7
    MAX_STANDING = 3

    def __init__(self):
        super().__init__("random_obstacle_spawner")

        # ------- Parameters -------
        self.declare_parameter("min_distance_from_robot", 1.5)
        self.declare_parameter("min_distance_from_goal",  1.0)
        self.declare_parameter("sector_half_angle_deg", 30.0)
        self.declare_parameter("working_zone_radius", 8.0)
        self.declare_parameter("safety_margin", 0.5)
        self.declare_parameter("fallback_radius", 7.0)
        self.declare_parameter("obstacle_prefix", "yolo_obstacle_")
        self.declare_parameter("walking_model_name", "person_walking")
        self.declare_parameter("standing_model_name", "person_standing")

        # Initial active counts (curriculum can change these)
        self.declare_parameter("initial_num_walking", 4)
        self.declare_parameter("initial_num_standing", 1)

        self.min_dist_robot = (
            self.get_parameter("min_distance_from_robot")
            .get_parameter_value().double_value
        )
        self.min_dist_goal = (
            self.get_parameter("min_distance_from_goal")
            .get_parameter_value().double_value
        )
        self.sector_half_angle = math.radians(
            self.get_parameter("sector_half_angle_deg")
            .get_parameter_value().double_value
        )
        self.working_zone_radius = (
            self.get_parameter("working_zone_radius")
            .get_parameter_value().double_value
        )
        self.safety_margin = (
            self.get_parameter("safety_margin")
            .get_parameter_value().double_value
        )
        self.fallback_radius = (
            self.get_parameter("fallback_radius")
            .get_parameter_value().double_value
        )
        self.obstacle_prefix = (
            self.get_parameter("obstacle_prefix")
            .get_parameter_value().string_value
        )
        self.walking_model = (
            self.get_parameter("walking_model_name")
            .get_parameter_value().string_value
        )
        self.standing_model = (
            self.get_parameter("standing_model_name")
            .get_parameter_value().string_value
        )

        # Desired active counts
        self.active_walking = min(
            int(self.get_parameter("initial_num_walking")
                .get_parameter_value().integer_value),
            self.MAX_WALKING
        )
        self.active_standing = min(
            int(self.get_parameter("initial_num_standing")
                .get_parameter_value().integer_value),
            self.MAX_STANDING
        )

        # All possible entity names
        self.walking_names: List[str] = [
            f"{self.obstacle_prefix}{i}" for i in range(self.MAX_WALKING)
        ]
        self.standing_names: List[str] = [
            f"{self.obstacle_prefix}{self.MAX_WALKING + i}"
            for i in range(self.MAX_STANDING)
        ]

        # Track which entities actually exist in Gazebo
        self.spawned_walking: Set[str] = set()
        self.spawned_standing: Set[str] = set()

        # Robot & goal poses
        self.robot_pose: Optional[Tuple[float, float]] = None
        self.goal_pose:  Optional[Tuple[float, float]] = None

        # Pending config from /obstacle_config topic
        self._pending_config = None

        # Subscriptions
        self.create_subscription(
            ModelStates, "/model_states", self._model_states_cb, 10
        )
        self.create_subscription(
            Odometry, "/goal_marker/odom", self._goal_odom_cb, 10
        )
        self.create_subscription(
            String, "/obstacle_config", self._config_cb, 10
        )

        # Services
        self.srv = self.create_service(
            Trigger, "reset_random_obstacles", self.handle_reset_obstacles
        )

        self.get_logger().info(
            f"RandomObstacleSpawner (On-Demand):\n"
            f"  Max capacity: {self.MAX_WALKING}w + {self.MAX_STANDING}s\n"
            f"  Initial active: {self.active_walking}w + {self.active_standing}s\n"
            f"  Models: walking={self.walking_model}, standing={self.standing_model}\n"
            f"  Nothing spawned yet — will spawn on first /reset_random_obstacles call."
        )

    # ---------- Callbacks ----------

    def _model_states_cb(self, msg: ModelStates):
        try:
            i = msg.name.index("my_robot")
        except ValueError:
            return
        p = msg.pose[i].position
        self.robot_pose = (p.x, p.y)

        # FIX: also take the goal position from /model_states (ground truth),
        # not just from /goal_marker/odom. /model_states is published by
        # Gazebo continuously and is unaffected by whether the goal-mover
        # node is enabled/disabled. /goal_marker/odom, by contrast, is only
        # published BY the mover node — while it's disabled (which the env
        # does during reset, right when this node's reset service fires),
        # self.goal_pose previously never received a new message and stayed
        # frozen at the goal's position from the END OF THE PREVIOUS EPISODE,
        # which is exactly the obstacles-spawn-at-the-wrong-place bug.
        try:
            gi = msg.name.index("goal_marker")
            gp = msg.pose[gi].position
            self.goal_pose = (gp.x, gp.y)
        except ValueError:
            pass

    def _goal_odom_cb(self, msg: Odometry):
        # Kept as a secondary source (e.g. higher publish rate while the
        # mover is active); harmless since it agrees with /model_states
        # whenever both are fresh, and /model_states above is what actually
        # guarantees correctness right after a teleport.
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    def _config_cb(self, msg: String):
        """
        Receive obstacle config from curriculum trainer.
        JSON: {"num_walking": 4, "num_standing": 1}
        Applied on next /reset_random_obstacles call.
        """
        try:
            config = json.loads(msg.data)
            self._pending_config = config
            self.get_logger().info(f"[OBSTACLES] Received config: {config}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"[OBSTACLES] Invalid JSON: {e}")

    # ---------- Gazebo helpers ----------

    def _call_cmd(self, cmd: list, timeout: float = 30.0) -> bool:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if out.returncode != 0:
                self.get_logger().warn(
                    f"Command failed: {' '.join(cmd)}\n"
                    f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            self.get_logger().warn(
                f"Command timed out after {timeout:.1f}s: {' '.join(cmd)}"
            )
            return False

    def _spawn_one_obstacle(
        self, name: str, model: str,
        x: float, y: float, yaw: float,
        max_attempts: int = 3,
    ) -> bool:
        self.get_logger().info(
            f"[OBSTACLES] Spawning {name} (model={model}) at "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )
        cmd = [
            "ros2", "run", "gazebo_ros", "spawn_entity.py",
            "-entity",   name,
            "-database", model,
            "-x", str(x), "-y", str(y), "-z", "0.0", "-Y", str(yaw),
        ]
        for attempt in range(max_attempts):
            self.get_logger().info(
                f"[OBSTACLES] spawn {name} attempt {attempt+1}/{max_attempts}"
            )
            if self._call_cmd(cmd, timeout=30.0):
                return True
        self.get_logger().error(
            f"[OBSTACLES] Failed to spawn {name} after {max_attempts} attempts."
        )
        return False

    def _delete_obstacle(self, name: str) -> bool:
        """Delete an entity from Gazebo."""
        self.get_logger().info(f"[OBSTACLES] Deleting {name} from Gazebo")
        cmd = [
            "ros2", "service", "call",
            "/delete_entity",
            "gazebo_msgs/srv/DeleteEntity",
            f"{{name: '{name}'}}"
        ]
        return self._call_cmd(cmd, timeout=10.0)

    def _teleport_obstacle(self, name: str, x: float, y: float, yaw: float = 0.0):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        cmd = [
            "ros2", "service", "call",
            "/set_entity_state",
            "gazebo_msgs/srv/SetEntityState",
            (
                "{state: {name: '" + name +
                "', pose: {position: {x: " + str(x) +
                ", y: " + str(y) +
                ", z: 0.0}, orientation: {z: " + str(qz) +
                ", w: " + str(qw) +
                "}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, "
                "angular: {x: 0.0, y: 0.0, z: 0.0}}}}"
            ),
        ]
        self._call_cmd(cmd, timeout=10.0)

    # ---------- Pose samplers (unchanged from original) ----------

    def _sample_working_zone(self, max_attempts: int = 100) -> Tuple[float, float, float]:
        if self.robot_pose is None:
            return self._sample_fallback()

        rx, ry = self.robot_pose
        gx, gy = self.goal_pose if self.goal_pose is not None else (None, None)

        for _ in range(max_attempts):
            r     = self.working_zone_radius * math.sqrt(random.random())
            theta = random.uniform(0.0, 2.0 * math.pi)
            x = rx + r * math.cos(theta)
            y = ry + r * math.sin(theta)

            if math.hypot(x - rx, y - ry) < self.min_dist_robot:
                continue
            if gx is not None and math.hypot(x - gx, y - gy) < self.min_dist_goal:
                continue

            yaw = random.uniform(-math.pi, math.pi)
            return x, y, yaw

        self.get_logger().warn("[OBSTACLES] Working-zone sampling failed, using fallback.")
        return self._sample_fallback()

    def _sample_sector(self, max_attempts: int = 200) -> Tuple[float, float, float]:
        if self.robot_pose is None or self.goal_pose is None:
            self.get_logger().warn("[OBSTACLES] Sector sample: no robot/goal pose – using working zone.")
            return self._sample_working_zone()

        rx, ry = self.robot_pose
        gx, gy = self.goal_pose

        robot_goal_dist = math.hypot(gx - rx, gy - ry)
        bearing_to_goal = math.atan2(gy - ry, gx - rx)

        r_min = self.min_dist_robot
        r_max = robot_goal_dist - self.min_dist_goal

        if r_max - r_min < self.safety_margin:
            self.get_logger().warn(
                f"[OBSTACLES] Robot-goal dist ({robot_goal_dist:.2f}m) too small "
                f"for sector placement – falling back to working zone."
            )
            return self._sample_working_zone()

        for _ in range(max_attempts):
            r = random.uniform(r_min, r_max)
            alpha = random.uniform(
                bearing_to_goal - self.sector_half_angle,
                bearing_to_goal + self.sector_half_angle,
            )
            x = rx + r * math.cos(alpha)
            y = ry + r * math.sin(alpha)

            if math.hypot(x - rx, y - ry) < self.min_dist_robot:
                continue
            if math.hypot(x - gx, y - gy) < self.min_dist_goal:
                continue

            yaw = random.uniform(-math.pi, math.pi)
            return x, y, yaw

        self.get_logger().warn("[OBSTACLES] Sector sampling failed, using working zone.")
        return self._sample_working_zone()

    def _sample_fallback(self) -> Tuple[float, float, float]:
        r     = random.uniform(self.min_dist_robot, self.fallback_radius)
        theta = random.uniform(-math.pi / 4.0, math.pi / 4.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yaw = random.uniform(-math.pi, math.pi)
        return x, y, yaw

    # ---------- Config application ----------

    def _apply_pending_config(self):
        """Apply pending config from /obstacle_config topic."""
        if self._pending_config is None:
            return

        config = self._pending_config
        self._pending_config = None
        self.sector_half_angle = math.radians(int(config["sector_half_angle_deg"]))  # default 30.0


        old_w = self.active_walking
        old_s = self.active_standing

        if "num_walking" in config:
            self.active_walking = min(int(config["num_walking"]), self.MAX_WALKING)
        if "num_standing" in config:
            self.active_standing = min(int(config["num_standing"]), self.MAX_STANDING)

        # If DECREASING walking count, delete excess from Gazebo immediately
        if self.active_walking < old_w:
            for i in range(self.active_walking, old_w):
                name = self.walking_names[i]
                if name in self.spawned_walking:
                    self._delete_obstacle(name)
                    self.spawned_walking.discard(name)
                    self.get_logger().info(
                        f"[OBSTACLES] Deleted excess walking person: {name}"
                    )

        # If DECREASING standing count, delete excess from Gazebo immediately
        if self.active_standing < old_s:
            for i in range(self.active_standing, old_s):
                name = self.standing_names[i]
                if name in self.spawned_standing:
                    self._delete_obstacle(name)
                    self.spawned_standing.discard(name)
                    self.get_logger().info(
                        f"[OBSTACLES] Deleted excess standing person: {name}"
                    )

        if self.active_walking != old_w or self.active_standing != old_s:
            self.get_logger().info(
                f"[OBSTACLES] Config applied: "
                f"walking {old_w}→{self.active_walking}, "
                f"standing {old_s}→{self.active_standing}"
            )

    # ---------- Service handler ----------

    def handle_reset_obstacles(self, request, response):
        """
        Reset obstacle positions for a new episode.

        - Applies any pending config first.
        - Spawns NEW obstacles that don't exist yet in Gazebo.
        - Walking persons that already exist: left in place (they wander).
        - Standing persons that already exist: teleported to new sector.
        """
        self.get_logger().info("[OBSTACLES] Reset request received.")

        # Apply pending curriculum config
        self._apply_pending_config()

        # ---- Walking persons ----
        for i in range(self.active_walking):
            name = self.walking_names[i]
            if name not in self.spawned_walking:
                # NEW obstacle: spawn it
                x, y, yaw = self._sample_working_zone()
                ok = self._spawn_one_obstacle(name, self.walking_model, x, y, yaw)
                if ok:
                    self.spawned_walking.add(name)
                    self.get_logger().info(f"[OBSTACLES] Spawned new walking: {name}")
            # else: already exists, leave it alone (walking persons wander)

        # ---- Standing persons ----
        for i in range(self.active_standing):
            name = self.standing_names[i]
            if name not in self.spawned_standing:
                # NEW obstacle: spawn it
                x, y, yaw = self._sample_sector()
                ok = self._spawn_one_obstacle(name, self.standing_model, x, y, yaw)
                if ok:
                    self.spawned_standing.add(name)
                    self.get_logger().info(f"[OBSTACLES] Spawned new standing: {name}")
            else:
                # Already exists: teleport to new sector position
                x, y, yaw = self._sample_sector()
                self._teleport_obstacle(name, x, y, yaw)

        response.success = True
        response.message = (
            f"Active: {self.active_walking}w (spawned={len(self.spawned_walking)}) + "
            f"{self.active_standing}s (spawned={len(self.spawned_standing)}). "
        )
        self.get_logger().info(
            f"[OBSTACLES] Reset done. "
            f"Walking: {len(self.spawned_walking)} spawned ({self.active_walking} active). "
            f"Standing: {len(self.spawned_standing)} spawned ({self.active_standing} active)."
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RandomObstacleSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RandomObstacleSpawner")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()