#!/usr/bin/env python3
"""
Curriculum-Compatible Random Obstacle Spawner

Changes from original:
1. NUM_WALKING and NUM_STANDING are now dynamic (configurable at runtime)
2. New service /configure_obstacles (std_srvs/SetBool is too limited,
   so we use a topic-based approach with JSON on /obstacle_config)
3. Supports spawning MORE obstacles mid-training and parking EXCESS
   obstacles far away when reducing count
4. Tracks all spawned entities to handle count changes cleanly

The key design challenge: Gazebo can't easily delete entities, so we
spawn the MAXIMUM count once, then "park" unused ones far away (z=-100).
"""

import math
import random
import json
import subprocess
from typing import List, Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates


# Hidden position for "parked" (inactive) obstacles
_PARK_X = 100.0
_PARK_Y = 100.0
_PARK_Z = -10.0  # Below ground so they're invisible


class RandomObstacleSpawner(Node):
    """
    Curriculum-compatible obstacle spawner for Gazebo.

    Services:
        /reset_random_obstacles  (std_srvs/Trigger)
            Reset obstacle positions for a new episode.
            Uses the current active_walking / active_standing counts.

        /configure_obstacles  (std_srvs/Trigger)
            Reads the latest config from /obstacle_config topic.
            (We use a topic because Trigger has no request fields.)

    Topic (subscriber):
        /obstacle_config  (std_msgs/String)
            JSON: {"num_walking": 4, "num_standing": 1}
            Publishes desired obstacle counts. Applied on next reset.

    Behaviour:
      * On FIRST reset call:
          - Spawns MAX_WALKING walking + MAX_STANDING standing persons.
          - Only active_walking / active_standing are placed in the arena.
          - The rest are parked far away (invisible).

      * On SUBSEQUENT reset calls:
          - Active walking persons are teleported to new working-zone positions.
          - Active standing persons are teleported to robot→goal sector.
          - Inactive (parked) persons stay parked.

      * On /obstacle_config message:
          - Updates active counts (applied on next /reset_random_obstacles).
          - If increasing count, previously parked obstacles become active.
          - If decreasing count, excess obstacles get parked.
    """

    # Maximum obstacle capacity (spawned once in Gazebo)
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

        # NEW: Initial active counts (can be overridden by /obstacle_config)
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

        # Active counts (how many are actually in the arena)
        self.active_walking = int(
            self.get_parameter("initial_num_walking")
            .get_parameter_value().integer_value
        )
        self.active_standing = int(
            self.get_parameter("initial_num_standing")
            .get_parameter_value().integer_value
        )

        # Clamp to maximums
        self.active_walking = min(self.active_walking, self.MAX_WALKING)
        self.active_standing = min(self.active_standing, self.MAX_STANDING)

        self.get_logger().info(
            f"RandomObstacleSpawner (Curriculum-Compatible):\n"
            f"  Max capacity: {self.MAX_WALKING} walking + {self.MAX_STANDING} standing\n"
            f"  Initial active: {self.active_walking} walking + {self.active_standing} standing\n"
            f"  Models: walking={self.walking_model}, standing={self.standing_model}"
        )

        # ---- Entity name lists ----
        # Walking persons: indices 0 .. MAX_WALKING-1
        self.walking_names: List[str] = [
            f"{self.obstacle_prefix}{i}" for i in range(self.MAX_WALKING)
        ]
        # Standing persons: indices MAX_WALKING .. MAX_WALKING+MAX_STANDING-1
        self.standing_names: List[str] = [
            f"{self.obstacle_prefix}{self.MAX_WALKING + i}"
            for i in range(self.MAX_STANDING)
        ]

        # Track spawn state
        self.all_spawned: bool = False

        # Track the previous active counts to detect curriculum changes
        self._prev_active_walking: int = self.active_walking
        self._prev_active_standing: int = self.active_standing

        # Robot & goal poses
        self.robot_pose: Optional[Tuple[float, float]] = None
        self.goal_pose:  Optional[Tuple[float, float]] = None

        # Subscriptions
        self.create_subscription(
            ModelStates, "/model_states", self._model_states_cb, 10
        )
        self.create_subscription(
            Odometry, "/goal_marker/odom", self._goal_odom_cb, 10
        )

        # NEW: Configuration topic
        self.create_subscription(
            String, "/obstacle_config", self._config_cb, 10
        )

        # Services
        self.srv = self.create_service(
            Trigger, "reset_random_obstacles", self.handle_reset_obstacles
        )
        # NEW: Explicit configure service (reads last config from topic)
        self.srv_configure = self.create_service(
            Trigger, "configure_obstacles", self.handle_configure
        )

        # Store latest config message
        self._pending_config: Optional[Dict] = None

        self.get_logger().info("RandomObstacleSpawner ready.")

    # ---------- Callbacks ----------

    def _model_states_cb(self, msg: ModelStates):
        try:
            i = msg.name.index("my_robot")
        except ValueError:
            return
        p = msg.pose[i].position
        self.robot_pose = (p.x, p.y)

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (p.x, p.y)

    def _config_cb(self, msg: String):
        """
        Receive obstacle configuration via topic.
        JSON format: {"num_walking": 4, "num_standing": 1}
        Applied on next reset or explicit /configure_obstacles call.
        """
        try:
            config = json.loads(msg.data)
            self._pending_config = config
            self.get_logger().info(
                f"[OBSTACLES] Received config: {config}"
            )
        except json.JSONDecodeError as e:
            self.get_logger().error(
                f"[OBSTACLES] Invalid JSON on /obstacle_config: {e}"
            )

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
        self,
        name: str,
        model: str,
        x: float,
        y: float,
        yaw: float,
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
            "-x", str(x),
            "-y", str(y),
            "-z", "0.0",
            "-Y", str(yaw),
        ]
        ok = False
        for attempt in range(max_attempts):
            self.get_logger().info(
                f"[OBSTACLES] spawn {name} attempt {attempt + 1}/{max_attempts}"
            )
            ok = self._call_cmd(cmd, timeout=30.0)
            if ok:
                break
        if not ok:
            self.get_logger().error(
                f"[OBSTACLES] Failed to spawn {name} after {max_attempts} attempts."
            )
        return ok

    def _teleport_obstacle(self, name: str, x: float, y: float, yaw: float = 0.0, z: float = 0.0):
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
                ", z: " + str(z) +
                "}, orientation: {z: " + str(qz) +
                ", w: " + str(qw) +
                "}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, "
                "angular: {x: 0.0, y: 0.0, z: 0.0}}}}"
            ),
        ]
        self._call_cmd(cmd, timeout=10.0)

    def _park_obstacle(self, name: str):
        """Move an obstacle far away so it's effectively invisible."""
        self.get_logger().debug(f"[OBSTACLES] Parking {name}")
        self._teleport_obstacle(name, _PARK_X, _PARK_Y, z=_PARK_Z)

    # ---------- Pose samplers (unchanged) ----------

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

        self.get_logger().warn(
            "[OBSTACLES] Working-zone sampling failed, using fallback."
        )
        return self._sample_fallback()

    def _sample_sector(self, max_attempts: int = 200) -> Tuple[float, float, float]:
        if self.robot_pose is None or self.goal_pose is None:
            self.get_logger().warn(
                "[OBSTACLES] Sector sample: no robot/goal pose – using working zone."
            )
            return self._sample_working_zone()

        rx, ry = self.robot_pose
        gx, gy = self.goal_pose

        robot_goal_dist = math.hypot(gx - rx, gy - ry)
        bearing_to_goal = math.atan2(gy - ry, gx - rx)

        r_min = self.min_dist_robot
        r_max = robot_goal_dist - self.min_dist_goal

        if r_max - r_min < self.safety_margin:
            self.get_logger().warn(
                f"[OBSTACLES] Robot-goal dist ({robot_goal_dist:.2f} m) too small "
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

        self.get_logger().warn(
            "[OBSTACLES] Sector sampling failed, using working zone."
        )
        return self._sample_working_zone()

    def _sample_fallback(self) -> Tuple[float, float, float]:
        r     = random.uniform(self.min_dist_robot, self.fallback_radius)
        theta = random.uniform(-math.pi / 4.0, math.pi / 4.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yaw = random.uniform(-math.pi, math.pi)
        return x, y, yaw

    # ---------- Configuration handler ----------

    def _apply_pending_config(self):
        """Apply any pending configuration from /obstacle_config topic."""
        if self._pending_config is None:
            return

        config = self._pending_config
        self._pending_config = None

        # Save previous counts to detect newly activated obstacles
        self._prev_active_walking = self.active_walking
        self._prev_active_standing = self.active_standing

        if "num_walking" in config:
            self.active_walking = min(
                int(config["num_walking"]), self.MAX_WALKING
            )
        if "num_standing" in config:
            self.active_standing = min(
                int(config["num_standing"]), self.MAX_STANDING
            )

        if self.active_walking != self._prev_active_walking or self.active_standing != self._prev_active_standing:
            self.get_logger().info(
                f"[OBSTACLES] Config applied: "
                f"walking {self._prev_active_walking}→{self.active_walking}, "
                f"standing {self._prev_active_standing}→{self.active_standing}"
            )

    # ---------- Service handlers ----------

    def handle_configure(self, request, response):
        """
        Explicit configure service. Applies the latest /obstacle_config.
        """
        self._apply_pending_config()
        response.success = True
        response.message = (
            f"Active obstacles: {self.active_walking} walking + "
            f"{self.active_standing} standing"
        )
        return response

    def handle_reset_obstacles(self, request, response):
        """
        Reset obstacle positions for a new episode.
        Applies any pending config first.
        """
        self.get_logger().info("[OBSTACLES] Reset request received.")

        # Apply any pending configuration change
        self._apply_pending_config()

        if not self.all_spawned:
            # ── FIRST EPISODE: Spawn ALL obstacles (max capacity) ──────────
            self.get_logger().info(
                f"[OBSTACLES] First call: spawning all "
                f"{self.MAX_WALKING} walking + {self.MAX_STANDING} standing "
                f"(active: {self.active_walking} + {self.active_standing})"
            )

            # --- Spawn walking persons ---
            for i, name in enumerate(self.walking_names):
                if i < self.active_walking:
                    # Active: place in arena
                    x, y, yaw = self._sample_working_zone()
                    self._spawn_one_obstacle(name, self.walking_model, x, y, yaw)
                else:
                    # Inactive: spawn parked
                    self._spawn_one_obstacle(
                        name, self.walking_model,
                        _PARK_X + i, _PARK_Y, 0.0
                    )

            # --- Spawn standing persons ---
            for i, name in enumerate(self.standing_names):
                if i < self.active_standing:
                    # Active: place in sector
                    x, y, yaw = self._sample_sector()
                    self._spawn_one_obstacle(name, self.standing_model, x, y, yaw)
                else:
                    # Inactive: spawn parked
                    self._spawn_one_obstacle(
                        name, self.standing_model,
                        _PARK_X + self.MAX_WALKING + i, _PARK_Y, 0.0
                    )

            self.all_spawned = True

        else:
            # ── SUBSEQUENT EPISODES ────────────────────────────────────────
            # Walking persons: STAY IN PLACE (they wander on their own).
            #   Only newly activated walking persons (from curriculum increase)
            #   get teleported into the arena.
            #   Newly deactivated ones get parked.
            # Standing persons: TELEPORTED every episode to new sector
            #   positions (because the goal marker has moved).
            self.get_logger().info(
                f"[OBSTACLES] Subsequent call: "
                f"active={self.active_walking} walking + {self.active_standing} standing"
            )

            # --- Walking persons: only handle changes ---
            for i, name in enumerate(self.walking_names):
                if i < self.active_walking and i >= self._prev_active_walking:
                    # NEWLY ACTIVATED by curriculum: place in arena
                    x, y, yaw = self._sample_working_zone()
                    self._teleport_obstacle(name, x, y, yaw)
                    self.get_logger().info(
                        f"[OBSTACLES] Activated walking person {name}"
                    )
                elif i >= self.active_walking:
                    # Inactive: ensure parked
                    self._park_obstacle(name)
                # else: already active, leave in place (they wander on their own)

            # --- Standing persons: always teleport to new sector ---
            for i, name in enumerate(self.standing_names):
                if i < self.active_standing:
                    # Active: teleport to robot→goal sector
                    x, y, yaw = self._sample_sector()
                    self._teleport_obstacle(name, x, y, yaw)
                else:
                    # Inactive: ensure parked
                    self._park_obstacle(name)

            # Update previous counts after handling
            self._prev_active_walking = self.active_walking
            self._prev_active_standing = self.active_standing

        response.success = True
        response.message = (
            f"Active: {self.active_walking} walking + {self.active_standing} standing. "
            f"Capacity: {self.MAX_WALKING} + {self.MAX_STANDING}."
        )
        self.get_logger().info(
            f"[OBSTACLES] Reset done. Active: {self.active_walking}w + {self.active_standing}s"
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
