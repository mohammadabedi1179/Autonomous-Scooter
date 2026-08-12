#!/usr/bin/env python3
"""
LSTM-Compatible Goal Following Environment (Curriculum Version)

Changes from original goal_env_lstm.py:
1. MAX_OBSTACLES is now a configurable __init__ parameter (default=10)
2. gt_obstacle_names generated dynamically based on max_obstacles
3. Added set_curriculum_stage() method for runtime difficulty changes
4. Observation space adapts to max_obstacles automatically

This is backward-compatible: if you pass max_obstacles=5, it behaves
identically to the original.
"""

import time
import math
import json
import subprocess
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from gymnasium import Env, spaces
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String, Float32

from detectors_msgs.msg import GoalMarkerState


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_from_quat(x, y, z, w) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _world_from_local(rx, ry, ryaw, lx, ly) -> Tuple[float, float]:
    wx = rx + lx * math.cos(ryaw) - ly * math.sin(ryaw)
    wy = ry + lx * math.sin(ryaw) + ly * math.cos(ryaw)
    return wx, wy


def _local_from_world(rx, ry, ryaw, wx, wy) -> Tuple[float, float]:
    dx = wx - rx
    dy = wy - ry
    lx =  math.cos(ryaw) * dx + math.sin(ryaw) * dy
    ly = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
    return lx, ly


class _ROS(Node):
    """ROS2 wrapper - same as goal_env_gt.py"""

    def __init__(
        self,
        cmd_topic: str,
        goal_state_topic: str,
        goal_odom_topic: str,
        obstacle_topic: str = "/follower_robot/obstacles_depth",
        ultrasonic_topics: dict | None = None,
        robot_model_name: str = "my_robot",
        goal_model_name: str = "goal_marker",
        gt_obstacle_names: Optional[List[str]] = None,
    ):
        super().__init__("sac_goal_env_lstm_node")

        self.robot_model_name = str(robot_model_name)
        self.goal_model_name = str(goal_model_name)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        self.goal_state: GoalMarkerState | None = None
        self.goal_pose = None
        self.robot_pose = None
        self._last_odom_vx = 0.0
        self._last_odom_wz = 0.0
        self.obstacles_json: str | None = None

        self.goal_pose_gt = None
        self.goal_vel_gt = None

        self.ultra = {
            "front_left": None,
            "front_right": None,
            "left_side": None,
            "right_side": None,
            "back": None,            # NEW: rear-facing 5th sensor
        }
        self.ultra_stamp = {k: 0.0 for k in self.ultra.keys()}

        self.create_subscription(GoalMarkerState, goal_state_topic, self._goal_state_cb, 10)
        self.create_subscription(Odometry, goal_odom_topic, self._goal_odom_cb, 10)
        self.create_subscription(ModelStates, "/model_states", self._model_states_cb, 10)
        self.create_subscription(String, obstacle_topic, self._obstacle_cb, 10)

        if ultrasonic_topics is None:
            ultrasonic_topics = {
                "front_left":  "/follower_robot/ultrasonic_bridge/front_left/distance_m",
                "front_right": "/follower_robot/ultrasonic_bridge/front_right/distance_m",
                "left_side":   "/follower_robot/ultrasonic_bridge/left_side/distance_m",
                "right_side":  "/follower_robot/ultrasonic_bridge/right_side/distance_m",
                "back":        "/follower_robot/ultrasonic_bridge/back/distance_m",   # NEW
            }

        for key, topic in ultrasonic_topics.items():
            self.create_subscription(
                Float32,
                topic,
                lambda msg, k=key: self._ultra_cb(k, msg),
                10,
            )

        # CHANGED: gt_obstacle_names is now passed in from the env
        # (generated dynamically based on max_obstacles)
        if gt_obstacle_names is None:
            gt_obstacle_names = [f"yolo_obstacle_{i}" for i in range(10)]
        self.gt_obstacle_names = [str(x) for x in gt_obstacle_names]
        self.gt_obstacles_world = {n: None for n in self.gt_obstacle_names}
        self.gt_obstacles_vel = {n: None for n in self.gt_obstacle_names}

    def _goal_state_cb(self, msg: GoalMarkerState):
        self.goal_state = msg

    def _goal_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.goal_pose = (float(p.x), float(p.y))
        # Also capture goal velocity from odom twist so perception path
        # can compute relative velocity without GT.
        # This is NOT a Gazebo cheat — it comes from the goal marker's own odom.
        v = msg.twist.twist.linear
        self.goal_vel_gt = (float(v.x), float(v.y))

    def _model_states_cb(self, msg: ModelStates):
        try:
            i = msg.name.index(self.robot_model_name)
        except ValueError:
            return

        p = msg.pose[i].position
        o = msg.pose[i].orientation
        yaw = _yaw_from_quat(o.x, o.y, o.z, o.w)

        self.robot_pose = (float(p.x), float(p.y), float(yaw))
        self._last_odom_vx = float(msg.twist[i].linear.x)
        self._last_odom_wz = float(msg.twist[i].angular.z)

        try:
            gi = msg.name.index(self.goal_model_name)
            pg = msg.pose[gi].position
            vg = msg.twist[gi].linear
            self.goal_pose_gt = (float(pg.x), float(pg.y))
            self.goal_vel_gt = (float(vg.x), float(vg.y))
        except ValueError:
            pass

        for name in self.gt_obstacle_names:
            try:
                j = msg.name.index(name)
                p2 = msg.pose[j].position
                v2 = msg.twist[j].linear
                self.gt_obstacles_world[name] = (float(p2.x), float(p2.y))
                self.gt_obstacles_vel[name] = (float(v2.x), float(v2.y))
            except ValueError:
                pass

    def _obstacle_cb(self, msg: String):
        self.obstacles_json = msg.data

    def send_cmd(self, v: float, w: float) -> None:
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def _ultra_cb(self, key: str, msg: Float32):
        self.ultra[key] = float(msg.data)
        self.ultra_stamp[key] = time.time()


class GoalFollowerLSTMEnv(Env):
    """
    LSTM-Compatible Goal Following Environment (Curriculum Version)
    
    Returns dict observations with:
    - 'fixed': [goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy, v, w, 
                ultra_fl, ultra_fr, ultra_ls, ultra_rs, prev_v, prev_w]  (12D)
    - 'obstacles': [[x, y, d, vx, vy], ...] for up to max_obstacles (max_obstacles×5)
    
    CHANGED from original:
    - MAX_OBSTACLES is now configurable via max_obstacles parameter (default=10)
    - gt_obstacle_names generated dynamically
    - Added set_curriculum_stage() for runtime difficulty changes
    """
    
    metadata = {"render_modes": []}

    # Normalization constants
    GOAL_DIST_MAX = 15.0
    OBS_COORD_MAX = 10.0
    VEL_MAX = 2.0

    # REMOVED: class-level MAX_OBSTACLES = 5
    # Now set in __init__ as self.MAX_OBSTACLES

    def __init__(
        self,
        cmd_topic="/follower_robot/cmd_vel",
        goal_state_topic="/follower_robot/depth_cam/goal_marker_state",
        goal_odom_topic="/goal_marker/odom",
        wheel_radius=0.0925,
        wheel_separation=0.66108,
        dt=0.1,
        success_radius=1.5,
        time_limit=80.0,
        R_goal=50.0,
        R_collision=-50.0,
        gamma_shaping=0.97,
        step_penalty=0.2,
        ttc_threshold=2.0,
        cam_x=0.4,
        cam_y=0.0,
        use_ground_truth_geometry: bool = True,
        goal_model_name: str = "goal_marker",
        gt_obstacle_names: Optional[List[str]] = None,
        smooth_alpha: float = 0.70,
        ttc_lateral_margin: float = 0.8,
        # ===== NEW PARAMETERS =====
        max_obstacles: int = 10,            # Was hardcoded to 5
        moving_goal: bool = False,          # Enable/disable moving goal
    ):
        super().__init__()

        self.use_ground_truth_geometry = bool(use_ground_truth_geometry)

        # CHANGED: MAX_OBSTACLES is now configurable
        self.MAX_OBSTACLES = int(max_obstacles)

        # Moving goal control: goal only moves during active episode steps
        self.moving_goal = bool(moving_goal)

        # --------- ACTION SPACE ----------
        self.v_max = 1.0
        self.w_max = 1.0
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

        self.max_obstacle_depth = 10.0

        # Ultrasonic
        # CHANGED: Gazebo SRF05 plugins now report up to 3 m (was 1 m). The env
        # cap must match the SDF <max>; the lowest of {SDF max, bridge clamp,
        # this cap} wins. All five sensors share the 3 m range.
        self.ultra_max_range = 3.0          # front_left/front_right/left_side/right_side
        self.ultra_back_max_range = 3.0     # rear-facing 5th sensor (NEW)
        self.ultra_stale_sec = 0.5

        # --------- OBSTACLE MEMORY ----------
        self.num_obs_mem = self.MAX_OBSTACLES  # CHANGED: uses configurable max
        self.obs_match_dist = 1.2
        self.obs_expire_s = 30.0
        self.obs_slots: List[Dict[str, Any]] = [
            {"valid": False, "wx": 0.0, "wy": 0.0, "vx": 0.0, "vy": 0.0, "last_seen": 0.0} 
            for _ in range(self.num_obs_mem)
        ]

        # --------- OBSERVATION SPACE (Dict) ----------
        # Fixed features: 15D  (added rear ultrasonic — was 14D)
        #   [goal_dist, goal_bearing, goal_rel_vx, goal_rel_vy,        # 4
        #    v, w,                                                      # 2
        #    ultra_fl, ultra_fr, ultra_ls, ultra_rs, ultra_back,       # 5  ← rear added
        #    prev_v, prev_w,                                            # 2
        #    shield_severity, shield_duration_norm]                     # 2
        # Obstacle sequence: MAX_OBSTACLES × 5
        #   Each obstacle: [x, y, depth, rel_vx, rel_vy]

        self.observation_space = spaces.Dict({
            'fixed': spaces.Box(
                low=np.array(
                    [0.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                     0.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.0,
                     0.0,  0.0],   # ultra(5)+prev(2)+shield(2)
                    dtype=np.float32),
                high=np.array(
                    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                     1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                     1.0,  1.0],
                    dtype=np.float32),
                dtype=np.float32
            ),
            'obstacles': spaces.Box(
                low=np.array([[-1.0, -1.0, 0.0, -1.0, -1.0]] * self.MAX_OBSTACLES, dtype=np.float32),
                high=np.array([[1.0, 1.0, 1.0, 1.0, 1.0]] * self.MAX_OBSTACLES, dtype=np.float32),
                dtype=np.float32
            )
        })

        # Kinematics
        self.r = wheel_radius
        self.L = wheel_separation
        self.dt = dt

        self.success_radius = success_radius
        self.time_limit = time_limit

        self.cam_x = float(cam_x)
        self.cam_y = float(cam_y)

        # Reward parameters
        self.R_goal = float(R_goal)
        self.R_collision = float(R_collision)
        self.gamma = float(gamma_shaping)
        self.step_penalty = float(step_penalty)
        self.ttc_threshold = float(ttc_threshold)

        # Action smoothing
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0
        self.c_progress = 2.0

        # Reward coefficients
        self.c_action_change = 0.2
        self.ttc_lateral_margin = float(ttc_lateral_margin)
        self.ultra_proximity_zone = 0.5
        self.c_ultra_proximity = 2.0
        self.c_bearing = 0.3          # legacy — kept for reference, not used in step()

        # ── Adaptive Heading Reward (Meng et al. 2025) ───────────────────────
        # Replaces the naive c_bearing * cos(bearing) term.
        # The robot is rewarded for aligning with the *best geometrically free*
        # direction toward the goal, NOT just for facing the goal while orbiting.
        self.c_heading           = 0.6    # w_θ  — reward weight (matches paper)
        self.heading_angle_max   = math.pi / 6   # θ_max — 30° max deviation for reward
        self.heading_obs_range   = 5.0    # ignore obstacles farther than this (m)
        self.heading_cone_half   = 0.25   # half-width of obstacle exclusion cone (rad)
        self.heading_sample_step = 0.1    # angular resolution for direction search (rad)
        self.heading_fov_half    = math.pi / 2   # forward FoV limit (±90°)

        # Search-on-reset
        self.enable_search_on_reset = True
        self.search_angular_speed = 1.0

        # Collision thresholds
        self.obstacle_collision_radius = 0.7
        self.ultra_collision_radius = 0.2

        # --------- SAFETY SHIELD ----------
        self.shield_enabled = True
        self.shield_critical_dist = 0.25
        self.shield_danger_dist = 0.50
        self.shield_caution_dist = 0.80
        self.shield_obs_critical = 0.9
        self.shield_obs_danger = 1.5
        self.shield_obs_caution = 2.5
        self.shield_caution_max_v = 0.4
        self.shield_danger_max_v = 0.1
        self.shield_danger_steer_bias = 0.5
        # NEW: rear clearance required before the shield/deadlock logic is allowed
        # to command a BACKWARD escape. If the rear sensor sees something closer
        # than this, the robot rotates in place instead of reversing into it.
        self.shield_back_clear = 0.5

        # ── Passive shield + Lagrangian per-step cost ────────────────────────
        # shield_mode:
        #   "active"  -> shield OVERRIDES the policy commands (original behaviour)
        #   "passive" -> shield ONLY observes + classifies + penalizes; the
        #                policy's own commands are executed unchanged. The policy
        #                is still TOLD it is in a zone via the shield_severity /
        #                shield_duration observation dims (see _obs).
        #   "off"     -> no shield, no penalty
        self.shield_mode = "passive"

        # Graded hand-off: in passive mode, ONLY these levels are observe+penalize.
        # Any level NOT listed here is still actively overridden — so by default
        # "critical" remains an active COLLISION BACKSTOP even in passive mode,
        # while the policy learns to handle caution/danger on its own. Once the
        # policy reliably reaches goals, add "critical" here for full hand-off.
        self.shield_passive_levels = {"caution", "danger", "critical"}

        # Per-step, severity-weighted cost c_t for the Lagrangian penalty.
        # Ratios mirror the old one-time penalties (3.0 / 1.5 / 0.4); the absolute
        # scale is irrelevant because lambda is auto-tuned by the PID-Lagrangian
        # controller in the training callback.
        self.shield_cost_weights = {
            "none": 0.0, "caution": 0.4, "danger": 1.5, "critical": 3.0,
        }
        # Lagrange multiplier (penalty weight). Updated externally each episode by
        # the callback via set_shield_lambda(). This is the ONLY knob, and it is
        # tuned automatically — do not hand-pick a fixed value.
        self.shield_lambda = 1.0
        # Per-episode constraint accumulators.
        self._ep_shield_cost = 0.0      # sum of c_t over the episode (the J_c signal)
        self._ep_critical_steps = 0     # raw count of steps spent in critical (KPI)

        # ── Per-episode shield-cost CAP (economics guard) ────────────────────
        # The previous run exploded ep_shield_cost to ~1300 (hundreds of phantom-
        # critical steps under noisy perception), so lambda*cost (~287) dwarfed
        # R_goal (100) and reaching the goal became unprofitable. We cap the cost
        # accrued per episode so the TOTAL penalty (lambda * ep_shield_cost) can
        # never exceed roughly half the goal reward, regardless of detector noise
        # or how long the robot lingers in a zone.
        #   pick cap so  lambda_max * cap  <=  ~0.5 * R_goal
        #   e.g. lambda_max=2, R_goal=100  ->  cap = 25
        self.shield_cost_cap = 25.0
        self._ep_shield_cost_raw = 0.0  # uncapped, for diagnostics only

        # ── Ground-truth -> perception blend schedule ────────────────────────
        # Teacher/student (privileged-information) training: start an episode with
        # clean GT geometry with probability p_gt, otherwise use real detections.
        # p_gt decays linearly from 1.0 to 0.0 over `_gt_blend_episodes` from the
        # moment start_gt_blend() is called (typically at each stage change). This
        # is set per-episode in reset(). _gt_blend_episodes=0 => always perception
        # (honours the constructor's use_ground_truth_geometry as a static value).
        self._gt_blend_episodes = 0
        self._gt_blend_counter = 0
        self._gt_static_mode = bool(self.use_ground_truth_geometry)
        self._shield_interventions = 0
        self._shield_interventions_danger = 0
        self._shield_interventions_caution = 0
        self._shield_interventions_critical = 0
        self._shield_active = False

        # FIX Design-Issue-5: track shield level + duration for observation
        self._current_shield_level = "none"   # last executed shield level
        self._steps_in_shield = 0             # consecutive steps shield was active

        # FIX Design-Issue-6: deadlock escape counter
        self.deadlock_escape_steps = 30       # steps before forced escape
        self._deadlock_counter = 0            # consecutive near-zero velocity steps
        self._forced_escapes = 0              # total forced escapes this episode

        # Episode-level metric accumulators
        self._ep_min_obstacle_depth = float("inf")   # min over all steps
        self._ep_min_ultrasonic_distance = float("inf")  # min over all steps
        self._ep_jerk_v_list: list = []   # |delta_v| each step (for mean jerk)
        self._ep_jerk_w_list: list = []   # |delta_w| each step (for mean jerk)
        self._ep_ttc_list: list = []      # TTC per step (inf replaced by 5)

        # FIX Design-Issue-7 + Bug-3: track policy intent separately from executed cmd
        # Smoothing is done from _prev_v_policy (what policy wanted),
        # NOT from the shield-modified output, so recovery is instant post-shield.
        self._prev_v_policy = 0.0
        self._prev_w_policy = 0.0

        # FIX Bug-2: one-time shield penalty flag — reset each episode
        self._shield_penalty_warned = {
            "critical": False,
            "danger":   False,
            "caution":  False,
        }

        # CHANGED: Generate gt_obstacle_names dynamically based on max_obstacles
        if gt_obstacle_names is None:
            gt_obstacle_names = [
                f"yolo_obstacle_{i}" for i in range(self.MAX_OBSTACLES)
            ]

        # ROS wrapper
        self.ros = _ROS(
            cmd_topic=cmd_topic,
            goal_state_topic=goal_state_topic,
            goal_odom_topic=goal_odom_topic,
            obstacle_topic="/follower_robot/obstacles_depth",
            ultrasonic_topics={
                "front_left":  "/follower_robot/ultrasonic_bridge/front_left/distance_m",
                "front_right": "/follower_robot/ultrasonic_bridge/front_right/distance_m",
                "left_side":   "/follower_robot/ultrasonic_bridge/left_side/distance_m",
                "right_side":  "/follower_robot/ultrasonic_bridge/right_side/distance_m",
                "back":        "/follower_robot/ultrasonic_bridge/back/distance_m",   # NEW
            },
            robot_model_name="my_robot",
            goal_model_name=goal_model_name,
            gt_obstacle_names=gt_obstacle_names,
        )
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.ros)

        # Episode state
        self._t0 = None
        self._prev_phi = 0.0

        # Logging
        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None
        self.prev_dist = None

        self.bearing_threshold = math.pi / 6  # 30 degrees
        self.obstacle_radius = 0.3  # meters


        # CHANGED: Default observation uses self.MAX_OBSTACLES (dynamic) + 15D fixed
        self._last_obs_valid = {
            'fixed': np.array([
                7.0 / self.GOAL_DIST_MAX, 0.0, 0.0, 0.0,  # goal (4)
                0.0, 0.0,                                   # robot vel (2)
                1.0, 1.0, 1.0, 1.0, 1.0,                   # ultrasonics (5) ← rear added
                0.0, 0.0,                                   # prev action (2)
                0.0, 0.0,                                   # shield_severity, shield_duration (2)
            ], dtype=np.float32),
            'obstacles': np.array(
                [[0.0, 0.0, 1.0, 0.0, 0.0]] * self.MAX_OBSTACLES,
                dtype=np.float32
            )
        }

        # --------- TEACHER DISTILLATION / BC ANCHOR ----------
        # In student training, the env can query a frozen teacher on the same
        # physical state but with ground-truth geometry.  The resulting action is
        # returned in info["bc_teacher_action"] and stored by the custom replay
        # buffer.  This deliberately does NOT change the student's observation;
        # it only supplies a supervised target for the SAC actor update.
        self.bc_teacher_model = None
        self.bc_teacher_enabled = False
        self.bc_teacher_deterministic = True


    # ===== TEACHER DISTILLATION HOOKS =====
    def set_bc_teacher(self, teacher_model=None, enabled: bool = True, deterministic: bool = True) -> None:
        """Attach/detach a frozen privileged teacher for BC distillation.

        The student still receives its normal observation (perception or
        curriculum-selected geometry).  At every step, before the student's
        command is executed, the env builds a *temporary* GT observation for the
        current physical state and queries this teacher.  The resulting action
        is emitted through info["bc_teacher_action"].
        """
        self.bc_teacher_model = teacher_model
        self.bc_teacher_enabled = bool(enabled and teacher_model is not None)
        self.bc_teacher_deterministic = bool(deterministic)
        state = "enabled" if self.bc_teacher_enabled else "disabled"
        try:
            self.ros.get_logger().info(f"[BC] Teacher distillation {state}")
        except Exception:
            pass

    def _copy_obs_dict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "fixed": np.array(obs["fixed"], dtype=np.float32, copy=True),
            "obstacles": np.array(obs["obstacles"], dtype=np.float32, copy=True),
        }

    def _build_gt_observation_for_teacher(self) -> Optional[Dict[str, np.ndarray]]:
        """Build a GT observation without changing the student's env state.

        _obs() normally mutates obs_slots and _last_obs_valid.  For distillation
        we need a privileged observation only for the teacher query, so this
        method snapshots those mutable fields, temporarily switches to GT
        geometry, calls _obs(), copies the result, and then restores the student
        state exactly as it was.
        """
        if self.ros.robot_pose is None or self.ros.goal_pose_gt is None:
            return None

        saved_use_gt = bool(self.use_ground_truth_geometry)
        saved_slots = [dict(slot) for slot in self.obs_slots]
        saved_last = self._copy_obs_dict(self._last_obs_valid)

        try:
            self.use_ground_truth_geometry = True
            gt_obs = self._obs(default=False)
            if gt_obs is None:
                return None
            if not (np.all(np.isfinite(gt_obs["fixed"])) and np.all(np.isfinite(gt_obs["obstacles"]))):
                return None
            return self._copy_obs_dict(gt_obs)
        except Exception as e:
            try:
                self.ros.get_logger().warn(f"[BC] Failed to build GT teacher obs: {e}")
            except Exception:
                pass
            return None
        finally:
            self.use_ground_truth_geometry = saved_use_gt
            self.obs_slots = saved_slots
            self._last_obs_valid = saved_last

    def _compute_bc_teacher_action(self) -> Tuple[Optional[np.ndarray], bool]:
        """Return frozen-teacher action for the current pre-step state."""
        if not self.bc_teacher_enabled or self.bc_teacher_model is None:
            return None, False

        gt_obs = self._build_gt_observation_for_teacher()
        if gt_obs is None:
            return None, False

        try:
            action, _ = self.bc_teacher_model.predict(
                gt_obs,
                deterministic=self.bc_teacher_deterministic,
            )
            action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
            action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
            return action, bool(np.all(np.isfinite(action)))
        except Exception as e:
            try:
                self.ros.get_logger().warn(f"[BC] Teacher prediction failed: {e}")
            except Exception:
                pass
            return None, False

    # ===== NEW METHOD: Curriculum stage setter =====
    def set_curriculum_stage(self, stage_config: dict):
        """
        Update environment parameters for curriculum learning.
        Call this between episodes (before or after reset).
        
        Args:
            stage_config: dict with optional keys:
                - 'time_limit': float (seconds)
                - 'success_radius': float (meters)
                - 'R_goal': float (goal reward)
                - 'R_collision': float (collision penalty)
                - 'step_penalty': float
        """
        if 'time_limit' in stage_config:
            self.time_limit = float(stage_config['time_limit'])
        if 'success_radius' in stage_config:
            self.success_radius = float(stage_config['success_radius'])
        if 'R_goal' in stage_config:
            self.R_goal = float(stage_config['R_goal'])
        if 'R_collision' in stage_config:
            self.R_collision = float(stage_config['R_collision'])
        if 'step_penalty' in stage_config:
            self.step_penalty = float(stage_config['step_penalty'])
        if 'moving_goal' in stage_config:
            self.moving_goal = bool(stage_config['moving_goal'])

        self.ros.get_logger().info(
            f"[CURRICULUM] Stage updated: time_limit={self.time_limit:.1f}s "
            f"success_radius={self.success_radius:.2f}m "
            f"R_goal={self.R_goal:.1f} R_collision={self.R_collision:.1f}"
        )

    def _set_goal_mover_enabled(self, enabled: bool):
        """Enable or disable the goal marker mover node via service call."""
        if not self.moving_goal:
            return
        val = "true" if enabled else "false"
        self._call_service_cli(
            "/goal_mover/enable",
            "std_srvs/srv/SetBool",
            f"{{data: {val}}}"
        )

    def _spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.exec.spin_once(timeout_sec=0.001)

    def _ultra_vector(self) -> Tuple[np.ndarray, float]:
        now = time.time()
        keys = ["front_left", "front_right", "left_side", "right_side"]
        vals = []
        for k in keys:
            v = self.ros.ultra.get(k, None)
            if v is None:
                vals.append(self.ultra_max_range)
                continue
            if (now - self.ros.ultra_stamp.get(k, 0.0)) > self.ultra_stale_sec:
                vals.append(self.ultra_max_range)
                continue
            if not math.isfinite(v):
                vals.append(self.ultra_max_range)
                continue
            v = max(0.0, min(float(v), self.ultra_max_range))
            vals.append(v)

        ultra_vec = np.array(vals, dtype=np.float32)
        ultra_min = float(np.min(ultra_vec)) if ultra_vec.size > 0 else float(self.ultra_max_range)
        return ultra_vec, ultra_min

    def _ultra_back(self) -> float:
        """Rear-facing ultrasonic reading in metres, clamped to its own range.

        Kept SEPARATE from _ultra_vector (which feeds the forward/side shield
        logic) so adding the rear sensor does not change any front/side mins.
        Returns ultra_back_max_range when missing / stale / non-finite
        (i.e. 'nothing detected behind').
        """
        now = time.time()
        v = self.ros.ultra.get("back", None)
        if v is None:
            return float(self.ultra_back_max_range)
        if (now - self.ros.ultra_stamp.get("back", 0.0)) > self.ultra_stale_sec:
            return float(self.ultra_back_max_range)
        if not math.isfinite(v):
            return float(self.ultra_back_max_range)
        return max(0.0, min(float(v), self.ultra_back_max_range))

    def _parse_obstacles_local(self) -> List[Tuple[float, float, float]]:
        if getattr(self.ros, "obstacles_json", None) is None:
            return []
        try:
            objs = json.loads(self.ros.obstacles_json)
        except Exception:
            return []
        if not isinstance(objs, list):
            return []

        out = []
        for o in objs:
            try:
                if "x_m" not in o or "y_m" not in o:
                    continue
                x = float(o.get("x_m"))
                y = float(o.get("y_m"))
                d = float(o.get("depth_m", math.hypot(x, y)))
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(d)):
                    continue
                if d <= 0.0:
                    continue
                d = max(0.0, min(d, self.max_obstacle_depth))
                out.append((x, y, d))
            except Exception:
                continue
        return out

    def _expire_obstacle_slots(self, now: float):
        for s in self.obs_slots:
            if s["valid"] and (now - float(s["last_seen"])) > self.obs_expire_s:
                s["valid"] = False

    def _update_obstacle_memory_gt(self):
        now = time.time()
        self._expire_obstacle_slots(now)

        if self.ros.robot_pose is None:
            return

        rx, ry, _ = self.ros.robot_pose

        pts = []
        for name in self.ros.gt_obstacle_names:
            p = self.ros.gt_obstacles_world.get(name, None)
            v = self.ros.gt_obstacles_vel.get(name, None)
            if p is None:
                continue
            vx, vy = (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)
            pts.append((float(p[0]), float(p[1]), vx, vy))

        pts.sort(key=lambda item: math.hypot(item[0] - rx, item[1] - ry))

        for j in range(self.num_obs_mem):
            if j < len(pts):
                wx, wy, vx, vy = pts[j]
                self.obs_slots[j]["valid"] = True
                self.obs_slots[j]["wx"] = wx
                self.obs_slots[j]["wy"] = wy
                self.obs_slots[j]["vx"] = vx
                self.obs_slots[j]["vy"] = vy
                self.obs_slots[j]["last_seen"] = now
            else:
                self.obs_slots[j]["valid"] = False

    def _slots_local_quintuplets_sorted_by_threat(self) -> List[Tuple[float, float, float, float, float]]:
        """Returns (lx, ly, d, rel_vx, rel_vy) for each obstacle slot"""
        if self.ros.robot_pose is None:
            return [(0.0, 0.0, self.max_obstacle_depth, 0.0, 0.0) for _ in range(self.num_obs_mem)]

        rx, ry, ryaw = self.ros.robot_pose
        
        robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
        robot_vy = self.ros._last_odom_vx * math.sin(ryaw)

        quintuplets = []
        for s in self.obs_slots:
            if not s["valid"]:
                quintuplets.append((0.0, 0.0, self.max_obstacle_depth, 0.0, 0.0))
                continue

            lx, ly = _local_from_world(rx, ry, ryaw, float(s["wx"]), float(s["wy"]))
            d = math.hypot(lx, ly)
            d = max(0.0, min(d, self.max_obstacle_depth))
            
            rel_vx_world = float(s["vx"]) - robot_vx
            rel_vy_world = float(s["vy"]) - robot_vy
            
            rel_vx_local = math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
            rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world
            
            quintuplets.append((float(lx), float(ly), float(d), float(rel_vx_local), float(rel_vy_local)))

        quintuplets_sorted = sorted(quintuplets, key=lambda t: t[2])
        return quintuplets_sorted[:self.num_obs_mem]

    def _gt_goal_dist_bearing_rel_vel(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if self.ros.robot_pose is None or self.ros.goal_pose_gt is None:
            return None, None, None, None
        
        rx, ry, ryaw = self.ros.robot_pose
        gx, gy = self.ros.goal_pose_gt
        
        dx = gx - rx
        dy = gy - ry
        dist = float(math.hypot(dx, dy))
        bearing = _wrap(math.atan2(dy, dx) - ryaw)
        
        if self.ros.goal_vel_gt is not None:
            goal_vx, goal_vy = self.ros.goal_vel_gt
            
            robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
            robot_vy = self.ros._last_odom_vx * math.sin(ryaw)
            
            rel_vx_world = goal_vx - robot_vx
            rel_vy_world = goal_vy - robot_vy
            
            rel_vx_local = math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
            rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world
        else:
            rel_vx_local = 0.0
            rel_vy_local = 0.0
        
        return dist, bearing, float(rel_vx_local), float(rel_vy_local)

    # ------------------------------------------------------------------
    # Perception-based (non-GT) obstacle memory update
    # ------------------------------------------------------------------

    def _update_obstacle_memory_perception(self):
        """
        Populate obs_slots from the obstacle-depth JSON published on
        /follower_robot/obstacles_depth.

        The preferred obstacle-node contract is:
            raw_x_m, raw_y_m  – direct stereo position in base_link frame
            x_m, y_m          – Kalman-filtered position in base_link frame
            vx_ms, vy_ms      – Kalman velocity estimate in base_link frame
            depth_m           – camera-forward depth, mainly for debugging

        For RL obstacle position we prefer raw_x_m/raw_y_m because validation
        showed lower position error than the lagged Kalman x_m/y_m.  We keep
        KF velocity as the auxiliary relative-velocity signal.
        """
        if getattr(self.ros, "obstacles_json", None) is None:
            return

        try:
            objs = json.loads(self.ros.obstacles_json)
        except Exception:
            return

        if not isinstance(objs, list) or len(objs) == 0:
            return

        if self.ros.robot_pose is None:
            return

        rx, ry, ryaw = self.ros.robot_pose
        now = time.time()
        self._expire_obstacle_slots(now)

        # Convert each detection from base_link-local into world frame so
        # that obs_slots stay world-frame (same convention as GT path).
        # base_link is robot-body frame: +x forward, +y left.
        # world frame is fixed.
        candidates = []
        for o in objs:
            try:
                # Position: obstacle node already publishes coordinates in base_link.
                # Prefer the direct stereo measurement for position because our
                # validation showed it is less lagged than KF x_m/y_m.  Fall back
                # to x_m/y_m for compatibility with older obstacle nodes.
                lx = float(o.get("raw_x_m", o["x_m"]))
                ly = float(o.get("raw_y_m", o["y_m"]))
                wx, wy = _world_from_local(rx, ry, ryaw, lx, ly)

                # Velocity: keep the KF velocity estimate as the auxiliary
                # velocity signal.  It is smoother than frame-to-frame raw
                # position differencing, but position itself should use raw_x/y.
                vlx = float(o.get("vx_ms", 0.0))
                vly = float(o.get("vy_ms", 0.0))
                # Rotate: world_vx = cos(yaw)*vlx - sin(yaw)*vly
                #         world_vy = sin(yaw)*vlx + cos(yaw)*vly
                wvx = math.cos(ryaw) * vlx - math.sin(ryaw) * vly
                wvy = math.sin(ryaw) * vlx + math.cos(ryaw) * vly

                # Use base-link planar range for sorting/threat logic, not
                # camera-forward depth_m.  depth_m is useful for debugging only.
                d = float(o.get("raw_range_m", math.hypot(lx, ly)))
                if not (math.isfinite(wx) and math.isfinite(wy)
                        and math.isfinite(wvx) and math.isfinite(wvy)
                        and math.isfinite(d) and d > 0.0):
                    continue

                candidates.append((wx, wy, wvx, wvy, d))
            except Exception:
                continue

        # Sort by depth (closest first) and fill slots
        candidates.sort(key=lambda c: c[4])

        for j in range(self.num_obs_mem):
            if j < len(candidates):
                wx, wy, wvx, wvy, _ = candidates[j]
                self.obs_slots[j]["valid"]     = True
                self.obs_slots[j]["wx"]        = wx
                self.obs_slots[j]["wy"]        = wy
                self.obs_slots[j]["vx"]        = wvx
                self.obs_slots[j]["vy"]        = wvy
                self.obs_slots[j]["last_seen"] = now
            else:
                # Don't invalidate stale slots immediately — let expiry handle it
                pass

    # ------------------------------------------------------------------
    # Perception-based (non-GT) goal distance / bearing / relative vel
    # ------------------------------------------------------------------

    def _perception_goal_dist_bearing_rel_vel(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute goal dist, bearing, and relative velocity without GT.

        Sources used (in priority order):
          1. goal_pose  – from /goal_marker/odom (Odometry), already subscribed.
             This is the real odometry of the goal marker, not a Gazebo cheat.
          2. goal_state – from GoalMarkerState (detection-based fallback for
             bearing if odom is stale).

        Relative velocity is computed as:
            goal_odom_vel (from /goal_marker/odom twist) − robot_vel (from
            /model_states or odometry).  If goal odom vel is unavailable we
            fall back to 0.
        """
        if self.ros.robot_pose is None:
            return None, None, None, None

        rx, ry, ryaw = self.ros.robot_pose

        # --- Goal position from odometry (not GT) ---
        goal_pos = self.ros.goal_pose          # set by _goal_odom_cb
        if goal_pos is None:
            # Last fallback: use GoalMarkerState bearing if available
            gs = self.ros.goal_state
            if gs is None or not getattr(gs, "visible", False):
                return None, None, None, None
            # GoalMarkerState gives bearing directly in robot frame
            # but no distance — return None so caller uses last valid obs
            return None, None, None, None

        gx, gy = goal_pos
        dx = gx - rx
        dy = gy - ry
        dist    = float(math.hypot(dx, dy))
        bearing = _wrap(math.atan2(dy, dx) - ryaw)

        # --- Relative velocity ---
        # goal_vel_gt is set by _goal_odom_cb from the goal marker's Odometry
        # twist — this is NOT a Gazebo cheat; it's published by the goal node.
        # NOTE: we reuse the same field that _goal_odom_cb writes to because
        # the subscriber already parses the twist from the odom message.
        # If your _goal_odom_cb doesn't parse twist, see the note below (*).
        goal_vel = getattr(self.ros, "goal_vel_gt", None)   # (vx_world, vy_world)

        robot_vx = self.ros._last_odom_vx * math.cos(ryaw)
        robot_vy = self.ros._last_odom_vx * math.sin(ryaw)

        if goal_vel is not None:
            rel_vx_world = float(goal_vel[0]) - robot_vx
            rel_vy_world = float(goal_vel[1]) - robot_vy
        else:
            rel_vx_world = -robot_vx
            rel_vy_world = -robot_vy

        rel_vx_local =  math.cos(ryaw) * rel_vx_world + math.sin(ryaw) * rel_vy_world
        rel_vy_local = -math.sin(ryaw) * rel_vx_world + math.cos(ryaw) * rel_vy_world

        return dist, bearing, float(rel_vx_local), float(rel_vy_local)

    # (*) If _goal_odom_cb does NOT parse twist, add this to it:
    #   vg = msg.twist.twist.linear
    #   self.ros.goal_vel_gt = (float(vg.x), float(vg.y))

    def _compute_ttc(self, obs_quintuplets: List[Tuple[float, float, float, float, float]], v_forward: float) -> float:
        if v_forward <= 1e-3:
            return float("inf")

        min_ttc = float("inf")
        for (ox, oy, d, rel_vx, rel_vy) in obs_quintuplets:
            if ox <= 0.0:
                continue
            if abs(oy) > self.ttc_lateral_margin:
                continue
            
            approach_speed = v_forward - rel_vx
            
            if approach_speed > 1e-3:
                ttc = float(ox) / approach_speed
                if ttc < min_ttc:
                    min_ttc = ttc
        return min_ttc

    def _compute_adaptive_heading_reward(
        self,
        bearing: float,
        obs_quintuplets: List[Tuple[float, float, float, float, float]],
        v_cmd: float,
        shield_level: str,
    ) -> float:
        """
        Adaptive Navigation Heading Reward — adapted from Meng et al. (2025).

        Instead of rewarding the robot for merely *facing* the goal (naive cos(bearing)),
        this method:
          1. Builds a set of blocked angular directions from nearby obstacles.
          2. Searches for the best *free* direction closest to the goal bearing.
          3. Awards reward only when the robot's heading matches that free direction
             within a tolerance (heading_angle_max = π/6 = 30°).

        Key differences from raw cos(bearing):
          - Zero reward when ALL forward directions are blocked (e.g. dense obstacle wall).
          - Zero reward when the robot is backing up (v_cmd ≤ 0) — no posture harvesting.
          - The reward degrades gracefully as the best free direction diverges from the goal,
            rather than always being +0.28/step regardless of obstacle context.

        Args:
            bearing:         Goal bearing in robot frame (rad), range [−π, π].
            obs_quintuplets: List of (lx, ly, depth, rel_vx, rel_vy) in robot frame.
            v_cmd:           Executed forward velocity after shield (m/s).
            shield_level:    Current shield level string.

        Returns:
            Scalar reward term (≥ 0).
        """
        # ── Gate 1: no heading reward when backing up or shield-critical ──────
        # During critical shield the robot moves at v=−0.15. Rewarding heading
        # alignment in that state is exactly the orbiting exploit we are fixing.
        if v_cmd <= 0.0 or shield_level == "critical":
            return 0.0

        # ── Build blocked angle set from nearby obstacles ─────────────────────
        # Obstacle cone width = heading_cone_half (0.25 rad ≈ 14°) each side.
        blocked: List[float] = []
        for (lx, ly, d, _rvx, _rvy) in obs_quintuplets:
            if d <= 0.0 or d > self.heading_obs_range:
                continue
            obs_angle = math.atan2(ly, lx)   # angle to obstacle in robot frame
            # Only consider obstacles in the forward hemisphere
            if abs(obs_angle) > self.heading_fov_half:
                continue
            # Cone width scales with proximity: closer → wider cone
            # At d=0.5m cone=0.40 rad; at d=5.0m cone=0.10 rad (linear interp)
            proximity_factor = max(0.0, 1.0 - d / self.heading_obs_range)
            cone = self.heading_cone_half * (1.0 + proximity_factor)
            blocked.append((obs_angle, cone))

        # ── Clip goal bearing to forward FoV ─────────────────────────────────
        theta_g = float(np.clip(bearing, -self.heading_fov_half, self.heading_fov_half))

        # ── Search for best free direction ────────────────────────────────────
        # Sweep candidate directions from most-preferred (θ_g) outward in both
        # directions until a free one is found.
        theta_best: Optional[float] = None

        # Build sorted candidate list: [θ_g, θ_g±step, θ_g±2·step, ...]
        max_offset = self.heading_fov_half
        offsets = [0.0]
        d_off = self.heading_sample_step
        while d_off <= max_offset + 1e-6:
            offsets.append(+d_off)
            offsets.append(-d_off)
            d_off += self.heading_sample_step

        for offset in offsets:
            theta = theta_g + offset
            if abs(theta) > self.heading_fov_half:
                continue
            # Check if theta lies inside any obstacle cone
            is_free = True
            for (obs_angle, cone) in blocked:
                angular_dist = abs(_wrap(theta - obs_angle))
                if angular_dist < cone:
                    is_free = False
                    break
            if is_free:
                theta_best = theta
                break

        # ── Compute reward ────────────────────────────────────────────────────
        if theta_best is None:
            # All forward directions blocked — zero reward (not negative, shield handles that)
            return 0.0

        deviation = abs(theta_best - theta_g)   # how far best free dir is from goal
        if deviation > self.heading_angle_max:
            # Best free direction is too far from goal to be meaningful
            return 0.0

        # Linear reward: max at deviation=0, zero at deviation=heading_angle_max
        reward = self.c_heading * (self.heading_angle_max - deviation)

        # ── Gate 2: scale by forward speed so slow motion earns less ─────────
        # v_cmd in [0, v_max=1.0] → factor in [0, 1]
        speed_factor = float(np.clip(v_cmd / self.v_max, 0.0, 1.0))
        reward *= speed_factor

        return float(reward)

    def start_gt_blend(self, n_episodes: int, already_elapsed: int = 0) -> None:
        self._gt_blend_episodes = int(max(0, n_episodes))
        self._gt_blend_counter = int(max(0, already_elapsed))

        # Important: n_episodes=0 should mean "no blend",
        # not "force perception".
        if self._gt_blend_episodes == 0:
            return

        if self._gt_blend_counter >= self._gt_blend_episodes:
            self._gt_blend_episodes = 0
            self._gt_static_mode = False
    def _select_geometry_mode_for_episode(self) -> None:
        """Decide GT vs perception for the episode about to start (called in
        reset()). Sets self.use_ground_truth_geometry accordingly."""
        if self._gt_blend_episodes <= 0:
            # No active blend: fall back to the static configured mode.
            self.use_ground_truth_geometry = self._gt_static_mode
            return
        # Linear decay of GT probability over the blend window.
        p_gt = 1.0 - (self._gt_blend_counter / float(self._gt_blend_episodes))
        p_gt = float(np.clip(p_gt, 0.0, 1.0))
        self.use_ground_truth_geometry = bool(np.random.random() < p_gt)
        self._gt_blend_counter += 1
        if self._gt_blend_counter >= self._gt_blend_episodes:
            # Blend finished: lock to pure perception from here on.
            self._gt_blend_episodes = 0
            self._gt_static_mode = False

    def set_shield_lambda(self, lam: float) -> None:
        """Set the Lagrange multiplier (penalty weight). Called by the
        PID-Lagrangian controller in the training callback once per episode."""
        self.shield_lambda = float(max(0.0, lam))

    def _shield_gate(self, v_cmd, w_cmd, safe_v, safe_w, level):
        """Decide what command actually gets executed, based on shield_mode.

        passive -> return the policy's ORIGINAL command (observe + penalize only).
        active  -> return the shield's corrected (safe) command.
        The classified `level` is returned unchanged in both cases so the
        observation, penalty and stats all still reflect the zone the robot is in.
        """
        if self.shield_mode == "passive" and level in self.shield_passive_levels:
            return v_cmd, w_cmd, level          # observe + penalize only
        return safe_v, safe_w, level            # active override (incl. critical backstop)

    def _apply_safety_shield(
        self,
        v_cmd: float,
        w_cmd: float,
        ultra_fl: float,
        ultra_fr: float,
        ultra_ls: float,
        ultra_rs: float,
        obs_quintuplets: List[Tuple[float, float, float, float, float]],
        ultra_back: float = float("inf"),     # NEW: rear clearance (metres)
    ) -> Tuple[float, float, str]:
        if not self.shield_enabled:
            return v_cmd, w_cmd, "none"

        safe_v = v_cmd
        safe_w = w_cmd
        level  = "none"

        min_front     = min(ultra_fl, ultra_fr)
        min_side      = min(ultra_ls, ultra_rs)
        min_ultra_all = min(min_front, min_side)

        ultra_left  = min(ultra_fl, ultra_ls)
        ultra_right = min(ultra_fr, ultra_rs)

        min_front_obs_depth = float("inf")
        closest_obs_oy = 0.0
        for (ox, oy, d, rel_vx, rel_vy) in obs_quintuplets:
            if ox > 0.0 and abs(oy) < self.ttc_lateral_margin:
                if ox < min_front_obs_depth:
                    min_front_obs_depth = ox
                    closest_obs_oy = oy

        # ── CRITICAL ────────────────────────────────────────────────────────
        # FIX Bug-1: Previously set safe_w = 0.0 here, causing complete deadlock.
        # Now the robot backs up slightly and rotates toward the clearer side so
        # it can actively escape rather than sitting frozen while the obstacle orbits.
        if min_ultra_all < self.shield_critical_dist or min_front_obs_depth < self.shield_obs_critical:
            # NEW: only reverse if the rear is clear; otherwise rotate in place
            # so the robot does not back into an obstacle behind it.
            safe_v = -0.15 if ultra_back > self.shield_back_clear else 0.0
            # Rotate toward the side with greater clearance
            if ultra_left >= ultra_right:
                safe_w = +self.w_max * 0.8   # turn left (CCW)
            else:
                safe_w = -self.w_max * 0.8   # turn right (CW)
            safe_w = max(-self.w_max, min(safe_w, self.w_max))
            level = "critical"
            self._shield_interventions += 1
            self._shield_interventions_critical += 1
            return self._shield_gate(v_cmd, w_cmd, safe_v, safe_w, level)

        # ── DANGER ──────────────────────────────────────────────────────────
        in_danger = (min_front < self.shield_danger_dist or
                     min_front_obs_depth < self.shield_obs_danger)

        if in_danger:
            if safe_v > self.shield_danger_max_v:
                safe_v = self.shield_danger_max_v

            if min_front < self.shield_danger_dist:
                if ultra_left < ultra_right:
                    safe_w = safe_w - self.shield_danger_steer_bias
                else:
                    safe_w = safe_w + self.shield_danger_steer_bias
            elif min_front_obs_depth < self.shield_obs_danger:
                if closest_obs_oy >= 0:
                    safe_w = safe_w - self.shield_danger_steer_bias
                else:
                    safe_w = safe_w + self.shield_danger_steer_bias

            safe_w = max(-self.w_max, min(safe_w, self.w_max))
            level = "danger"
            self._shield_interventions += 1
            self._shield_interventions_danger += 1
            return self._shield_gate(v_cmd, w_cmd, safe_v, safe_w, level)

        # ── CAUTION ─────────────────────────────────────────────────────────
        in_caution = (min_front < self.shield_caution_dist or
                      min_front_obs_depth < self.shield_obs_caution or
                      min_side < self.shield_danger_dist)

        if in_caution:
            if safe_v > self.shield_caution_max_v:
                safe_v = self.shield_caution_max_v
            level = "caution"
            self._shield_interventions += 1
            self._shield_interventions_caution += 1
            return self._shield_gate(v_cmd, w_cmd, safe_v, safe_w, level)

        return safe_v, safe_w, level

    def _obs(self, default: bool = False) -> Dict[str, np.ndarray]:
        """
        Returns dict observation:
        {
            'fixed': [12D] - goal, robot, ultrasonics, prev_action
            'obstacles': [MAX_OBSTACLES×5] - obstacle sequence (x, y, d, vx, vy)
        }
        """
        if default:
            return self._last_obs_valid.copy()

        if self.use_ground_truth_geometry:
            self._update_obstacle_memory_gt()
        else:
            # Perception path: fill obs_slots from Kalman depth node JSON
            self._update_obstacle_memory_perception()

        # Goal dist / bearing / relative velocity
        if self.use_ground_truth_geometry:
            dist, bearing, goal_rel_vx, goal_rel_vy = self._gt_goal_dist_bearing_rel_vel()
        else:
            dist, bearing, goal_rel_vx, goal_rel_vy = self._perception_goal_dist_bearing_rel_vel()

        if dist is None or bearing is None:
            return self._last_obs_valid.copy()

        ultra_vec, _ = self._ultra_vector()
        ultra_fl, ultra_fr, ultra_ls, ultra_rs = map(float, ultra_vec.tolist())
        ultra_back = self._ultra_back()   # NEW: rear sensor (metres)

        obs_quintuplets = self._slots_local_quintuplets_sorted_by_threat()

        v = float(self.ros._last_odom_vx)
        w = float(self.ros._last_odom_wz)

        # Build fixed features (15D — rear ultrasonic added; shield state appended)
        # Map shield level to a scalar severity in [0, 1]
        shield_severity_map = {"none": 0.0, "caution": 0.33, "danger": 0.67, "critical": 1.0}
        shield_severity   = shield_severity_map.get(self._current_shield_level, 0.0)
        shield_dur_norm   = float(min(self._steps_in_shield / 50.0, 1.0))

        fixed = np.array([
            np.clip(dist / self.GOAL_DIST_MAX, 0.0, 1.0),
            bearing / math.pi,
            np.clip(goal_rel_vx / self.VEL_MAX, -1.0, 1.0),
            np.clip(goal_rel_vy / self.VEL_MAX, -1.0, 1.0),
            np.clip(v / self.v_max, -1.0, 1.0),
            np.clip(w / self.w_max, -1.0, 1.0),
            ultra_fl / self.ultra_max_range,
            ultra_fr / self.ultra_max_range,
            ultra_ls / self.ultra_max_range,
            ultra_rs / self.ultra_max_range,
            ultra_back / self.ultra_back_max_range,   # NEW: rear sensor [0,1]
            np.clip(self._prev_v_policy / self.v_max, -1.0, 1.0),   # FIX Design-Issue-7: policy intent
            np.clip(self._prev_w_policy / self.w_max, -1.0, 1.0),
            float(shield_severity),     # NEW: 0=none, 0.33=caution, 0.67=danger, 1.0=critical
            float(shield_dur_norm),     # NEW: how long shield has been active (0–1)
        ], dtype=np.float32)

        # Build obstacle sequence (MAX_OBSTACLES×5)
        obstacles = np.zeros((self.MAX_OBSTACLES, 5), dtype=np.float32)
        for i, (x, y, d, rvx, rvy) in enumerate(obs_quintuplets):
            obstacles[i, 0] = np.clip(x / self.OBS_COORD_MAX, -1.0, 1.0)
            obstacles[i, 1] = np.clip(y / self.OBS_COORD_MAX, -1.0, 1.0)
            obstacles[i, 2] = np.clip(d / self.max_obstacle_depth, 0.0, 1.0)
            obstacles[i, 3] = np.clip(rvx / self.VEL_MAX, -1.0, 1.0)
            obstacles[i, 4] = np.clip(rvy / self.VEL_MAX, -1.0, 1.0)

        obs_dict = {
            'fixed': fixed,
            'obstacles': obstacles
        }

        # Validate
        if not (np.all(np.isfinite(fixed)) and np.all(np.isfinite(obstacles))):
            return self._last_obs_valid.copy()

        self._last_obs_valid = obs_dict
        return obs_dict

    def _reset_entity_with_retry(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
        yaw: float = 0.0,
        max_attempts: int = 3,
    ) -> bool:
        for attempt in range(max_attempts):
            self.ros.get_logger().info(f"[{name}] Reset attempt {attempt + 1}/{max_attempts}")
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)

            req = (
                "{state: {"
                f"name: '{name}', "
                "pose: {position: {"
                f"x: {x}, y: {y}, z: {z}"
                "}, orientation: {"
                f"z: {qz}, w: {qw}"
                "}}, "
                "twist: {"
                "linear: {x: 0.0, y: 0.0, z: 0.0}, "
                "angular: {x: 0.0, y: 0.0, z: 0.0}"
                "}"
                "}}"
            )

            cmd = ["ros2", "service", "call", "/set_entity_state", "gazebo_msgs/srv/SetEntityState", req]

            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if "success: True" in out.stdout or "success=True" in out.stdout:
                    self.ros.get_logger().info(f"[{name}] Reset confirmed by CLI")
                    return True
                else:
                    self.ros.get_logger().warn(
                        f"[{name}] Reset not confirmed.\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                    )
            except subprocess.TimeoutExpired:
                self.ros.get_logger().warn(f"[{name}] CLI call timed out")

        self.ros.get_logger().error(f"[{name}] Failed to reset after {max_attempts}")
        return False

    def _wait_for_pose_sync(
        self,
        goal_x: float,
        goal_y: float,
        robot_x: float,
        robot_y: float,
        timeout: float = 1.5,
        tol: float = 0.15,
    ) -> bool:
        """Block until self.ros.robot_pose and self.ros.goal_pose_gt actually
        reflect the freshly-teleported positions, or until `timeout` elapses.

        WHY THIS EXISTS:
        `/set_entity_state` succeeding only confirms Gazebo accepted the
        teleport request synchronously. It says nothing about whether
        `/model_states` has published a fresh message reflecting that new
        pose yet. Anything that reads pose immediately after a teleport
        (e.g. the random-obstacle spawner, which places obstacles "between"
        robot and goal) can otherwise see stale poses left over from the end
        of the PREVIOUS episode.

        IMPORTANT: this checks `goal_pose_gt` (sourced from `/model_states`,
        published unconditionally by Gazebo) rather than `goal_pose`
        (sourced from `/goal_marker/odom`, published only by the goal-mover
        node). The mover is intentionally kept disabled during this part of
        reset(), so `goal_pose` will NOT update here by design — that's
        expected, not a bug. Only call this while physics is UNPAUSED;
        `/model_states` does not refresh while the world is paused.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.exec.spin_once(timeout_sec=0.02)

            goal_ok = (
                self.ros.goal_pose_gt is not None
                and math.hypot(self.ros.goal_pose_gt[0] - goal_x, self.ros.goal_pose_gt[1] - goal_y) < tol
            )
            robot_ok = (
                self.ros.robot_pose is not None
                and math.hypot(self.ros.robot_pose[0] - robot_x, self.ros.robot_pose[1] - robot_y) < tol
            )

            if goal_ok and robot_ok:
                return True

        self.ros.get_logger().warn(
            "[RESET] Timed out waiting for goal_pose_gt/robot_pose to reflect the new "
            "teleport before spawning obstacles; obstacle placement may use a stale pose. "
            f"goal_pose_gt={self.ros.goal_pose_gt} target=({goal_x:.2f},{goal_y:.2f}) "
            f"robot_pose={self.ros.robot_pose} target=({robot_x:.2f},{robot_y:.2f})"
        )
        return False

    def _search_full_rotation(self) -> bool:
        if self.ros.robot_pose is None:
            self.ros.get_logger().warn("[SEARCH] No robot pose available, skipping search.")
            return False

        _, _, yaw0 = self.ros.robot_pose
        target_rotation = 4.0 * math.pi
        w = self.search_angular_speed

        self.ros.get_logger().info(f"[SEARCH] Starting search at {w:.2f} rad/s")

        accumulated = 0.0
        prev_yaw = yaw0

        while accumulated < target_rotation:
            self.ros.send_cmd(0.0, w)
            self._spin(self.dt)

            if self.ros.goal_state is not None and getattr(self.ros.goal_state, "visible", False):
                self.ros.get_logger().info("[SEARCH] Goal seen during search.")
                break

            if self.ros.robot_pose is not None:
                _, _, yaw = self.ros.robot_pose
                dyaw = _wrap(yaw - prev_yaw)
                accumulated += abs(dyaw)
                prev_yaw = yaw

        self.ros.send_cmd(0.0, 0.0)
        self._spin(0.1)

        self.ros.get_logger().info(f"[SEARCH] Finished search; rot={accumulated:.2f} rad")
        return True

    def _call_service_cli(self, service, srv_type, args="{}"):
        try:
            cmd = ["ros2", "service", "call", service, srv_type, args]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            if out.returncode != 0:
                self.ros.get_logger().warn(
                    f"Service call failed: {' '.join(cmd)}\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}"
                )
            return out.returncode == 0
        except subprocess.TimeoutExpired:
            self.ros.get_logger().warn(f"Service call timed out: {' '.join(cmd)}")
            return False

    def _hard_stop_robot(self, duration=0.5):
        t_end = time.time() + duration
        while time.time() < t_end:
            self.ros.send_cmd(0.0, 0.0)
            self._spin(0.05)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # STOP goal movement during reset phases
        self._set_goal_mover_enabled(False)

        self._t0 = None
        self._prev_phi = 0.0

        self._ep_robot_traj = []
        self._robot_start = None
        self._goal_start = None
        self._robot_last = None
        self._goal_last = None
        self._min_dist = None
        self.prev_dist = None

        self._prev_v_cmd = 0.0
        self._prev_w_cmd = 0.0
        self._shield_interventions = 0
        self._shield_interventions_danger = 0
        self._shield_interventions_caution = 0
        self._shield_interventions_critical = 0
        self._shield_active = False

        # FIX: reset all new per-episode state
        self._current_shield_level = "none"
        self._steps_in_shield = 0
        # Reset Lagrangian per-episode accumulators
        self._ep_shield_cost = 0.0
        self._ep_shield_cost_raw = 0.0
        self._ep_critical_steps = 0

        # Choose ground-truth vs perception geometry for THIS episode (blend).
        self._select_geometry_mode_for_episode()
        self._deadlock_counter = 0
        self._forced_escapes = 0
        self._prev_v_policy = 0.0
        self._prev_w_policy = 0.0
        self._shield_penalty_warned = {
            "critical": False,
            "danger":   False,
            "caution":  False,
        }

        # Reset episode-level metric accumulators
        self._ep_min_obstacle_depth = float("inf")
        self._ep_min_ultrasonic_distance = float("inf")
        self._ep_jerk_v_list = []
        self._ep_jerk_w_list = []
        self._ep_ttc_list = []

        for s in self.obs_slots:
            s["valid"] = False
            s["wx"] = 0.0
            s["wy"] = 0.0
            s["vx"] = 0.0
            s["vy"] = 0.0
            s["last_seen"] = 0.0

        self._call_service_cli("/pause_physics", "std_srvs/srv/Empty", "{}")
        self._hard_stop_robot(duration=0.5)

        self._reset_entity_with_retry("my_robot", 0.0, 0.0, 0.3, 0.7854)

        radius = 7.0
        angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        gx = radius * math.cos(angle)
        gy = radius * math.sin(angle)
        self._reset_entity_with_retry("goal_marker", gx, gy, 0.75, 0.0)

        # FIX: unpause physics BEFORE waiting for / spawning obstacles, not
        # after. /model_states is published by gazebo_ros_state, which does
        # NOT publish fresh data while the world is paused — so polling for
        # an updated pose (or expecting the obstacle spawner to see one)
        # while physics is still paused can never succeed, regardless of
        # which topic anything reads from. This was the actual root cause:
        # robot_pose was stale too, and robot_pose was never gated by the
        # goal mover at all — it's pure /model_states, same as goal_pose_gt.
        self._call_service_cli("/unpause_physics", "std_srvs/srv/Empty", "{}")

        # Don't trust a fixed sleep: poll until robot_pose/goal_pose_gt
        # actually reflect the fresh teleport (within tolerance), or time
        # out. Deliberately checks goal_pose_gt (/model_states), not
        # goal_pose (/goal_marker/odom) — the goal mover stays disabled
        # through this whole block on purpose (see below), so goal_pose
        # will not update here.
        self._wait_for_pose_sync(gx, gy, 0.0, 0.0, timeout=1.5, tol=0.15)

        subprocess.run(
            ["ros2", "service", "call", "/reset_random_obstacles", "std_srvs/srv/Trigger", "{}"],
            capture_output=True,
            text=True,
        )
        self.ros.get_logger().info("[ENV] Random obstacles reset/teleported")

        self._spin(0.3)

        self._hard_stop_robot(duration=1.5)
        self.ros.obstacles_json = None

        if self.enable_search_on_reset:
            _ = self._search_full_rotation()

        now = time.time()
        self._t0 = now

        # START goal movement now that episode is fully set up. Kept here
        # (NOT moved earlier) on purpose: the goal mover drives its own
        # autonomous motion once enabled, so enabling it before obstacles
        # are placed would let it walk the goal away from the position
        # obstacles were just sampled around.
        self._set_goal_mover_enabled(True)

        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_start = (rx, ry)
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))

        if self.ros.goal_pose_gt is not None:
            gx0, gy0 = self.ros.goal_pose_gt
            self._goal_start = (gx0, gy0)
            self._goal_last = (gx0, gy0)
        elif self.ros.goal_pose is not None:
            gx0, gy0 = self.ros.goal_pose
            self._goal_start = (gx0, gy0)
            self._goal_last = (gx0, gy0)

        obs = self._obs(default=False)
        self._min_dist = float(obs['fixed'][0]) * self.GOAL_DIST_MAX

        self.ros.get_logger().info(
            f"[RESET] gamma={self.gamma:.3f} step_penalty={self.step_penalty:.3f} ttc_threshold={self.ttc_threshold:.2f} "
            f"smooth_alpha={self.smooth_alpha:.2f}"
        )
        return obs, {}
    def _is_goal_reached(self, dist: float, bearing: float, obs_quintuplets: list) -> bool:
        """
        Goal is reached if:
        1. Distance is within success radius
        2. Robot is facing the goal (bearing is small)
        3. No obstacle blocks line-of-sight to goal
        """
        # 1. Distance check
        if dist > self.success_radius:
            return False
        
        # 2. Orientation check: robot must face goal within threshold
        bearing_threshold = math.pi / 6  # 30 degrees
        if abs(bearing) > bearing_threshold:
            return False
        
        # 3. Line-of-sight check: no obstacle between robot and goal
        # Goal is at distance `dist` and bearing `bearing` in robot frame
        goal_x = dist * math.cos(bearing)
        goal_y = dist * math.sin(bearing)
        
        for ox, oy, od, _, _ in obs_quintuplets:
            # Skip invalid obstacles (at max depth)
            if od >= self.max_obstacle_depth - 0.01:
                continue
            
            # Check if obstacle is between robot (0,0) and goal (goal_x, goal_y)
            # Using point-to-segment distance
            t = max(0.0, min(1.0, (ox * goal_x + oy * goal_y) / (dist * dist + 1e-6)))
            closest_x = t * goal_x
            closest_y = t * goal_y
            dist_to_line = math.hypot(ox - closest_x, oy - closest_y)
            
            # If obstacle is close to the line and between robot and goal
            obstacle_radius = 0.3  # assume obstacle has ~30cm radius
            if dist_to_line < obstacle_radius and od < dist:
                return False  # obstacle blocks line-of-sight
        
        return True


    def step(self, action):
        a = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        if not np.all(np.isfinite(a)):
            self.ros.get_logger().error(f"[STEP] Non-finite action: {a}, zeroing.")
            a = np.array([0.0, 0.0], dtype=np.float32)

        # Distillation target for the replay entry that will be created from
        # this transition.  This must be computed BEFORE executing the action,
        # because SB3 stores info together with the *pre-step* observation.
        bc_teacher_action, bc_teacher_valid = self._compute_bc_teacher_action()

        v_des = float(a[0]) * self.v_max
        w_des = float(a[1]) * self.w_max

        # FIX Design-Issue-7: smooth from _prev_v_policy (policy intent),
        # NOT from the shield-modified command, so recovery is immediate post-shield.
        v_cmd = (1.0 - self.smooth_alpha) * self._prev_v_policy + self.smooth_alpha * v_des
        w_cmd = (1.0 - self.smooth_alpha) * self._prev_w_policy + self.smooth_alpha * w_des

        # delta for action-change penalty uses policy-level smooth commands
        delta_v = abs(v_cmd - self._prev_v_policy)
        delta_w = abs(w_cmd - self._prev_w_policy)

        # Store policy intent BEFORE shield modifies anything
        v_policy = v_cmd
        w_policy = w_cmd
        self._prev_v_policy = v_cmd   # FIX Design-Issue-7: store intent, not shield output
        self._prev_w_policy = w_cmd

        pre_ultra_vec, _ = self._ultra_vector()
        pre_ultra_fl, pre_ultra_fr, pre_ultra_ls, pre_ultra_rs = map(float, pre_ultra_vec.tolist())
        pre_ultra_back = self._ultra_back()   # NEW: rear clearance pre-shield

        if self.use_ground_truth_geometry:
            self._update_obstacle_memory_gt()
        else:
            # Perception path: use Kalman node JSON for pre-shield obstacle check
            self._update_obstacle_memory_perception()
        pre_obs_quintuplets_raw = self._slots_local_quintuplets_sorted_by_threat()

        v_cmd, w_cmd, shield_level = self._apply_safety_shield(
            v_cmd, w_cmd,
            pre_ultra_fl, pre_ultra_fr, pre_ultra_ls, pre_ultra_rs,
            pre_obs_quintuplets_raw,
            ultra_back=pre_ultra_back,        # NEW
        )
        self._shield_active = (shield_level != "none")
        self._current_shield_level = shield_level

        # FIX Design-Issue-6: deadlock escape — if robot is near-stationary for too long,
        # force a rotation toward the side with more clearance.
        if abs(v_cmd) < 0.05 and abs(w_cmd) < 0.05:
            self._deadlock_counter += 1
        else:
            self._deadlock_counter = 0

        if self._deadlock_counter >= self.deadlock_escape_steps:
            # Pick rotation direction toward clearer side
            ultra_left  = min(pre_ultra_fl, pre_ultra_ls)
            ultra_right = min(pre_ultra_fr, pre_ultra_rs)
            escape_dir  = +1.0 if ultra_left >= ultra_right else -1.0
            # NEW: only nudge backward if the rear is clear; else pure rotation
            v_cmd = -0.10 if pre_ultra_back > self.shield_back_clear else 0.0
            w_cmd = escape_dir * self.w_max     # full rotation toward clear side
            self._deadlock_counter = 0
            self._forced_escapes += 1
            self.ros.get_logger().warn(
                f"[DEADLOCK] Forced escape #{self._forced_escapes} "
                f"dir={'left' if escape_dir > 0 else 'right'}"
            )

        # Update shield duration counter (for observation)
        if shield_level != "none":
            self._steps_in_shield += 1
        else:
            self._steps_in_shield = 0

        # NOTE: _prev_v_cmd kept for logging / backward-compat only — NOT used for smoothing
        self._prev_v_cmd = v_cmd
        self._prev_w_cmd = w_cmd

        self.ros.send_cmd(v_cmd, w_cmd)
        self._spin(self.dt)

        obs = self._obs()
        if not (np.all(np.isfinite(obs['fixed'])) and np.all(np.isfinite(obs['obstacles']))):
            self.ros.get_logger().error(f"[STEP] Non-finite obs, using last valid.")
            obs = self._last_obs_valid.copy()

        # Un-normalize for reward computation
        dist    = float(obs['fixed'][0]) * self.GOAL_DIST_MAX
        bearing = float(obs['fixed'][1]) * math.pi

        v_obs = float(obs['fixed'][4]) * self.v_max
        w_obs = float(obs['fixed'][5]) * self.w_max

        # Un-normalize obstacles
        obs_quintuplets = []
        for i in range(self.MAX_OBSTACLES):
            ox  = float(obs['obstacles'][i, 0]) * self.OBS_COORD_MAX
            oy  = float(obs['obstacles'][i, 1]) * self.OBS_COORD_MAX
            od  = float(obs['obstacles'][i, 2]) * self.max_obstacle_depth
            ovx = float(obs['obstacles'][i, 3]) * self.VEL_MAX
            ovy = float(obs['obstacles'][i, 4]) * self.VEL_MAX
            obs_quintuplets.append((ox, oy, od, ovx, ovy))

        min_obs_depth = float(min(o[2] for o in obs_quintuplets))

        ultra_fl = float(obs['fixed'][6]) * self.ultra_max_range
        ultra_fr = float(obs['fixed'][7]) * self.ultra_max_range
        ultra_ls = float(obs['fixed'][8]) * self.ultra_max_range
        ultra_rs = float(obs['fixed'][9]) * self.ultra_max_range
        ultra_back = float(obs['fixed'][10]) * self.ultra_back_max_range   # NEW (index shifted)
        min_ultra = float(min(ultra_fl, ultra_fr, ultra_ls, ultra_rs))

        # ── Episode-level metric accumulation ──────────────────────────────
        # Track minimum proximity values across all steps
        if min_obs_depth < self._ep_min_obstacle_depth:
            self._ep_min_obstacle_depth = min_obs_depth
        if min_ultra < self._ep_min_ultrasonic_distance:
            self._ep_min_ultrasonic_distance = min_ultra

        # Jerk = |change in commanded velocity| per step (policy-level, before shield)
        self._ep_jerk_v_list.append(delta_v)
        self._ep_jerk_w_list.append(delta_w)

        # Logging
        if self.ros.robot_pose is not None:
            rx, ry, _ = self.ros.robot_pose
            self._robot_last = (rx, ry)
            self._ep_robot_traj.append((rx, ry))

        if self.ros.goal_pose is not None:
            gx, gy = self.ros.goal_pose
            self._goal_last = (gx, gy)

        if self._min_dist is None or dist < self._min_dist:
            self._min_dist = dist

        reward    = 0.0
        terminated = False
        truncated  = False
        reason     = ""

        reward -= self.step_penalty

        # FIX Bug-3: TTC penalty uses v_cmd (EXECUTED velocity after shield),
        # NOT v_policy (intended velocity). Previously the policy was penalised
        # for intending to go fast even when the shield stopped it completely.
        ttc = self._compute_ttc(obs_quintuplets, max(0.0, v_cmd))
        # Accumulate TTC for episode mean (replace inf with 5.0)
        self._ep_ttc_list.append(min(ttc, 50.0) if math.isfinite(ttc) else 50.0)
        if v_cmd > 0.1 and ttc < self.ttc_threshold:
            reward -= float(np.exp(-ttc))

        if self.prev_dist is None:
            self.prev_dist = dist
        progress = self.prev_dist - dist
        reward += self.c_progress * progress
        self.prev_dist = dist

        # REPLACED: naive `c_bearing * cos(bearing)` — see detailed comment in
        # _compute_adaptive_heading_reward(). The old term was always +0.28/step
        # even when the robot was orbiting under shield-critical, which caused
        # timeout episodes to outscore goal episodes (Episode 2 vs Episode 3 bug).
        reward += self._compute_adaptive_heading_reward(
            bearing, obs_quintuplets, v_cmd, shield_level
        )

        reward -= self.c_action_change * (delta_v + delta_w)

        if min_ultra < self.ultra_proximity_zone:
            reward -= self.c_ultra_proximity * (self.ultra_proximity_zone - min_ultra)

        # ── Lagrangian per-step shield cost (replaces the old one-time penalty) ──
        # c_t is a per-step, severity-weighted cost. The policy is penalized
        #     reward -= lambda * c_t
        # on EVERY step it occupies a shield zone, so it gets a dense gradient to
        # leave the zone and to avoid entering it — unlike the old one-time penalty,
        # which had zero gradient on the *number* of activations.
        #
        # lambda is NOT hand-tuned: the PID-Lagrangian controller in the training
        # callback raises it when the average cost exceeds the budget and lowers it
        # once the budget is met, so the policy is never penalized more than needed
        # (this is what preserves the shortest-path objective).
        # c_t is severity-weighted. We CAP the accrued per-episode cost at
        # shield_cost_cap so that lambda * ep_shield_cost can never dominate
        # R_goal (the economics-inversion bug). The uncapped value is still
        # logged (ep_shield_cost_raw) for diagnostics.
        c_t_raw = self.shield_cost_weights.get(shield_level, 0.0)
        self._ep_shield_cost_raw += c_t_raw
        remaining = max(0.0, self.shield_cost_cap - self._ep_shield_cost)
        c_t = min(c_t_raw, remaining)        # capped contribution this step
        self._ep_shield_cost += c_t
        if shield_level == "critical":
            self._ep_critical_steps += 1
        reward -= self.shield_lambda * c_t

        if min_ultra < self.ultra_collision_radius or min_obs_depth < self.obstacle_collision_radius:
            reward += self.R_collision
            terminated = True
            reason = "collision"
            self._set_goal_mover_enabled(False)

        elif self._is_goal_reached(dist, bearing, obs_quintuplets):
            reward += self.R_goal
            terminated = True
            reason = "goal"
            self._set_goal_mover_enabled(False)

        elif (time.time() - self._t0) >= self.time_limit:
            truncated = True
            reward -= 10.0
            reason = "timeout"
            self._set_goal_mover_enabled(False)

        info = {
            "reason": reason,
            "robot_start": self._robot_start,
            "goal_start": self._goal_start,
            "robot_final": self._robot_last,
            "goal_final": self._goal_last,
            "min_dist": self._min_dist,
            "robot_traj": self._ep_robot_traj,
            "obstacles_local_sorted": obs_quintuplets[:3],
            "obstacles_world_slots": [
                {"valid": s["valid"], "wx": s["wx"], "wy": s["wy"], "vx": s["vx"], "vy": s["vy"], "last_seen": s["last_seen"]}
                for s in self.obs_slots
            ],
            "min_obstacle_depth": min_obs_depth,
            "ultra_distances": {
                "front_left": ultra_fl,
                "front_right": ultra_fr,
                "left_side": ultra_ls,
                "right_side": ultra_rs,
            },
            "min_ultrasonic_distance": min_ultra,
            "ttc": float(ttc),
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd),
            "v_policy": float(v_policy),
            "w_policy": float(w_policy),
            "shield_level": shield_level,
            "shield_interventions_ep": self._shield_interventions,
            "shield_interventions_danger_ep": self._shield_interventions_danger,
            "shield_interventions_caution_ep": self._shield_interventions_caution,
            "shield_interventions_critical_ep": self._shield_interventions_critical,
            # ── Lagrangian constraint signals ──
            "ep_shield_cost": float(self._ep_shield_cost),     # J_c: sum of c_t this episode (CAPPED)
            "ep_shield_cost_raw": float(self._ep_shield_cost_raw),  # uncapped, diagnostics
            "use_ground_truth_geometry": bool(self.use_ground_truth_geometry),
            "ep_critical_steps": int(self._ep_critical_steps), # raw critical-step count (KPI)
            "shield_lambda": float(self.shield_lambda),        # current multiplier
            "shield_mode": self.shield_mode,
            "bc_teacher_valid": bool(bc_teacher_valid),
            "bc_teacher_action": (
                bc_teacher_action.astype(np.float32) if bc_teacher_valid and bc_teacher_action is not None
                else np.zeros(self.action_space.shape, dtype=np.float32)
            ),
            "steps_in_shield": self._steps_in_shield,        # consecutive shield steps
            "forced_escapes_ep": self._forced_escapes,       # deadlock escapes this episode
            "deadlock_counter": self._deadlock_counter,       # current deadlock count
            # Episode-level aggregate metrics (valid when episode ends)
            "ep_min_obstacle_depth": self._ep_min_obstacle_depth,
            "ep_min_ultrasonic_distance": self._ep_min_ultrasonic_distance,
            "ep_mean_jerk_v": float(np.mean(self._ep_jerk_v_list)) if self._ep_jerk_v_list else 0.0,
            "ep_mean_jerk_w": float(np.mean(self._ep_jerk_w_list)) if self._ep_jerk_w_list else 0.0,
            "ep_mean_ttc": float(np.mean(self._ep_ttc_list)) if self._ep_ttc_list else 5.0,
            "v_obs": float(v_obs),
            "w_obs": float(w_obs),
            "gamma_shaping": float(self.gamma),
            "step_penalty": float(self.step_penalty),
            "heading_reward": float(self._compute_adaptive_heading_reward(   # diagnostic
                bearing, obs_quintuplets, v_cmd, shield_level
            )),
        }

        step_robot_pose = self.ros.robot_pose
        step_goal_pose = self.ros.goal_pose
        step_gt_obstacles = getattr(self.ros, "gt_obstacles_world", None)

        seen_local = self._parse_obstacles_local()
        seen_world = []
        if step_robot_pose is not None:
            rx, ry, ryaw = step_robot_pose
            for (lx, ly, d) in seen_local:
                wx, wy = _world_from_local(rx, ry, ryaw, lx, ly)
                seen_world.append((float(wx), float(wy), float(d)))

        info["step_robot_pose"] = step_robot_pose
        info["step_goal_pose"] = step_goal_pose
        info["step_goal_pose_gt"] = self.ros.goal_pose_gt
        info["step_goal_vel_gt"] = self.ros.goal_vel_gt
        info["step_gt_obstacles_world"] = step_gt_obstacles
        info["step_gt_obstacles_vel"] = self.ros.gt_obstacles_vel
        info["step_seen_obstacles_local"] = seen_local
        info["step_seen_obstacles_world"] = seen_world

        self.ros.get_logger().info(
            f"[STEP] dist={dist:.2f} bearing={bearing:.2f} v={v_cmd:.2f} w={w_cmd:.2f} "
            f"min_obs={min_obs_depth:.2f} min_ultra={min_ultra:.2f} ttc={ttc:.2f} "
            f"shield={shield_level} heading_r={info['heading_reward']:.3f} "
            f"R={reward:.3f} term={terminated} trunc={truncated} reason={reason}"
        )

        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ros.send_cmd(0.0, 0.0)
        except Exception as e:
            print(f"Error sending stop cmd during close: {e}")

        try:
            if self.exec is not None:
                self.exec.shutdown()
        except Exception as e:
            print(f"Error shutting down executor: {e}")

        try:
            if self.ros is not None:
                self.ros.destroy_node()
        except Exception as e:
            print(f"Error destroying node: {e}")