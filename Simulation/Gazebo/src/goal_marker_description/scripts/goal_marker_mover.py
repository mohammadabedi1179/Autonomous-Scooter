#!/usr/bin/env python3
"""
Goal Marker Mover Node

Moves the goal marker during training episodes to create a "moving goal"
scenario for curriculum learning.

The goal moves in a random-walk pattern:
- Picks a random direction, moves that way for a few seconds
- Changes direction periodically
- Stays within a configurable bounding area
- Bounces off boundaries

Control:
    /goal_mover/enable   (std_srvs/SetBool)  - Enable/disable movement
    /goal_mover/config   (std_msgs/String)    - JSON config: {"speed": 0.3}

The node publishes to /goal_marker/cmd_vel to move the goal entity.

IMPORTANT: Your goal_marker model.sdf must have a velocity/movement plugin
(e.g., libgazebo_ros_planar_move.so or libgazebo_ros_diff_drive.so) that
subscribes to cmd_vel. If it doesn't, you'll need to use /set_entity_state
instead (see the alternative approach at the bottom of this file).
"""

import math
import random
import time
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import SetBool
from std_msgs.msg import String


class GoalMarkerMover(Node):
    """
    Moves the goal marker in a random-walk pattern.
    
    Can be enabled/disabled by the curriculum training callback.
    When disabled, publishes zero velocity (goal stays still).
    """
    
    def __init__(self):
        super().__init__("goal_marker_mover")
        
        # Parameters
        self.declare_parameter("cmd_vel_topic", "/goal_marker/cmd_vel")
        self.declare_parameter("speed", 0.3)           # m/s
        self.declare_parameter("max_speed", 0.8)        # m/s cap
        self.declare_parameter("direction_change_sec", 4.0)  # seconds between direction changes
        self.declare_parameter("boundary_radius", 10.0)  # Stay within this radius from origin
        self.declare_parameter("boundary_margin", 1.0)   # Start turning when this close to boundary
        self.declare_parameter("update_rate", 10.0)      # Hz
        self.declare_parameter("enabled", False)         # Start disabled
        self.declare_parameter("use_cmd_vel", True)      # True=cmd_vel, False=set_entity_state
        
        self.cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic")
            .get_parameter_value().string_value
        )
        self.speed = (
            self.get_parameter("speed")
            .get_parameter_value().double_value
        )
        self.max_speed = (
            self.get_parameter("max_speed")
            .get_parameter_value().double_value
        )
        self.direction_change_sec = (
            self.get_parameter("direction_change_sec")
            .get_parameter_value().double_value
        )
        self.boundary_radius = (
            self.get_parameter("boundary_radius")
            .get_parameter_value().double_value
        )
        self.boundary_margin = (
            self.get_parameter("boundary_margin")
            .get_parameter_value().double_value
        )
        update_rate = (
            self.get_parameter("update_rate")
            .get_parameter_value().double_value
        )
        self.enabled = (
            self.get_parameter("enabled")
            .get_parameter_value().bool_value
        )
        self.use_cmd_vel = (
            self.get_parameter("use_cmd_vel")
            .get_parameter_value().bool_value
        )
        
        # State
        self.goal_pose = None          # (x, y) from model_states
        self.current_direction = random.uniform(0, 2 * math.pi)
        self.last_direction_change = time.time()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )
        
        # Subscribers
        self.create_subscription(
            ModelStates, "/model_states", self._model_states_cb, 10
        )
        self.create_subscription(
            String, "/goal_mover/config", self._config_cb, 10
        )
        
        # Services
        self.create_service(
            SetBool, "/goal_mover/enable", self._enable_cb
        )
        
        # Timer
        self.timer = self.create_timer(1.0 / update_rate, self._tick)
        
        self.get_logger().info(
            f"GoalMarkerMover initialized:\n"
            f"  Topic: {self.cmd_vel_topic}\n"
            f"  Speed: {self.speed} m/s\n"
            f"  Direction change: every {self.direction_change_sec}s\n"
            f"  Boundary: {self.boundary_radius}m radius\n"
            f"  Enabled: {self.enabled}\n"
            f"  Mode: {'cmd_vel' if self.use_cmd_vel else 'set_entity_state'}"
        )
    
    def _model_states_cb(self, msg: ModelStates):
        try:
            i = msg.name.index("goal_marker")
            p = msg.pose[i].position
            self.goal_pose = (float(p.x), float(p.y))
        except ValueError:
            pass
    
    def _config_cb(self, msg: String):
        """
        Receive configuration updates.
        JSON: {"speed": 0.3, "enabled": true, "direction_change_sec": 5.0}
        """
        try:
            config = json.loads(msg.data)
            if "speed" in config:
                self.speed = min(float(config["speed"]), self.max_speed)
            if "enabled" in config:
                self.enabled = bool(config["enabled"])
            if "direction_change_sec" in config:
                self.direction_change_sec = float(config["direction_change_sec"])
            
            self.get_logger().info(
                f"[GOAL_MOVER] Config updated: speed={self.speed}, "
                f"enabled={self.enabled}, dir_change={self.direction_change_sec}s"
            )
        except Exception as e:
            self.get_logger().error(f"[GOAL_MOVER] Invalid config: {e}")
    
    def _enable_cb(self, request, response):
        """Enable/disable goal movement."""
        self.enabled = request.data
        
        if not self.enabled:
            # Send stop command immediately
            self._send_stop()
        else:
            # Reset direction on enable
            self.current_direction = random.uniform(0, 2 * math.pi)
            self.last_direction_change = time.time()
        
        response.success = True
        response.message = f"Goal mover {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(f"[GOAL_MOVER] {response.message}")
        return response
    
    def _send_stop(self):
        """Send zero velocity."""
        tw = Twist()
        tw.linear.x = 0.0
        tw.linear.y = 0.0
        tw.angular.z = 0.0
        self.cmd_vel_pub.publish(tw)
    
    def _tick(self):
        """Main update loop."""
        if not self.enabled:
            return
        
        now = time.time()
        
        # Change direction periodically
        if now - self.last_direction_change > self.direction_change_sec:
            # Smooth direction change: ±60° from current
            delta = random.uniform(-math.pi / 3, math.pi / 3)
            self.current_direction += delta
            self.last_direction_change = now
        
        # Boundary enforcement: steer away from edges
        if self.goal_pose is not None:
            gx, gy = self.goal_pose
            dist_from_origin = math.hypot(gx, gy)
            
            if dist_from_origin > (self.boundary_radius - self.boundary_margin):
                # Steer back toward origin
                angle_to_origin = math.atan2(-gy, -gx)
                # Blend current direction toward origin
                blend = min(1.0, (dist_from_origin - self.boundary_radius + self.boundary_margin) / self.boundary_margin)
                self.current_direction = (
                    (1.0 - blend) * self.current_direction +
                    blend * angle_to_origin
                )
        
        # Compute velocity
        vx = self.speed * math.cos(self.current_direction)
        vy = self.speed * math.sin(self.current_direction)
        
        if self.use_cmd_vel:
            # Publish cmd_vel (assumes planar move plugin)
            tw = Twist()
            tw.linear.x = float(vx)
            tw.linear.y = float(vy)
            tw.angular.z = 0.0
            self.cmd_vel_pub.publish(tw)
        else:
            # Alternative: use set_entity_state for direct position control
            # This is more reliable but less smooth
            self._move_via_set_state(vx, vy)
    
    def _move_via_set_state(self, vx: float, vy: float):
        """
        Alternative movement method using /set_entity_state.
        Use this if your goal_marker SDF doesn't have a velocity plugin.
        Sets the entity's velocity directly in Gazebo.
        """
        if self.goal_pose is None:
            return
        
        import subprocess
        
        gx, gy = self.goal_pose
        cmd = [
            "ros2", "service", "call",
            "/set_entity_state",
            "gazebo_msgs/srv/SetEntityState",
            (
                "{state: {name: 'goal_marker', "
                "pose: {position: {x: " + str(gx) +
                ", y: " + str(gy) +
                ", z: 0.15}}, "
                "twist: {linear: {x: " + str(vx) +
                ", y: " + str(vy) +
                ", z: 0.0}}}}"
            ),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = GoalMarkerMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down GoalMarkerMover")
    finally:
        # Stop the goal before shutting down
        tw = Twist()
        node.cmd_vel_pub.publish(tw)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
