#!/usr/bin/env python3
"""
Standalone observation-age probe.

Subscribes to the goal_marker_state topic and, using the SIM clock, reports how
old each message is at the moment an independent observer receives it. This
isolates *topic + clock* freshness from any extra lag your RL env / policy node
adds on top of it.

Run it (workspace sourced, sim running, robot ideally moving):

    python3 obs_age_probe.py --ros-args -p use_sim_time:=true

What to look for:
  - mean age steady around the per-frame latency (~50-100 ms)  -> topic is fresh
  - occasional SPIKE lines up near ~300-350 ms                 -> sim-time stutter
        (these should line up with the gaps you saw in `ros2 topic hz /clock`)
If THIS probe is clean but you still feel lag, the lag is being added by the
consumer (env / policy loop), not by perception. Use the in-env snippet for that.
"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter

from detectors_msgs.msg import GoalMarkerState

TOPIC = "/follower_robot/depth_cam/goal_marker_state"
WINDOW = 10          # messages per summary line (~1 s at 10 Hz)
SPIKE_MS = 150.0     # flag any single message older than this


class ObsAgeProbe(Node):
    def __init__(self):
        super().__init__("obs_age_probe")
        # Belt-and-suspenders: force sim time even if the CLI flag is forgotten.
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.create_subscription(GoalMarkerState, TOPIC, self.cb, 10)
        self._reset_window()
        self.get_logger().info(f"obs_age_probe listening on {TOPIC}")

    def _reset_window(self):
        self._n = 0
        self._sum = 0.0
        self._max = -1e9
        self._min = 1e9

    def cb(self, msg: GoalMarkerState):
        now = self.get_clock().now()
        age_ms = (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9 * 1e3

        self._n += 1
        self._sum += age_ms
        self._max = max(self._max, age_ms)
        self._min = min(self._min, age_ms)

        if abs(age_ms) > SPIKE_MS:
            self.get_logger().warn(
                f"SPIKE: obs age = {age_ms:+.0f} ms (visible={msg.visible})"
            )

        if self._n >= WINDOW:
            self.get_logger().info(
                f"obs_age over last {self._n}: "
                f"mean={self._sum / self._n:+.0f}ms  "
                f"min={self._min:+.0f}ms  max={self._max:+.0f}ms"
            )
            self._reset_window()


def main():
    rclpy.init()
    node = ObsAgeProbe()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
