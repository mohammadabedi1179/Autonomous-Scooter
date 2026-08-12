#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def stamp_to_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class StereoTimeCheck(Node):
    def __init__(self):
        super().__init__("stereo_time_check")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.left_sub = Subscriber(
            self, Image,
            "/follower_robot/depth_cam/left/image_raw",
            qos_profile=qos,
        )
        self.right_sub = Subscriber(
            self, Image,
            "/follower_robot/depth_cam/right/image_raw",
            qos_profile=qos,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=20,
            slop=0.10,
        )
        self.sync.registerCallback(self.cb)

        self.n = 0
        self.max_dt = 0.0

    def cb(self, left, right):
        dt = abs(stamp_to_sec(left) - stamp_to_sec(right))
        self.max_dt = max(self.max_dt, dt)
        self.n += 1

        if self.n % 10 == 0:
            self.get_logger().info(
                f"left-right dt = {dt*1000:.2f} ms | max = {self.max_dt*1000:.2f} ms"
            )


def main():
    rclpy.init()
    node = StereoTimeCheck()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()