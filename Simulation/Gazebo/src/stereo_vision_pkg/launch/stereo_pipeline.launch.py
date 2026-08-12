#!/usr/bin/env python3
"""
stereo_pipeline.launch.py
=========================
Launches the full stereo vision pipeline:
  1. pi_camera_receiver_node   — receives TCP frames from the Pi
  2. stereo_disparity_node     — rectifies + computes disparity
  3. yolo_stereo_detector      — YOLO detection on left frame
  4. stereo_depth_kalman       — fuses detections + disparity → depth JSON

Usage
-----
  ros2 launch stereo_vision_pkg stereo_pipeline.launch.py

Optional overrides
------------------
  ros2 launch stereo_vision_pkg stereo_pipeline.launch.py \
      calib_json:=/absolute/path/to/stereo_calibration_params.json \
      port:=8000
"""

import os
from ament_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = get_package_share_directory('stereo_vision_pkg')
    default_params = os.path.join(pkg_share, 'config', 'pipeline_params.yaml')

    # ── Launch arguments (override on command line) ───────────────────────
    calib_arg = DeclareLaunchArgument(
        'calib_json',
        default_value='/home/user/stereo_calibration_params.json',
        description='Absolute path to stereo_calibration_params.json'
    )
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='8000',
        description='TCP port that the Pi streams to'
    )

    calib_json = LaunchConfiguration('calib_json')
    port       = LaunchConfiguration('port')

    # ── Node 1: TCP receiver ──────────────────────────────────────────────
    receiver_node = Node(
        package='stereo_vision_pkg',
        executable='pi_camera_receiver_node',
        name='pi_camera_receiver',
        namespace='follower_robot',
        parameters=[
            default_params,
            {'port': port},
        ],
        output='screen',
    )

    # ── Node 2: Stereo disparity ──────────────────────────────────────────
    disparity_node = Node(
        package='stereo_vision_pkg',
        executable='stereo_disparity_node',
        name='stereo_disparity_node',
        namespace='follower_robot',
        parameters=[
            default_params,
            {'calib_json': calib_json},
        ],
        output='screen',
    )

    # ── Node 3: YOLO detector ─────────────────────────────────────────────
    yolo_node = Node(
        package='stereo_vision_pkg',
        executable='yolo_stereo_detector',
        name='yolo_stereo_detector',
        namespace='follower_robot',
        parameters=[default_params],
        output='screen',
    )

    # ── Node 4: Depth + Kalman ────────────────────────────────────────────
    depth_node = Node(
        package='stereo_vision_pkg',
        executable='stereo_depth_kalman',
        name='stereo_box_depth_kalman_synced',
        namespace='follower_robot',
        parameters=[
            default_params,
            {'calib_json': calib_json},
        ],
        output='screen',
    )

    return LaunchDescription([
        calib_arg,
        port_arg,
        receiver_node,
        disparity_node,
        yolo_node,
        depth_node,
    ])
