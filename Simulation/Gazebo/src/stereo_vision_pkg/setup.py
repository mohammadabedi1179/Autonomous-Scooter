from setuptools import setup, find_packages

package_name = 'stereo_vision_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Register with ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        ('share/' + package_name + '/launch', ['launch/stereo_pipeline.launch.py']),
        # Install config files
        ('share/' + package_name + '/config', ['config/pipeline_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Stereo vision pipeline for Raspberry Pi + IMX219-83',
    license='MIT',
    entry_points={
        'console_scripts': [
            # ros2 run stereo_vision_pkg <name>
            'pi_camera_receiver_node = stereo_vision_pkg.pi_camera_receiver_node:main',
            'stereo_disparity_node   = stereo_vision_pkg.stereo_disparity_node:main',
            'stereo_disp_4panel_viewer   = stereo_vision_pkg.stereo_disp_4panel_viewer:main',
            'yolo_stereo_detector    = stereo_vision_pkg.yolo_stereo_detector_disparity_light:main',
            'stereo_depth_kalman     = stereo_vision_pkg.stereo_box_depth_kalman_synced:main',
            'goal_marker_node	     = stereo_vision_pkg.goal_marker_apriltag_depth_node:main',
        ],
    },
)
