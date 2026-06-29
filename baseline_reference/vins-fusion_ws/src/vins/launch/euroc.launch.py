"""ROS2 launch file for VINS-Fusion on EuRoC datasets."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Default config path
    pkg_share = get_package_share_directory('vins')
    default_config = os.path.join(
        pkg_share, 'config', 'euroc', 'euroc_stereo_imu_config.yaml')

    config_path_arg = DeclareLaunchArgument(
        'config_path', default_value=default_config,
        description='Path to config YAML file')

    vins_node = Node(
        package='vins',
        executable='vins_node',
        name='vins_estimator',
        output='screen',
        arguments=[LaunchConfiguration('config_path')],
        remappings=[
            ('imu_propagate',      '/vins_estimator/imu_propagate'),
            ('path',               '/vins_estimator/path'),
            ('odometry',           '/vins_estimator/odometry'),
            ('point_cloud',        '/vins_estimator/point_cloud'),
            ('margin_cloud',       '/vins_estimator/margin_cloud'),
            ('key_poses',          '/vins_estimator/key_poses'),
            ('camera_pose',        '/vins_estimator/camera_pose'),
            ('camera_pose_visual', '/vins_estimator/camera_pose_visual'),
            ('keyframe_pose',      '/vins_estimator/keyframe_pose'),
            ('keyframe_point',     '/vins_estimator/keyframe_point'),
            ('extrinsic',          '/vins_estimator/extrinsic'),
            ('image_track',        '/vins_estimator/image_track'),
        ],
    )

    return LaunchDescription([
        config_path_arg,
        vins_node,
    ])
