"""ROS2 launch file for VINS-Fusion RViz visualization."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('vins')
    rviz_config = os.path.join(pkg_share, 'config', 'vins_rviz_config.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rvizvisualisation',
        output='log',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([rviz_node])
