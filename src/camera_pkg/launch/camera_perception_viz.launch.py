#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # 1) Camera node
    camera_node = Node(
        package='camera_pkg',           # The package name
        executable='camera_node',       # The console_script name from setup.py
        name='camera_node',
        output='screen',
        # If you have parameters for the camera_node, you can pass them here:
        # parameters=[{'use_webcam': True, 'fps': 10}],
    )

    # 2) Perception node
    perception_node = Node(
        package='perception_pkg',
        executable='perception_node',
        name='perception_node',
        output='screen'
    )

    # 3) Visualization node
    visualization_node = Node(
        package='visualization_pkg',
        executable='visualization_node',
        name='visualization_node',
        output='screen'
    )

    # Return the LaunchDescription with all 3 nodes
    return LaunchDescription([
        camera_node,
        perception_node,
        visualization_node
    ])
