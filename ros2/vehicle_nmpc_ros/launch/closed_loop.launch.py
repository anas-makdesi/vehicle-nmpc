"""Simulator + NMPC in closed loop.

    ros2 launch vehicle_nmpc_ros closed_loop.launch.py mu:=0.6 mu_plan:=0.57

``mu`` is the true road friction in the simulator; ``mu_plan`` the friction the NMPC
plans with (e.g. the value certified by ``examples/grip_certificate.py``).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("vehicle_nmpc_ros")
    track = os.path.join(share, "config", "track.csv")
    mu = LaunchConfiguration("mu")
    mu_plan = LaunchConfiguration("mu_plan")
    return LaunchDescription(
        [
            DeclareLaunchArgument("mu", default_value="1.0", description="true road friction"),
            DeclareLaunchArgument(
                "mu_plan", default_value="1.0", description="friction the NMPC plans with"
            ),
            Node(
                package="vehicle_nmpc_ros",
                executable="sim_node",
                name="vehicle_sim",
                parameters=[{"track_csv": track, "mu": mu, "state_rate_hz": 50.0, "v0": 15.0}],
                output="screen",
            ),
            Node(
                package="vehicle_nmpc_ros",
                executable="nmpc_node",
                name="nmpc",
                parameters=[{"mu_model": mu_plan, "mu_limit": mu_plan}],
                output="screen",
            ),
        ]
    )
