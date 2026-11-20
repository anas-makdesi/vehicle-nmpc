"""Launch test: simulator and NMPC nodes drive the car in real time for 20 s.

Checks that the car makes progress, stays inside the corridor, every solve succeeds, and
the controller answers well within the 20 ms control period.
"""

import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vehicle_nmpc_msgs.msg import ControllerStats, FrenetState

RUN_S = 20.0


@pytest.mark.launch_test
def generate_test_description():
    share = get_package_share_directory("vehicle_nmpc_ros")
    sim = launch_ros.actions.Node(
        package="vehicle_nmpc_ros",
        executable="sim_node",
        parameters=[{"track_csv": os.path.join(share, "config", "track.csv"), "mu": 1.0}],
    )
    nmpc = launch_ros.actions.Node(package="vehicle_nmpc_ros", executable="nmpc_node")
    return launch.LaunchDescription([sim, nmpc, launch_testing.actions.ReadyToTest()])


class TestClosedLoop(unittest.TestCase):
    def test_drives_in_real_time(self):
        rclpy.init()
        node = rclpy.create_node("closed_loop_probe")
        states, stats = [], []
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(FrenetState, "/vehicle/state", states.append, qos)
        node.create_subscription(ControllerStats, "/nmpc/stats", stats.append, 10)
        t_end = time.time() + RUN_S
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()

        self.assertGreater(len(states), 0.8 * RUN_S * 50)
        self.assertGreater(len(stats), 0.8 * RUN_S * 50)
        progress = states[-1].s - states[0].s
        self.assertGreater(progress, 250.0)  # about 1/5 of the lap in 20 s
        self.assertLess(max(abs(m.n) for m in states), 2.5)
        self.assertTrue(all(m.status == 0 for m in stats[1:]))
        lat = sorted(m.latency_ms for m in stats[1:])
        self.assertLess(lat[len(lat) // 2], 10.0)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
