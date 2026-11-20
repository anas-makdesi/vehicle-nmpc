#!/usr/bin/env python3
"""Record one lap of the running ROS 2 closed loop and print a summary (JSON).

ros2 launch vehicle_nmpc_ros closed_loop.launch.py &
python3 lap_probe.py --length 1321 [--out lap.json]
"""

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vehicle_nmpc_msgs.msg import ControllerStats, FrenetState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, required=True, help="track length (m)")
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("lap_probe")
    S, T = [], []
    qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(
        FrenetState,
        "/vehicle/state",
        lambda m: S.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, m.s, m.n, m.vx)),
        qos,
    )
    node.create_subscription(
        ControllerStats,
        "/nmpc/stats",
        lambda m: T.append((m.solve_time_ms, m.step_time_ms, m.latency_ms, m.status)),
        50,
    )
    t_end = time.time() + a.timeout
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if S and S[-1][1] - S[0][1] >= a.length:
            break
    node.destroy_node()
    rclpy.shutdown()
    S, T = np.array(S), np.array(T[1:])
    done = S[-1, 1] - S[0, 1] >= a.length
    q = lambda v, p: round(float(np.percentile(v, p)), 3)  # noqa: E731
    out = dict(
        finished=bool(done),
        lap_time_s=round(float(S[-1, 0] - S[0, 0]), 2) if done else None,
        distance_m=round(float(S[-1, 1] - S[0, 1]), 1),
        max_offset_m=round(float(np.abs(S[:, 2]).max()), 3),
        steps=int(len(T)),
        failed_solves=int((T[:, 3] != 0).sum()),
        solve_ms_median=q(T[:, 0], 50),
        solve_ms_p99=q(T[:, 0], 99),
        solve_ms_max=round(float(T[:, 0].max()), 3),
        step_ms_median=q(T[:, 1], 50),
        latency_ms_median=q(T[:, 2], 50),
        latency_ms_p99=q(T[:, 2], 99),
        latency_ms_max=round(float(T[:, 2].max()), 3),
    )
    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
