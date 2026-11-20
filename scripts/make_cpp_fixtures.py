"""Reference data for the C++ tests (ros2/vehicle_nmpc_ros/test/fixtures).

    python scripts/make_cpp_fixtures.py

plant_traj.csv  open-loop plant trajectory under a fixed input sequence (friction 0.8)
solver_seq.csv  closed-loop states and the inputs the Python NMPC (RTI) returned for them
The C++ plant and the C++ solver wrapper must reproduce these.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from vnmpc.model import plant_step  # noqa: E402
from vnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from vnmpc.params import Vehicle  # noqa: E402
from vnmpc.track import Track  # noqa: E402

FIX = ROOT / "ros2" / "vehicle_nmpc_ros" / "test" / "fixtures"


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    veh, tr = Vehicle(), Track.default()
    # plant: steering-rate sine and an accelerate / brake pattern
    x = np.array([0.0, 0.3, 0.02, 14.0, 0.0, 0.0, 0.0])
    rows = []
    for k in range(300):
        t = 0.02 * k
        u = np.array([0.4 * np.sin(1.3 * t), 2.0 if t < 2 else (-4.0 if t < 3.5 else 0.5)])
        xn = plant_step(x, u, 0.02, veh, 0.8, tr)
        rows.append(np.r_[x, u, xn])
        x = xn
    np.savetxt(FIX / "plant_traj.csv", np.array(rows), delimiter=",", fmt="%.17g",
               header="x0(7),u(2),x1(7); mu=0.8, dt=0.02, 10 RK4 substeps")
    # solver: closed loop from the default start
    c = NMPC(veh, tr, NMPCConfig())
    x = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0])
    rows = []
    for k in range(150):
        u = c.step(x)
        rows.append(np.r_[x, u])
        x = plant_step(x, u, 0.02, veh, 1.0, tr)
    np.savetxt(FIX / "solver_seq.csv", np.array(rows), delimiter=",", fmt="%.17g",
               header="x(7),u(2); RTI, friction 1.0, default config")
    print("fixtures written to", FIX)


if __name__ == "__main__":
    main()
