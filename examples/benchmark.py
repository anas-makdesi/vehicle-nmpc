"""Timing benchmark of the controllers, in closed loop on the dry-road lap.

    python examples/benchmark.py            # writes results/timing.json and a figure

For each controller we record the solver time of every control step over a full lap:
median, 99th percentile, worst case, and the share of steps over the 20 ms (50 Hz) and
10 ms (100 Hz) budgets. Also:

* horizon sweep: the same 2 s horizon with 20, 40 and 80 shooting intervals, and a
  100 Hz variant (first interval 10 ms);
* real-time iteration vs SQP iterated to convergence (time, iterations, lap result);
* SQP from a cold start vs warm start at the same states;
* IPOPT (CasADi) on the same problem, warm-started along consecutive states, and the
  difference between its first input and the acados SQP one;
* Python call overhead: wall time of ``NMPC.step`` vs acados' internal time.

Timings vary by machine (and by about +-30 % between runs on a shared machine): compare
ratios, and re-run on the target hardware for absolute numbers.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from vnmpc.ltv_mpc import LTVMPC  # noqa: E402
from vnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from vnmpc.params import Vehicle  # noqa: E402
from vnmpc.reference import IpoptReference  # noqa: E402
from vnmpc.sim import run_lap  # noqa: E402
from vnmpc.track import Track  # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "docs" / "figures"


def stats(ms, extra=None):
    ms = np.asarray(ms, float)
    d = dict(steps=int(ms.size), median_ms=round(float(np.median(ms)), 3),
             p99_ms=round(float(np.percentile(ms, 99)), 3), max_ms=round(float(ms.max()), 3),
             over_20ms_pct=round(100 * float(np.mean(ms > 20)), 2),
             over_10ms_pct=round(100 * float(np.mean(ms > 10)), 2))
    d.update(extra or {})
    return d


def main(fig: bool = True, quick: bool = False):
    OUT.mkdir(exist_ok=True)
    veh, tr = Vehicle(), Track.default()
    T = {}
    laps = {}

    def lap(key, ctrl, label):
        r = run_lap(ctrl, veh, tr, 1.0, label, t_max=20 if quick else 150)
        laps[key] = r
        return r

    # ---- acados RTI: horizon sweep (same 2 s horizon) and 100 Hz
    variants = {
        "rti_N20": NMPCConfig(n_fine=10, dt_tail=0.18, n_tail=10),
        "rti_N40": NMPCConfig(),
        "rti_N80": NMPCConfig(n_fine=10, dt_tail=1.8 / 70, n_tail=70),
        "rti_N50_100Hz": NMPCConfig(dt0=0.01, n_fine=20, dt_tail=0.06, n_tail=30),
    }
    for key, cfg in variants.items():
        c = NMPC(veh, tr, cfg)
        r = run_lap(c, veh, tr, 1.0, key, dt=cfg.dt0, t_max=20 if quick else 150)
        laps[key] = r
        T[key] = stats(r.solve_ms, dict(
            N=cfg.N, horizon_s=round(cfg.horizon, 2), rate_hz=round(1 / cfg.dt0),
            lap_time=None if not r.finished else round(r.lap_time, 2),
            max_offset_m=round(r.max_offset, 2)))
        if key == "rti_N40":  # Python call overhead around the C solver
            T[key]["python_step_median_ms"] = round(1e3 * float(np.median(c.wall_times)), 3)

    # ---- acados SQP to convergence
    sqp = NMPC(veh, tr, NMPCConfig(mode="sqp", max_iter=100))
    r = lap("sqp_N40", sqp, "sqp")
    T["sqp_N40"] = stats(r.solve_ms, dict(
        iterations_median=int(np.median(sqp.iters)), iterations_max=int(np.max(sqp.iters)),
        lap_time=None if not r.finished else round(r.lap_time, 2),
        max_offset_m=round(r.max_offset, 2)))

    # ---- linear MPC (OSQP)
    ltv = LTVMPC(veh, tr, use=0.65)
    r = lap("ltv_osqp", ltv, "ltv")
    T["ltv_osqp"] = stats(r.solve_ms, dict(
        python_step_median_ms=round(1e3 * float(np.median(ltv.wall_times)), 3),
        lap_time=None if not r.finished else round(r.lap_time, 2),
        max_offset_m=round(r.max_offset, 2), note="speed profile at 65% of grip"))

    # ---- cold vs warm SQP and IPOPT, on windows of consecutive closed-loop states
    X = laps["rti_N40"].log.arrays()["x"]
    starts = [400, 1300, 2200] if not quick else [100]
    win = 15 if not quick else 5
    cold = NMPC(veh, tr, NMPCConfig(mode="sqp", max_iter=100), build=False)
    warm = NMPC(veh, tr, NMPCConfig(mode="sqp", max_iter=100), build=False)
    ip = IpoptReference(veh, tr)
    cold_ms, cold_it, warm_ms, warm_it, ip_ms, ip_it, du = [], [], [], [], [], [], []
    for s0 in starts:
        warm.initialized = False
        Xg = Ug = None
        for k in range(s0, s0 + win):
            cold.initialized = False
            cold.solver.reset()
            cold.step(X[k])
            cold_ms.append(cold.solve_times[-1] * 1e3); cold_it.append(cold.iters[-1])
            u_w = warm.step(X[k])
            if k > s0:
                warm_ms.append(warm.solve_times[-1] * 1e3); warm_it.append(warm.iters[-1])
            Xw, Uw = warm.prediction()
            if Xg is None:  # first state of a window: start IPOPT from acados' solution
                Xg, Ug = Xw, Uw
            u_i, _, Xg, Ug = ip.solve(X[k], Xg, Ug)
            if k > s0:
                ip_ms.append(ip.solve_times[-1] * 1e3); ip_it.append(ip.iters[-1])
            du.append(float(np.max(np.abs(u_i - u_w) / np.array([veh.delta_rate_max, 9.0]))))
    T["sqp_cold"] = stats(cold_ms, dict(iterations_median=int(np.median(cold_it))))
    T["sqp_warm"] = stats(warm_ms, dict(iterations_median=int(np.median(warm_it))))
    T["ipopt_warm"] = stats(ip_ms, dict(
        iterations_median=int(np.median(ip_it)),
        max_first_input_diff_vs_acados=float(f"{max(du):.1e}")))

    (OUT / "timing.json").write_text(json.dumps(T, indent=2))
    for k, v in T.items():
        print(k, v)
    if fig:
        figure(T, laps)
    return T


def figure(T, laps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vnmpc.plotting import INK2, MUTED, SERIES, style

    style()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = [
        ("acados RTI, N = 20", "rti_N20", 0),
        ("acados RTI, N = 40 (default)", "rti_N40", 0),
        ("acados RTI, N = 80", "rti_N80", 0),
        ("acados RTI, N = 50, 100 Hz", "rti_N50_100Hz", 0),
        ("Linear MPC, OSQP (QP only)", "ltv_osqp", 1),
        ("acados SQP iterated to convergence", "sqp_N40", 2),
        ("IPOPT (CasADi), warm start, 42 states", "ipopt_warm", 3),
    ]
    ros = ROOT / "results" / "ros2_lap_dry.json"
    if ros.exists():  # state-to-command latency of the C++ node in the ROS 2 loop
        R = json.loads(ros.read_text())
        T = dict(T, ros2_latency=dict(median_ms=R["latency_ms_median"],
                                      p99_ms=R["latency_ms_p99"], max_ms=R["latency_ms_max"]))
        rows.insert(2, ("ROS 2 C++ node, state -> command", "ros2_latency", 0))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, key, c) in enumerate(rows[::-1]):
        d = T[key]
        ax.plot([d["median_ms"], d["p99_ms"]], [i, i], color=SERIES[c], lw=3,
                solid_capstyle="round")
        ax.plot([d["p99_ms"], d["max_ms"]], [i, i], color=SERIES[c], lw=1, alpha=0.6)
        ax.plot(d["median_ms"], i, "o", ms=8, color=SERIES[c], mec="white", mew=1.5)
        med = d["median_ms"]
        ax.text(d["max_ms"] * 1.15, i, f"{med:.2g} ms" if med < 10 else f"{med:.0f} ms",
                va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xscale("log")
    for b, txt, dy in ((10, "100 Hz\nbudget", 0), (20, "50 Hz\nbudget", 0)):
        ax.axvline(b, color=MUTED, lw=1, ls="--")
        ax.text(b * 1.05, -0.9, txt, color=INK2, fontsize=8.5, va="bottom")
    ax.set_ylim(-1.0, len(rows) - 0.5)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("time per control step (ms), log scale: dot = median, "
                  "thick = up to p99, thin = worst case")
    ax.set_title("Solve time per control step over a full lap")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.2, 2000)
    fig.tight_layout()
    fig.savefig(FIG / "timing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--fig-only" in sys.argv:
        figure(json.loads((OUT / "timing.json").read_text()), None)
    else:
        main(fig="--no-fig" not in sys.argv, quick="--quick" in sys.argv)
