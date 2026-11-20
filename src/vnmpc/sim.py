"""Closed-loop simulation and lap metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import body_accel, plant_forces, plant_step
from .params import G, Vehicle
from .track import Track


@dataclass
class Log:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    u: list = field(default_factory=list)
    a: list = field(default_factory=list)  # (a_long, a_lat) from the tyre forces
    mu_planned: list = field(default_factory=list)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


@dataclass
class Result:
    name: str
    finished: bool
    lap_time: float  # s (nan if not finished)
    distance: float  # m covered
    max_offset: float  # max |n| (m)
    rms_offset: float  # m
    corridor_violation_time: float  # s spent with |n| > half width
    spun: bool  # |heading error| > 60 deg or vx < 2 m/s
    max_friction_use: float  # max |tyre-force acceleration| / (mu_true g)
    solve_ms: np.ndarray  # per-step solver time (ms)
    log: Log

    def summary(self) -> dict:
        st = self.solve_ms
        return dict(
            name=self.name, finished=self.finished,
            lap_time=None if not self.finished else round(self.lap_time, 2),
            distance=round(self.distance, 1), max_offset=round(self.max_offset, 2),
            rms_offset=round(self.rms_offset, 3),
            corridor_violation_time=round(self.corridor_violation_time, 2),
            spun=self.spun, max_friction_use=round(self.max_friction_use, 2),
            solve_ms_median=round(float(np.median(st)), 3) if st.size else None,
            solve_ms_p99=round(float(np.percentile(st, 99)), 3) if st.size else None,
            solve_ms_max=round(float(st.max()), 3) if st.size else None,
        )


def run_lap(controller, veh: Vehicle, track: Track, mu_true: float, name: str = "",
            x0=None, dt: float = 0.02, t_max: float = 150.0, laps: float = 1.0,
            on_step=None) -> Result:
    """Drive ``laps`` laps (or until the car spins / leaves the road by > 5 m).

    ``on_step(k, x, u)`` is called after each control step (e.g. to feed a learner)."""
    x = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0]) if x0 is None else np.asarray(x0, float)
    log = Log()
    s_goal = x[0] + laps * track.length
    spun = finished = False
    t = 0.0
    for k in range(int(t_max / dt)):
        u = controller.step(x)
        log.t.append(t); log.x.append(x.copy()); log.u.append(u.copy())
        log.a.append(body_accel(x, u, veh, mu_true))
        log.mu_planned.append(getattr(controller, "mu_limit", getattr(controller, "mu", np.nan)))
        if on_step is not None:
            on_step(k, x, u)
        x = plant_step(x, u, dt, veh, mu_true, track)
        t += dt
        if x[0] >= s_goal:
            finished = True
            # interpolate the crossing time
            s_prev = log.x[-1][0]
            t -= dt * (x[0] - s_goal) / max(x[0] - s_prev, 1e-9)
            break
        if abs(x[2]) > np.deg2rad(60) or x[3] < 2.0 or abs(x[1]) > track.half_width + 5:
            spun = abs(x[2]) > np.deg2rad(60) or x[3] < 2.0
            break
    L = log.arrays()
    X = L["x"]
    n = X[:, 1]
    over = np.abs(n) > track.half_width
    use = np.hypot(L["a"][:, 0], L["a"][:, 1]) / (mu_true * G)
    st = np.asarray(getattr(controller, "solve_times", []), float) * 1e3
    return Result(
        name=name, finished=finished, lap_time=t if finished else np.nan,
        distance=float(x[0] - X[0, 0]), max_offset=float(np.abs(n).max()),
        rms_offset=float(np.sqrt(np.mean(n ** 2))),
        corridor_violation_time=float(over.sum() * dt) + (0.0 if finished or spun else 0.0),
        spun=spun or (not finished and abs(x[1]) > track.half_width + 5),
        max_friction_use=float(use.max()), solve_ms=st, log=log,
    )


def axle_samples(x, u, veh: Vehicle, mu_true: float, rng, noise: float = 0.02):
    """Measured (|alpha|, |Fy|/Fz, ax) of both axles at one step (sensor noise added)."""
    Fxf, Fxr, Fyf, Fyr, af, ar = plant_forces(x, u, veh, mu_true)
    e = rng.uniform(-noise, noise, 2)
    return (np.array([af, ar]), np.array([Fyf / veh.Fzf + e[0], Fyr / veh.Fzr + e[1]]),
            np.array([u[1], u[1]]))
