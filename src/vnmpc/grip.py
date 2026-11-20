"""Certified tyre grip from data, with ddma.

The NMPC needs the road friction ``mu``. Planning with a value that is too high is
dangerous (the car asks for grip the road cannot give); a value that is too low only
costs lap time. This module turns measured tyre data into a friction value that is
*guaranteed* to be available.

Data: pairs (slip angle |alpha|, normalised lateral force |Fy| / Fz) for each axle,
e.g. from a short calibration manoeuvre and later from normal driving. Assumptions:

* the normalised lateral force is a function f(alpha) of the slip angle (odd, so we fold
  the data onto alpha >= 0), with bounded slope ``g_lo <= f' <= g_hi`` (physical prior:
  no tyre is stiffer than ``g_hi``; the force cannot drop faster than ``g_lo`` past the
  peak);
* each measurement is within ``[f - e_lo, f + e_hi]`` (sensor noise, and the force
  reduction while a little longitudinal force is transmitted; only samples with small
  longitudinal acceleration are used).

ddma's :class:`ddma.GrowthMap` turns this into lower and upper bounds on f on a fine
partition of the slip axis that hold for *every* function consistent with the data and
the assumptions. The certified friction is the largest lower bound: some slip angle
provably gives at least that much lateral force. The NMPC plans with it (tyre model and
friction-ellipse constraint), so it never asks for more grip than the data certify.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import plant_forces, plant_step
from .params import G, Vehicle


@dataclass
class GripCertificate:
    veh: Vehicle
    alpha_max: float = 0.3  # slip range covered by the model (rad)
    cells: int = 300
    mu_prior: float = 0.3  # friction assumed before any data (e.g. a "packed snow" floor)
    mu_bounds: tuple = (0.1, 1.2)  # range of plausible friction, for the slope prior
    noise: float = 0.02  # sensor noise bound on |Fy| / Fz
    ax_gate: float = 0.3  # use samples with |ax| below this (m/s^2)
    derate_margin: float = 0.03  # force reduction allowed by the gate, in |Fy| / Fz
    model: object = field(init=False, repr=False)
    n_used: int = field(init=False, default=0)

    def __post_init__(self):
        from ddma import GrowthMap, RectangularPartition

        t = self.veh.tyre
        a = np.linspace(0.0, self.alpha_max, 3001)
        shape_slope = t.D * t.C * t.B * np.cos(t.C * np.arctan(t.B * a)) / (1 + (t.B * a) ** 2)
        mu_lo, mu_hi = self.mu_bounds
        g_hi = mu_hi * shape_slope.max()
        g_lo = min(0.0, mu_hi * shape_slope.min())
        self.g_lo, self.g_hi = float(g_lo), float(g_hi)
        part = RectangularPartition([np.linspace(0.0, self.alpha_max, self.cells + 1)])
        self.model = GrowthMap(part, g_lo=[self.g_lo], g_hi=[self.g_hi],
                               d_lo=-(self.noise + self.derate_margin), d_hi=self.noise)

    # ----------------------------------------------------------------- data
    def add(self, alpha, force, ax=None):
        """Add samples: slip angles (rad), normalised lateral forces |Fy| / Fz, and
        optionally the longitudinal acceleration for the gate. Returns the number used."""
        alpha = np.abs(np.atleast_1d(np.asarray(alpha, float)))
        force = np.abs(np.atleast_1d(np.asarray(force, float)))
        keep = alpha <= self.alpha_max
        if ax is not None:
            keep &= np.abs(np.atleast_1d(ax)) <= self.ax_gate
        if not keep.any():
            return 0
        self.model.update(alpha[keep, None], force[keep])
        self.n_used += int(keep.sum())
        return int(keep.sum())

    # ------------------------------------------------------------ certificate
    def bounds(self):
        """(cell centres, certified lower bound on f, upper bound on f)."""
        lo, hi = self.model.f_bounds()
        e = np.linspace(0.0, self.alpha_max, self.cells + 1)
        return 0.5 * (e[1:] + e[:-1]), np.asarray(lo).ravel(), np.asarray(hi).ravel()

    @property
    def mu_certified(self) -> float:
        """Largest friction the data guarantee (never below the prior)."""
        _, lo, _ = self.bounds()
        lo = lo[np.isfinite(lo)]
        best = float(lo.max()) / self.veh.tyre.D if lo.size else -np.inf
        return max(self.mu_prior, best)


def calibration_run(veh: Vehicle, track, mu_true: float, speed: float = 12.0,
                    duration: float = 12.0, dt: float = 0.02, noise: float = 0.02,
                    seed: int = 0):
    """Ramp-steer manoeuvre at constant speed on an open pad (ISO 4138 style).

    The steering angle ramps up until either axle passes ~1.5x its peak slip, while a
    speed loop holds ``speed``. Returns the samples (alpha, |Fy|/Fz, ax) of both axles,
    with uniform sensor noise of amplitude ``noise`` on the force.
    """
    rng = np.random.default_rng(seed)

    class _Flat:  # open pad: straight reference line, no curvature
        length = 1e9

        @staticmethod
        def kappa_at(s):
            return 0.0

    x = np.array([0.0, 0.0, 0.0, speed, 0.0, 0.0, 0.0])
    A, F, AX = [], [], []
    rate = 0.15 / duration * 2  # rad/s of road-wheel angle
    a_stop = 1.5 * veh.tyre.peak_slip
    for k in range(int(duration / dt)):
        u = np.array([rate, float(np.clip(1.0 * (speed - x[3]), -2.0, 2.0))])
        Fxf, Fxr, Fyf, Fyr, af, ar = plant_forces(x, u, veh, mu_true)
        A += [af, ar]
        e = rng.uniform(-noise, noise, 2)
        F += [Fyf / veh.Fzf + e[0], Fyr / veh.Fzr + e[1]]
        AX += [u[1], u[1]]
        if max(abs(af), abs(ar)) > a_stop:
            break
        x = plant_step(x, u, dt, veh, mu_true, _Flat)
    return np.array(A), np.array(F), np.array(AX)


__all__ = ["GripCertificate", "calibration_run", "G"]
