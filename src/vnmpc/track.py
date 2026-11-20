"""Closed race track as a smooth centre line, with curvature and a speed profile.

The centre line is a periodic cubic spline through hand-placed points, resampled by arc
length. Everything the controllers need is a function of the arc length ``s``:
curvature ``kappa(s)``, the reference speed ``v_ref(s)`` and the corridor half width.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .params import G

DEFAULT_POINTS = np.array(
    [[0, 0], [200, 0], [320, 10], [360, 70], [330, 130], [250, 140], [200, 190],
     [225, 260], [180, 310], [60, 300], [-20, 240], [-30, 170], [0, 115], [-40, 60],
     [-60, 10]],
    dtype=float,
)


@dataclass
class Track:
    s: np.ndarray  # arc length samples (m), s[0] = 0, last sample < length
    xy: np.ndarray  # centre line points (K, 2)
    psi: np.ndarray  # heading of the centre line (rad, unwrapped)
    kappa: np.ndarray  # curvature (1/m), positive = left turn
    length: float
    half_width: float  # corridor half width for the vehicle's centre (m)

    @classmethod
    def default(cls, half_width: float = 2.5, ds: float = 0.5) -> "Track":
        return cls.from_points(DEFAULT_POINTS, half_width=half_width, ds=ds)

    @classmethod
    def from_points(cls, P, half_width: float = 2.5, ds: float = 0.5) -> "Track":
        P = np.vstack([P, P[:1]])
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        cs = CubicSpline(d, P, bc_type="periodic")
        # arc length of the spline parameter, on a fine grid
        t = np.linspace(0.0, d[-1], 200_001)
        dp = cs(t, 1)
        speed = np.hypot(dp[:, 0], dp[:, 1])
        arc = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))]
        length = float(arc[-1])
        n = int(round(length / ds))
        s = np.linspace(0.0, length, n, endpoint=False)
        ts = np.interp(s, arc, t)
        xy = cs(ts)
        d1, d2 = cs(ts, 1), cs(ts, 2)
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3
        psi = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
        return cls(s=s, xy=xy, psi=psi, kappa=kappa, length=length, half_width=half_width)

    # ----------------------------------------------------------------- lookups
    def _wrap(self, s):
        return np.mod(s, self.length)

    def kappa_at(self, s):
        return np.interp(self._wrap(s), self.s, self.kappa, period=self.length)

    def frenet_to_xy(self, s, n):
        """Cartesian position of the point at arc length s and lateral offset n (left > 0)."""
        s = self._wrap(np.asarray(s, float))
        x = np.interp(s, self.s, self.xy[:, 0], period=self.length)
        y = np.interp(s, self.s, self.xy[:, 1], period=self.length)
        c, si = np.cos(self.psi), np.sin(self.psi)
        cx = np.interp(s, self.s, c, period=self.length)
        sx = np.interp(s, self.s, si, period=self.length)
        h = np.hypot(cx, sx)
        return x - n * sx / h, y + n * cx / h

    def heading_at(self, s):
        s = self._wrap(np.asarray(s, float))
        c = np.interp(s, self.s, np.cos(self.psi), period=self.length)
        si = np.interp(s, self.s, np.sin(self.psi), period=self.length)
        return np.arctan2(si, c)

    # ----------------------------------------------------------- speed profile
    def speed_profile(self, mu: float = 1.0, v_max: float = 30.0, use: float = 0.85,
                      a_acc: float = 3.0) -> np.ndarray:
        """Classic forward/backward pass on a friction circle ("g-g diagram"): the fastest
        speed with total acceleration ``sqrt(a_long^2 + a_lat^2) <= use * mu * g``, with
        traction further capped at ``a_acc``. Returns v at ``self.s``."""
        a_tot = use * mu * G
        kap = np.maximum(np.abs(self.kappa), 1e-9)
        v = np.minimum(v_max, np.sqrt(a_tot / kap))
        ds = np.diff(np.r_[self.s, self.length])

        def a_long(vv, k):
            return np.sqrt(max(a_tot ** 2 - (vv ** 2 * k) ** 2, 0.0))

        K = len(v)
        for _ in range(2):  # two rounds so the closed loop wraps correctly
            for i in range(K):  # forward: traction
                j = i - 1
                a = min(a_acc, a_long(v[j], kap[j]))
                v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a * ds[j]))
            for i in range(K - 1, -1, -1):  # backward: braking
                j = (i + 1) % K
                v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_long(v[j], kap[j]) * ds[i]))
        return v
