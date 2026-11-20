"""Linear time-varying MPC with OSQP: the classic industrial baseline.

Lateral and longitudinal control are decoupled, as in many production path-tracking
stacks:

* **Speed**: a proportional law on the track's speed profile with feedforward,
  ``ax = clip(a_ff(s) + k_v (v_ref(s) - vx))``.
* **Steering**: a linear MPC on the lateral error model with *linear* tyres (cornering
  stiffness), linearised along the predicted speed and curvature,
      z = [n, xi, vy, r, delta],   input delta_dot,
      n'  = vx xi + vy,             xi' = r - kappa vx,
      vy' = (Fyf + Fyr)/m - vx r,   r'  = (lf Fyf - lr Fyr)/Iz,
      Fyf = Cf (delta - (vy + lf r)/vx),  Fyr = -Cr (vy - lr r)/vx,
  discretised exactly (zero-order hold) on the same grid as the NMPC, with the same
  corridor (soft), steering limits and slip-angle limits (linear in z for the predicted
  speed). The QP has a fixed
  sparsity pattern; each step only its values change (``osqp.update``), and it is warm
  started from the previous solution.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from .params import Vehicle
from .track import Track

NZ = 5


class LTVMPC:
    def __init__(self, veh: Vehicle, track: Track, mu: float = 1.0, dt0: float = 0.02,
                 n_fine: int = 10, dt_tail: float = 0.06, n_tail: int = 30,
                 v_max: float = 30.0, use: float = 0.85, k_v: float = 1.0,
                 W=(2.0, 2.0, 1.0, 20.0), slack_w: float = 1e3, slip_limit: bool = True):
        import osqp

        self.veh, self.track, self.mu = veh, track, mu
        self.dts = np.r_[np.full(n_fine, dt0), np.full(n_tail, dt_tail)]
        self.N = N = len(self.dts)
        self.v_profile = track.speed_profile(mu=mu, v_max=v_max, use=use)
        self.k_v = k_v
        self.Cf = mu * veh.Fzf * veh.tyre.max_slope
        self.Cr = mu * veh.Fzr * veh.tyre.max_slope
        wn, wxi, wd, wdd = W
        hw = track.half_width

        # variables: z_0..z_N (NZ each), u_0..u_{N-1}, eps_0..eps_N (slack on |n|),
        # sig_1..sig_N (slack on the slip angles)
        self.nz_tot = (N + 1) * NZ
        self.nv = self.nz_tot + N + (N + 1) + N
        iz = lambda k: k * NZ  # noqa: E731
        self.iu = lambda k: self.nz_tot + k  # noqa: E731
        self.ie = lambda k: self.nz_tot + N + k  # noqa: E731
        self.isg = lambda k: self.nz_tot + 2 * N + 1 + (k - 1)  # noqa: E731, k = 1..N

        # cost (dt-weighted like acados)
        P = np.zeros(self.nv)
        for k in range(N + 1):
            w = self.dts[k] if k < N else 1.0
            P[iz(k) + 0] = 2 * wn * w
            P[iz(k) + 1] = 2 * wxi * w
            P[iz(k) + 4] = 2 * wd * w
            if k < N:
                P[self.iu(k)] = 2 * wdd * w
            P[self.ie(k)] = 2 * slack_w
            if k >= 1:
                P[self.isg(k)] = 2 * slack_w * 100
        self.P = sp.diags(P).tocsc()
        q = np.zeros(self.nv)
        for k in range(N + 1):
            q[self.ie(k)] = slack_w
            if k >= 1:
                q[self.isg(k)] = slack_w * 100
        self.q = q

        # constraints:
        #   rows 0..NZ-1: z_0 = x0
        #   dynamics: z_{k+1} - Ad_k z_k - Bd_k u_k = c_k       (N*NZ rows, dense 5x5 blocks)
        #   n_k - eps_k <= hw, n_k + eps_k >= -hw              (2(N+1) rows)
        #   |delta_k| <= delta_max (N+1), |u_k| <= rate (N), eps >= 0 (N+1)
        rows, cols = [], []
        r = 0
        for i in range(NZ):
            rows.append(r + i); cols.append(iz(0) + i)
        r += NZ
        self.dyn_row0 = r
        for k in range(N):
            for i in range(NZ):
                rows.append(r + i); cols.append(iz(k + 1) + i)  # identity
                for j in range(NZ):
                    rows.append(r + i); cols.append(iz(k) + j)  # -Ad
                rows.append(r + i); cols.append(self.iu(k))  # -Bd
            r += NZ
        for k in range(N + 1):
            rows += [r, r]; cols += [iz(k), self.ie(k)]; r += 1  # n - eps <= hw
            rows += [r, r]; cols += [iz(k), self.ie(k)]; r += 1  # n + eps >= -hw
        for k in range(N + 1):
            rows.append(r); cols.append(iz(k) + 4); r += 1
        for k in range(N):
            rows.append(r); cols.append(self.iu(k)); r += 1
        for k in range(N + 1):
            rows.append(r); cols.append(self.ie(k)); r += 1
        # slip angles at k = 1..N (linear in z for the predicted speed):
        #   alpha_f = delta - (vy + lf r)/vx,  alpha_r = -(vy - lr r)/vx
        self.slip_row0 = r
        for k in range(1, N + 1):
            for _ in range(2):  # alpha_f - sig <= limit, alpha_f + sig >= -limit
                rows += [r] * 4; cols += [iz(k) + 2, iz(k) + 3, iz(k) + 4, self.isg(k)]; r += 1
            for _ in range(2):
                rows += [r] * 3; cols += [iz(k) + 2, iz(k) + 3, self.isg(k)]; r += 1
        # sig >= 0
        for k in range(1, N + 1):
            rows.append(r); cols.append(self.isg(k)); r += 1
        self.m = r
        # build with placeholder values, then remember the CSC data order
        self._rows, self._cols = np.array(rows), np.array(cols)
        vals = self._values(np.zeros((N, NZ, NZ)), np.zeros((N, NZ)))
        A = sp.csc_matrix((vals, (self._rows, self._cols)), shape=(self.m, self.nv))
        # CSC stores values sorted by column, then row: data = vals[order]
        self._order = np.lexsort((self._rows, self._cols))
        assert A.nnz == len(vals)
        A = sp.csc_matrix((vals[self._order], (self._rows[self._order],
                                               self._cols[self._order])), shape=A.shape)
        self.l = np.full(self.m, -np.inf)
        self.u = np.full(self.m, np.inf)
        r = self.dyn_row0 + N * NZ
        for k in range(N + 1):
            self.u[r] = hw; r += 1
            self.l[r] = -hw; r += 1
        for k in range(N + 1):
            self.l[r], self.u[r] = -veh.delta_max, veh.delta_max; r += 1
        for k in range(N):
            self.l[r], self.u[r] = -veh.delta_rate_max, veh.delta_rate_max; r += 1
        for k in range(N + 1):
            self.l[r] = 0.0; r += 1
        a_pk = veh.tyre.peak_slip if slip_limit else np.inf
        for k in range(1, N + 1):
            for _ in range(2):
                self.u[r] = a_pk; r += 1  # alpha - sig <= limit
                self.l[r] = -a_pk; r += 1  # alpha + sig >= -limit
        for k in range(1, N + 1):
            self.l[r] = 0.0; r += 1
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.P, q=self.q, A=A, l=self.l, u=self.u, verbose=False,
                          warm_starting=True, polishing=False, eps_abs=1e-4, eps_rel=1e-4,
                          max_iter=4000)
        self.solve_times: list[float] = []  # OSQP internal (setup update + solve) (s)
        self.wall_times: list[float] = []  # whole step() incl. linearisation (s)
        self.status: list[str] = []
        self.last = None
        self._shift = 0

    # ----------------------------------------------------------------- helpers
    def _values(self, Ad, Bd, vk=None):
        """Constraint matrix values in triplet order."""
        N = self.N
        vk = np.ones(N + 1) if vk is None else vk
        v = [1.0] * NZ
        for k in range(N):
            for i in range(NZ):
                v.append(1.0)
                v.extend(-Ad[k, i])
                v.append(-Bd[k, i])
        v += [1.0, -1.0, 1.0, 1.0] * (N + 1)
        v += [1.0] * (N + 1) + [1.0] * N + [1.0] * (N + 1)
        lf, lr = self.veh.lf, self.veh.lr
        for k in range(1, N + 1):
            v += [-1 / vk[k], -lf / vk[k], 1.0, -1.0]
            v += [-1 / vk[k], -lf / vk[k], 1.0, 1.0]
            v += [-1 / vk[k], lr / vk[k], -1.0]
            v += [-1 / vk[k], lr / vk[k], 1.0]
        v += [1.0] * N
        return np.array(v)

    def _continuous(self, vx):
        veh = self.veh
        Cf, Cr, m, Iz, lf, lr = self.Cf, self.Cr, veh.m, veh.Iz, veh.lf, veh.lr
        A = np.zeros((NZ, NZ))
        A[0, 1], A[0, 2] = vx, 1.0
        A[1, 3] = 1.0
        A[2, 2] = -(Cf + Cr) / (m * vx)
        A[2, 3] = -(Cf * lf - Cr * lr) / (m * vx) - vx
        A[2, 4] = Cf / m
        A[3, 2] = -(lf * Cf - lr * Cr) / (Iz * vx)
        A[3, 3] = -(lf ** 2 * Cf + lr ** 2 * Cr) / (Iz * vx)
        A[3, 4] = lf * Cf / Iz
        return A

    def speed_command(self, s, vx):
        v_ref = np.interp(np.mod(s, self.track.length), self.track.s, self.v_profile,
                          period=self.track.length)
        # feedforward: d(v^2/2)/ds of the profile
        ds = 1.0
        v_ahead = np.interp(np.mod(s + ds, self.track.length), self.track.s, self.v_profile,
                            period=self.track.length)
        a_ff = (v_ahead ** 2 - v_ref ** 2) / (2 * ds)
        return float(np.clip(a_ff + self.k_v * (v_ref - vx), self.veh.ax_min,
                             self.veh.ax_max))

    # -------------------------------------------------------------------- step
    def step(self, x) -> np.ndarray:
        t0 = time.perf_counter()
        s, n, xi, vx, vy, r, delta = x
        ax0 = self.speed_command(s, vx)
        # predicted speed and position along the horizon (speed law rolled out)
        N = self.N
        Ad = np.empty((N, NZ, NZ))
        Bd = np.empty((N, NZ))
        c = np.empty((N, NZ))
        sk, vk = s, max(vx, 1.0)
        vks = np.empty(N + 1)
        vks[0] = vk
        for k in range(N):
            h = self.dts[k]
            kap = self.track.kappa_at(sk)
            A = self._continuous(vk)
            M = np.zeros((NZ + 2, NZ + 2))
            M[:NZ, :NZ] = A
            M[4, NZ] = 1.0  # delta' = u
            M[1, NZ + 1] = -kap * vk  # affine curvature term
            E = expm(M * h)
            Ad[k], Bd[k], c[k] = E[:NZ, :NZ], E[:NZ, NZ], E[:NZ, NZ + 1]
            ak = self.speed_command(sk, vk)
            sk += vk * h
            vk = max(vk + ak * h, 1.0)
            vks[k + 1] = vk
        vals = self._values(Ad, Bd, vks)[self._order]
        z0 = np.array([n, xi, vy, r, delta])
        self.l[:NZ] = self.u[:NZ] = z0
        r0 = self.dyn_row0
        for k in range(N):
            self.l[r0 + k * NZ: r0 + (k + 1) * NZ] = c[k]
            self.u[r0 + k * NZ: r0 + (k + 1) * NZ] = c[k]
        t1 = time.perf_counter()
        self.solver.update(Ax=vals, l=self.l, u=self.u)
        res = self.solver.solve()
        t2 = time.perf_counter()
        self.solve_times.append(t2 - t1)
        self.status.append(res.info.status)
        if res.x is None or res.info.status_val not in (1, 2):
            # fall back on the previous plan, shifted by one step
            self._shift += 1
            ddelta = 0.0
            if self.last is not None and self._shift < N:
                ddelta = float(self.last[self.iu(self._shift)])
        else:
            ddelta = float(res.x[self.iu(0)])
            self.last, self._shift = res.x, 0
        self.wall_times.append(time.perf_counter() - t0)
        return np.array([ddelta, ax0])
