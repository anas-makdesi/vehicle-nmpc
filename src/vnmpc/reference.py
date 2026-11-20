"""Reference solution of the same optimal control problem with CasADi + IPOPT.

Independent of acados: the same model, grid, cost and (soft) constraints, transcribed by
multiple shooting with the same RK4 step and solved by an interior-point method to tight
tolerance. Used to check that acados finds the same optimum, and as the "slow but exact"
point in the timing benchmark.
"""
from __future__ import annotations

import time

import numpy as np

from .model import NU, NX, casadi_model, kappa_interpolant
from .nmpc import NMPCConfig, axle_accel_limits
from .params import G, Vehicle
from .track import Track


class IpoptReference:
    def __init__(self, veh: Vehicle, track: Track, cfg: NMPCConfig | None = None,
                 mu_model: float = 1.0, mu_limit: float | None = None):
        import casadi as ca

        self.cfg = cfg = cfg or NMPCConfig()
        mu_limit = mu_model if mu_limit is None else mu_limit
        kap = kappa_interpolant(track)
        m = casadi_model(veh, kap)
        v1 = track.speed_profile(mu=1.0, v_max=1e3)
        sv = np.concatenate([track.s + k * track.length for k in range(-1, 3)])
        vref_fun = ca.interpolant("vref_s_ip", "bspline", [sv], np.tile(v1, 4))
        f = ca.Function("f", [m["x"], m["u"], m["p"]], [m["f"]])
        hfun = ca.Function("h", [m["x"], m["u"], m["p"]],
                           [ca.vertcat((m["u"][1] ** 2 + m["ay"] ** 2) / (m["p"][1] * G) ** 2,
                                       m["af"], m["ar"])])
        p = ca.DM([mu_model, mu_limit, np.sqrt(mu_limit)])
        N, dts = cfg.N, cfg.time_steps
        W = np.array(cfg.W)
        We = np.array(cfg.W_e)
        hw, a_pk = track.half_width, veh.tyre.peak_slip
        ax_lo, ax_hi = axle_accel_limits(veh)

        opti = ca.Opti()
        X = opti.variable(NX, N + 1)
        U = opti.variable(NU, N)
        Sn = opti.variable(1, N + 1)  # slack on |n|
        Sh = opti.variable(3, N)  # slacks on h
        x0 = opti.parameter(NX)
        J = 0
        z1, z2 = cfg.slack_l1, cfg.slack_l2

        def y_stage(x, u):
            v_ref = ca.fmin(cfg.v_max, p[2] * vref_fun(x[0]))
            return ca.vertcat(x[1], x[2], x[3] - v_ref, x[6], u[0], u[1])

        def rk4(x, u, h):
            k1 = f(x, u, p); k2 = f(x + h / 2 * k1, u, p)
            k3 = f(x + h / 2 * k2, u, p); k4 = f(x + h * k3, u, p)
            return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            xk, uk, h = X[:, k], U[:, k], float(dts[k])
            y = y_stage(xk, uk)
            J += h * 0.5 * ca.sumsqr(ca.sqrt(W) * y)
            opti.subject_to(X[:, k + 1] == rk4(xk, uk, h))
            opti.subject_to(opti.bounded(-veh.delta_rate_max, uk[0], veh.delta_rate_max))
            opti.subject_to(opti.bounded(veh.ax_min, uk[1], veh.ax_max))
            opti.subject_to(opti.bounded(-veh.delta_max, xk[6], veh.delta_max))
            hk = hfun(xk, uk, p)
            opti.subject_to(hk[0] <= 1 + Sh[0, k])
            opti.subject_to(opti.bounded(ax_lo * mu_limit * G, uk[1], ax_hi * mu_limit * G))
            opti.subject_to(opti.bounded(-a_pk - Sh[1:, k], hk[1:], a_pk + Sh[1:, k]))
            J += h * (z1 * ca.sum1(Sh[:, k]) + 0.5 * z2 * ca.sumsqr(Sh[:, k]))
        for k in range(N + 1):
            wk = float(dts[k]) if k < N else 1.0
            if k > 0:
                opti.subject_to(opti.bounded(-hw - Sn[k], X[1, k], hw + Sn[k]))
            J += wk * (z1 * Sn[k] + 0.5 * z2 * Sn[k] ** 2)
        xe = X[:, N]
        ye = ca.vertcat(xe[1], xe[2], xe[3] - ca.fmin(cfg.v_max, p[2] * vref_fun(xe[0])))
        J += 0.5 * ca.sumsqr(ca.sqrt(We) * ye)
        opti.subject_to(ca.vec(Sn) >= 0)
        opti.subject_to(ca.vec(Sh) >= 0)
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "tol": 1e-8, "max_iter": 500, "sb": "yes",
                     "warm_start_init_point": "no"})
        self.opti, self.X, self.U, self.x0, self.J = opti, X, U, x0, J
        self.Sn, self.Sh = Sn, Sh
        self.solve_times: list[float] = []
        self.iters: list[int] = []

    def solve(self, x0, X_guess=None, U_guess=None):
        """Solve from x0 (optionally warm-started). Returns (u0, cost, X, U)."""
        o = self.opti
        o.set_value(self.x0, x0)
        if X_guess is not None:
            o.set_initial(self.X, np.asarray(X_guess).T)
        if U_guess is not None:
            o.set_initial(self.U, np.asarray(U_guess).T)
        o.set_initial(self.Sn, 0)
        o.set_initial(self.Sh, 0)
        t0 = time.perf_counter()
        sol = o.solve()
        self.solve_times.append(time.perf_counter() - t0)
        self.iters.append(int(sol.stats()["iter_count"]))
        X = np.array(sol.value(self.X)).T
        U = np.array(sol.value(self.U)).T
        return U[0], float(sol.value(self.J)), X, U
