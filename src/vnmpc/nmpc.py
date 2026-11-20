"""Nonlinear MPC with acados (multiple shooting, Gauss-Newton SQP, HPIPM).

The optimal control problem
    minimise   sum_k ||y(x_k, u_k)||_W^2 + ||y_N(x_N)||_W_N^2
    subject to x_{k+1} = F(x_k, u_k)           (RK4 of :func:`vnmpc.model.casadi_model`)
               |n_k| <= corridor half width     (soft)
               ax^2 + ay^2 <= (mu_limit g)^2    (friction ellipse, soft)
               |alpha_f|, |alpha_r| <= peak slip (stay on the stable side of the tyre, soft)
               no axle asked for more longitudinal force than its friction allows
               |delta| <= delta_max, |delta_dot| <= rate limit, ax in [ax_min, ax_max]
with y = [n, xi, vx - v_ref(s), delta, delta_dot, ax]. The reference speed is
``min(v_max, v_scale * v_1(s))``, where ``v_1`` is the track's speed profile for friction 1
and ``v_scale = sqrt(mu)`` (the profile's v^2 scales with the friction coefficient).

The horizon is 2 s on a non-uniform grid: 10 steps of one control period (20 ms at 50 Hz)
followed by 30 steps of 60 ms, so the first move is resolved at the control rate and the
tail looks far enough ahead to brake for corners.

``mode="rti"`` runs one real-time iteration per control step (the production setting);
``mode="sqp"`` iterates to convergence (used as a check and in the benchmarks).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import NU, NX, casadi_model, kappa_interpolant
from .params import G, Vehicle
from .track import Track

BUILD_DIR = Path(os.environ.get("VNMPC_BUILD_DIR", Path.cwd() / "build"))


@dataclass
class NMPCConfig:
    dt0: float = 0.02  # control period = first steps of the grid (s)
    n_fine: int = 10
    dt_tail: float = 0.06
    n_tail: int = 30
    v_max: float = 30.0
    mode: str = "rti"  # "rti" or "sqp"
    max_iter: int = 50  # for mode="sqp"
    hessian: str = "EXACT"  # or "GAUSS_NEWTON"
    qp_cond_N: int = 10  # partial condensing horizon
    W: tuple = (2.0, 2.0, 1.0, 1.0, 20.0, 0.05)  # n, xi, vx err, delta, delta_dot, ax
    W_e: tuple = (5.0, 5.0, 1.0)
    slack_l1: float = 1e3
    slack_l2: float = 1e3
    name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.n_fine + self.n_tail

    @property
    def time_steps(self) -> np.ndarray:
        return np.r_[np.full(self.n_fine, self.dt0), np.full(self.n_tail, self.dt_tail)]

    @property
    def horizon(self) -> float:
        return float(self.time_steps.sum())


def axle_accel_limits(veh: Vehicle):
    """Bounds on ax / (mu g) so that no axle needs more longitudinal force than its own
    friction allows: traction on the rear axle, braking split 60/40 front/rear."""
    from .model import BRAKE_FRONT

    L = veh.L
    hi = veh.lf / L  # rear load share
    lo = -min(veh.lr / (BRAKE_FRONT * L), veh.lf / ((1 - BRAKE_FRONT) * L))
    return lo, hi


def build_ocp(veh: Vehicle, track: Track, cfg: NMPCConfig):
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    kap = kappa_interpolant(track)
    m = casadi_model(veh, kap)
    v1 = track.speed_profile(mu=1.0, v_max=1e3)
    sv = np.concatenate([track.s + k * track.length for k in range(-1, 3)])
    vref_fun = ca.interpolant("vref_s", "bspline", [sv], np.tile(v1, 4))
    x, u, p = m["x"], m["u"], m["p"]
    v_ref = ca.fmin(cfg.v_max, p[2] * vref_fun(x[0]))

    model = AcadosModel()
    model.name = cfg.name or f"vnmpc_{cfg.mode}_N{cfg.N}_dt{int(round(cfg.dt0 * 1000))}"
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = m["f"]
    model.xdot = ca.SX.sym("xdot", NX)
    model.f_impl_expr = model.xdot - m["f"]
    mu_lim = p[1]
    model.con_h_expr_0 = model.con_h_expr = ca.vertcat(
        (u[1] ** 2 + m["ay"] ** 2) / (mu_lim * G) ** 2,
        m["af"],
        m["ar"],
        u[1] / (mu_lim * G),  # per-axle traction / braking limit (hard)
    )
    model.cost_y_expr = ca.vertcat(x[1], x[2], x[3] - v_ref, x[6], u[0], u[1])
    model.cost_y_expr_e = ca.vertcat(x[1], x[2], x[3] - v_ref)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = cfg.N
    ocp.solver_options.tf = cfg.horizon
    ocp.solver_options.time_steps = cfg.time_steps
    ocp.parameter_values = np.array([1.0, 1.0, 1.0])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(cfg.W)
    ocp.cost.W_e = np.diag(cfg.W_e)
    ocp.cost.yref = np.zeros(len(cfg.W))
    ocp.cost.yref_e = np.zeros(len(cfg.W_e))

    c = ocp.constraints
    c.x0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    # state bounds: n (soft), delta (hard)
    hw = track.half_width
    c.idxbx = np.array([1, 6])
    c.lbx = np.array([-hw, -veh.delta_max])
    c.ubx = np.array([hw, veh.delta_max])
    c.idxsbx = np.array([0])
    c.idxbx_e = np.array([1])
    c.lbx_e = np.array([-hw])
    c.ubx_e = np.array([hw])
    c.idxsbx_e = np.array([0])
    c.idxbu = np.array([0, 1])
    c.lbu = np.array([-veh.delta_rate_max, veh.ax_min])
    c.ubu = np.array([veh.delta_rate_max, veh.ax_max])
    a_pk = veh.tyre.peak_slip
    lo_ax, hi_ax = axle_accel_limits(veh)
    c.lh = np.array([-1e3, -a_pk, -a_pk, lo_ax])
    c.uh = np.array([1.0, a_pk, a_pk, hi_ax])
    c.idxsh = np.array([0, 1, 2])
    # acados needs the path constraints of the first stage declared separately; without
    # them the input that is actually applied would not be constrained by h.
    c.lh_0, c.uh_0, c.idxsh_0 = c.lh.copy(), c.uh.copy(), c.idxsh.copy()
    ocp.cost.zl_0 = ocp.cost.zu_0 = cfg.slack_l1 * np.ones(3)
    ocp.cost.Zl_0 = ocp.cost.Zu_0 = cfg.slack_l2 * np.ones(3)
    ns, ns_e = 4, 1
    ocp.cost.zl = cfg.slack_l1 * np.ones(ns)
    ocp.cost.zu = cfg.slack_l1 * np.ones(ns)
    ocp.cost.Zl = cfg.slack_l2 * np.ones(ns)
    ocp.cost.Zu = cfg.slack_l2 * np.ones(ns)
    ocp.cost.zl_e = cfg.slack_l1 * np.ones(ns_e)
    ocp.cost.zu_e = cfg.slack_l1 * np.ones(ns_e)
    ocp.cost.Zl_e = cfg.slack_l2 * np.ones(ns_e)
    ocp.cost.Zu_e = cfg.slack_l2 * np.ones(ns_e)

    so = ocp.solver_options
    so.integrator_type = "ERK"
    so.sim_method_num_stages = 4
    so.sim_method_num_steps = 1
    # Gauss-Newton full steps oscillate on this model (the tyre and kinematic terms carry
    # large multipliers), so use the exact Hessian, projected to be positive definite.
    so.hessian_approx = cfg.hessian
    if cfg.hessian == "EXACT":
        so.regularize_method = "PROJECT"
        so.reg_epsilon = 1e-4
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_cond_N = cfg.qp_cond_N
    so.qp_solver_warm_start = 1
    so.qp_solver_iter_max = 50
    if cfg.mode == "rti":
        so.nlp_solver_type = "SQP_RTI"
    else:
        so.nlp_solver_type = "SQP"
        so.nlp_solver_max_iter = cfg.max_iter
        so.tol = 1e-6
        so.globalization = "MERIT_BACKTRACKING"
    so.print_level = 0
    for k, v in cfg.extra.items():
        setattr(so, k, v)
    return ocp


class NMPC:
    """acados NMPC controller. ``step(x)`` returns the input to apply now."""

    def __init__(self, veh: Vehicle, track: Track, cfg: NMPCConfig | None = None,
                 mu_model: float = 1.0, mu_limit: float | None = None, build: bool = True):
        from acados_template import AcadosOcpSolver

        self.veh, self.track = veh, track
        self.cfg = cfg = cfg or NMPCConfig()
        self.ocp = build_ocp(veh, track, cfg)
        out = BUILD_DIR / self.ocp.model.name
        out.mkdir(parents=True, exist_ok=True)
        self.ocp.code_export_directory = str(out / "c_generated_code")
        self.ocp.json_file = "acados_ocp.json"  # written inside code_export_directory
        self.solver = AcadosOcpSolver(self.ocp, build=build, generate=build, verbose=False)
        self.set_friction(mu_model, mu_limit)
        self.initialized = False
        self.solve_times: list[float] = []  # acados internal time per step (s)
        self.wall_times: list[float] = []  # Python wall time of step() (s)
        self.status: list[int] = []
        self.iters: list[int] = []

    # ------------------------------------------------------------------ setup
    def set_friction(self, mu_model: float, mu_limit: float | None = None):
        """Friction used by the planning model and by the constraints (no rebuild needed)."""
        mu_limit = mu_model if mu_limit is None else mu_limit
        self.mu_model, self.mu_limit = float(mu_model), float(mu_limit)
        p = np.array([self.mu_model, self.mu_limit, np.sqrt(self.mu_limit)])
        for k in range(self.cfg.N + 1):
            self.solver.set(k, "p", p)

    def _init_guess(self, x0):
        """Straight-line guess: keep speed, follow the centre line."""
        t = np.r_[0.0, np.cumsum(self.cfg.time_steps)]
        for k in range(self.cfg.N + 1):
            xk = x0.copy()
            xk[0] = x0[0] + x0[3] * t[k]
            xk[1:3] = 0.0 if k else x0[1:3]
            xk[4:6] = 0.0 if k else x0[4:6]
            self.solver.set(k, "x", xk)
            if k < self.cfg.N:
                self.solver.set(k, "u", np.zeros(NU))
        self.initialized = True

    # ------------------------------------------------------------------- step
    def step(self, x0) -> np.ndarray:
        x0 = np.asarray(x0, float)
        if not self.initialized:
            self._init_guess(x0)
            if self.cfg.mode == "rti":  # a few iterations to start from a good point
                for _ in range(5):
                    self.solver.solve_for_x0(x0, fail_on_nonzero_status=False,
                                             print_stats_on_failure=False)
        t0 = time.perf_counter()
        u0 = self.solver.solve_for_x0(x0, fail_on_nonzero_status=False,
                                      print_stats_on_failure=False)
        self.wall_times.append(time.perf_counter() - t0)
        self.solve_times.append(float(self.solver.get_stats("time_tot")))
        self.status.append(int(self.solver.status))
        it = self.solver.get_stats("sqp_iter") if self.cfg.mode == "sqp" else 1
        self.iters.append(int(it))
        return np.asarray(u0, float)

    def prediction(self):
        X = np.array([self.solver.get(k, "x") for k in range(self.cfg.N + 1)])
        U = np.array([self.solver.get(k, "u") for k in range(self.cfg.N)])
        return X, U

    def cost(self) -> float:
        return float(self.solver.get_cost())
