"""Dynamic bicycle model in path (Frenet) coordinates.

State  x = [s, n, xi, vx, vy, r, delta]
    s      progress along the centre line (m)
    n      lateral offset from the centre line, left > 0 (m)
    xi     heading relative to the centre line (rad)
    vx, vy body-frame velocities (m/s)
    r      yaw rate (rad/s)
    delta  road-wheel steering angle (rad)
Input  u = [delta_dot, ax]
    delta_dot  steering rate (rad/s)
    ax         longitudinal acceleration command (m/s^2)

Two versions of the same equations:

* :func:`casadi_model` - what the controllers plan with, with friction ``mu`` as a
  parameter.
* :func:`plant_rhs` - the simulated "real" car (NumPy, mirrored in the ROS 2 C++
  simulator), with the true friction.

Both use magic-formula lateral tyres whose force is derated by the friction circle when
the axle also transmits longitudinal force (braking split 60/40 front/rear, traction on
the rear axle). The plant also caps the longitudinal force at the friction limit; the
controllers keep it below that limit through the friction-ellipse constraint.
"""
from __future__ import annotations

import numpy as np

from .params import G, Vehicle

NX, NU = 7, 2
STATE_NAMES = ("s", "n", "xi", "vx", "vy", "r", "delta")
INPUT_NAMES = ("delta_dot", "ax")
BRAKE_FRONT = 0.6


def slip_angles(vx, vy, r, delta, veh: Vehicle, atan2):
    af = delta - atan2(vy + veh.lf * r, vx)
    ar = -atan2(vy - veh.lr * r, vx)
    return af, ar


# --------------------------------------------------------------------------- CasADi
def casadi_model(veh: Vehicle, kappa_fun):
    """Symbolic model for the controllers.

    ``kappa_fun`` maps a CasADi scalar s to the curvature (e.g. a CasADi interpolant).
    Parameters p = [mu_model, mu_limit, v_scale]: friction used by the tyre model, friction
    used by the constraints, and the scaling of the reference speed (see :mod:`vnmpc.nmpc`).
    Returns a dict with symbols and expressions.
    """
    import casadi as ca

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", 3)
    s, n, xi, vx, vy, r, delta = (x[i] for i in range(NX))
    ddelta, ax = u[0], u[1]
    mu = p[0]
    t = veh.tyre
    af, ar = slip_angles(vx, vy, r, delta, veh, ca.atan2)
    # longitudinal force split as in the plant: traction on the rear axle, braking 60/40
    Fx = veh.m * ax
    Fxf = ca.if_else(Fx >= 0, 0, BRAKE_FRONT * Fx)
    Fxr = ca.if_else(Fx >= 0, Fx, (1 - BRAKE_FRONT) * Fx)
    capf, capr = mu * veh.Fzf, mu * veh.Fzr
    derf = ca.sqrt(1 - ca.fmin((Fxf / capf) ** 2, 0.99))
    derr = ca.sqrt(1 - ca.fmin((Fxr / capr) ** 2, 0.99))
    Fyf = derf * mu * veh.Fzf * t.D * ca.sin(t.C * ca.atan(t.B * af))
    Fyr = derr * mu * veh.Fzr * t.D * ca.sin(t.C * ca.atan(t.B * ar))
    kap = kappa_fun(s)
    sdot = (vx * ca.cos(xi) - vy * ca.sin(xi)) / (1 - n * kap)
    cd, sd = ca.cos(delta), ca.sin(delta)
    f = ca.vertcat(
        sdot,
        vx * ca.sin(xi) + vy * ca.cos(xi),
        r - kap * sdot,
        (Fxf * cd - Fyf * sd + Fxr) / veh.m + vy * r,
        (Fxf * sd + Fyf * cd + Fyr) / veh.m - vx * r,
        (veh.lf * (Fxf * sd + Fyf * cd) - veh.lr * Fyr) / veh.Iz,
        ddelta,
    )
    ay = (Fxf * sd + Fyf * cd + Fyr) / veh.m
    return dict(x=x, u=u, p=p, f=f, af=af, ar=ar, ay=ay, Fyf=Fyf, Fyr=Fyr, kappa=kap)


def kappa_interpolant(track, laps: int = 4):
    """CasADi B-spline of the curvature over ``laps`` laps starting at -1 lap, so that the
    prediction horizon may run past the finish line (and the state may start before 0)."""
    import casadi as ca

    s = np.concatenate([track.s + k * track.length for k in range(-1, laps - 1)])
    k = np.tile(track.kappa, laps)
    return ca.interpolant("kappa_s", "bspline", [s], k)


# --------------------------------------------------------------------------- plant
def plant_forces(x, u, veh: Vehicle, mu: float):
    """Tyre forces of the simulated car: (Fxf, Fxr, Fyf, Fyr, af, ar)."""
    _, _, _, vx, vy, r, delta = x
    ax = u[1]
    t = veh.tyre
    af, ar = slip_angles(vx, vy, r, delta, veh, np.arctan2)
    Fx = veh.m * ax
    if Fx >= 0:
        Fxf, Fxr = 0.0, Fx
    else:
        Fxf, Fxr = BRAKE_FRONT * Fx, (1 - BRAKE_FRONT) * Fx
    capf, capr = mu * veh.Fzf, mu * veh.Fzr
    Fxf = float(np.clip(Fxf, -capf, capf))
    Fxr = float(np.clip(Fxr, -capr, capr))
    Fyf = mu * veh.Fzf * t.D * np.sin(t.C * np.arctan(t.B * af))
    Fyr = mu * veh.Fzr * t.D * np.sin(t.C * np.arctan(t.B * ar))
    Fyf *= np.sqrt(max(0.0, 1.0 - (Fxf / capf) ** 2))
    Fyr *= np.sqrt(max(0.0, 1.0 - (Fxr / capr) ** 2))
    return Fxf, Fxr, Fyf, Fyr, af, ar


def plant_rhs(x, u, veh: Vehicle, mu: float, kappa: float):
    s, n, xi, vx, vy, r, delta = x
    Fxf, Fxr, Fyf, Fyr, _, _ = plant_forces(x, u, veh, mu)
    cd, sd = np.cos(delta), np.sin(delta)
    sdot = (vx * np.cos(xi) - vy * np.sin(xi)) / (1 - n * kappa)
    return np.array([
        sdot,
        vx * np.sin(xi) + vy * np.cos(xi),
        r - kappa * sdot,
        (Fxf * cd - Fyf * sd + Fxr) / veh.m + vy * r,
        (Fxf * sd + Fyf * cd + Fyr) / veh.m - vx * r,
        (veh.lf * (Fxf * sd + Fyf * cd) - veh.lr * Fyr) / veh.Iz,
        u[0],
    ])


def plant_step(x, u, dt, veh: Vehicle, mu: float, track, substeps: int = 10):
    """Integrate the plant over dt with RK4 (zero-order hold on u)."""
    h = dt / substeps
    x = np.asarray(x, float).copy()
    for _ in range(substeps):
        k = track.kappa_at(x[0])
        k1 = plant_rhs(x, u, veh, mu, k)
        k2 = plant_rhs(x + 0.5 * h * k1, u, veh, mu, k)
        k3 = plant_rhs(x + 0.5 * h * k2, u, veh, mu, k)
        k4 = plant_rhs(x + h * k3, u, veh, mu, k)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def body_accel(x, u, veh: Vehicle, mu: float):
    """Tyre-force accelerations of the plant in the body frame: (a_long, a_lat)."""
    Fxf, Fxr, Fyf, Fyr, _, _ = plant_forces(x, u, veh, mu)
    d = x[6]
    return ((Fxf * np.cos(d) - Fyf * np.sin(d) + Fxr) / veh.m,
            (Fxf * np.sin(d) + Fyf * np.cos(d) + Fyr) / veh.m)


__all__ = ["NX", "NU", "STATE_NAMES", "INPUT_NAMES", "casadi_model", "kappa_interpolant",
           "plant_forces", "plant_rhs", "plant_step", "body_accel", "G"]
