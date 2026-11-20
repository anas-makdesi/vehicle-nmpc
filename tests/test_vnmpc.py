"""Tests for the vehicle NMPC package (Python side).

The acados tests generate and compile a solver (a few seconds each, cached by name in
``build/``); they are skipped if acados is not installed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vnmpc.grip import GripCertificate, calibration_run
from vnmpc.ltv_mpc import LTVMPC
from vnmpc.model import casadi_model, kappa_interpolant, plant_rhs, plant_step
from vnmpc.params import Vehicle
from vnmpc.sim import run_lap
from vnmpc.track import Track

warnings.filterwarnings("ignore")

try:
    import acados_template  # noqa: F401

    HAVE_ACADOS = True
except Exception:  # pragma: no cover
    HAVE_ACADOS = False
needs_acados = pytest.mark.skipif(not HAVE_ACADOS, reason="acados not installed")
try:
    import ddma  # noqa: F401

    HAVE_DDMA = True
except Exception:  # pragma: no cover
    HAVE_DDMA = False
needs_ddma = pytest.mark.skipif(not HAVE_DDMA, reason="ddma not installed")


@pytest.fixture(scope="module")
def veh():
    return Vehicle()


@pytest.fixture(scope="module")
def track():
    return Track.default()


# --------------------------------------------------------------------- track
def test_track_closed_and_smooth(track):
    assert 1200 < track.length < 1450
    x0, y0 = track.frenet_to_xy(0.0, 0.0)
    x1, y1 = track.frenet_to_xy(track.length - 1e-6, 0.0)
    assert np.hypot(x1 - x0, y1 - y0) < 0.6
    assert 1 / np.abs(track.kappa).max() > 10.0  # tightest corner radius above 10 m
    # curvature equals the heading rate
    dpsi = np.gradient(track.psi, track.s)
    assert np.max(np.abs(dpsi - track.kappa)[5:-5]) < 2e-3


def test_speed_profile_respects_friction_circle(track):
    for mu in (0.6, 1.0):
        v = track.speed_profile(mu=mu, use=0.85)
        a_lat = v ** 2 * np.abs(track.kappa)
        assert np.all(a_lat <= 0.85 * mu * 9.81 + 1e-6)
        assert v.max() <= 30.0 + 1e-9
    assert track.speed_profile(0.6).mean() < track.speed_profile(1.0).mean()


# --------------------------------------------------------------------- model
def test_casadi_model_matches_plant_without_saturation(veh, track):
    """The planning model and the plant agree when no axle force is clipped."""
    import casadi as ca

    kap = kappa_interpolant(track)
    m = casadi_model(veh, kap)
    f = ca.Function("f", [m["x"], m["u"], m["p"]], [m["f"]])
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = np.array([rng.uniform(0, track.length), rng.uniform(-2, 2), rng.uniform(-0.1, 0.1),
                      rng.uniform(8, 28), rng.uniform(-1, 1), rng.uniform(-0.4, 0.4),
                      rng.uniform(-0.1, 0.1)])
        u = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-3, 1.5)])
        mu = rng.uniform(0.5, 1.0)
        fm = np.array(f(x, u, [mu, mu, np.sqrt(mu)])).ravel()
        k = float(kap(x[0]))
        fp = plant_rhs(x, u, veh, mu, k)
        np.testing.assert_allclose(fm, fp, rtol=1e-6, atol=1e-6)


def test_plant_straight_line(veh):
    class Flat:
        @staticmethod
        def kappa_at(s):
            return 0.0

    x = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
    x1 = plant_step(x, np.array([0.0, 1.0]), 1.0, veh, 1.0, Flat, substeps=100)
    assert x1[0] == pytest.approx(20.5, abs=1e-6)
    assert x1[3] == pytest.approx(21.0, abs=1e-6)
    np.testing.assert_allclose(x1[[1, 2, 4, 5, 6]], 0.0, atol=1e-12)


# ---------------------------------------------------------------------- grip
@needs_ddma
def test_grip_certificate_is_sound_and_tight(veh):
    for mu in (0.4, 0.6, 0.8):
        A, F, AX = calibration_run(veh, None, mu, seed=3)
        g = GripCertificate(veh)
        assert g.add(A, F, AX) > 100
        assert g.mu_certified <= mu  # never above the true friction
        assert g.mu_certified >= 0.85 * mu  # and not far below it
        a, lo, hi = g.bounds()
        t = veh.tyre
        true = mu * t.D * np.sin(t.C * np.arctan(t.B * a))
        ok = np.isfinite(lo)
        assert np.all(lo[ok] <= true[ok] + 1e-9)
        assert np.all(hi[ok] >= true[ok] - 1e-9)


@needs_ddma
def test_grip_prior_without_data(veh):
    assert GripCertificate(veh, mu_prior=0.3).mu_certified == 0.3


# ------------------------------------------------------------------ baseline
def test_ltv_mpc_finishes_a_lap_with_margin(veh, track):
    r = run_lap(LTVMPC(veh, track, use=0.65), veh, track, 1.0, t_max=80)
    assert r.finished and r.max_offset < track.half_width


# ---------------------------------------------------------------------- NMPC
@needs_acados
def test_nmpc_rti_drives_a_lap(veh, track):
    from vnmpc.nmpc import NMPC, NMPCConfig

    c = NMPC(veh, track, NMPCConfig(name="test_rti"))
    r = run_lap(c, veh, track, 1.0, t_max=80)
    assert r.finished
    assert r.max_offset < 0.5
    assert r.max_friction_use < 0.95
    assert all(s == 0 for s in c.status)
    assert np.median(r.solve_ms) < 20.0


@needs_acados
def test_nmpc_matches_ipopt(veh, track):
    """acados (SQP to convergence) and IPOPT find the same optimum."""
    from vnmpc.nmpc import NMPC, NMPCConfig
    from vnmpc.reference import IpoptReference

    sqp = NMPC(veh, track, NMPCConfig(mode="sqp", max_iter=200, name="test_sqp"))
    ip = IpoptReference(veh, track)
    for x in (np.array([30.0, 0.2, 0.01, 25.0, 0.1, 0.05, 0.01]),
              np.array([700.0, -0.5, -0.03, 18.0, -0.2, 0.2, 0.03])):
        sqp.initialized = False
        u = sqp.step(x)
        X, U = sqp.prediction()
        ui, ci, _, _ = ip.solve(x, X, U)
        np.testing.assert_allclose(u, ui, atol=1e-4)
        assert sqp.cost() == pytest.approx(ci, rel=1e-3, abs=1e-4)


@needs_acados
@needs_ddma
def test_certified_friction_keeps_the_car_on_a_low_grip_road(veh, track):
    from vnmpc.nmpc import NMPC, NMPCConfig

    A, F, AX = calibration_run(veh, None, 0.6)
    g = GripCertificate(veh)
    g.add(A, F, AX)
    c = NMPC(veh, track, NMPCConfig(name="test_rti"), mu_model=g.mu_certified)
    r = run_lap(c, veh, track, 0.6, t_max=40)  # the first 40 s include the hairpins
    assert not r.spun
    assert r.max_offset < track.half_width
    nominal = NMPC(veh, track, NMPCConfig(name="test_rti"), mu_model=1.0)
    r = run_lap(nominal, veh, track, 0.6, t_max=40)
    assert r.spun  # planning with the nominal friction loses the car
