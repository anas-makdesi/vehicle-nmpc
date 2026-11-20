"""Lap experiments: NMPC (acados) vs the linear MPC baseline (OSQP), and driving on a
low-grip road with a friction value certified from data (ddma).

    python examples/lap_comparison.py          # writes results/laps.json and figures

Experiment 1 - dry road (true friction 1.0, known to the controllers):
  * NMPC, one real-time iteration per step (RTI), reference speed at 85 % of the grip;
  * NMPC iterated to convergence (SQP) on the same problem, to see what RTI gives up;
  * linear MPC with linear tyres and a separate speed loop, same speed profile;
  * the same linear MPC with the speed profile lowered until it completes the lap.
Experiment 2 - low-grip road (true friction 0.6):
  * NMPC that assumes the nominal friction 1.0;
  * NMPC with the friction certified by ddma from a short (about 6 s) calibration run;
  * NMPC that knows the true friction (oracle, for reference).
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

from vnmpc.grip import GripCertificate, calibration_run  # noqa: E402
from vnmpc.ltv_mpc import LTVMPC  # noqa: E402
from vnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from vnmpc.params import Vehicle  # noqa: E402
from vnmpc.sim import run_lap  # noqa: E402
from vnmpc.track import Track  # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "docs" / "figures"


def main(fig: bool = True):
    OUT.mkdir(exist_ok=True)
    veh, tr = Vehicle(), Track.default()
    res = {}

    # ---------------------------------------------------------------- dry road
    res["nmpc_rti"] = run_lap(NMPC(veh, tr, NMPCConfig()), veh, tr, 1.0, "NMPC (RTI)")
    res["nmpc_sqp"] = run_lap(NMPC(veh, tr, NMPCConfig(mode="sqp")), veh, tr, 1.0,
                              "NMPC (converged SQP)")
    res["ltv_85"] = run_lap(LTVMPC(veh, tr, use=0.85), veh, tr, 1.0, "Linear MPC, 85% profile")
    use_ok = None
    for use in (0.80, 0.75, 0.70, 0.65, 0.60):
        r = run_lap(LTVMPC(veh, tr, use=use), veh, tr, 1.0, f"Linear MPC, {use:.0%} profile")
        if r.finished and r.corridor_violation_time == 0:
            res["ltv_ok"], use_ok = r, use
            break

    # ----------------------------------------------------------- low-grip road
    mu_true = 0.6
    A, F, AX = calibration_run(veh, tr, mu_true)
    cert = GripCertificate(veh)
    cert.add(A, F, AX)
    mu_c = cert.mu_certified
    res["low_nominal"] = run_lap(NMPC(veh, tr, NMPCConfig(), mu_model=1.0), veh, tr, mu_true,
                                 "NMPC, assumes friction 1.0")
    res["low_cert"] = run_lap(NMPC(veh, tr, NMPCConfig(), mu_model=mu_c), veh, tr, mu_true,
                              f"NMPC, certified friction {mu_c:.2f}")
    res["low_oracle"] = run_lap(NMPC(veh, tr, NMPCConfig(), mu_model=mu_true), veh, tr, mu_true,
                                "NMPC, true friction 0.6 (oracle)")

    summary = {k: v.summary() for k, v in res.items()}
    summary["_meta"] = dict(track_length=round(tr.length, 1), half_width=tr.half_width,
                            ltv_profile_that_finishes=use_ok, mu_true_low=mu_true,
                            mu_certified=round(mu_c, 3), calibration_samples=int(len(A)),
                            calibration_samples_used=cert.n_used)
    (OUT / "laps.json").write_text(json.dumps(summary, indent=2, default=float))
    for k, v in summary.items():
        print(k, v)
    if fig:
        figures(veh, tr, res, cert, A, F, AX)
    return summary


def figures(veh, tr, res, cert, A, F, AX):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from vnmpc.plotting import CRITICAL, INK2, SERIES, draw_track, speed_cmap, style

    style()
    FIG.mkdir(parents=True, exist_ok=True)

    def xy_of(r):
        X = r.log.arrays()["x"]
        return np.array(tr.frenet_to_xy(X[:, 0], X[:, 1])).T, X

    # --- hero: NMPC lap coloured by speed, linear MPC's spin marked
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    draw_track(ax, tr)
    P, X = xy_of(res["nmpc_rti"])
    seg = np.stack([P[:-1], P[1:]], axis=1)
    lc = LineCollection(seg, cmap=speed_cmap(), linewidths=2.4, zorder=3)
    lc.set_array(X[:-1, 3] * 3.6)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.06)
    cb.set_label("speed (km/h)")
    cb.outline.set_visible(False)
    Pl, _ = xy_of(res["ltv_85"])
    ax.plot(*Pl[-1], marker="X", ms=11, color=CRITICAL, mec="white", mew=1.5, zorder=5)
    ax.annotate("linear MPC (85% profile)\nspins here", Pl[-1], xytext=(-14, 14),
                textcoords="offset points", color=INK2, fontsize=10, ha="right")
    ax.set_title(f"NMPC lap, dry road: {res['nmpc_rti'].lap_time:.1f} s, "
                 f"max {res['nmpc_rti'].max_offset:.2f} m off the centre line")
    ax = axs[1]
    for i, key in enumerate(["nmpc_rti", "ltv_ok", "ltv_85"]):
        r = res[key]
        X = r.log.arrays()["x"]
        ax.plot(X[:, 0], X[:, 3] * 3.6, color=SERIES[i], lw=1.8, label=r.name)
    ax.axvline(res["ltv_85"].distance, color=CRITICAL, lw=1, ls=":")
    ax.set_xlabel("distance along the track (m)")
    ax.set_ylabel("speed (km/h)")
    ax.set_title("Speed along the lap")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "lap_dry.png", dpi=150)
    plt.close(fig)

    # --- low grip: three NMPCs on a road with friction 0.6
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    draw_track(ax, tr)
    for i, key in enumerate(["low_cert", "low_oracle", "low_nominal"]):
        r = res[key]
        P, _ = xy_of(r)
        ax.plot(P[:, 0], P[:, 1], color=SERIES[i], lw=2.4 if i == 0 else 1.2,
                ls="-" if i != 1 else (0, (4, 3)), zorder=6 - i, label=r.name)
        if not r.finished:
            ax.plot(*P[-1], marker="X", ms=11, color=CRITICAL, mec="white", mew=1.5, zorder=8)
            ax.annotate("spins", P[-1], xytext=(10, -14), textcoords="offset points",
                        color=INK2, fontsize=10)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 1.02))
    ax.set_title("Road friction 0.6")
    ax = axs[1]
    a, lo, hi = cert.bounds()
    al = np.abs(A)
    ff = np.abs(F)
    keep = np.abs(AX) <= cert.ax_gate
    ax.scatter(np.rad2deg(al[keep]), ff[keep], s=9, color=SERIES[0], alpha=0.5, lw=0,
               label="measured samples", zorder=2)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(np.rad2deg(a[ok]), lo[ok], hi[ok], color=SERIES[0], alpha=0.15, lw=0,
                    label="ddma guaranteed band", zorder=1)
    t = veh.tyre
    aa = np.linspace(0, cert.alpha_max, 400)
    ax.plot(np.rad2deg(aa), 0.6 * t.D * np.sin(t.C * np.arctan(t.B * aa)), color=INK2,
            lw=1.4, ls="--", label="true tyre curve (unknown)")
    ax.axhline(cert.mu_certified, color=SERIES[1], lw=1.8)
    ax.text(np.rad2deg(cert.alpha_max) * 0.58, cert.mu_certified + 0.02,
            f"certified grip {cert.mu_certified:.2f}", ha="right", color=INK2, fontsize=10)
    ax.set_xlim(0, np.rad2deg(cert.alpha_max) * 0.6)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("slip angle (deg)")
    ax.set_ylabel("lateral force / load")
    ax.set_title(f"Grip certified from a {len(A) // 2 * 0.02:.0f} s calibration run "
                 f"({cert.n_used} samples)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "lap_low_grip.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(fig="--no-fig" not in sys.argv)
