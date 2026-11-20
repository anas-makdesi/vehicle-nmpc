"""Short demo for the README and the website: two NMPCs on a low-grip road (friction 0.6).

One assumes the nominal friction 1.0; the other uses the friction certified by ddma from a
short calibration run. Same controller, same track, same car: only the friction value in the
model differs. Runs the two laps of examples/lap_comparison.py (experiment 2), then draws them.

    python examples/demo_gif.py      (writes docs/figures/demo.gif; needs acados and ddma)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from vnmpc.grip import GripCertificate, calibration_run  # noqa: E402
from vnmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from vnmpc.params import Vehicle  # noqa: E402
from vnmpc.sim import run_lap  # noqa: E402
from vnmpc.track import Track  # noqa: E402

INK, TEAL, ORANGE, MUTED, RED = "#0B1220", "#2a9d8f", "#FB923C", "#6B7280", "#C2410C"
PAPER, ROAD, EDGE = "#F7F7F5", "#E4E4DF", "#B9B8B3"
SPEEDUP = 6          # playback speed
FRAME_MS = 100
MU_TRUE = 0.6


def run():
    veh, tr = Vehicle(), Track.default()
    A, F, AX = calibration_run(veh, tr, MU_TRUE)
    cert = GripCertificate(veh)
    cert.add(A, F, AX)
    mu_c = cert.mu_certified
    laps = {
        "nominal": run_lap(NMPC(veh, tr, NMPCConfig(), mu_model=1.0), veh, tr, MU_TRUE,
                           "assumes friction 1.0"),
        "cert": run_lap(NMPC(veh, tr, NMPCConfig(), mu_model=mu_c), veh, tr, MU_TRUE,
                        f"certified friction {mu_c:.2f}"),
    }
    for k, r in laps.items():
        print(k, r.summary())
    return tr, laps, mu_c


def draw(tr, laps, mu_c):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from vnmpc.plotting import track_edges

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    runs = []
    for key, color, label in (("cert", TEAL, f"NMPC, friction certified by ddma ({mu_c:.2f})"),
                              ("nominal", ORANGE, "NMPC, assumes friction 1.0")):
        r = laps[key]
        L = r.log.arrays()
        X, t = L["x"], L["t"]
        P = np.array(tr.frenet_to_xy(X[:, 0], X[:, 1])).T
        heading = tr.heading_at(X[:, 0]) + X[:, 2]
        runs.append(dict(r=r, t=t, P=P, X=X, h=heading, color=color, label=label))
    t_end = max(d["t"][-1] for d in runs)
    times = np.r_[np.arange(0, t_end, SPEEDUP * FRAME_MS / 1000), t_end + 1e-6]
    (lx, ly), (rx, ry) = track_edges(tr)

    fig = plt.figure(figsize=(7.2, 4.2), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0.0, 0.0, 0.62, 0.92])
    side = fig.add_axes([0.62, 0.0, 0.38, 0.92])
    frames = []
    for T in times:
        ax.clear()
        side.clear()
        side.axis("off")
        ax.fill(np.r_[lx, rx[::-1]], np.r_[ly, ry[::-1]], color=ROAD, lw=0)
        ax.plot(lx, ly, color=EDGE, lw=0.8)
        ax.plot(rx, ry, color=EDGE, lw=0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        y = 0.92
        for d in runs:
            k = int(np.searchsorted(d["t"], T, side="right")) - 1
            k = max(0, min(k, len(d["t"]) - 1))
            ax.plot(d["P"][: k + 1, 0], d["P"][: k + 1, 1], color=d["color"], lw=2.2, alpha=0.9)
            over = T >= d["t"][-1]
            if over and not d["r"].finished:
                ax.plot(*d["P"][-1], "X", color=RED, ms=14, mec="white", mew=1.5, zorder=6)
                status, scol = f"spun after {d['r'].distance:.0f} m", RED
            elif over:
                status, scol = f"lap finished in {d['r'].lap_time:.1f} s", INK
            else:
                ax.plot(*d["P"][k], marker=(3, 0, np.rad2deg(d["h"][k]) - 90), ms=11,
                        color=d["color"], mec="white", mew=1, zorder=6)
                status, scol = f"{d['X'][k, 3] * 3.6:.0f} km/h", INK
            side.plot([0.02, 0.1], [y - 0.01, y - 0.01], color=d["color"], lw=4,
                      transform=side.transAxes)
            side.text(0.13, y, d["label"], transform=side.transAxes, fontsize=9, color=INK,
                      va="center", wrap=True)
            side.text(0.13, y - 0.09, status, transform=side.transAxes, fontsize=10.5,
                      color=scol, va="center", fontweight="bold")
            y -= 0.3
        side.text(0.02, 0.12, f"t = {T:5.1f} s   ({SPEEDUP}x speed)\nroad friction {MU_TRUE}",
                  transform=side.transAxes, fontsize=9.5, color=MUTED, va="center")
        fig.suptitle("Real-time NMPC on a low-grip road", x=0.02, y=0.98, ha="left",
                     fontweight="bold", fontsize=11, color=INK)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    durations = [FRAME_MS] * len(frames)
    durations[-1] = 3000
    images = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
              for f in frames]
    out = ROOT / "docs" / "figures" / "demo.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"{len(frames)} frames; saved {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    draw(*run())
