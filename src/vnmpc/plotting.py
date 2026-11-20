"""Shared figure style (matplotlib)."""
from __future__ import annotations

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
CRITICAL = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b3"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": "#ebeae6",
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def speed_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("speed", BLUE_RAMP[1:])


def track_edges(track):
    """Left and right road edges (x, y) for drawing."""
    s = np.r_[track.s, track.s[:1]]
    left = track.frenet_to_xy(s, np.full_like(s, track.half_width + 1.0))
    right = track.frenet_to_xy(s, np.full_like(s, -track.half_width - 1.0))
    return left, right


def draw_track(ax, track):
    (lx, ly), (rx, ry) = track_edges(track)
    ax.fill(np.r_[lx, rx[::-1]], np.r_[ly, ry[::-1]], color="#ebeae6", lw=0, zorder=0)
    ax.plot(lx, ly, color=MUTED, lw=0.8, zorder=1)
    ax.plot(rx, ry, color=MUTED, lw=0.8, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")
