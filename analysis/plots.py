"""Shared plotting helpers: barycentric simplex scatter (the program's
standard rendering: RGB = belief components), used from Phase 1 onward."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Vertices of the 2-simplex in the plane (equilateral triangle).
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def to_xy(beliefs: np.ndarray) -> np.ndarray:
    """(n, 3) simplex points -> (n, 2) barycentric plane coordinates."""
    return np.asarray(beliefs) @ _V


def simplex_scatter(ax, beliefs, colors=None, s=1.0, alpha=0.5, title=None):
    """Scatter beliefs in the triangle; default color = RGB(belief)."""
    xy = to_xy(beliefs)
    c = np.clip(np.asarray(beliefs), 0, 1) if colors is None else colors
    ax.scatter(xy[:, 0], xy[:, 1], c=c, s=s, alpha=alpha, linewidths=0)
    tri = np.vstack([_V, _V[0]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=0.8)
    for i, lbl in enumerate(["s0", "s1", "s2"]):
        off = (_V[i] - _V.mean(axis=0)) * 0.12
        ax.annotate(lbl, _V[i] + off, ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax
