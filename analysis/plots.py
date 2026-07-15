"""Reusable plotting helpers for three-component simplex data."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_VERTICES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]
)


def to_xy(
    points: np.ndarray,
    *,
    vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Map ``(n, 3)`` simplex points to two-dimensional coordinates."""
    vertices = DEFAULT_VERTICES if vertices is None else np.asarray(vertices)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("simplex points must have shape (n, 3)")
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")
    return points @ vertices


def simplex_scatter(
    ax,
    points,
    colors=None,
    s=1.0,
    alpha=0.5,
    title=None,
    *,
    labels=None,
    vertices=None,
):
    """Scatter three-component simplex points in a triangle."""
    vertices = (
        DEFAULT_VERTICES if vertices is None else np.asarray(vertices)
    )
    points = np.asarray(points)
    xy = to_xy(points, vertices=vertices)
    c = np.clip(points, 0, 1) if colors is None else colors
    ax.scatter(xy[:, 0], xy[:, 1], c=c, s=s, alpha=alpha, linewidths=0)
    tri = np.vstack([vertices, vertices[0]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=0.8)
    if labels is not None:
        if len(labels) != 3:
            raise ValueError("labels must contain three entries")
        for i, label in enumerate(labels):
            offset = (vertices[i] - vertices.mean(axis=0)) * 0.12
            ax.annotate(
                label,
                vertices[i] + offset,
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax
