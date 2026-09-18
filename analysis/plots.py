"""Reusable plotting helpers for three-component simplex data."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.probes.controls import score_prediction

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


def _plot_array(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numbers") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _plot_limits(values, *, equal=False):
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    center = low / 2 + high / 2
    half_width = high / 2 - low / 2
    if equal:
        half_width = np.full_like(half_width, np.max(half_width))
    half_width = np.where(half_width > 0, half_width, np.maximum(np.abs(center) * 0.05, 0.5))
    with np.errstate(over="ignore", invalid="ignore"):
        limits = np.stack([center - half_width * 1.3, center + half_width * 1.3], axis=-1)
    if not np.isfinite(limits).all():
        raise ValueError("plot limits exceed the finite float64 range")
    return limits


def plot_belief_comparison(
    targets,
    predicted,
    *,
    state_labels=None,
    coordinates=None,
    point_colors=None,
    title=None,
    axes=None,
    seed=42,
) -> Figure:
    targets = _plot_array(targets, "targets")
    predicted = _plot_array(predicted, "predicted")
    if targets.ndim != 2 or targets.shape[0] == 0 or targets.shape[1] < 2:
        raise ValueError("targets must have nonempty shape (n, n_states), with at least two states")
    if predicted.shape != targets.shape:
        raise ValueError("predicted and targets must have matching shapes")
    if np.any((targets < 0) | (targets > 1)):
        raise ValueError("targets must contain probabilities in [0, 1]")
    for name, values in (("targets", targets), ("predicted", predicted)):
        if not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0):
            raise ValueError(f"{name} rows must sum to 1 within absolute tolerance 1e-8")
    n, n_states = targets.shape
    if state_labels is None:
        state_labels = [f"State {index}" for index in range(n_states)]
    elif isinstance(state_labels, str) or len(state_labels) != n_states:
        raise ValueError("state_labels must contain one label per state")
    state_labels = [str(label) for label in state_labels]
    if any(not label.strip() for label in state_labels):
        raise ValueError("state_labels must be nonempty")
    supplied_projection = coordinates is not None
    if supplied_projection:
        vertices = _plot_array(coordinates, "coordinates")
        if vertices.ndim != 2 or vertices.shape[0] != n_states or vertices.shape[1] not in (2, 3):
            raise ValueError("coordinates must have shape (n_states, 2) or (n_states, 3)")
    elif n_states == 2:
        vertices = np.array([[0.0, 0.0], [1.0, 0.0]])
    elif n_states == 3:
        vertices = DEFAULT_VERTICES
    elif n_states == 4:
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
        ])
    else:
        raise ValueError("explicit coordinates are required for more than four states; supply a projection or caller-selected marginals")
    dimension = vertices.shape[1]
    full_simplex = np.linalg.matrix_rank(vertices[1:] - vertices[0]) == n_states - 1
    if point_colors is None:
        palette = np.eye(3) if n_states == 3 else matplotlib.colormaps["turbo"](
            np.linspace(0.05, 0.95, n_states)
        )[:, :3]
        colors = targets @ palette
    else:
        colors = _plot_array(point_colors, "point_colors")
        if colors.ndim != 2 or colors.shape[0] != n or colors.shape[1] not in (3, 4):
            raise ValueError("point_colors must have shape (n, 3) or (n, 4)")
        if np.any((colors < 0) | (colors > 1)):
            raise ValueError("point_colors must contain RGB or RGBA values in [0, 1]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    order = np.random.default_rng(seed).permutation(n)
    with np.errstate(over="ignore", invalid="ignore"):
        target_points = targets @ vertices
        predicted_points = predicted @ vertices
    if not np.isfinite(target_points).all() or not np.isfinite(predicted_points).all():
        raise ValueError("embedded coordinates exceed the finite float64 range")
    limits = _plot_limits(np.vstack([vertices, target_points, predicted_points]), equal=True)
    score = score_prediction(predicted, targets)
    outside_fraction = np.mean(np.any((predicted < -1e-8) | (predicted > 1 + 1e-8), axis=1))
    sum_error = np.max(np.abs(predicted.sum(axis=1) - 1))
    if axes is None:
        figure, axes = plt.subplots(
            1, 2, figsize=(11, 6.5), dpi=160,
            subplot_kw={"projection": "3d"} if dimension == 3 else {},
        )
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) != 2 or any(not isinstance(ax, Axes) for ax in axes):
            raise ValueError("axes must contain exactly two Matplotlib axes")
        if axes[0] is axes[1] or axes[0].figure is not axes[1].figure:
            raise ValueError("axes must be distinct and belong to the same figure")
        expected_name = "3d" if dimension == 3 else "rectilinear"
        if any(ax.name != expected_name for ax in axes):
            raise ValueError(f"axes must both use the {expected_name} projection")
        figure = axes[0].figure
    figure.set_layout_engine(None)
    positions = [ax.get_position(original=True).frozen() for ax in axes]
    left, bottom = min(pos.x0 for pos in positions), min(pos.y0 for pos in positions)
    width = max(pos.x1 for pos in positions) - left
    height = max(pos.y1 for pos in positions) - bottom
    for ax, position in zip(axes, positions):
        ax.set_position([
            0.04 + 0.92 * (position.x0 - left) / width,
            0.25 + 0.59 * (position.y0 - bottom) / height,
            0.92 * position.width / width,
            0.59 * position.height / height,
        ])
    for ax, points, panel_title in zip(axes, (target_points, predicted_points), ("True targets", "Raw predictions")):
        scatter_kwargs = {"depthshade": False} if dimension == 3 else {}
        ax.scatter(*points[order].T, c=colors[order], s=12, linewidths=0, **scatter_kwargs)
        if full_simplex:
            for first, second in combinations(range(n_states), 2):
                ax.plot(*vertices[[first, second]].T, color="0.35", linewidth=0.8)
        ax.scatter(*vertices.T, c="0.2", s=12, **scatter_kwargs)
        for vertex, label in zip(vertices, state_labels):
            position = vertex + (vertex - vertices.mean(axis=0)) * 0.1
            ax.text(*position, label, ha="center", va="center", fontsize=9)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        if dimension == 3:
            ax.set_zlim(limits[2])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-58)
        else:
            ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(panel_title, fontsize=12, pad=12)
    figure.suptitle(title if title is not None else "Belief comparison", fontsize=15, y=0.96)
    mse = "N/A" if score["mse"] is None else f"{score['mse']:.6g}"
    r_squared = "N/A" if score["r_squared"] is None else f"{score['r_squared']:.6g}"
    figure.text(0.5, 0.17, f"Raw MSE = {mse}    |    R² = {r_squared}    |    n = {n}", ha="center", va="center", fontsize=11)
    figure.text(0.5, 0.115, f"Outside-simplex fraction = {outside_fraction:.6g}    |    Max sum error = {sum_error:.6g}", ha="center", va="center", fontsize=10)
    geometry = (
        f"Supplied projection ({dimension}D; {n_states} states), not a full simplex"
        if not full_simplex else f"Full simplex: affine dimension {n_states - 1} ({n_states} states)"
    )
    color_label = "Caller-supplied paired colors" if point_colors is not None else "Colors from true targets"
    figure.text(0.5, 0.055, f"{geometry}\n{color_label}; predictions are not clipped or renormalized.", ha="center", va="center", fontsize=9)
    return figure


def plot_learning_curve(
    steps,
    values,
    *,
    reference=None,
    reference_label=None,
    reference_interval=None,
    intervals=None,
    title=None,
    ylabel="Task metric",
    require_initialization=True,
    xscale="linear",
) -> Figure:
    steps = _plot_array(steps, "steps")
    values = _plot_array(values, "values")
    if steps.ndim != 1 or len(steps) == 0 or values.shape != steps.shape:
        raise ValueError("steps and values must be aligned, nonempty one-dimensional arrays")
    if np.any(steps < 0) or len(np.unique(steps)) != len(steps):
        raise ValueError("steps must be nonnegative and unique")
    if not isinstance(require_initialization, (bool, np.bool_)):
        raise ValueError("require_initialization must be a boolean")
    if require_initialization and not np.any(steps == 0):
        raise ValueError("an actual step-zero initialization measurement is required")
    if xscale not in ("linear", "symlog"):
        raise ValueError("xscale must be linear or symlog to support step zero")
    if intervals is not None:
        intervals = _plot_array(intervals, "intervals")
        if intervals.shape != (len(steps), 2):
            raise ValueError("intervals must have shape (n, 2)")
        if np.any(intervals[:, 0] > values) or np.any(intervals[:, 1] < values):
            raise ValueError("intervals must be ordered and contain their corresponding values")
    if reference is None:
        if reference_label is not None or reference_interval is not None:
            raise ValueError("reference_label and reference_interval require a reference")
    else:
        reference = _plot_array(reference, "reference")
        if reference.ndim != 0:
            raise ValueError("reference must be a finite scalar")
        reference = float(reference)
        if not isinstance(reference_label, str) or not reference_label.strip():
            raise ValueError("a reference requires an explicit nonempty reference_label")
        if reference_interval is not None:
            reference_interval = _plot_array(reference_interval, "reference_interval")
            if reference_interval.shape != (2,):
                raise ValueError("reference_interval must contain two endpoints")
            if not reference_interval[0] <= reference <= reference_interval[1]:
                raise ValueError("reference_interval must be ordered and contain the reference")
    order = np.argsort(steps, kind="stable")
    steps, values = steps[order], values[order]
    y_extent = [values]
    if intervals is not None:
        intervals = intervals[order]
        y_extent.append(intervals.ravel())
    if reference is not None:
        y_extent.append(np.array([reference]))
    if reference_interval is not None:
        y_extent.append(reference_interval)
    y_limits = _plot_limits(np.concatenate(y_extent))
    figure, ax = plt.subplots(figsize=(8.8, 5.5), dpi=160, layout="constrained")
    ax.plot(steps, values, marker="o", markersize=5, linewidth=1.5, color="C0", label="Measured checkpoints")
    if intervals is not None:
        ax.errorbar(steps, values, yerr=np.stack([values - intervals[:, 0], intervals[:, 1] - values]), fmt="none", color="C0", capsize=3, label="Supplied point intervals")
    if steps[0] == 0:
        ax.plot(steps[:1], values[:1], marker="D", markersize=6, linestyle="none", color="black", label="Initialization (step 0)")
    if reference is not None:
        ax.axhline(reference, linestyle="--", color="C1", linewidth=1.2, label=reference_label)
    if reference_interval is not None:
        ax.axhspan(*reference_interval, color="C1", alpha=0.15, label=f"{reference_label} interval")
    ax.set_xscale(xscale)
    ax.set_ylim(y_limits)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else "Learning curve")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    return figure
