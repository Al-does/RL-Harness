from io import BytesIO
from itertools import combinations

import numpy as np
import pytest

from analysis import plots
from analysis.plots import (
    DEFAULT_VERTICES,
    plot_belief_comparison,
    plot_learning_curve,
    simplex_scatter,
    to_xy,
)
from analysis.probes.controls import score_prediction

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def figure_text(figure):
    return "\n".join(text.get_text() for text in figure.texts)


def png_bytes(figure):
    output = BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()


def assert_contains(limits, values):
    assert limits[0] < np.min(values)
    assert limits[1] > np.max(values)


def test_existing_triangle_helpers_remain_compatible():
    targets = np.eye(3)
    np.testing.assert_array_equal(to_xy(targets), DEFAULT_VERTICES)
    figure, ax = plt.subplots()
    assert simplex_scatter(ax, targets, labels=["A", "B", "C"]) is ax
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), DEFAULT_VERTICES)
    with pytest.raises(ValueError, match="shape"):
        to_xy(np.eye(4))
    assert isinstance(figure, Figure)


def test_three_state_comparison_preserves_raw_points_colors_metrics_and_limits():
    targets = np.array([[0.7, 0.1, 0.2], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    predicted = np.array([[-12.0, 15.0, -2.0], [0.2, 0.3, 0.5], [3.0, -1.0, -1.0]])
    original_targets, original_predicted = targets.copy(), predicted.copy()
    figure = plot_belief_comparison(targets, predicted, state_labels=["A", "B", "C"], seed=8)
    order = np.random.default_rng(8).permutation(3)
    assert isinstance(figure, Figure)
    assert figure.dpi == 160
    assert len(figure.axes) == 2
    left, right = figure.axes
    np.testing.assert_allclose(left.collections[0].get_offsets(), targets[order] @ DEFAULT_VERTICES)
    np.testing.assert_allclose(right.collections[0].get_offsets(), predicted[order] @ DEFAULT_VERTICES)
    np.testing.assert_array_equal(left.collections[0].get_facecolors(), right.collections[0].get_facecolors())
    np.testing.assert_allclose(left.collections[0].get_facecolors()[:, :3], targets[order])
    assert left.get_xlim() == right.get_xlim()
    assert left.get_ylim() == right.get_ylim()
    all_points = np.vstack([targets @ DEFAULT_VERTICES, predicted @ DEFAULT_VERTICES, DEFAULT_VERTICES])
    assert_contains(left.get_xlim(), all_points[:, 0])
    assert_contains(left.get_ylim(), all_points[:, 1])
    for ax in figure.axes:
        assert [label.get_text() for label in ax.texts] == ["A", "B", "C"]
        assert len(ax.lines) == 3
        assert ax.get_aspect() == 1
    score = score_prediction(predicted, targets)
    text = figure_text(figure)
    assert f"Raw MSE = {score['mse']:.6g}" in text
    assert f"R² = {score['r_squared']:.6g}" in text
    assert "Outside-simplex fraction = 0.666667" in text
    assert "Max sum error = 0" in text
    assert "affine dimension 2 (3 states)" in text
    np.testing.assert_array_equal(targets, original_targets)
    np.testing.assert_array_equal(predicted, original_predicted)


def test_four_states_use_a_full_tetrahedron_with_six_edges_and_raw_outliers():
    targets = np.vstack([np.eye(4), [0.1, 0.2, 0.3, 0.4]])
    predicted = targets.copy()
    predicted[-1] = [-7, 2, -4, 10]
    figure = plot_belief_comparison(targets, predicted, state_labels=["A", "B", "C", "D"])
    left, right = figure.axes
    assert left.name == right.name == "3d"
    vertices = np.asarray(np.column_stack(left.collections[1]._offsets3d))
    assert np.linalg.matrix_rank(vertices[1:] - vertices[0]) == 3
    np.testing.assert_allclose([np.linalg.norm(vertices[i] - vertices[j]) for i, j in combinations(range(4), 2)], 1)
    order = np.random.default_rng(42).permutation(len(targets))
    np.testing.assert_allclose(np.column_stack(left.collections[0]._offsets3d), targets[order] @ vertices)
    np.testing.assert_allclose(np.column_stack(right.collections[0]._offsets3d), predicted[order] @ vertices)
    np.testing.assert_array_equal(left.collections[0]._facecolors, right.collections[0]._facecolors)
    for ax in figure.axes:
        assert len(ax.lines) == 6
        assert len(ax.texts) == 4
        assert not ax.collections[0].get_depthshade()
        np.testing.assert_allclose(ax.get_box_aspect(), np.repeat(ax.get_box_aspect()[0], 3))
    assert (left.elev, left.azim) == (right.elev, right.azim)
    all_points = np.vstack([vertices, targets @ vertices, predicted @ vertices])
    for dimension, getter in enumerate(("get_xlim", "get_ylim", "get_zlim")):
        assert getattr(left, getter)() == getattr(right, getter)()
        assert_contains(getattr(left, getter)(), all_points[:, dimension])
    text = figure_text(figure)
    assert "affine dimension 3 (4 states)" in text
    assert "projection" not in text.lower()
    assert "Outside-simplex fraction = 0.2" in text
    assert png_bytes(figure).startswith(b"\x89PNG\r\n\x1a\n")


def test_two_states_use_a_line_without_discarding_either_probability():
    targets = np.array([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
    predicted = np.array([[1.2, -0.2], [-0.2, 1.2], [0.3, 0.7]])
    figure = plot_belief_comparison(targets, predicted)
    order = np.random.default_rng(42).permutation(3)
    right = figure.axes[1]
    np.testing.assert_allclose(right.collections[0].get_offsets()[:, 0], predicted[order, 1])
    np.testing.assert_array_equal(right.collections[0].get_offsets()[:, 1], 0)
    assert len(right.lines) == 1
    assert "affine dimension 1" in figure_text(figure)


@pytest.mark.parametrize("dimension", [2, 3])
def test_supplied_projection_uses_every_state_and_is_labeled_honestly(dimension):
    targets = np.eye(7)
    predicted = np.roll(targets, 1, axis=1)
    vertices = np.arange(7 * dimension, dtype=float).reshape(7, dimension)
    vertices[:, 1] = np.arange(7) ** 2
    figure = plot_belief_comparison(targets, predicted, coordinates=vertices)
    order = np.random.default_rng(42).permutation(7)
    for ax, values in zip(figure.axes, (targets, predicted)):
        observed = np.column_stack(ax.collections[0]._offsets3d) if dimension == 3 else ax.collections[0].get_offsets()
        np.testing.assert_allclose(observed, values[order] @ vertices)
        assert len(ax.texts) == 7
        assert not ax.lines
    assert f"Supplied projection ({dimension}D; 7 states), not a full simplex" in figure_text(figure)
    assert f"Raw MSE = {score_prediction(predicted, targets)['mse']:.6g}" in figure_text(figure)


def test_explicit_four_state_two_dimensional_coordinates_are_only_a_projection():
    figure = plot_belief_comparison(np.eye(4), np.eye(4), coordinates=[[0, 0], [1, 0], [1, 1], [0, 1]])
    assert all(ax.name == "rectilinear" for ax in figure.axes)
    assert "Supplied projection (2D; 4 states), not a full simplex" in figure_text(figure)


def test_multifactor_structure_is_not_assumed_or_automatically_marginalized():
    first = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]])
    second = np.array([[0.3, 0.4, 0.3], [0.7, 0.1, 0.2]])
    joint = (first[:, :, None] * second[:, None, :]).reshape(2, 9)
    with pytest.raises(ValueError, match="explicit coordinates"):
        plot_belief_comparison(joint, joint)
    marginal = joint.reshape(2, 3, 3).sum(axis=2)
    figure = plot_belief_comparison(marginal, marginal)
    assert "3 states" in figure_text(figure)
    with pytest.raises(ValueError, match="sum to 1"):
        plot_belief_comparison(np.concatenate([first, second], axis=1), np.concatenate([first, second], axis=1))


@pytest.mark.parametrize("channels", [3, 4])
def test_caller_rgb_or_rgba_colors_are_paired_without_modification(channels):
    colors = np.linspace(0.1, 0.9, 3 * channels).reshape(3, channels)
    figure = plot_belief_comparison(np.eye(3), np.roll(np.eye(3), 1, axis=0), point_colors=colors)
    order = np.random.default_rng(42).permutation(3)
    for ax in figure.axes:
        np.testing.assert_array_equal(ax.collections[0].get_facecolors()[:, :channels], colors[order])
    assert "Caller-supplied paired colors" in figure_text(figure)


def test_zero_variance_targets_display_na_and_score_prediction_is_used(monkeypatch):
    targets = np.tile([0.2, 0.3, 0.5], (4, 1))
    predicted = np.tile([0.1, -0.1, 1.0], (4, 1))
    calls = []

    def recording_score(prediction, target):
        calls.append((prediction.copy(), target.copy()))
        return score_prediction(prediction, target)

    monkeypatch.setattr(plots, "score_prediction", recording_score)
    figure = plot_belief_comparison(targets, predicted)
    assert "R² = N/A" in figure_text(figure)
    assert "nan" not in figure_text(figure).lower()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], predicted)
    np.testing.assert_array_equal(calls[0][1], targets)


def test_sum_tolerance_does_not_renormalize_predictions():
    targets = np.eye(3)
    predicted = targets.copy()
    predicted[0, 1] = 5e-9
    figure = plot_belief_comparison(targets, predicted)
    order = np.random.default_rng(42).permutation(3)
    np.testing.assert_array_equal(figure.axes[1].collections[0].get_offsets(), predicted[order] @ DEFAULT_VERTICES)
    assert "Max sum error = 5e-09" in figure_text(figure)
    predicted[0, 1] = 2e-8
    with pytest.raises(ValueError, match="sum to 1"):
        plot_belief_comparison(targets, predicted)


@pytest.mark.parametrize("targets,predicted", [
    ([], []),
    ([1, 0, 0], [1, 0, 0]),
    (np.empty((0, 3)), np.empty((0, 3))),
    ([[1]], [[1]]),
    (np.eye(3), np.eye(4)),
    ([[0.5, 0.6]], [[0.5, 0.5]]),
    ([[0.5, 0.5]], [[0.5, 0.6]]),
    ([[-0.1, 1.1]], [[0.5, 0.5]]),
    ([[np.nan, 0.5]], [[0.5, 0.5]]),
    ([[0.5, 0.5]], [[np.nan, 0.5]]),
    ([[0.5, 0.5]], [[np.inf, -np.inf]]),
    ([[0.5 + 0j, 0.5]], [[0.5, 0.5]]),
    ([[0.5, 0.5]], [[0.5 + 1j, 0.5 - 1j]]),
])
def test_belief_comparison_rejects_invalid_arrays(targets, predicted):
    with pytest.raises(ValueError):
        plot_belief_comparison(targets, predicted)
    assert not plt.get_fignums()


@pytest.mark.parametrize("kwargs", [
    {"state_labels": ["A", "B"]},
    {"state_labels": "ABC"},
    {"state_labels": ["A", "", "C"]},
    {"coordinates": np.zeros((2, 2))},
    {"coordinates": np.zeros((3, 4))},
    {"coordinates": [[0, 0], [1, np.nan], [0, 1]]},
    {"point_colors": np.zeros((3, 2))},
    {"point_colors": np.zeros((2, 3))},
    {"point_colors": np.full((3, 3), 1.2)},
    {"point_colors": np.full((3, 4), np.nan)},
    {"seed": None},
    {"seed": -1},
    {"seed": True},
])
def test_belief_comparison_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        plot_belief_comparison(np.eye(3), np.eye(3), **kwargs)
    assert not plt.get_fignums()


@pytest.mark.parametrize("n_states", [3, 4])
@pytest.mark.parametrize("layout", [None, "constrained", "manual"])
def test_comparison_renders_into_caller_axes_and_returns_their_figure(n_states, layout):
    projection = {"projection": "3d"} if n_states == 4 else {}
    if layout == "manual":
        figure = plt.figure(figsize=(11, 6.5), dpi=160)
        axes = [figure.add_axes([left, 0.1, 0.35, 0.8], **projection) for left in (0.1, 0.55)]
    else:
        figure, axes = plt.subplots(1, 2, figsize=(11, 6.5), dpi=160, subplot_kw=projection, layout=layout)
    numbers = plt.get_fignums()
    returned = plot_belief_comparison(np.eye(n_states), np.eye(n_states), axes=axes)
    assert returned is figure
    assert plt.get_fignums() == numbers
    assert len(figure.axes) == 2
    assert all(len(ax.collections) == 2 for ax in axes)
    assert png_bytes(returned).startswith(b"\x89PNG\r\n\x1a\n")
    renderer = figure.canvas.get_renderer()
    for text in figure.texts[1:]:
        assert all(not text.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)) for ax in axes)


def test_comparison_rejects_incompatible_or_unpaired_axes():
    figure, axes = plt.subplots(1, 2)
    other, other_axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    for invalid in ([axes[0]], [axes[0], axes[0]], [axes[0], other_axes[0]], other_axes, [1, 2]):
        with pytest.raises(ValueError, match="axes"):
            plot_belief_comparison(np.eye(3), np.eye(3), axes=invalid)
    with pytest.raises(ValueError, match="projection"):
        plot_belief_comparison(np.eye(4), np.eye(4), axes=axes)
    assert all(not ax.collections for ax in figure.axes + other.axes)


@pytest.mark.parametrize("n_states", [3, 4])
def test_belief_titles_metrics_and_footnotes_do_not_overlap(n_states):
    figure = plot_belief_comparison(np.eye(n_states), np.eye(n_states), title="Generic held-out belief comparison")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = list(figure.texts) + [ax.title for ax in figure.axes]
    boxes = [text.get_window_extent(renderer) for text in texts]
    for first, second in combinations(boxes, 2):
        assert not first.overlaps(second)
    for box in boxes:
        assert figure.bbox.contains(box.x0, box.y0)
        assert figure.bbox.contains(box.x1, box.y1)
    for text in figure.texts[1:]:
        assert all(not text.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)) for ax in figure.axes)


@pytest.mark.parametrize("n_states", [3, 4])
def test_belief_plot_png_bytes_are_deterministic_and_do_not_use_global_randomness(n_states):
    targets = np.random.default_rng(4).dirichlet(np.ones(n_states), size=20)
    predicted = targets * 2 - 1 / n_states
    np.random.seed(11)
    expected = np.random.random(5)
    np.random.seed(11)
    first = plot_belief_comparison(targets, predicted)
    second = plot_belief_comparison(targets, predicted)
    np.testing.assert_array_equal(np.random.random(5), expected)
    assert png_bytes(first) == png_bytes(second)


def test_learning_curve_keeps_raw_units_sorts_pairs_and_does_not_cap_reference_overshoots():
    steps = np.array([100, 0, 10])
    values = np.array([81.0, 22.0, 74.0])
    original_steps, original_values = steps.copy(), values.copy()
    figure = plot_learning_curve(steps, values, reference=70.0, reference_label="Analytic expected reference", ylabel="Success (%)")
    ax = figure.axes[0]
    curve, initialization, reference = ax.lines
    np.testing.assert_array_equal(curve.get_xdata(), [0, 10, 100])
    np.testing.assert_array_equal(curve.get_ydata(), [22, 74, 81])
    np.testing.assert_array_equal(initialization.get_xdata(), [0])
    np.testing.assert_array_equal(initialization.get_ydata(), [22])
    np.testing.assert_array_equal(reference.get_ydata(), [70, 70])
    assert reference.get_label() == "Analytic expected reference"
    assert curve.get_marker() == "o"
    assert "Initialization (step 0)" in initialization.get_label()
    assert_contains(ax.get_ylim(), values)
    assert ax.get_xlabel() == "Training step"
    assert ax.get_ylabel() == "Success (%)"
    assert "Bayes" not in " ".join(text.get_text() for text in ax.legend_.texts)
    np.testing.assert_array_equal(steps, original_steps)
    np.testing.assert_array_equal(values, original_values)


def test_learning_curve_intervals_keep_endpoints_and_sort_with_measurements():
    figure = plot_learning_curve(
        [10, 0, 20], [0.9, 0.4, 1.1],
        intervals=[[0.8, 1.2], [-0.2, 0.7], [1.0, 1.5]],
        reference=0.8, reference_label="Formal reference",
        reference_interval=[0.75, 0.85],
    )
    ax = figure.axes[0]
    np.testing.assert_allclose(ax.collections[0].get_segments(), [
        [[0, -0.2], [0, 0.7]],
        [[10, 0.8], [10, 1.2]],
        [[20, 1.0], [20, 1.5]],
    ])
    assert_contains(ax.get_ylim(), [-0.2, 1.5])
    band = ax.patches[0]
    assert band.get_label() == "Formal reference interval"
    assert band.get_y() == 0.75
    assert band.get_height() == pytest.approx(0.1)
    assert "confidence" not in " ".join(text.get_text() for text in ax.legend_.texts).lower()


def test_learning_curve_reference_interval_is_included_in_limits():
    figure = plot_learning_curve([0, 1], [0.4, 0.6], reference=0.7, reference_label="Formal range", reference_interval=[-5, 8])
    assert_contains(figure.axes[0].get_ylim(), [-5, 8])


@pytest.mark.parametrize("xscale", ["linear", "symlog"])
def test_learning_curve_preserves_actual_zero_for_supported_scales(xscale):
    figure = plot_learning_curve([0, 1, 1000], [-20, 10, 40], xscale=xscale)
    ax = figure.axes[0]
    figure.canvas.draw()
    assert ax.get_xscale() == xscale
    assert np.isfinite(ax.transData.transform([[0, -20]])).all()
    assert ax.get_xlim()[0] <= 0 <= ax.get_xlim()[1]
    assert 0 in ax.get_xticks()
    assert ax.get_ylabel() == "Task metric"
    assert len(ax.lines) == 2
    assert_contains(ax.get_ylim(), [-20, 40])


def test_learning_curve_without_initialization_does_not_invent_a_measurement():
    with pytest.raises(ValueError, match="initialization"):
        plot_learning_curve([5, 10], [2, 3])
    figure = plot_learning_curve([5, 10], [2, 3], require_initialization=False)
    ax = figure.axes[0]
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [5, 10])
    assert "initialization" not in " ".join(text.get_text() for text in ax.legend_.texts).lower()


def test_single_initialization_measurement_has_finite_limits_and_no_assumed_maximum():
    figure = plot_learning_curve([0], [500])
    ax = figure.axes[0]
    assert_contains(ax.get_ylim(), [500])
    assert ax.get_ylim()[0] > 1
    assert np.isfinite(ax.get_xlim()).all()
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [500])


@pytest.mark.parametrize("steps,values,kwargs", [
    ([], [], {}),
    ([0, 1], [0.2], {}),
    ([[0, 1]], [[0.2, 0.3]], {}),
    ([0, 0], [0.2, 0.3], {}),
    ([-1, 0], [0.2, 0.3], {}),
    ([0, np.inf], [0.2, 0.3], {}),
    ([0, 1], [0.2, np.nan], {}),
    ([0, 1], [0.2, 0.3j], {}),
    ([0, 1], [0.2, 0.3], {"xscale": "log"}),
    ([0, 1], [0.2, 0.3], {"require_initialization": 1}),
    ([0, 1], [0.2, 0.3], {"reference": 1}),
    ([0, 1], [0.2, 0.3], {"reference": 1, "reference_label": "  "}),
    ([0, 1], [0.2, 0.3], {"reference": [1], "reference_label": "Ref"}),
    ([0, 1], [0.2, 0.3], {"reference": np.inf, "reference_label": "Ref"}),
    ([0, 1], [0.2, 0.3], {"reference_label": "Ref"}),
    ([0, 1], [0.2, 0.3], {"reference_interval": [0, 1]}),
    ([0, 1], [0.2, 0.3], {"reference": 1, "reference_label": "Ref", "reference_interval": [2, 0]}),
    ([0, 1], [0.2, 0.3], {"reference": 1, "reference_label": "Ref", "reference_interval": [0, 0.5]}),
    ([0, 1], [0.2, 0.3], {"reference": 1, "reference_label": "Ref", "reference_interval": [0, np.nan]}),
    ([0, 1], [0.2, 0.3], {"reference": 1, "reference_label": "Ref", "reference_interval": [0, 1, 2]}),
    ([0, 1], [0.2, 0.3], {"intervals": [[0, 1]]}),
    ([0, 1], [0.2, 0.3], {"intervals": [[0.5, 0.1], [0, 1]]}),
    ([0, 1], [0.2, 0.3], {"intervals": [[0.3, 0.4], [0, 1]]}),
    ([0, 1], [0.2, 0.3], {"intervals": [[0, 0.1], [0, 1]]}),
    ([0, 1], [0.2, 0.3], {"intervals": [[0, np.nan], [0, 1]]}),
])
def test_learning_curve_rejects_invalid_inputs(steps, values, kwargs):
    with pytest.raises(ValueError):
        plot_learning_curve(steps, values, **kwargs)
    assert not plt.get_fignums()


def test_learning_curve_png_is_deterministic_and_owned_by_caller():
    kwargs = {"reference": 20, "reference_label": "Expected reference", "intervals": [[-12, -8], [25, 35]]}
    first = plot_learning_curve([0, 5], [-10, 30], **kwargs)
    second = plot_learning_curve([0, 5], [-10, 30], **kwargs)
    output = png_bytes(first)
    assert output.startswith(b"\x89PNG\r\n\x1a\n")
    assert output == png_bytes(second)
    assert plt.fignum_exists(first.number)
    plt.close(first)
    assert not plt.fignum_exists(first.number)


def test_custom_injective_vertices_are_a_full_simplex_and_ignore_roundoff():
    target = np.eye(3)
    prediction = target + np.array([-1e-10, 1e-10, 0])
    figure = plot_belief_comparison(target, prediction, coordinates=[[0, 0], [0.5, 1], [1, 0]])
    assert "Full simplex: affine dimension 2" in figure_text(figure)
    assert "Outside-simplex fraction = 0" in figure_text(figure)
    assert all(len(axis.lines) == 3 for axis in figure.axes)
    assert prediction.min() < 0
