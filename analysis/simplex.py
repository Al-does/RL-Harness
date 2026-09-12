"""Component-block metrics and offline viewers for aligned belief probes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np

from analysis.probes.controls import score_prediction


def _array(value: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _components(components: Mapping[str, Sequence[int]], width: int) -> dict[str, list[int]]:
    if not components or any(not isinstance(name, str) or not name for name in components):
        raise ValueError("components must have nonempty string names")
    result = {}
    for name, indices in components.items():
        values = np.asarray(indices)
        if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
            raise ValueError("component indices must be nonempty integer vectors")
        result[name] = values.tolist()
    flattened = [index for indices in result.values() for index in indices]
    if sorted(flattened) != list(range(width)):
        raise ValueError("components must partition all belief coordinates exactly once")
    return result


def geometry_metrics(
    prediction: np.ndarray,
    beliefs: np.ndarray,
    *,
    components: Mapping[str, Sequence[int]],
    coordinate_tolerance: float = 1e-7,
    sum_tolerance: float = 1e-6,
) -> dict:
    """Score raw affine coordinates and their component masses without projection."""
    prediction = _array(prediction, "prediction")
    beliefs = _array(beliefs, "beliefs")
    if beliefs.ndim != 2 or not all(beliefs.shape) or prediction.shape != beliefs.shape:
        raise ValueError("prediction and beliefs must have equal nonempty (rows, states) shapes")
    groups = _components(components, beliefs.shape[1])
    if any(not np.isfinite(t) or t < 0 for t in (coordinate_tolerance, sum_tolerance)):
        raise ValueError("simplex tolerances must be finite and nonnegative")
    return {
        **score_prediction(prediction, beliefs),
        "outside_simplex_fraction": float(np.mean(
            (prediction.min(axis=1) < -coordinate_tolerance)
            | (prediction.max(axis=1) > 1 + coordinate_tolerance)
            | (np.abs(prediction.sum(axis=1) - 1) > sum_tolerance)
        )),
        "coordinate_min": float(prediction.min()),
        "coordinate_max": float(prediction.max()),
        "mass_sum_max_error": float(np.abs(prediction.sum(axis=1) - 1).max()),
        "component_posterior": score_prediction(
            np.stack([prediction[:, indices].sum(axis=1) for indices in groups.values()], axis=1),
            np.stack([beliefs[:, indices].sum(axis=1) for indices in groups.values()], axis=1),
        ),
    }


def _indices(values: Sequence[int], size: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if (
        result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
        or (result < 0).any() or (result >= size).any()
        or len(np.unique(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct in-range integer indices")
    return result


def _text_grid(values: Sequence[Sequence[str]], shape: tuple[int, int], name: str) -> list:
    if len(values) != shape[0] or any(len(row) != shape[1] for row in values):
        raise ValueError(f"{name} must match (episodes, positions)")
    if any(not isinstance(value, str) for row in values for value in row):
        raise ValueError(f"{name} must contain strings")
    return [list(row) for row in values]


def build_simplex_run(
    *,
    name: str,
    targets: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    components: Mapping[str, Sequence[int]],
    primary_site: str,
    cloud_rows: Sequence[int],
    example_episodes: Sequence[int],
    tokens: Sequence[Sequence[str]] | None = None,
    state_labels: Sequence[str] | None = None,
    description: str = "",
    episode_labels: Sequence[str] | None = None,
    position_notes: Sequence[Sequence[str]] | None = None,
) -> dict:
    """Build one run from complete held-out episodes and matched raw predictions.

    Arrays have shape (episodes, positions, states). Metrics use every supplied
    row; cloud/example selection affects display only. Components may have any
    names/order, but the 3D viewer requires exactly three coordinates per block.
    Token labels and optional position notes are experiment-supplied display
    text. No action, BOS, terminal, or target-timing convention is inferred.
    """
    targets = _array(targets, "targets")
    if targets.ndim != 3 or not all(targets.shape):
        raise ValueError("targets must have nonempty (episodes, positions, states) shape")
    if (targets < 0).any() or not np.allclose(targets.sum(axis=2), 1, rtol=0, atol=1e-8):
        raise ValueError("targets must contain probability rows")
    groups = _components(components, targets.shape[2])
    if any(len(indices) != 3 for indices in groups.values()):
        raise ValueError("the 3D viewer requires three coordinates per component")
    episodes, positions, width = targets.shape
    cloud = _indices(cloud_rows, episodes * positions, "cloud_rows")
    examples = _indices(example_episodes, episodes, "example_episodes")
    token_labels = (
        [[str(t) for t in range(positions)] for _ in range(episodes)] if tokens is None
        else _text_grid(tokens, (episodes, positions), "tokens")
    )
    states = list(state_labels) if state_labels is not None else [
        f"State {index}" for index in range(width)
    ]
    if len(states) != width or any(not isinstance(label, str) for label in states):
        raise ValueError("state_labels must contain one string per state coordinate")
    notes = (
        [[""] * positions for _ in range(episodes)] if position_notes is None
        else _text_grid(position_notes, (episodes, positions), "position_notes")
    )
    labels = list(episode_labels) if episode_labels is not None else [
        f"Episode {index}" for index in range(episodes)
    ]
    if len(labels) != episodes or any(not isinstance(label, str) for label in labels):
        raise ValueError("episode_labels must contain one string per episode")
    if not isinstance(name, str) or not name or not isinstance(description, str):
        raise ValueError("name and description must be strings; name cannot be empty")
    if not predictions:
        raise ValueError("at least one checkpoint is required")
    sites = list(next(iter(predictions.values())))
    if not sites or primary_site not in sites or any(not isinstance(site, str) or not site for site in sites):
        raise ValueError("sites must be nonempty names and include primary_site")
    packed, metrics = {}, {}
    for checkpoint, values in predictions.items():
        if not isinstance(checkpoint, str) or not checkpoint or set(values) != set(sites):
            raise ValueError("checkpoints must have nonempty names and the same representation sites")
        packed[checkpoint], metrics[checkpoint] = {}, {}
        for site, value in values.items():
            value = _array(value, f"{checkpoint}/{site}")
            if value.shape != targets.shape:
                raise ValueError("predictions must match the complete target episode shape")
            flattened = value.reshape(-1, width)
            packed[checkpoint][site] = {
                "cloud": flattened[cloud].tolist(),
                "sequences": value[examples].tolist(),
            }
            metrics[checkpoint][site] = geometry_metrics(
                flattened, targets.reshape(-1, width), components=groups,
            )
    return {
        "name": name, "description": description,
        "components": [
            {"name": label, "indices": indices, "state_labels": [states[index] for index in indices]}
            for label, indices in groups.items()
        ],
        "sites": sites, "primary_site": primary_site,
        "checkpoints": list(predictions), "metrics": metrics,
        "cloud": {"targets": targets.reshape(-1, width)[cloud].tolist(), "rows": cloud.tolist()},
        "sequences": [
            {
                "id": int(index), "label": labels[index],
                "tokens": token_labels[index], "notes": notes[index],
                "targets": targets[index].tolist(),
            }
            for index in examples
        ],
        "predictions": packed,
    }


def write_simplex_viewer(
    output: Path,
    runs: Sequence[dict],
    *,
    title: str = "Nonergodic belief geometry",
    description: str = "",
    reports: Mapping[str, dict] | None = None,
) -> Path:
    """Bundle build_simplex_run results and Plotly for offline file:// use.

    Refuses to overwrite any viewer-owned file. An existing directory may hold
    experiment-owned raw arrays and reports; these are never changed.
    """
    if not runs:
        raise ValueError("at least one run is required")
    payload = {"schema": 2, "title": title, "description": description, "runs": list(runs)}
    report_files = {
        f"report_{index}.json": json.dumps(report, indent=2, allow_nan=False) + "\n"
        for index, report in enumerate((reports or {}).values())
    }
    payload["reports"] = [
        {"name": name, "path": filename}
        for name, filename in zip(reports or {}, report_files)
    ]
    data = "window.SIMPLEX_DATA=" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n"
    assets = files("analysis").joinpath("simplex_viewer")
    contents = {name: assets.joinpath(name).read_text() for name in ("index.html", "viewer.js", "style.css")}
    try:
        contents["plotly.min.js"] = files("plotly").joinpath("package_data/plotly.min.js").read_text()
    except ModuleNotFoundError as error:
        raise RuntimeError("Install rl-harness[visualization] to export a simplex viewer") from error
    contents.update({"data.js": data, **report_files})
    output = Path(output)
    if any((output / name).exists() for name in contents):
        raise FileExistsError("viewer output files already exist; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (output / name).write_text(content, encoding="utf-8")
    return output / "index.html"
