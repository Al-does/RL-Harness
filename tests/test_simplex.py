from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.belief_geometry import evaluate_belief_geometry
from analysis.simplex import build_simplex_run, geometry_metrics, write_simplex_viewer


@pytest.fixture
def inputs():
    targets = np.random.default_rng(7).dirichlet(np.ones(9), size=(3, 5))
    targets[0, -1] = [1 - 1e-26, 0, 0, 1e-26, 0, 0, 0, 0, 0]
    prediction = targets.copy()
    prediction[2, -1, :2] = [1.4, -0.4]
    return {
        "name": "Passive observations",
        "targets": targets,
        "predictions": {"epoch 0": {"embedding": targets}, "epoch 7": {"embedding": prediction}},
        "components": {"warm": [2, 0, 1], "cool": [3, 4, 5], "other": [6, 7, 8]},
        "primary_site": "embedding",
        "tokens": [["BOS", "red", "red", "blue", "green"]] * 3,
        "cloud_rows": [0, 4, 10],
        "example_episodes": [0, 2],
    }


def test_paired_sampling_preserves_raw_values_and_scores_all_rows(inputs):
    before = inputs["predictions"]["epoch 7"]["embedding"].copy()
    run = build_simplex_run(**inputs)
    assert run["components"][0] == {
        "name": "warm", "indices": [2, 0, 1], "state_labels": ["State 2", "State 0", "State 1"],
    }
    assert run["sequences"][0]["notes"] == [""] * 5
    assert "actions" not in run["sequences"][0]
    assert run["cloud"]["targets"][1][3] == 1e-26
    assert [sequence["id"] for sequence in run["sequences"]] == [0, 2]
    assert run["predictions"]["epoch 7"]["embedding"]["sequences"][1][-1][:2] == [1.4, -0.4]
    np.testing.assert_array_equal(inputs["predictions"]["epoch 7"]["embedding"], before)
    for checkpoint, values in inputs["predictions"].items():
        np.testing.assert_array_equal(
            run["predictions"][checkpoint]["embedding"]["cloud"], values["embedding"].reshape(-1, 9)[[0, 4, 10]],
        )
    assert run["metrics"]["epoch 7"]["embedding"]["outside_simplex_fraction"] == pytest.approx(1 / 15)


@pytest.mark.parametrize("change, message", [
    ({"components": {"a": [0, 1, 2], "b": [2, 3, 4]}}, "partition"),
    ({"components": {"a": list(range(9))}}, "three coordinates"),
    ({"cloud_rows": [15]}, "cloud_rows"),
    ({"cloud_rows": [0, 0]}, "cloud_rows"),
    ({"example_episodes": [True]}, "example_episodes"),
    ({"example_episodes": [-1]}, "example_episodes"),
    ({"example_episodes": []}, "example_episodes"),
    ({"tokens": [["BOS"]]}, "tokens"),
    ({"tokens": [[0] * 5] * 3}, "strings"),
    ({"episode_labels": ["one"]}, "episode_labels"),
    ({"state_labels": ["one"]}, "state_labels"),
    ({"primary_site": "missing"}, "primary_site"),
    ({"predictions": {}}, "checkpoint"),
    ({"predictions": {"a": {"embedding": np.zeros((3, 4, 9))}}}, "shape"),
    ({"predictions": {"a": {"embedding": np.full((3, 5, 9), np.nan)}}}, "finite"),
    ({"predictions": {"a": {"embedding": np.zeros((3, 5, 9), dtype=complex)}}}, "real"),
    ({"targets": np.zeros((3, 5, 9))}, "probability"),
])
def test_rejects_unaligned_or_ambiguous_data(inputs, change, message):
    with pytest.raises(ValueError, match=message):
        build_simplex_run(**{**inputs, **change})


def test_checkpoint_sites_must_match(inputs):
    inputs["predictions"]["epoch 7"] = {"another": inputs["targets"]}
    with pytest.raises(ValueError, match="same representation"):
        build_simplex_run(**inputs)


def test_optional_labels_and_notes(inputs):
    run = build_simplex_run(**{
        **inputs, "tokens": None,
        "state_labels": [f"hidden {index}" for index in range(9)],
        "position_notes": [["no action"] * 5] * 3,
    })
    assert run["components"][0]["state_labels"] == ["hidden 2", "hidden 0", "hidden 1"]
    assert run["sequences"][0]["tokens"] == ["0", "1", "2", "3", "4"]
    assert run["sequences"][0]["notes"] == ["no action"] * 5


def test_metrics_support_general_component_sizes_and_constant_targets():
    target = np.tile([0.1, 0.2, 0.3, 0.4], (3, 1))
    prediction = target.copy()
    prediction[0, :2] = [0.5, -0.2]
    metrics = geometry_metrics(prediction, target, components={"one": [0, 1], "two": [2, 3]})
    assert metrics["coordinate_min"] == -0.2
    assert metrics["r_squared"] is None
    assert metrics["component_posterior"]["mse"] == pytest.approx(0)
    assert metrics["component_posterior"]["r_squared"] is None
    assert metrics["outside_simplex_fraction"] == pytest.approx(1 / 3)


def test_probe_battery_outputs_compose_with_viewer(inputs):
    train = np.random.default_rng(9).dirichlet(np.ones(9), size=(4, 5))
    target = inputs["targets"]
    battery = evaluate_belief_geometry(
        {"embedding": train.reshape(-1, 9)},
        {"embedding": target.reshape(-1, 9)},
        train.reshape(-1, 9), target.reshape(-1, 9),
        train_groups=np.repeat(np.arange(4), 5),
        test_groups=np.repeat(np.arange(4, 7), 5),
        n_null_repeats=1, n_resamples=5,
    )
    run = build_simplex_run(**{
        **inputs,
        "predictions": {"checkpoint": {"embedding": battery.predictions["embedding"].reshape(target.shape)}},
    })
    assert run["metrics"]["checkpoint"]["embedding"]["r_squared"] == pytest.approx(1)


def test_export_bundles_assets_reports_and_refuses_overwrites(inputs, tmp_path):
    pytest.importorskip("plotly")
    run = build_simplex_run(**inputs)
    output = tmp_path / "viewer"
    index = write_simplex_viewer(output, [run], reports={"Full report": {"metric": None}})
    assert index.is_file()
    assert {file.name for file in output.iterdir()} == {
        "index.html", "viewer.js", "style.css", "plotly.min.js", "data.js", "report_0.json",
    }
    script = (output / "data.js").read_text()
    payload = json.loads(script.removeprefix("window.SIMPLEX_DATA=").removesuffix(";\n"))
    assert payload["schema"] == 2
    assert payload["reports"] == [{"name": "Full report", "path": "report_0.json"}]
    assert payload["runs"][0]["cloud"]["targets"][1][3] == 1e-26
    with pytest.raises(FileExistsError):
        write_simplex_viewer(output, [run])
    assert (output / "data.js").read_text() == script
    with pytest.raises(ValueError):
        write_simplex_viewer(tmp_path / "bad", [run], reports={"bad": {"value": np.inf}})
    assert not (tmp_path / "bad").exists()
