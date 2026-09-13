from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.belief_geometry import evaluate_belief_geometry
from analysis.simplex import (
    ExplorerScore,
    add_explorer_scores,
    build_simplex_run,
    geometry_metrics,
    write_nonergodic_belief_explorer,
)


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


def test_checkpoint_sites_can_differ(inputs):
    inputs["predictions"]["epoch 7"] = {"another": inputs["targets"]}
    run = build_simplex_run(**inputs)
    assert run["sites"] == ["embedding", "another"]
    assert list(run["metrics"]["epoch 7"]) == ["another"]
    assert list(run["predictions"]["epoch 0"]) == ["embedding"]


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
    index = write_nonergodic_belief_explorer(output, [run], reports={"Full report": {"metric": None}})
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
        write_nonergodic_belief_explorer(output, [run])
    assert (output / "data.js").read_text() == script
    with pytest.raises(ValueError):
        write_nonergodic_belief_explorer(tmp_path / "bad", [run], reports={"bad": {"value": np.inf}})
    assert not (tmp_path / "bad").exists()


def test_optional_scores_preserve_zero_missing_values_and_geometry(inputs, monkeypatch):
    run = build_simplex_run(**inputs)

    def no_scoring(*args, **kwargs):
        pytest.fail("attaching saved scores must not recalculate geometry metrics")

    monkeypatch.setattr("analysis.simplex.geometry_metrics", no_scoring)
    enriched = add_explorer_scores(
        run,
        layer_scores={"epoch 7": {"score-only site": {
            "NTP R²": ExplorerScore(0.0, "Independent held-out episodes"),
            "log NTP R²": ExplorerScore(None, "Constant target; undefined"),
        }}},
        task_scores={"Bayes maximum reward occupancy": ExplorerScore(0.75, "Fixture optimum", "percent")},
    )
    assert "layer_scores" not in run
    assert enriched["metrics"] is run["metrics"]
    assert enriched["predictions"] is run["predictions"]
    assert enriched["sites"] == ["embedding"]
    assert enriched["layer_scores"]["epoch 7"]["score-only site"]["NTP R²"]["value"] == 0
    assert enriched["layer_scores"]["epoch 7"]["score-only site"]["log NTP R²"]["value"] is None
    assert enriched["task_scores"]["Bayes maximum reward occupancy"]["format"] == "percent"
    json.dumps(enriched, allow_nan=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1j, True, "0.4"])
def test_rejects_invalid_optional_scores(inputs, value):
    with pytest.raises(ValueError, match="score values"):
        build_simplex_run(**inputs, task_scores={"Reward occupancy": ExplorerScore(value, "fixture")})


def test_scores_validate_checkpoint_and_format(inputs):
    run = build_simplex_run(**inputs)
    with pytest.raises(ValueError, match="checkpoint"):
        add_explorer_scores(run, layer_scores={"missing": {}})
    with pytest.raises(ValueError, match="formats"):
        add_explorer_scores(run, task_scores={"occupancy": ExplorerScore(1.0, "", "invalid")})
    assert "task_scores" not in run
