"""Focused scientific and smoke tests for the supervised MESS3 replication."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.MESS3_supervised import experiment
from harness.context import RunContext
from harness.hardware import PROFILES


EXPECTED_TRANSITIONS = np.array(
    [
        [
            [0.765, 0.00375, 0.00375],
            [0.0425, 0.0675, 0.00375],
            [0.0425, 0.00375, 0.0675],
        ],
        [
            [0.0675, 0.0425, 0.00375],
            [0.00375, 0.765, 0.00375],
            [0.00375, 0.0425, 0.0675],
        ],
        [
            [0.0675, 0.00375, 0.0425],
            [0.00375, 0.0675, 0.0425],
            [0.00375, 0.00375, 0.765],
        ],
    ]
)


def test_paper_mess3_matrices_and_path_distribution_are_exact():
    matrices = experiment.paper_transition_matrices()
    np.testing.assert_allclose(matrices, EXPECTED_TRANSITIONS)

    sequences = experiment.enumerate_token_sequences(6)
    probabilities = experiment.sequence_probabilities(sequences, matrices)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    assert np.all(probabilities > 0.0)


def test_beliefs_align_with_observed_prefixes():
    matrices = experiment.paper_transition_matrices()
    sequence = np.array([[0, 1, 2]], dtype=np.int64)
    beliefs = experiment.bayesian_beliefs(sequence, matrices)

    expected = np.full(3, 1.0 / 3.0)
    for position, token in enumerate(sequence[0]):
        expected = expected @ matrices[token]
        expected /= expected.sum()
        np.testing.assert_allclose(beliefs[0, position], expected)


def test_model_is_fresh_causal_and_has_reported_scale():
    first = experiment.build_model(42, "cpu")
    second = experiment.build_model(42, "cpu")
    assert first is not second
    assert sum(parameter.numel() for parameter in first.parameters()) == 143_075

    tokens = torch.tensor([[0, 1, 2, 0], [0, 1, 0, 2]])
    logits, residuals, normalized = first(tokens, return_residuals=True)
    assert logits.shape == (2, 4, 3)
    assert len(residuals) == 4
    assert residuals[-1].shape == (2, 4, 64)
    assert normalized.shape == (2, 4, 64)
    torch.testing.assert_close(logits[0, :2], logits[1, :2])


def test_probe_split_holds_out_whole_sequences():
    train, test = experiment.split_probe_sequences(3**10, seed=42)
    assert len(train) == 11_809
    assert len(test) == 47_240
    assert not np.intersect1d(train, test).size
    assert len(np.union1d(train, test)) == 3**10


def test_smoke_run_writes_complete_compact_outputs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        run_id="smoke",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    summary = experiment.run(context)

    assert summary["smoke"] is True
    assert summary["optimizer_updates"] == experiment.SMOKE_UPDATES
    assert context.results_dir.joinpath("fig_mess3_simplex.png").is_file()
    assert context.results_dir.joinpath("fig_training_curve.png").is_file()
    assert context.results_dir.joinpath("fig_probe_by_layer.png").is_file()
    assert context.results_dir.joinpath("probe_metrics.json").is_file()
    assert context.artifacts_dir.joinpath(
        "checkpoints", "checkpoint_final.pt"
    ).is_file()
    persisted = json.loads(
        context.results_dir.joinpath("summary.json").read_text()
    )
    assert persisted["scientific_passed"] is None
