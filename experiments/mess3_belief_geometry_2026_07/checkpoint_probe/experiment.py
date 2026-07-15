"""Fit and report MESS3 belief probes for one public RLlib checkpoint."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_algorithm
from analysis.plots import simplex_scatter
from analysis.probes import probe_predict
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    evaluate_probe,
    within_branch_action_variance_fraction,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _device(context: RunContext) -> str:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if (
        profile.learner_device == "mps"
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def run(context: RunContext):
    if context.resume_from is None:
        raise ValueError("checkpoint probing requires --resume-from CHECKPOINT")
    if context.seed is None:
        raise ValueError("checkpoint probing requires a resolved seed")
    device = _device(context)
    train_steps = 256 if context.smoke else 120_000
    test_steps = 128 if context.smoke else 60_000
    warmup = 4 if context.smoke else 64

    with load_algorithm(context.resume_from) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        environment_class = algorithm.config.env
        environment_config = dict(algorithm.config.env_config)

        def make_environment():
            return environment_class(environment_config)

        train = collect_probe_data(
            module,
            make_environment,
            n_steps=train_steps,
            seed=context.seed + 7_000_000,
            policy_mode="policy",
            device=device,
            warmup=warmup,
        )
        test = collect_probe_data(
            module,
            make_environment,
            n_steps=test_steps,
            seed=context.seed + 7_500_000,
            policy_mode="policy",
            device=device,
            warmup=warmup,
        )
        greedy = collect_probe_data(
            module,
            make_environment,
            n_steps=test_steps,
            seed=context.seed + 7_900_000,
            policy_mode="greedy",
            device=device,
            warmup=warmup,
        )

    metrics = evaluate_probe(train, test)
    weight, bias = metrics.pop("probe")
    metrics.update(
        reward_mean=float(test.rewards.mean()),
        reward_greedy=float(greedy.rewards.mean()),
        within_branch_action_variance_fraction=(
            within_branch_action_variance_fraction(test)
        ),
    )
    (context.results_dir / "probe_result.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    predicted = probe_predict(weight, bias, test.activations)
    display = np.clip(predicted, 0, None)
    display /= np.maximum(display.sum(axis=1, keepdims=True), 1e-9)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ("s0", "s1", "s2")
    simplex_scatter(
        axes[0],
        test.beliefs,
        s=0.5,
        alpha=0.4,
        title="true beliefs",
        labels=labels,
    )
    simplex_scatter(
        axes[1],
        display,
        s=0.5,
        alpha=0.4,
        title=(
            f"decoded\n(global={metrics['r2_global']:.3f}, "
            f"fine={metrics['r2_fine']:.3f})"
        ),
        labels=labels,
    )
    figure.tight_layout()
    figure.savefig(context.results_dir / "fig_probe.png", dpi=160)
    plt.close(figure)
    return metrics
