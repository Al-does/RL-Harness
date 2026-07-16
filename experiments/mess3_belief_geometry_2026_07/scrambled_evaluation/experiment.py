"""Compare normal and scrambled evaluation of a reward-only checkpoint."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_module
from analysis.plots import simplex_scatter
from analysis.probes import probe_predict
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    evaluate_probe,
)
from experiments.mess3_belief_geometry_2026_07.shared import (
    CONTINUOUS_ENV_BASE,
)
from harness.context import RunContext
from harness.hardware import PROFILES


PROBE_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "tokens": True,
}
NORMAL_ENV_CONFIG = {
    **CONTINUOUS_ENV_BASE,
    "diagnostics": PROBE_DIAGNOSTICS,
}
SCRAMBLED_ENV_CONFIG = {
    **CONTINUOUS_ENV_BASE,
    "observation": {"token_scrambling": "uniform"},
    "diagnostics": PROBE_DIAGNOSTICS,
}


def make_normal_environment() -> HMMEnv:
    return HMMEnv(NORMAL_ENV_CONFIG)


def make_scrambled_environment() -> HMMEnv:
    return HMMEnv(SCRAMBLED_ENV_CONFIG)


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


def _evaluate_condition(
    module,
    env_factory,
    *,
    seed: int,
    device: str,
    smoke: bool,
):
    train = collect_probe_data(
        module,
        env_factory,
        n_steps=256 if smoke else 120_000,
        seed=seed,
        device=device,
        warmup=4 if smoke else 64,
    )
    test = collect_probe_data(
        module,
        env_factory,
        n_steps=128 if smoke else 60_000,
        seed=seed + 10_000,
        device=device,
        warmup=4 if smoke else 64,
    )
    metrics = evaluate_probe(train, test)
    weight, bias = metrics.pop("probe")
    metrics["reward_mean"] = float(test.rewards.mean())
    return metrics, probe_predict(weight, bias, test.activations), test


def run(context: RunContext):
    if context.resume_from is None:
        raise ValueError(
            "scrambled evaluation requires --resume-from CHECKPOINT"
        )
    if context.seed is None:
        raise ValueError("scrambled evaluation requires a resolved seed")
    device = _device(context)
    with load_module(context.resume_from) as module:
        normal = _evaluate_condition(
            module,
            make_normal_environment,
            seed=context.seed + 777_000,
            device=device,
            smoke=context.smoke,
        )
        scrambled = _evaluate_condition(
            module,
            make_scrambled_environment,
            seed=context.seed + 777_000,
            device=device,
            smoke=context.smoke,
        )

    conditions = {"normal": normal, "scrambled": scrambled}
    payload = {
        name: condition[0]
        for name, condition in conditions.items()
    }
    (context.results_dir / "scrambled_evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = ("s0", "s1", "s2")
    simplex_scatter(
        axes[0],
        normal[2].beliefs,
        s=0.5,
        alpha=0.4,
        title="true beliefs",
        labels=labels,
    )
    for axis, name in zip(axes[1:], ("normal", "scrambled")):
        metrics, predicted, _ = conditions[name]
        display = np.clip(predicted, 0, None)
        display /= np.maximum(display.sum(axis=1, keepdims=True), 1e-9)
        simplex_scatter(
            axis,
            display,
            s=0.5,
            alpha=0.4,
            title=(
                f"{name} input\n"
                f"global={metrics['r2_global']:.3f}, "
                f"fine={metrics['r2_fine']:.3f}"
            ),
            labels=labels,
        )
    figure.tight_layout()
    figure.savefig(
        context.results_dir / "fig_scrambled_evaluation.png",
        dpi=160,
    )
    plt.close(figure)
    return payload
