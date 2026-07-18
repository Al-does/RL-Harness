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
    make_transducer_target,
)
from experiments.mess3_belief_geometry_2026_07.shared import (
    CONTINUOUS_ENV_BASE,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import SeedSource, named_seed_sequences


_STREAM_KEYS = {
    "paired_condition_evaluation": (0,),
}
_CONDITION_STREAM_KEYS = {
    "probe_train": (0,),
    "probe_test": (1,),
}
# Explicit spawn keys are order-independent; never renumber or reuse a key.
# paired_condition_evaluation is deliberately reused for normal and scrambled.
PROBE_DIAGNOSTICS = {
    "state": True,
    "belief": True,
    "tokens": True,
    "transitions": True,
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


def _transducer_target(env_factory):
    environment = env_factory()
    try:
        return make_transducer_target(environment)
    finally:
        environment.close()


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
    seed: SeedSource,
    device: str,
    smoke: bool,
    initial_belief: np.ndarray,
    action_outcome_operator,
    initial_outcome_operator,
):
    streams = named_seed_sequences(seed, _CONDITION_STREAM_KEYS)
    train = collect_probe_data(
        module,
        env_factory,
        n_steps=256 if smoke else 120_000,
        seed=streams["probe_train"],
        device=device,
        warmup=4 if smoke else 64,
        initial_belief=initial_belief,
        action_outcome_operator=action_outcome_operator,
        initial_outcome_operator=initial_outcome_operator,
    )
    test = collect_probe_data(
        module,
        env_factory,
        n_steps=128 if smoke else 60_000,
        seed=streams["probe_test"],
        device=device,
        warmup=4 if smoke else 64,
        initial_belief=initial_belief,
        action_outcome_operator=action_outcome_operator,
        initial_outcome_operator=initial_outcome_operator,
    )
    metrics = evaluate_probe(train, test)
    weight, bias = metrics.pop("probe")
    target_error = max(
        float(np.max(np.abs(data.beliefs - data.diagnostic_beliefs)))
        for data in (train, test)
    )
    if target_error > 1e-10:
        raise AssertionError(
            "transducer target is misaligned with environment diagnostics: "
            f"max absolute error {target_error:.3e}"
        )
    metrics["target"] = "predictive_transducer_belief"
    metrics["target_consistency_max_abs"] = target_error
    metrics["reward_mean"] = float(test.rewards.mean())
    return metrics, probe_predict(weight, bias, test.activations), test


def run(context: RunContext):
    if context.resume_from is None:
        raise ValueError(
            "scrambled evaluation requires --resume-from CHECKPOINT"
        )
    if context.seed is None:
        raise ValueError("scrambled evaluation requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    device = _device(context)
    normal_target = _transducer_target(make_normal_environment)
    scrambled_target = _transducer_target(make_scrambled_environment)
    with load_module(context.resume_from) as module:
        normal = _evaluate_condition(
            module,
            make_normal_environment,
            seed=streams["paired_condition_evaluation"],
            device=device,
            smoke=context.smoke,
            initial_belief=normal_target[0],
            action_outcome_operator=normal_target[1],
            initial_outcome_operator=normal_target[2],
        )
        scrambled = _evaluate_condition(
            module,
            make_scrambled_environment,
            seed=streams["paired_condition_evaluation"],
            device=device,
            smoke=context.smoke,
            initial_belief=scrambled_target[0],
            action_outcome_operator=scrambled_target[1],
            initial_outcome_operator=scrambled_target[2],
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
