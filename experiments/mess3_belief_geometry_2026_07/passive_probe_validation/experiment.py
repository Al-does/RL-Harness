"""Validate supervised representation probing on passive MESS3."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter
from analysis.probes import probe_predict
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
    evaluate_probe,
)
from experiments.mess3_belief_geometry_2026_07.supervised import (
    train_supervised,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import named_seed_sequences
from learners.models.next_token import NextTokenAuxHead
from learners.models.transformer import (
    TransformerModel,
    TransformerModelConfig,
)


class ExperimentModule(NextTokenAuxHead, TransformerModel):
    """Transformer with the validation task's prediction head."""


TOTAL_ENV_STEPS = 3_000_000
ENV_CONFIG = {
    "model": {
        "factory": "envs.mess3.model:passive_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": "envs.mess3.tasks.passive:PassiveTask",
        "kwargs": {"action_limit": 5.0},
    },
    "delay": 1,
    "episode_length": 1024,
    "diagnostics": {
        "state": True,
        "belief": True,
        "tokens": True,
    },
}
MODEL_CONFIG = {
    **TransformerModelConfig(
        d_model=96,
        n_layers=3,
        n_heads=4,
        context_len=64,
    ).to_dict(),
    "next_token_aux": {"num_classes": 3},
}
_STREAM_KEYS = {
    "training": (0,),
    "probe_train": (1,),
    "probe_test": (2,),
}
# Explicit spawn keys are order-independent; never renumber or reuse a key.


def make_environment() -> HMMEnv:
    return HMMEnv(ENV_CONFIG)


def next_token_logits(
    module: ExperimentModule,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    return module.next_token_aux_head(embeddings)


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
    if context.seed is None:
        raise ValueError("passive probe validation requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    module = train_supervised(
        context,
        seed=streams["training"],
        env_factory=make_environment,
        module_class=ExperimentModule,
        model_config=MODEL_CONFIG,
        logits_from_embeddings=next_token_logits,
        target="next_token",
        total_steps=8192 if context.smoke else TOTAL_ENV_STEPS,
        num_classes=3,
        batch_episodes=8,
        learning_rate=3e-4,
        fresh_data_episodes=8 if context.smoke else 512,
        log_every=1 if context.smoke else 25,
    )

    train_steps = 256 if context.smoke else 120_000
    test_steps = 128 if context.smoke else 60_000
    warmup = 4 if context.smoke else 64
    device = _device(context)
    train = collect_probe_data(
        module,
        make_environment,
        n_steps=train_steps,
        seed=streams["probe_train"],
        policy_mode="random",
        device=device,
        warmup=warmup,
    )
    test = collect_probe_data(
        module,
        make_environment,
        n_steps=test_steps,
        seed=streams["probe_test"],
        policy_mode="random",
        device=device,
        warmup=warmup,
    )
    metrics = evaluate_probe(train, test)
    weight, bias = metrics.pop("probe")
    metrics["prior_expectation"] = 0.994
    metrics["passed"] = bool(metrics["r2_global"] >= 0.98)
    (context.results_dir / "probe_result.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    predicted = probe_predict(weight, bias, test.activations)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    simplex_scatter(
        axes[0],
        test.beliefs,
        s=0.5,
        alpha=0.4,
        title="ground-truth beliefs",
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        axes[1],
        np.clip(predicted, 0, 1),
        s=0.5,
        alpha=0.4,
        title=f"decoded (global R²={metrics['r2_global']:.3f})",
        labels=("s0", "s1", "s2"),
    )
    figure.tight_layout()
    figure.savefig(
        context.results_dir / "fig_passive_probe.png",
        dpi=160,
    )
    plt.close(figure)
    return metrics
