"""Two-factor 64-dim transformer, 50M steps, entropy 0.008."""

from __future__ import annotations

from experiments.factored_mess3_beliefs_2026_08.shared_longrun import (
    MODEL_CONFIG_64D,
    build_config as _build_config,
    run_independent,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return _build_config(context, n_factors=2, model_config=MODEL_CONFIG_64D)


def run(context: RunContext):
    return run_independent(
        context,
        n_factors=2,
        model_config=MODEL_CONFIG_64D,
    )


__all__ = ["build_config", "run"]
