"""Supervised next-visible-token prediction under random actions."""

from __future__ import annotations

import torch

from envs.mess3.env_continuous import Mess3ContinuousEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    CONTINUOUS_ENV_BASE,
)
from experiments.mess3_belief_geometry_2026_07.supervised import (
    train_supervised,
)
from harness.context import RunContext
from learners.models.next_token import NextTokenAuxHead
from learners.models.transformer import (
    TransformerModel,
    TransformerModelConfig,
)


class ExperimentModule(NextTokenAuxHead, TransformerModel):
    """Transformer encoder with this condition's token prediction head."""


TOTAL_ENV_STEPS = 10_000_000
ENV_CONFIG = dict(CONTINUOUS_ENV_BASE)
MODEL_CONFIG = {
    **TransformerModelConfig(
        d_model=96,
        n_layers=3,
        n_heads=4,
        context_len=64,
    ).to_dict(),
    "next_token_aux": {"num_classes": 3},
}


def make_environment() -> Mess3ContinuousEnv:
    return Mess3ContinuousEnv(ENV_CONFIG)


def next_token_logits(
    module: ExperimentModule,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    return module.next_token_aux_head(embeddings)


def run(context: RunContext):
    return train_supervised(
        context,
        env_factory=make_environment,
        module_class=ExperimentModule,
        model_config=MODEL_CONFIG,
        logits_from_embeddings=next_token_logits,
        target="next_token",
        total_steps=8192 if context.smoke else TOTAL_ENV_STEPS,
        num_classes=3,
        batch_episodes=8,
        learning_rate=3e-4,
        fresh_data_episodes=8 if context.smoke else 256,
        log_every=1 if context.smoke else 25,
    )
