"""Supervised hidden-state twin of the delayed state-guess RL condition."""

from __future__ import annotations

import torch

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    STATE_GUESS_ENV_BASE,
)
from experiments.mess3_belief_geometry_2026_07.supervised import (
    train_supervised,
)
from harness.context import RunContext
from learners.models.state_aux import StateAuxHead
from learners.models.transformer import (
    TransformerModel,
    TransformerModelConfig,
)


class ExperimentModule(StateAuxHead, TransformerModel):
    """Transformer encoder with the state target head for this condition."""


TOTAL_ENV_STEPS = 5_000_000
ENV_CONFIG = {
    **STATE_GUESS_ENV_BASE,
    "randomize_first_episode_length": False,
    "diagnostics": {"state": True},
}
MODEL_CONFIG = {
    **TransformerModelConfig(
        d_model=96,
        n_layers=3,
        n_heads=4,
        context_len=64,
    ).to_dict(),
    "state_aux": {"num_classes": 3},
}


def make_environment() -> HMMEnv:
    return HMMEnv(ENV_CONFIG)


def state_logits(
    module: ExperimentModule,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    return module.state_aux_head(embeddings)


def run(context: RunContext):
    return train_supervised(
        context,
        env_factory=make_environment,
        module_class=ExperimentModule,
        model_config=MODEL_CONFIG,
        logits_from_embeddings=state_logits,
        target="state",
        total_steps=8192 if context.smoke else TOTAL_ENV_STEPS,
        num_classes=3,
        batch_episodes=8,
        learning_rate=3e-4,
        fresh_data_episodes=8 if context.smoke else 256,
        log_every=1 if context.smoke else 25,
    )
