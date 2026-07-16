"""Delayed state-guess PPO with gamma zero."""

from __future__ import annotations

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    SMOKE_ENV_STEPS,
    STATE_GUESS_ENV_BASE,
    apply_runtime_resources,
)
from harness.context import RunContext
from harness.runners import run_tune
from learners.models import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 2_500_000
ENV_CONFIG = dict(STATE_GUESS_ENV_BASE)
MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()


def build_config(context: RunContext) -> PPOConfig:
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=ENV_CONFIG)
        .training(
            lr=3e-4,
            gamma=0.0,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.003,
            train_batch_size_per_learner=(
                2048 if context.smoke else 32_768
            ),
            minibatch_size=256 if context.smoke else 4096,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=MODEL_CONFIG,
            )
        )
        .debugging(seed=context.seed)
    )
    return apply_runtime_resources(
        config,
        context,
        default_env_runners=4,
    )


def run(context: RunContext):
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    return run_tune(
        build_config(context),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": target_steps,
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=3,
                checkpoint_frequency=1 if context.smoke else 10,
                checkpoint_at_end=True,
            ),
        },
    )
