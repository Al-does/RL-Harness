"""MLP PPO with exact Bayesian belief observations."""

from __future__ import annotations

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.mess3.env_continuous import Mess3ContinuousEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    CONTINUOUS_ENV_BASE,
    SMOKE_ENV_STEPS,
    apply_runtime_resources,
)
from harness.context import RunContext
from harness.runners import run_tune
from learners.models import MLPModel, MLPModelConfig


TOTAL_ENV_STEPS = 5_000_000
ENV_CONFIG = {**CONTINUOUS_ENV_BASE, "obs_mode": "belief"}
MODEL_CONFIG = MLPModelConfig(hidden_dims=(128, 128)).to_dict()


def build_config(context: RunContext) -> PPOConfig:
    config = (
        PPOConfig()
        .environment(Mess3ContinuousEnv, env_config=ENV_CONFIG)
        .training(
            lr=3e-4,
            gamma=0.99,
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
                module_class=MLPModel,
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
