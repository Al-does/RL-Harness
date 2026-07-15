"""Reward PPO plus next-visible-token prediction at weight 0.1."""

from __future__ import annotations

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
    PPOTorchLearner,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.mess3.env_continuous import Mess3ContinuousEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    CONTINUOUS_ENV_BASE,
    SMOKE_ENV_STEPS,
    apply_runtime_resources,
    next_visible_token_targets,
)
from harness.context import RunContext
from harness.runners import run_tune
from learners.models.next_token import NextTokenAuxHead
from learners.models.transformer import (
    TransformerModel,
    TransformerModelConfig,
)
from losses.next_token import NextTokenAuxLossMixin


class ExperimentModule(NextTokenAuxHead, TransformerModel):
    """Transformer actor-critic with this experiment's auxiliary head."""


class ExperimentLearner(NextTokenAuxLossMixin, PPOTorchLearner):
    """PPO Learner with this experiment's auxiliary objective."""


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
LEARNER_CONFIG = {
    "next_token_aux/lambda": 0.1,
    "next_token_aux/target_extractor": next_visible_token_targets,
}


def build_config(context: RunContext) -> PPOConfig:
    config = (
        PPOConfig()
        .environment(Mess3ContinuousEnv, env_config=ENV_CONFIG)
        .learners(
            learner_class=ExperimentLearner,
            learner_config_dict=LEARNER_CONFIG,
        )
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
                module_class=ExperimentModule,
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
