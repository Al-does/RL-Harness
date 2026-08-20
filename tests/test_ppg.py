"""Focused tests for reusable Phasic Policy Gradient components."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from learners import PPGConfig
from learners.models import MLPModel, PPGAuxiliaryValueHead
from learners.models.ppg import AUX_VALUE_PREDICTIONS


class TinyPPGModel(PPGAuxiliaryValueHead, MLPModel):
    """Inline composition proving the PPG head is model-agnostic."""


def _module() -> TinyPPGModel:
    return TinyPPGModel(
        observation_space=gym.spaces.Box(
            -1.0,
            1.0,
            shape=(4,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(3),
        model_config={"hidden_dims": (16, 16)},
    )


def test_ppg_auxiliary_head_is_training_only_and_updates_encoder():
    module = _module()
    batch = {Columns.OBS: torch.randn(8, 4)}

    train_output = module._forward_train(batch)
    assert train_output[AUX_VALUE_PREDICTIONS].shape == (8,)
    train_output[AUX_VALUE_PREDICTIONS].square().mean().backward()
    assert module.encoder[0].weight.grad is not None
    assert module.heads.policy.weight.grad is None

    rollout_output = module._forward(batch)
    assert AUX_VALUE_PREDICTIONS not in rollout_output


def test_ppg_config_exposes_canonical_phase_defaults():
    config = PPGConfig()

    assert config.policy_iterations_per_aux == 32
    assert config.aux_epochs == 6
    assert config.aux_minibatch_size == 128
    assert config.beta_clone == 1.0
    assert config.aux_value_loss_coeff == 1.0
    assert config.aux_true_value_loss_coeff == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_iterations_per_aux", 0),
        ("aux_epochs", 0),
        ("aux_minibatch_size", -1),
        ("beta_clone", -1.0),
        ("aux_value_loss_coeff", -1.0),
        ("aux_true_value_loss_coeff", -1.0),
    ],
)
def test_ppg_config_rejects_invalid_phase_settings(field, value):
    config = PPGConfig().environment("CartPole-v1")
    setattr(config, field, value)

    with pytest.raises(ValueError):
        config.validate()
