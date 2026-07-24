"""Focused tests for reusable implicit-quantile value components."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from learners import IQNPPOTorchLearner
from learners.components import (
    IQNValueConfig,
    IQNValueHead,
    midpoint_taus,
    sample_taus,
)
from learners.models import (
    IQNValueMixin,
    MLPModel,
    MLPModelConfig,
    TransformerModel,
    TransformerModelConfig,
)
from learners.models.iqn_value import FWD_QUANTILES, FWD_TAUS
from losses import quantile_huber_loss


class IQNMLPModel(IQNValueMixin, MLPModel):
    pass


class IQNTransformerModel(IQNValueMixin, TransformerModel):
    pass


def spaces():
    return (
        gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(5,),
            dtype=np.float32,
        ),
        gym.spaces.Discrete(3),
    )


def test_iqn_config_and_tau_helpers_validate_and_stay_on_device():
    with pytest.raises(ValueError, match="positive"):
        IQNValueConfig(train_quantiles=0)
    with pytest.raises(ValueError, match="positive"):
        sample_taus(torch.zeros(2, 4), 0)

    reference = torch.zeros(2, 3, 4, dtype=torch.float64)
    sampled = sample_taus(reference, 5)
    fixed = midpoint_taus(reference, 4)

    assert sampled.shape == (2, 3, 5)
    assert sampled.dtype == reference.dtype
    assert sampled.device == reference.device
    torch.testing.assert_close(
        fixed[0, 0],
        torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float64),
    )


def test_iqn_head_is_device_native_and_differentiable():
    head = IQNValueHead(embedding_dim=8, n_cosines=16)
    embeddings = torch.randn(2, 3, 8, requires_grad=True)
    taus = torch.rand(2, 3, 5)

    quantiles = head(embeddings, taus)

    assert quantiles.shape == (2, 3, 5)
    assert quantiles.device == embeddings.device
    quantiles.mean().backward()
    assert embeddings.grad is not None
    assert embeddings.grad.shape == embeddings.shape


def test_quantile_huber_loss_matches_simple_case_and_masks_padding():
    quantiles = torch.zeros(1, 2, 2, requires_grad=True)
    taus = torch.tensor([[[0.25, 0.75], [0.25, 0.75]]])
    targets = torch.tensor([[1.0, 100.0]])
    valid = torch.tensor([[True, False]])

    loss = quantile_huber_loss(
        quantiles,
        taus,
        targets,
        kappa=1.0,
        valid=valid,
    )

    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    assert quantiles.grad is not None

    all_invalid = quantile_huber_loss(
        quantiles.detach(),
        taus,
        targets,
        valid=torch.zeros_like(valid),
    )
    torch.testing.assert_close(all_invalid, torch.tensor(0.0))


def test_iqn_value_mixin_composes_with_memoryless_model():
    observation_space, action_space = spaces()
    module = IQNMLPModel(
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            **MLPModelConfig(hidden_dims=(16, 8)).to_dict(),
            "iqn_value": {
                "train_quantiles": 7,
                "value_quantiles": 4,
                "n_cosines": 8,
            },
        },
    )
    batch = {Columns.OBS: torch.randn(3, 5)}

    outputs = module._forward_train(batch)
    values = module.compute_values(batch, embeddings=outputs[Columns.EMBEDDINGS])

    assert outputs[FWD_QUANTILES].shape == (3, 7)
    assert outputs[FWD_TAUS].shape == (3, 7)
    assert values.shape == (3,)
    assert isinstance(module.heads.value, torch.nn.Identity)


def test_iqn_value_mixin_composes_with_stateful_transformer():
    observation_space, action_space = spaces()
    module = IQNTransformerModel(
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            **TransformerModelConfig(
                d_model=16,
                n_layers=1,
                n_heads=2,
                context_len=4,
                max_seq_len=3,
            ).to_dict(),
            "iqn_value": {
                "train_quantiles": 5,
                "value_quantiles": 4,
                "n_cosines": 8,
            },
        },
    )
    initial = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    batch = {
        Columns.OBS: torch.randn(1, 2, 5),
        Columns.STATE_IN: initial,
    }

    outputs = module._forward_train(batch)
    values = module.compute_values(batch, embeddings=outputs[Columns.EMBEDDINGS])

    assert outputs[FWD_QUANTILES].shape == (1, 2, 5)
    assert outputs[FWD_TAUS].shape == (1, 2, 5)
    assert values.shape == (1, 2)


def test_iqn_ppo_learner_rejects_scalar_value_loss_before_build():
    learner = object.__new__(IQNPPOTorchLearner)
    learner.config = SimpleNamespace(vf_loss_coeff=0.5)

    with pytest.raises(ValueError, match="vf_loss_coeff=0.0"):
        learner.build()
