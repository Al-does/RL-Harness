"""Focused tests for the fixed-quantile PPO value critic."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from learners import QRPPOTorchLearner
from learners.components import QRValueConfig, QRValueHead, midpoint_taus
from learners.models import (
    MLPModel,
    MLPModelConfig,
    QRValueMixin,
    TransformerModel,
    TransformerModelConfig,
)
from learners.models.qr_value import FWD_QUANTILES
from losses import quantile_huber_loss


class QRMLPModel(QRValueMixin, MLPModel):
    pass


class QRTransformerModel(QRValueMixin, TransformerModel):
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


def test_qr_config_and_head_validate_and_backpropagate():
    with pytest.raises(ValueError, match="positive"):
        QRValueConfig(num_quantiles=0)
    with pytest.raises(ValueError, match="positive"):
        QRValueHead(8, num_quantiles=0)

    head = QRValueHead(embedding_dim=8, num_quantiles=7)
    embeddings = torch.randn(2, 3, 8, requires_grad=True)
    quantiles = head(embeddings)

    assert quantiles.shape == (2, 3, 7)
    assert quantiles.device == embeddings.device
    quantiles.mean().backward()
    assert embeddings.grad is not None


def test_qr_fixed_fractions_drive_quantile_huber_loss_on_device():
    quantiles = torch.zeros(1, 2, 4, requires_grad=True)
    targets = torch.tensor([[1.0, -1.0]])
    taus = midpoint_taus(quantiles, quantiles.shape[-1])

    loss = quantile_huber_loss(quantiles, taus, targets)

    assert taus.shape == quantiles.shape
    assert taus.device == quantiles.device
    assert torch.isfinite(loss)
    loss.backward()
    assert quantiles.grad is not None


def test_qr_value_mixin_composes_with_memoryless_model():
    observation_space, action_space = spaces()
    module = QRMLPModel(
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            **MLPModelConfig(hidden_dims=(16, 8)).to_dict(),
            "qr_value": {"num_quantiles": 7},
        },
    )
    batch = {Columns.OBS: torch.randn(3, 5)}

    outputs = module._forward_train(batch)
    values = module.compute_values(batch, embeddings=outputs[Columns.EMBEDDINGS])

    assert outputs[FWD_QUANTILES].shape == (3, 7)
    assert values.shape == (3,)
    assert isinstance(module.heads.value, torch.nn.Identity)
    torch.testing.assert_close(
        values,
        outputs[FWD_QUANTILES].mean(dim=-1),
    )


def test_qr_value_mixin_composes_with_stateful_transformer():
    observation_space, action_space = spaces()
    module = QRTransformerModel(
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
            "qr_value": {"num_quantiles": 5},
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
    assert values.shape == (1, 2)


def test_qr_ppo_learner_rejects_scalar_value_loss_before_build():
    learner = object.__new__(QRPPOTorchLearner)
    learner.config = SimpleNamespace(vf_loss_coeff=0.5)

    with pytest.raises(ValueError, match="vf_loss_coeff=0.0"):
        learner.build()
