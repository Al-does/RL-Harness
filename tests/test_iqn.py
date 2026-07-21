"""Unit and composition tests for reusable IQN value critics."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from learners import HUBER_KAPPA_KEY, LOSS_COEFFICIENT_KEY
from learners.components.iqn import IQNValueHead
from learners.models import (
    FWD_QUANTILES,
    FWD_TAUS,
    IQNMLPModel,
    IQNTransformerModel,
    IQNValueConfig,
    NextTokenAuxHead,
)
from learners.ppo_iqn import IQNPPOTorchLearner
from losses import quantile_huber_loss
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing


class RecordingMetrics:
    def __init__(self):
        self.calls = []

    def log_dict(self, values, *, key, window):
        self.calls.append((values, key, window))


class StubIQNLearner(IQNPPOTorchLearner):
    def __init__(self):
        self.metrics = RecordingMetrics()


class TransformerWithNextTokenAndIQN(
    NextTokenAuxHead,
    IQNTransformerModel,
):
    pass


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
    quantiles = torch.zeros(1, 2, 2)
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


def test_quantile_huber_loss_validates_shapes_and_kappa():
    quantiles = torch.zeros(2, 3)
    taus = torch.zeros(2, 3)
    targets = torch.zeros(2)

    with pytest.raises(ValueError, match="positive"):
        quantile_huber_loss(quantiles, taus, targets, kappa=0.0)
    with pytest.raises(ValueError, match="matching"):
        quantile_huber_loss(quantiles, taus[:, :2], targets, kappa=1.0)
    with pytest.raises(ValueError, match="valid mask"):
        quantile_huber_loss(
            quantiles,
            taus,
            targets,
            kappa=1.0,
            valid=torch.ones(2, 1),
        )


def test_iqn_config_defaults_and_validation():
    assert IQNValueConfig.from_dict({}).to_dict() == {
        "train_quantiles": 32,
        "value_quantiles": 64,
        "n_cosines": 64,
    }
    with pytest.raises(ValueError, match="positive"):
        IQNValueConfig(train_quantiles=0)


def test_iqn_mlp_samples_training_taus_and_uses_fixed_value_taus():
    module = IQNMLPModel(
        observation_space=gym.spaces.Box(
            -1.0,
            1.0,
            shape=(4,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(2),
        model_config={
            "hidden_dims": (16, 16),
            "iqn_value": {
                "train_quantiles": 5,
                "value_quantiles": 7,
                "n_cosines": 8,
            },
        },
    )
    batch = {Columns.OBS: torch.randn(6, 4)}

    first = module._forward_train(batch)
    second = module._forward_train(batch)
    first_values = module.compute_values(batch)
    second_values = module.compute_values(batch)

    assert first[FWD_TAUS].shape == (6, 5)
    assert first[FWD_QUANTILES].shape == (6, 5)
    assert not torch.equal(first[FWD_TAUS], second[FWD_TAUS])
    torch.testing.assert_close(first_values, second_values)
    assert first_values.shape == (6,)
    assert isinstance(module.heads.value, nn.Identity)
    assert FWD_QUANTILES not in module._forward(batch)


def test_iqn_transformer_composes_with_an_auxiliary_head():
    module = TransformerWithNextTokenAndIQN(
        observation_space=gym.spaces.Box(
            -1.0,
            1.0,
            shape=(5,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(3),
        model_config={
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "context_len": 4,
            "max_seq_len": 3,
            "iqn_value": {
                "train_quantiles": 3,
                "value_quantiles": 4,
                "n_cosines": 8,
            },
            "next_token_aux": {"num_classes": 3},
        },
    )
    state = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    batch = {
        Columns.OBS: torch.randn(1, 2, 5),
        Columns.STATE_IN: state,
    }

    output = module._forward_train(batch)

    assert output[FWD_QUANTILES].shape == (1, 2, 3)
    assert output["next_token_aux/logits"].shape == (1, 2, 3)
    assert module.compute_values(batch).shape == (1, 2)


def test_iqn_ppo_learner_adds_loss_and_logs_metrics(monkeypatch):
    base_loss = torch.tensor(2.0)

    def base_compute_loss(self, *, module_id, config, batch, fwd_out):
        return base_loss

    monkeypatch.setattr(
        PPOTorchLearner,
        "compute_loss_for_module",
        base_compute_loss,
    )
    learner = StubIQNLearner()
    quantiles = torch.zeros(1, 2, 2, requires_grad=True)
    config = SimpleNamespace(
        learner_config_dict={
            LOSS_COEFFICIENT_KEY: 0.5,
            HUBER_KAPPA_KEY: 1.0,
        }
    )
    batch = {
        Postprocessing.VALUE_TARGETS: torch.ones(1, 2),
        Columns.LOSS_MASK: torch.tensor([[True, False]]),
    }

    total = learner.compute_loss_for_module(
        module_id="policy",
        config=config,
        batch=batch,
        fwd_out={
            FWD_QUANTILES: quantiles,
            FWD_TAUS: torch.tensor(
                [[[0.25, 0.75], [0.25, 0.75]]]
            ),
        },
    )

    torch.testing.assert_close(total, torch.tensor(2.125))
    values, key, window = learner.metrics.calls[0]
    assert set(values) == {
        "iqn_value/loss",
        "iqn_value/mean_quantile_spread",
    }
    assert key == "policy"
    assert window == 1
    total.backward()
    assert quantiles.grad is not None
