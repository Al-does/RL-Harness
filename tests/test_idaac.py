"""Focused tests for reusable DAAC and IDAAC components."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from learners import IDAACConfig
from learners.idaac import add_temporal_order_pairs
from learners.models import IDAACModel
from learners.models.idaac import (
    ADVANTAGE_PREDICTIONS,
    ORDER_TARGETS,
    PAIRED_EMBEDDINGS,
    PAIRED_OBSERVATIONS,
    PAIR_POSITIONS,
    PAIR_VALID_MASK,
)
from losses.idaac import invariance_losses


def _module() -> IDAACModel:
    return IDAACModel(
        observation_space=gym.spaces.Box(
            -5.0,
            5.0,
            shape=(4,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(2),
        model_config={
            "encoder_type": "mlp",
            "hidden_dims": (16, 16),
            "order_hidden_dims": (8,),
        },
    )


def test_idaac_model_decouples_policy_and_value_gradients():
    module = _module()
    batch = {
        Columns.OBS: torch.randn(12, 4),
        Columns.ACTIONS: torch.randint(0, 2, (12,)),
        PAIRED_OBSERVATIONS: torch.randn(12, 4),
    }

    outputs = module._forward_train(batch)
    assert outputs[Columns.ACTION_DIST_INPUTS].shape == (12, 2)
    assert outputs[ADVANTAGE_PREDICTIONS].shape == (12,)
    assert outputs[PAIRED_EMBEDDINGS].shape == (12, 16)

    outputs[ADVANTAGE_PREDICTIONS].square().mean().backward()
    assert module.policy_encoder[0].weight.grad is not None
    assert module.advantage_head.weight.grad is not None
    assert module.value_encoder[0].weight.grad is None
    assert module.value_head.weight.grad is None

    module.zero_grad(set_to_none=True)
    module.compute_values(batch).square().mean().backward()
    assert module.policy_encoder[0].weight.grad is None
    assert module.policy_head.weight.grad is None
    assert module.value_encoder[0].weight.grad is not None
    assert module.value_head.weight.grad is not None


def test_idaac_adversary_isolates_encoder_and_classifier_gradients():
    module = _module()
    first = module.policy_encoder(torch.randn(10, 4))
    second = module.policy_encoder(torch.randn(10, 4))
    targets = torch.randint(0, 2, (10,), dtype=torch.bool)

    discriminator_logits = module.order_logits(
        first,
        second,
        detach_embeddings=True,
    )
    encoder_logits = module.order_logits(
        first,
        second,
        detach_classifier=True,
    )
    discriminator_loss, encoder_loss = invariance_losses(
        discriminator_logits,
        encoder_logits,
        targets,
    )

    encoder_loss.backward(retain_graph=True)
    assert module.policy_encoder[0].weight.grad is not None
    assert all(
        parameter.grad is None
        for parameter in module.order_classifier.parameters()
    )

    module.zero_grad(set_to_none=True)
    discriminator_loss.backward()
    assert module.policy_encoder[0].weight.grad is None
    assert all(
        parameter.grad is not None
        for parameter in module.order_classifier.parameters()
    )


def test_temporal_pairs_never_cross_episode_boundaries():
    torch.manual_seed(7)
    observations = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    batch = {
        Columns.OBS: observations,
        Columns.TERMINATEDS: torch.tensor(
            [False, False, True, False, False, False, True, False]
        ),
        Columns.TRUNCATEDS: torch.zeros(8, dtype=torch.bool),
        Columns.LOSS_MASK: torch.tensor(
            [True, True, True, True, True, True, True, False]
        ),
    }

    add_temporal_order_pairs(batch)

    paired = batch[PAIRED_OBSERVATIONS].squeeze(-1).to(dtype=torch.long)
    valid = batch[PAIR_VALID_MASK]
    episode = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2])
    indices = observations.squeeze(-1).to(dtype=torch.long)
    assert torch.all(episode[paired[valid]] == episode[indices[valid]])
    assert torch.all(paired[valid] != indices[valid])
    assert torch.equal(
        batch[ORDER_TARGETS][valid],
        indices[valid] > paired[valid],
    )
    assert not valid[-1]


def test_decoupled_transformer_matches_chunked_and_stepwise_policy_encoding():
    torch.manual_seed(3)
    module = IDAACModel(
        observation_space=gym.spaces.Box(
            -5.0,
            5.0,
            shape=(4,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(2),
        model_config={
            "encoder_type": "transformer",
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 2,
            "context_len": 2,
            "max_seq_len": 5,
        },
    )
    observations = torch.randn(2, 5, 4)
    initial_state = {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(2, *([1] * value.ndim))
        for key, value in module.get_initial_state().items()
    }
    pair_positions = torch.tensor(
        [[1, 0, 3, 2, 3], [1, 0, 1, 4, 3]],
        dtype=torch.long,
    )

    outputs = module._forward_train(
        {
            Columns.OBS: observations,
            Columns.ACTIONS: torch.randint(0, 2, (2, 5)),
            Columns.STATE_IN: initial_state,
            PAIR_POSITIONS: pair_positions,
        }
    )
    expected = outputs[Columns.EMBEDDINGS]
    gathered = torch.gather(
        expected,
        1,
        pair_positions.unsqueeze(-1).expand(2, 5, 16),
    )
    assert torch.allclose(outputs[PAIRED_EMBEDDINGS], gathered)
    assert module.compute_values(
        {Columns.OBS: observations, Columns.STATE_IN: initial_state}
    ).shape == (2, 5)

    step_state = initial_state
    step_embeddings = []
    for timestep in range(observations.shape[1]):
        embedding, step_state = module.encode_step(
            observations[:, timestep],
            step_state,
        )
        step_embeddings.append(embedding)
    actual = torch.stack(step_embeddings, dim=1)
    assert torch.allclose(actual, expected, atol=1e-5)
    assert set(step_state) == set(initial_state)


def test_idaac_config_uses_paper_wide_defaults():
    config = IDAACConfig()

    assert config.num_epochs == 1
    assert config.value_num_epochs == 9
    assert config.value_update_frequency == 1
    assert config.advantage_loss_coeff == 0.25
    assert config.invariance_loss_coeff == 0.001
    assert config.gamma == 0.999
    assert config.lambda_ == 0.95


def test_idaac_model_defaults_to_token_guess_scale_transformer():
    module = IDAACModel(
        observation_space=gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
        action_space=gym.spaces.Discrete(2),
        model_config={"max_seq_len": 32},
    )

    assert module.config.encoder_type == "transformer"
    assert module.config.d_model == 64
    assert module.config.n_layers == 4
    assert module.config.n_heads == 1
    assert module.config.context_len == 10
    assert module.policy_encoder is not module.value_encoder
    assert module.policy_encoder.d_model == module.value_encoder.d_model == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value_num_epochs", 0),
        ("value_update_frequency", 0),
        ("value_minibatch_size", -1),
        ("advantage_loss_coeff", -0.1),
        ("invariance_loss_coeff", -0.1),
        ("adam_epsilon", 0.0),
    ],
)
def test_idaac_config_rejects_invalid_settings(field, value):
    config = IDAACConfig().environment("CartPole-v1")
    setattr(config, field, value)

    with pytest.raises(ValueError):
        config.validate()
