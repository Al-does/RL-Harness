"""Integration tests for reusable RLModule and head composition."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from learners.models import (
    MLPModel,
    NextTokenAuxHead,
    StateAuxHead,
    TransformerModel,
)


OBS_DIM = 5
CONTEXT_LENGTH = 8
NEXT_TOKEN_LOGITS = "next_token_aux/logits"
STATE_LOGITS = "state_aux/logits"


class TransformerWithNextTokenAux(NextTokenAuxHead, TransformerModel):
    pass


class TransformerWithTwoAuxHeads(
    NextTokenAuxHead,
    StateAuxHead,
    TransformerModel,
):
    pass


def make_module(
    action_space,
    *,
    module_class=TransformerModel,
    mixin_config=None,
):
    model_config = {
        "context_len": CONTEXT_LENGTH,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 2,
        "max_seq_len": 5,
        **(mixin_config or {}),
    }
    return module_class(
        observation_space=gym.spaces.Box(
            -5.0,
            5.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        ),
        action_space=action_space,
        model_config=model_config,
    )


def initial_state(module, batch_size=1):
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        for key, value in module.get_initial_state().items()
    }


def rollout_embeddings(module, observations):
    state = initial_state(module)
    embeddings, states = [], [state]
    for observation in observations:
        embedding, state = module.encode_step(
            observation.unsqueeze(0),
            state,
        )
        embeddings.append(embedding[0])
        states.append(state)
    return torch.stack(embeddings), states


@pytest.mark.parametrize("total_length", [3, 13, 20])
def test_chunked_train_forward_matches_stepwise_rollout(total_length):
    torch.manual_seed(0)
    module = make_module(
        gym.spaces.Box(-5.0, 5.0, (2,), np.float32)
    ).eval()
    observations = torch.randn(total_length, OBS_DIM)
    expected, states = rollout_embeddings(module, observations)

    with torch.no_grad():
        for start in range(0, total_length, 5):
            chunk = observations[start : start + 5].unsqueeze(0)
            output = module._forward_train(
                {
                    Columns.OBS: chunk,
                    Columns.STATE_IN: states[start],
                }
            )
            actual = output[Columns.EMBEDDINGS][0]
            assert torch.allclose(
                actual,
                expected[start : start + 5],
                atol=1e-5,
            )


def test_padding_does_not_change_earlier_embeddings():
    module = make_module(
        gym.spaces.Box(-5.0, 5.0, (2,), np.float32)
    ).eval()
    observations = torch.randn(1, 3, OBS_DIM)
    padded = torch.cat(
        [observations, torch.zeros(1, 2, OBS_DIM)],
        dim=1,
    )
    state = initial_state(module)
    with torch.no_grad():
        original = module._forward_train(
            {Columns.OBS: observations, Columns.STATE_IN: state}
        )[Columns.EMBEDDINGS]
        with_padding = module._forward_train(
            {Columns.OBS: padded, Columns.STATE_IN: state}
        )[Columns.EMBEDDINGS]
    assert torch.allclose(original[0], with_padding[0, :3], atol=1e-5)


def test_base_outputs_and_head_mixins_compose_cooperatively():
    base = make_module(gym.spaces.Discrete(3))
    state = initial_state(base)
    batch = {
        Columns.OBS: torch.randn(1, 2, OBS_DIM),
        Columns.STATE_IN: state,
    }
    base_output = base._forward_train(batch)
    assert Columns.ACTION_DIST_INPUTS in base_output
    assert NEXT_TOKEN_LOGITS not in base_output

    composed = make_module(
        gym.spaces.Discrete(3),
        module_class=TransformerWithTwoAuxHeads,
        mixin_config={
            "next_token_aux": {"num_classes": 3},
            "state_aux": {"num_classes": 4},
        },
    )
    composed_output = composed._forward_train(
        {
            Columns.OBS: torch.randn(1, 2, OBS_DIM),
            Columns.STATE_IN: initial_state(composed),
        }
    )
    assert composed_output[NEXT_TOKEN_LOGITS].shape == (1, 2, 3)
    assert composed_output[STATE_LOGITS].shape == (1, 2, 4)


def test_auxiliary_head_is_training_only_and_gradients_reach_encoder():
    module = make_module(
        gym.spaces.Box(-5.0, 5.0, (2,), np.float32),
        module_class=TransformerWithNextTokenAux,
        mixin_config={"next_token_aux": {"num_classes": 3}},
    )
    observations = torch.zeros(2, 5, OBS_DIM)
    observations[:, :, 0] = 1.0
    state = initial_state(module, batch_size=2)
    output = module._forward_train(
        {Columns.OBS: observations, Columns.STATE_IN: state}
    )
    next_tokens = observations[:, 1:, :3]
    targets = next_tokens.argmax(dim=-1)
    valid = next_tokens.sum(dim=-1) > 0.5
    loss = torch.nn.functional.cross_entropy(
        output[NEXT_TOKEN_LOGITS][:, :-1, :][valid],
        targets[valid],
    )
    loss.backward()

    assert module.encoder.input_projection.weight.grad is not None
    assert module.heads.policy.weight.grad is None
    rollout_output = module._forward(
        {
            Columns.OBS: torch.randn(2, 1, OBS_DIM),
            Columns.STATE_IN: state,
        }
    )
    assert NEXT_TOKEN_LOGITS not in rollout_output


def test_mlp_module_remains_memoryless():
    module = MLPModel(
        observation_space=gym.spaces.Box(
            -5.0,
            5.0,
            shape=(3,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(3),
        model_config={"hidden_dims": (32, 32)},
    )
    output = module._forward_train({Columns.OBS: torch.randn(7, 3)})

    assert output[Columns.ACTION_DIST_INPUTS].shape == (7, 3)
    assert module.compute_values({Columns.OBS: torch.randn(7, 3)}).shape == (
        7,
    )
    assert not module.is_stateful()
