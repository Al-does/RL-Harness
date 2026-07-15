"""Configuration and construction tests for reusable RLModule pieces."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from learners.models import (
    MLPModelConfig,
    NextTokenAuxHead,
    TransformerModel,
    TransformerModelConfig,
)


class TransformerWithTestHead(NextTokenAuxHead, TransformerModel):
    pass


def make_module(module_class=TransformerModel, model_config=None):
    config = TransformerModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        context_len=4,
        max_seq_len=5,
    ).to_dict()
    config.update(model_config or {})
    return module_class(
        observation_space=gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(5,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(3),
        model_config=config,
    )


def test_model_configs_validate_component_constraints():
    with pytest.raises(ValueError, match="divisible"):
        TransformerModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError, match="positive widths"):
        MLPModelConfig(hidden_dims=())


def test_base_transformer_contains_only_policy_and_value_outputs():
    module = make_module()
    initial = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    output = module._forward_train(
        {
            Columns.OBS: torch.randn(1, 2, 5),
            Columns.STATE_IN: initial,
        }
    )

    assert Columns.ACTION_DIST_INPUTS in output
    assert Columns.EMBEDDINGS in output
    assert "next_token_aux/logits" not in output


def test_head_mixin_uses_namespaced_component_configuration():
    module = make_module(
        TransformerWithTestHead,
        {"next_token_aux": {"num_classes": 4}},
    )
    initial = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in module.get_initial_state().items()
    }
    output = module._forward_train(
        {
            Columns.OBS: torch.randn(1, 2, 5),
            Columns.STATE_IN: initial,
        }
    )

    assert output["next_token_aux/logits"].shape == (1, 2, 4)
