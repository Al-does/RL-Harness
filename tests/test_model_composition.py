"""Configuration and reconstruction tests for composable RLlib models."""

from __future__ import annotations

import json

import pytest
import torch

from analysis.checkpoints import load_module, module_from_blueprint
from blueprints.base import ModelSpec, get
from learners.models import (
    MLPModelConfig,
    TransformerModel,
    TransformerModelConfig,
    TransformerWithNextTokenAux,
    TransformerWithStateAux,
)
from learners.ppo import PPOWithNextTokenAux
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
    PPOTorchLearner,
)


def test_blueprint_selects_class_and_supports_immutable_overrides():
    blueprint = get("a_main")
    assert blueprint.model.model_class is TransformerModel

    wider = blueprint.model.with_config(d_model=128)
    assert wider.config.d_model == 128
    assert blueprint.model.config.d_model == 96
    assert blueprint.learner_class is PPOTorchLearner
    assert blueprint.aux_config == {}
    assert wider.to_dict()["class"] == (
        "learners.models.transformer:TransformerModel"
    )
    json.dumps(wider.to_dict())


def test_aux_blueprint_composes_namespaced_model_and_learner_mixins():
    blueprint = get("a_aux_0p1")
    assert blueprint.model.model_class is TransformerWithNextTokenAux
    assert blueprint.learner_class is PPOWithNextTokenAux
    assert blueprint.aux_config == {"next_token_aux/lambda": 0.1}
    assert blueprint.model.to_model_config()["next_token_aux"] == {
        "num_classes": 3
    }


def test_supervised_blueprints_select_semantic_head_mixins():
    state_blueprint = get("b_sl")
    token_blueprint = get("a_pred")

    assert state_blueprint.model.model_class is TransformerWithStateAux
    assert state_blueprint.model.to_model_config()["state_aux"] == {
        "num_classes": 3
    }
    assert (
        token_blueprint.model.model_class
        is TransformerWithNextTokenAux
    )
    assert token_blueprint.aux_config == {}


def test_model_configs_validate_component_constraints():
    with pytest.raises(ValueError, match="divisible"):
        TransformerModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError, match="positive widths"):
        MLPModelConfig(hidden_dims=())


def test_new_blueprint_schema_reconstructs_and_loads_checkpoint(tmp_path):
    model = ModelSpec(
        TransformerWithNextTokenAux,
        TransformerModelConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            context_len=4,
            max_seq_len=5,
        ),
        mixin_config={"next_token_aux": {"num_classes": 3}},
    )
    blueprint = {
        "env_entry": "envs.mess3.env_continuous:Mess3ContinuousEnv",
        "env_kwargs": {"episode_length": 16},
        "scramble_tokens": False,
        "model": model.to_dict(),
    }
    module = module_from_blueprint(blueprint)
    assert isinstance(module, TransformerWithNextTokenAux)
    assert module.encoder.d_model == 32
    assert module.next_token_aux_head.out_features == 3

    (tmp_path / "blueprint.json").write_text(json.dumps(blueprint))
    checkpoint = tmp_path / "module_state_00000000.pt"
    torch.save(
        {"state_dict": module.state_dict(), "env_steps": 0}, checkpoint
    )
    loaded = load_module(tmp_path, checkpoint)
    for expected, actual in zip(module.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
