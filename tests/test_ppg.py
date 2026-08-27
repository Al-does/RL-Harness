"""Focused tests for reusable Phasic Policy Gradient components."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from learners import PPGConfig, PPGTorchLearner
from learners.models import (
    MLPModel,
    PPGAuxiliaryValueHead,
    PPGMLPModel,
    PPGTransformerModel,
    TransformerModel,
)
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


def test_policy_phase_value_loss_detaches_the_shared_encoder():
    module = _module()
    forward = module._forward_train({Columns.OBS: torch.randn(8, 4)})
    distribution = module.get_train_action_dist_cls().from_logits(
        forward[Columns.ACTION_DIST_INPUTS]
    )
    actions = distribution.sample()
    batch = {
        Columns.ACTIONS: actions,
        Columns.ACTION_LOGP: distribution.logp(actions).detach(),
        Columns.ACTION_DIST_INPUTS: forward[
            Columns.ACTION_DIST_INPUTS
        ].detach(),
        Postprocessing.ADVANTAGES: torch.zeros(8),
        Postprocessing.VALUE_TARGETS: torch.randn(8),
    }

    class Wrapper:
        def unwrapped(self):
            return module

    class ZeroScheduler:
        @staticmethod
        def get_current_value():
            return 0.0

    class Metrics:
        @staticmethod
        def log_dict(*args, **kwargs):
            return None

    learner = type(
        "PolicyLossHarness",
        (),
        {
            "module": {"default_policy": Wrapper()},
            "entropy_coeff_schedulers_per_module": {
                "default_policy": ZeroScheduler()
            },
            "metrics": Metrics(),
        },
    )()
    config = type(
        "PolicyLossConfig",
        (),
        {
            "clip_param": 0.2,
            "vf_clip_param": 1e9,
            "vf_loss_coeff": 1.0,
            "use_kl_loss": False,
        },
    )()

    loss = PPGTorchLearner._compute_policy_loss(
        learner,
        module_id="default_policy",
        config=config,
        batch=batch,
        fwd_out=forward,
    )
    loss.backward()

    encoder_grad = module.encoder[0].weight.grad
    assert encoder_grad is None or torch.count_nonzero(encoder_grad) == 0
    assert module.heads.value.weight.grad is not None
    assert torch.count_nonzero(module.heads.value.weight.grad) > 0
    assert module.ppg_auxiliary_value_head.weight.grad is None


def test_ppg_config_exposes_canonical_phase_defaults():
    config = PPGConfig()

    assert config.get_default_rl_module_spec().module_class is PPGMLPModel
    assert config.policy_iterations_per_aux == 32
    assert config.aux_epochs == 6
    assert config.aux_minibatch_size == 128
    assert config.beta_clone == 1.0
    assert config.aux_value_loss_coeff == 1.0
    assert config.aux_true_value_loss_coeff == 1.0


def test_ppg_exports_ready_to_use_stock_model_compositions():
    assert issubclass(PPGMLPModel, PPGAuxiliaryValueHead)
    assert issubclass(PPGMLPModel, MLPModel)
    assert issubclass(PPGTransformerModel, PPGAuxiliaryValueHead)
    assert issubclass(PPGTransformerModel, TransformerModel)


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


def test_ppg_config_rejects_multiple_learners():
    config = (
        PPGConfig()
        .environment("CartPole-v1")
        .learners(num_learners=2, num_gpus_per_learner=0)
    )

    with pytest.raises(ValueError, match="at most one Learner"):
        config.validate()


def test_ppg_config_rejects_old_api_stack():
    config = PPGConfig().environment("CartPole-v1").api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    with pytest.raises(ValueError, match="requires RLlib's new API stack"):
        config.validate()
