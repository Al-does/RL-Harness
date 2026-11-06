"""Recurrent and memoryless baseline contracts, including real PPO updates."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from learners.models import (
    GRUModel, GRUModelConfig, ResidualMLPModel, ResidualMLPModelConfig,
)


def make_module(cls, **config):
    return cls(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        action_space=gym.spaces.Discrete(3),
        model_config={"hidden_dim": 16, "d_model": 16, "n_layers": 2, **config},
    )


def initial(module, batch=2):
    return {key: torch.from_numpy(value).unsqueeze(0).repeat(batch, 1, 1)
            for key, value in module.get_initial_state().items()}


def test_gru_chunk_matches_steps_and_carries_state():
    module = make_module(GRUModel)
    assert module.is_stateful()
    assert module.get_initial_state()["h"].shape == (2, 16)
    obs = torch.randn(2, 7, 5)
    state = initial(module)
    whole = module._forward_train({Columns.OBS: obs, Columns.STATE_IN: state})
    steps = []
    for token in obs.unbind(1):
        step = module._forward({Columns.OBS: token[:, None], Columns.STATE_IN: state})
        steps.append(step[Columns.ACTION_DIST_INPUTS])
        state = step[Columns.STATE_OUT]
    torch.testing.assert_close(torch.cat(steps, 1), whole[Columns.ACTION_DIST_INPUTS])
    torch.testing.assert_close(state["h"], whole[Columns.STATE_OUT]["h"])
    first = module._forward_train({Columns.OBS: obs[:, :3], Columns.STATE_IN: initial(module)})
    second = module._forward_train({Columns.OBS: obs[:, 3:], Columns.STATE_IN: first[Columns.STATE_OUT]})
    torch.testing.assert_close(second[Columns.ACTION_DIST_INPUTS], whole[Columns.ACTION_DIST_INPUTS][:, 3:])
    torch.testing.assert_close(second[Columns.STATE_OUT]["h"], state["h"])
    whole[Columns.EMBEDDINGS].square().sum().backward()
    assert module.encoder.gru.weight_hh_l0.grad is not None


def test_memoryless_timestep_independence():
    module = make_module(ResidualMLPModel)
    assert not module.is_stateful()
    assert not module.get_initial_state()
    obs = torch.randn(2, 5, 5)
    original = module._forward_train({Columns.OBS: obs})
    altered = torch.randn_like(obs)
    altered[:, 2] = obs[:, 2]
    changed = module._forward_train({Columns.OBS: altered})
    torch.testing.assert_close(original[Columns.ACTION_DIST_INPUTS][:, 2], changed[Columns.ACTION_DIST_INPUTS][:, 2])
    step, _ = module.encode_step(obs[:, 2])
    torch.testing.assert_close(step, original[Columns.EMBEDDINGS][:, 2])
    assert Columns.STATE_OUT not in original


@pytest.mark.parametrize("cls", [GRUModel, ResidualMLPModel])
def test_feature_selection_and_probe_norm(cls):
    module = make_module(cls, obs_feature_indices=(0, 2))
    obs = torch.randn(2, 5)
    changed = obs.clone()
    changed[:, [1, 3, 4]] += 100
    state = initial(module) if module.is_stateful() else None
    embedding, next_state = module.encode_step(obs, state)
    other, _ = module.encode_step(changed, state)
    torch.testing.assert_close(embedding, other)
    residual, residual_state = module.encode_step_pre_final_norm(obs, state)
    torch.testing.assert_close(module.encoder.final_norm(residual), embedding)
    if module.is_stateful():
        torch.testing.assert_close(next_state["h"], residual_state["h"])
    assert module.action_distribution_inputs(embedding).shape == (2, 3)


@pytest.mark.parametrize("config", [GRUModelConfig, ResidualMLPModelConfig])
def test_config_roundtrip_and_invalid_features(config):
    value = config.from_dict({"n_layers": 3, "obs_feature_indices": [0, 2], "unrelated": 9})
    assert config.from_dict(value.to_dict()) == value
    assert value.override(n_layers=1).n_layers == 1
    assert value.obs_feature_indices == (0, 2)
    for indices in ((), (-1,), (1.5,)):
        with pytest.raises(ValueError, match="obs_feature_indices"):
            config(obs_feature_indices=indices)
    with pytest.raises(ValueError, match="positive"):
        config(n_layers=0)


@pytest.mark.parametrize("cls", [GRUModel, ResidualMLPModel])
def test_out_of_bounds_features(cls):
    with pytest.raises(ValueError, match="observation dimension"):
        make_module(cls, obs_feature_indices=(5,))


@pytest.mark.parametrize("cls", [GRUModel, ResidualMLPModel])
def test_ppo_cpu_train(cls, tmp_path):
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1, rollout_fragment_length=32)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(train_batch_size_per_learner=32, minibatch_size=16, num_epochs=1)
        .rl_module(rl_module_spec=RLModuleSpec(module_class=cls, model_config={
            "hidden_dim": 16, "d_model": 16, "n_layers": 2, "max_seq_len": 8,
            "obs_feature_indices": (0, 2),
        }))
        .debugging(seed=42)
    )
    algo = config.build_algo()
    try:
        module = algo.get_module()
        print(f"{cls.__name__}: {sum(p.numel() for p in module.parameters())} parameters")
        before = {name: p.detach().clone() for name, p in module.named_parameters()}
        result = algo.train()
        assert result["training_iteration"] == 1
        assert any(not torch.equal(before[name], p) for name, p in module.named_parameters())
        assert np.isfinite(result["learners"]["default_policy"]["total_loss"])
    finally:
        algo.stop()
