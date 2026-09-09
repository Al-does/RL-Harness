from __future__ import annotations

import gc
import weakref

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

from learners.models.transformer import TransformerModel


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, config):
        self.horizon = config.get("horizon")
        self.truncate = config.get("truncate", False)
        self.steps = 0
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.resets += 1
        return np.zeros(2, dtype=np.float32), {"steps": self.steps, "resets": self.resets}

    def step(self, action):
        self.steps += 1
        done = self.horizon is not None and self.steps >= self.horizon
        return (
            np.array([self.steps % 2, action], dtype=np.float32),
            1.0,
            done and not self.truncate,
            done and self.truncate,
            {"steps": self.steps, "resets": self.resets},
        )


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**environment):
    return (
        PPOConfig()
        .environment(CountingEnv, env_config=environment)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=2, rollout_fragment_length=8)
        .rl_module(rl_module_spec=RLModuleSpec(
            module_class=TransformerModel,
            model_config={"d_model": 8, "n_layers": 1, "n_heads": 1, "context_len": 8, "max_seq_len": 4},
        ))
        .debugging(seed=42)
    )


def metric_value(metrics, key):
    value = metrics[key]
    return value.peek() if hasattr(value, "peek") else value


def kv_reference(chunk):
    data = chunk.extra_model_outputs[Columns.STATE_OUT].data
    return weakref.ref(data["kv_k"] if isinstance(data, dict) else data[0]["kv_k"])


def test_upstream_retains_every_stateful_chunk_even_after_metrics_polling():
    runner = SingleAgentEnvRunner(config=config())
    try:
        references = []
        for _ in range(8):
            chunks = runner.sample(num_timesteps=16, explore=False)
            assert all(Columns.STATE_OUT in chunk.extra_model_outputs for chunk in chunks)
            references.extend(kv_reference(chunk) for chunk in chunks)
            del chunks
            runner.get_metrics()
        gc.collect()
        assert sum(len(chunks) for chunks in runner._ongoing_episodes_for_metrics.values()) == 16
        assert all(reference() is not None for reference in references)
    finally:
        runner.stop()


@pytest.mark.parametrize("poll_metrics", (False, True))
@pytest.mark.parametrize("to_numpy", (False, True))
def test_continuing_runner_releases_chunks_without_resetting_state(poll_metrics, to_numpy):
    from harness.env_runners import ContinuingSingleAgentEnvRunner

    settings = config().env_runners(episodes_to_numpy=to_numpy)
    runner = ContinuingSingleAgentEnvRunner(config=settings)
    try:
        references = []
        reported_steps = 0
        episode_ids = None
        for sample in range(32):
            chunks = runner.sample(num_timesteps=16, explore=False)
            assert len(chunks) == 2
            ids = [chunk.id_ for chunk in chunks]
            episode_ids = ids if episode_ids is None else episode_ids
            assert ids == episode_ids
            for chunk in chunks:
                assert len(chunk) == 8 and not chunk.is_done
                assert chunk.get_infos(-1) == {"steps": (sample + 1) * 8, "resets": 1}
                assert Columns.STATE_OUT in chunk.extra_model_outputs
                state = chunk.get_extra_model_outputs(Columns.STATE_OUT, -1)
                assert state["kv_len"].item() == min((sample + 1) * 8, runner.module.encoder.cache_len)
                assert np.isfinite(state["kv_k"]).all()
                np.testing.assert_array_equal(chunk.get_rewards(), np.ones(8))
            del chunk, state
            references.extend(kv_reference(chunk) for chunk in chunks)
            assert not runner._ongoing_episodes_for_metrics
            del chunks
            if poll_metrics:
                metrics = runner.get_metrics()
                reported_steps += metric_value(metrics, "num_env_steps_sampled_lifetime")
                assert metric_value(metrics, "num_episodes_lifetime") == 0
        gc.collect()
        assert all(reference() is None for reference in references)
        metrics = runner.get_metrics()
        reported_steps += metric_value(metrics, "num_env_steps_sampled_lifetime")
        assert reported_steps == 512
        assert metric_value(metrics, "num_episodes_lifetime") == 0
        assert "episode_return_mean" not in metrics
    finally:
        runner.stop()


def trajectory(runner_class):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        runner = runner_class(config=config())
    observations, actions, states = [], [], []
    try:
        for _ in range(8):
            for chunk in runner.sample(num_timesteps=16, explore=False):
                observations.append(chunk.get_observations())
                actions.append(chunk.get_actions())
                states.append(chunk.get_extra_model_outputs(Columns.STATE_OUT, -1))
            runner.get_metrics()
        return observations, actions, states
    finally:
        runner.stop()


def test_sampler_outputs_and_recurrent_context_match_upstream():
    from harness.env_runners import ContinuingSingleAgentEnvRunner

    before = trajectory(SingleAgentEnvRunner)
    after = trajectory(ContinuingSingleAgentEnvRunner)
    for old, new in zip(before[:2], after[:2]):
        np.testing.assert_array_equal(old, new)
    for old, new in zip(before[2], after[2]):
        assert old.keys() == new.keys()
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])


def test_rejects_episode_count_sampling_before_entering_upstream(monkeypatch):
    from harness.env_runners import ContinuingSingleAgentEnvRunner

    runner = ContinuingSingleAgentEnvRunner(config=config())
    try:
        def forbidden_sample(*args, **kwargs):
            pytest.fail("episode-count sampling would never finish")

        monkeypatch.setattr(SingleAgentEnvRunner, "sample", forbidden_sample)
        with pytest.raises(ValueError, match="num_episodes"):
            runner.sample(num_episodes=1)
    finally:
        runner.stop()


def test_sampling_failure_also_releases_metrics_references(monkeypatch):
    from harness.env_runners import ContinuingSingleAgentEnvRunner

    runner = ContinuingSingleAgentEnvRunner(config=config())
    try:
        def failed_sample(self, **kwargs):
            self._ongoing_episodes_for_metrics["partial"].append(object())
            raise RuntimeError("sampling failed")

        monkeypatch.setattr(SingleAgentEnvRunner, "sample", failed_sample)
        with pytest.raises(RuntimeError, match="sampling failed"):
            runner.sample(num_timesteps=16)
        assert not runner._ongoing_episodes_for_metrics
    finally:
        runner.stop()


def test_rejects_complete_episode_batch_mode():
    from harness.env_runners import ContinuingSingleAgentEnvRunner

    with pytest.raises(ValueError, match="truncate_episodes"):
        ContinuingSingleAgentEnvRunner(config=config().env_runners(batch_mode="complete_episodes"))


@pytest.mark.parametrize("truncate", (False, True))
def test_unexpected_episode_end_fails_instead_of_reporting_partial_metrics(truncate):
    from harness.env_runners import ContinuingSingleAgentEnvRunner

    runner = ContinuingSingleAgentEnvRunner(config=config(horizon=12, truncate=truncate))
    try:
        runner.sample(num_timesteps=16, explore=False)
        with pytest.raises(ValueError, match="non-terminating"):
            runner.sample(num_timesteps=16, explore=False)
        assert not runner._ongoing_episodes_for_metrics
    finally:
        runner.stop()


@pytest.mark.parametrize("truncate", (False, True))
def test_standard_runner_keeps_complete_finite_episode_metrics(truncate):
    runner = SingleAgentEnvRunner(config=config(horizon=12, truncate=truncate))
    try:
        runner.sample(num_timesteps=16, explore=False)
        runner.get_metrics()
        runner.sample(num_timesteps=16, explore=False)
        metrics = runner.get_metrics()
        assert metric_value(metrics, "episode_len_mean") == 12
        assert metric_value(metrics, "episode_return_mean") == 12.0
        assert metric_value(metrics, "num_episodes_lifetime") == 2
    finally:
        runner.stop()
