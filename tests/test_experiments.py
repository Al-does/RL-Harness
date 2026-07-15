"""Import and construction smoke tests for concrete experiment recipes."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from envs.mess3.env_continuous import Mess3ContinuousEnv
from experiments.mess3_belief_geometry_2026_07.probe import (
    collect_probe_data,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models import TransformerModel


FAMILY = "experiments.mess3_belief_geometry_2026_07"
FAMILY_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "mess3_belief_geometry_2026_07"
)


def experiment_modules() -> list[str]:
    return sorted(
        f"{FAMILY}.{path.parent.name}.experiment"
        for path in FAMILY_PATH.glob("*/experiment.py")
    )


def test_all_migrated_experiment_leaves_import():
    modules = experiment_modules()

    assert len(modules) == 23
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert callable(module.run)


def test_all_rllib_recipes_build_fresh_smoke_configs(tmp_path):
    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
        hardware=PROFILES["cpu"],
    )
    built = 0

    for module_name in experiment_modules():
        module = importlib.import_module(module_name)
        if not hasattr(module, "build_config"):
            continue
        first = module.build_config(context)
        second = module.build_config(context)
        built += 1

        assert first is not second
        assert first.seed == 42
        assert first.num_env_runners == 0
        assert first.train_batch_size_per_learner == 2048

    assert built == 15


def test_mess3_probe_uses_batched_generic_rollout_collection():
    environment_config = {"episode_length": 3}

    def make_environment():
        return Mess3ContinuousEnv(environment_config)

    environment = make_environment()
    try:
        module = TransformerModel(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config={
                "context_len": 4,
                "d_model": 24,
                "n_layers": 1,
                "n_heads": 3,
                "max_seq_len": 3,
            },
        )
    finally:
        environment.close()

    for policy_mode in ("random", "greedy", "policy"):
        data = collect_probe_data(
            module,
            make_environment,
            n_steps=7,
            seed=42,
            policy_mode=policy_mode,
            n_envs=2,
            warmup=1,
        )

        assert data.activations.shape == (7, 24)
        assert data.beliefs.shape == (7, 3)
        assert data.actions.shape == (7, 2)
        assert data.tokens.shape == (7,)
        assert data.previous_tokens.shape == (7,)
        assert data.states.shape == (7,)
        assert data.rewards.shape == (7,)
        np.testing.assert_allclose(data.beliefs.sum(axis=1), 1.0)
        assert np.all(np.abs(data.actions) <= 5.0)
