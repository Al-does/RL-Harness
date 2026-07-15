"""Import and construction smoke tests for concrete experiment recipes."""

from __future__ import annotations

import importlib
from pathlib import Path

from harness.context import RunContext
from harness.hardware import PROFILES


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
