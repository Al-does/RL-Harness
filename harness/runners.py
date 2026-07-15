"""Thin execution helpers for direct RLlib and Tune-managed runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from harness.artifacts import (
    RunArtifacts,
    flatten_scalar_metrics,
    record_result,
    update_run_manifest,
)
from harness.context import RunContext
from harness.hardware import configure_hardware, shutdown_ray_if_owned


class AlgorithmConfigLike(Protocol):
    algo_class: type

    def build_algo(self): ...

    def to_dict(self) -> dict[str, Any]: ...


StopCondition = Callable[[Mapping[str, Any]], bool]
ResultRecorder = Callable[[RunContext, Mapping[str, Any]], None]


def save_algorithm_checkpoint(
    algorithm: Any,
    context: RunContext,
    *,
    label: str,
) -> Path:
    """Save an Algorithm through RLlib's public Checkpointable API."""
    root = RunArtifacts.from_context(context).checkpoints_dir
    root.mkdir(parents=True, exist_ok=True)
    saved_path = algorithm.save_to_path(root / label)
    return Path(saved_path)


def _build_or_restore_algorithm(
    config: AlgorithmConfigLike, context: RunContext
) -> Any:
    if context.resume_from is None:
        return config.build_algo()
    from ray.rllib.algorithms.algorithm import Algorithm

    return Algorithm.from_checkpoint(str(context.resume_from))


def run_algorithm(
    config: AlgorithmConfigLike,
    context: RunContext,
    *,
    should_stop: StopCondition,
    recorder: ResultRecorder = record_result,
    checkpoint_interval: int | None = None,
    checkpoint_at_end: bool = False,
) -> Mapping[str, Any]:
    """Run an ordinary direct RLlib loop with guaranteed cleanup."""
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    started_ray = (
        configure_hardware(context.hardware)
        if context.hardware is not None
        else False
    )
    algorithm = None
    iteration = 0
    try:
        algorithm = _build_or_restore_algorithm(config, context)
        while True:
            result = algorithm.train()
            iteration += 1
            recorder(context, result)
            if checkpoint_interval and iteration % checkpoint_interval == 0:
                save_algorithm_checkpoint(
                    algorithm,
                    context,
                    label=f"iteration_{iteration:06d}",
                )
            if should_stop(result):
                if checkpoint_at_end and not (
                    checkpoint_interval
                    and iteration % checkpoint_interval == 0
                ):
                    save_algorithm_checkpoint(
                        algorithm,
                        context,
                        label=f"iteration_{iteration:06d}_final",
                    )
                return result
    finally:
        try:
            if algorithm is not None:
                algorithm.stop()
        finally:
            shutdown_ray_if_owned(started_ray)


def build_tuner(
    config: AlgorithmConfigLike,
    context: RunContext,
    *,
    stop: Mapping[str, Any] | Callable[[str, Mapping[str, Any]], bool] | None,
    tune_config: Any = None,
    run_config_kwargs: Mapping[str, Any] | None = None,
):
    """Construct one RLlib Tune run rooted under this run's artifacts."""
    from ray import tune

    kwargs = dict(run_config_kwargs or {})
    if "storage_path" in kwargs or "local_dir" in kwargs:
        raise ValueError(
            "Tune storage is owned by RunContext.artifacts_dir"
        )
    kwargs.setdefault("name", "tune")
    run_config = tune.RunConfig(
        storage_path=str(context.artifacts_dir),
        stop=stop,
        **kwargs,
    )
    param_space = config.to_dict()
    if context.resume_from is not None:
        return tune.Tuner.restore(
            str(context.resume_from),
            trainable=config.algo_class,
            param_space=param_space,
        )
    return tune.Tuner(
        trainable=config.algo_class,
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )


def run_tune(
    config: AlgorithmConfigLike,
    context: RunContext,
    *,
    stop: Mapping[str, Any] | Callable[[str, Mapping[str, Any]], bool] | None,
    tune_config: Any = None,
    run_config_kwargs: Mapping[str, Any] | None = None,
):
    """Build and fit a Tune-managed RLlib run with owned Ray cleanup."""
    started_ray = (
        configure_hardware(context.hardware)
        if context.hardware is not None
        else False
    )
    try:
        tuner = build_tuner(
            config,
            context,
            stop=stop,
            tune_config=tune_config,
            run_config_kwargs=run_config_kwargs,
        )
        result_grid = tuner.fit()
        trials = []
        for result in result_grid:
            result_metrics = dict(result.metrics or {})
            result_metrics.pop("config", None)
            metrics = flatten_scalar_metrics(result_metrics)
            checkpoint = getattr(result, "checkpoint", None)
            checkpoint_path = (
                str(checkpoint.path)
                if checkpoint is not None
                and getattr(checkpoint, "path", None) is not None
                else None
            )
            config_values = getattr(result, "config", {}) or {}
            trials.append(
                {
                    "trial_id": metrics.get("trial_id"),
                    "path": getattr(result, "path", None),
                    "status": (
                        "failed"
                        if getattr(result, "error", None) is not None
                        else "completed"
                    ),
                    "error": (
                        str(result.error)
                        if getattr(result, "error", None) is not None
                        else None
                    ),
                    "resolved_seed": config_values.get("seed"),
                    "checkpoint": checkpoint_path,
                    "metrics": metrics,
                }
            )
        summary = {"num_trials": len(trials), "trials": trials}
        RunArtifacts.from_context(context).write_json(
            "tune_summary.json",
            summary,
        )
        update_run_manifest(
            context,
            trials=[
                {
                    "trial_id": trial["trial_id"],
                    "resolved_seed": trial["resolved_seed"],
                    "status": trial["status"],
                    "checkpoint": trial["checkpoint"],
                }
                for trial in trials
            ],
        )
        return result_grid
    finally:
        shutdown_ray_if_owned(started_ray)
