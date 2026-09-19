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


def _record_tune_history(context: RunContext, result: Any) -> None:
    """Persist every Tune scalar-metric row as compact generic progress."""

    dataframe = getattr(result, "metrics_dataframe", None)
    if dataframe is None:
        return
    trial_id = flatten_scalar_metrics(dict(result.metrics or {})).get(
        "trial_id"
    )
    artifacts = RunArtifacts.from_context(context)
    for _, row in dataframe.iterrows():
        values = {
            str(key): value
            for key, value in row.to_dict().items()
            if str(key) != "config"
            and not str(key).startswith("config/")
        }
        if trial_id is not None:
            values.setdefault("trial_id", trial_id)
        artifacts.append_result(values)


def _should_upload_checkpoint(context: RunContext, upload: bool | None) -> bool:
    if upload is None:
        if context.smoke and not context.publish_smoke:
            return False
        from harness.storage import is_b2_configured

        return is_b2_configured()
    return upload


def _upload_checkpoint(
    context: RunContext, path: Path, *, upload: bool | None
) -> None:
    """Best-effort B2 upload of a freshly saved checkpoint.

    Failures are logged and never abort training; the end-of-run
    ``maybe_upload_run_artifacts`` pass remains the durability backstop.
    """
    from harness.storage import is_b2_configured, upload_artifact_directory

    if upload is True and not is_b2_configured():
        raise RuntimeError(
            "checkpoint upload was requested but B2 is not configured. "
            "Set B2_BUCKET, B2_ENDPOINT, B2_APPLICATION_KEY_ID, and "
            "B2_APPLICATION_KEY."
        )
    if not _should_upload_checkpoint(context, upload):
        return
    try:
        summary = upload_artifact_directory(context, path)
    except Exception as error:  # noqa: BLE001 - upload must not kill training
        print(f"[checkpoint-upload] FAILED {path.name}: {error}", flush=True)
        return
    print(
        f"[checkpoint-upload] {path.name}: {summary['file_count']} files "
        f"({summary['total_bytes']} bytes) -> {summary['base_uri']}",
        flush=True,
    )


def save_algorithm_checkpoint(
    algorithm: Any,
    context: RunContext,
    *,
    label: str,
    root: Path | None = None,
    upload: bool | None = None,
) -> Path:
    """Save an Algorithm through RLlib's public Checkpointable API.

    Saves under ``root`` (default: the run's ``checkpoints/`` directory) and,
    unless ``upload`` is ``False``, uploads the saved directory to B2 right
    away when B2 is configured. Throwaway smoke runs skip the upload unless
    ``upload`` is explicitly ``True``. Tune-internal checkpoints are saved by
    the framework itself and remain covered by the end-of-run upload.
    """
    root = root or RunArtifacts.from_context(context).checkpoints_dir
    root.mkdir(parents=True, exist_ok=True)
    saved_path = Path(algorithm.save_to_path(root / label))
    _upload_checkpoint(context, saved_path, upload=upload)
    return saved_path


def _build_or_restore_algorithm(
    config: AlgorithmConfigLike, context: RunContext
) -> Any:
    algorithm = config.build_algo()
    if context.resume_from is not None:
        algorithm.restore(str(context.resume_from))
    return algorithm


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
            raw_iter = result.get("training_iteration")
            iter_label = int(raw_iter) if raw_iter is not None else iteration
            if checkpoint_interval and iter_label % checkpoint_interval == 0:
                save_algorithm_checkpoint(
                    algorithm,
                    context,
                    label=f"iteration_{iter_label:06d}",
                )
            if should_stop(result):
                if checkpoint_at_end and not (
                    checkpoint_interval
                    and iter_label % checkpoint_interval == 0
                ):
                    save_algorithm_checkpoint(
                        algorithm,
                        context,
                        label=f"iteration_{iter_label:06d}_final",
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
            _record_tune_history(context, result)
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
