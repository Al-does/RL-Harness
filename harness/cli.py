"""Command-line entry point for importable experiment recipes."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

# Ray inspects this at import time, so set it before loading an experiment that
# imports RLlib. Workers should use the already-synchronized project runtime.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from harness.artifacts import (
    finish_run_manifest,
    maybe_upload_run_artifacts,
    start_run_manifest,
)
from harness.context import DEFAULT_SEED, RunContext, new_run_id
from harness.hardware import PROFILES, HardwareProfile, detect_profile


@dataclass(frozen=True, slots=True)
class LoadedExperiment:
    module_name: str
    module: ModuleType
    file: Path
    run: Callable[[RunContext], Any]

    @property
    def directory(self) -> Path:
        return self.file.parent


def load_experiment(module_name: str) -> LoadedExperiment:
    """Import and validate one ``...experiment`` recipe module."""
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise TypeError(f"{module_name!r} has no source file")
    path = Path(module_file).resolve()
    if path.name != "experiment.py":
        raise ValueError(
            f"{module_name!r} must resolve to a leaf experiment.py"
        )
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError(f"{module_name!r} must define callable run(context)")
    return LoadedExperiment(module_name, module, path, run)


def _hardware_profile(name: str) -> HardwareProfile:
    resolved = detect_profile() if name == "auto" else name
    return PROFILES[resolved]


def make_run_context(
    experiment: LoadedExperiment,
    *,
    seed: int | None = DEFAULT_SEED,
    run_id: str | None = None,
    smoke: bool = False,
    resume_from: Path | None = None,
    results_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    hardware_profile: str = "auto",
) -> RunContext:
    """Resolve operational CLI inputs into the immutable run context."""
    resolved_run_id = run_id or new_run_id()
    result_path = (
        Path(results_dir)
        if results_dir is not None
        else experiment.directory / "results" / resolved_run_id
    )
    artifact_path = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else experiment.directory / "artifacts" / resolved_run_id
    )
    return RunContext(
        experiment_dir=experiment.directory,
        results_dir=result_path,
        artifacts_dir=artifact_path,
        seed=seed,
        run_id=resolved_run_id,
        smoke=smoke,
        resume_from=resume_from,
        hardware=_hardware_profile(hardware_profile),
    )


def execute_experiment(
    experiment: LoadedExperiment,
    context: RunContext,
    *,
    command: Sequence[str] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    upload_artifacts: bool | None = None,
) -> Any:
    """Run an experiment while recording start, completion, and failure."""
    start_run_manifest(
        context,
        experiment_module=experiment.module_name,
        experiment_file=experiment.file,
        command=list(command) if command is not None else None,
        runtime_overrides=runtime_overrides,
    )
    try:
        result = experiment.run(context)
    except BaseException as error:
        maybe_upload_run_artifacts(
            context,
            upload=upload_artifacts,
            experiment_module=experiment.module_name,
        )
        finish_run_manifest(context, status="failed", error=error)
        raise
    maybe_upload_run_artifacts(
        context,
        upload=upload_artifacts,
        experiment_module=experiment.module_name,
    )
    finish_run_manifest(context, status="completed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an importable RL experiment recipe."
    )
    parser.add_argument(
        "experiment",
        help="dotted module path ending in .experiment",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="request the experiment's minimal wiring-check budget",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument(
        "--hardware-profile",
        default="auto",
        choices=["auto", *sorted(PROFILES)],
    )
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "upload artifacts/ to Backblaze B2 when configured "
            "(default: upload when B2_* environment variables are set)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment = load_experiment(args.experiment)
    context = make_run_context(
        experiment,
        seed=args.seed,
        run_id=args.run_id,
        smoke=args.smoke,
        resume_from=args.resume_from,
        results_dir=args.results_dir,
        artifacts_dir=args.artifacts_dir,
        hardware_profile=args.hardware_profile,
    )
    execute_experiment(
        experiment,
        context,
        runtime_overrides={
            "hardware_profile": args.hardware_profile,
            "seed": args.seed,
            "smoke": args.smoke,
            "resume_from": args.resume_from,
            "upload_artifacts": args.upload_artifacts,
        },
        upload_artifacts=args.upload_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
