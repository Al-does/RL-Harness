"""Generic result, artifact, and run-provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from harness.context import RunContext


MANIFEST_FILENAME = "run_manifest.json"
PROGRESS_FILENAME = "progress.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_value(repository: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_state(repository: Path | None) -> dict[str, Any]:
    if repository is None:
        return {"root": None, "commit": None, "dirty": None}
    status = _git_value(repository, "status", "--porcelain")
    return {
        "root": str(repository),
        "commit": _git_value(repository, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _library_package_info() -> dict[str, Any]:
    """Locate the installed rl-harness checkout and record commit + version."""
    import harness as harness_package

    harness_file = getattr(harness_package, "__file__", None)
    library_root = (
        _repository_root(Path(harness_file).resolve())
        if harness_file is not None
        else None
    )
    version: str | None
    try:
        version = metadata.version("rl-harness")
    except metadata.PackageNotFoundError:
        version = None
    info = _git_state(library_root)
    info["version"] = version
    info["package"] = "rl-harness"
    return info


def _framework_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in ("ray", "torch", "gymnasium", "numpy", "rl-harness"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions

def _hardware_summary(context: RunContext) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    if context.hardware is not None:
        summary["profile"] = (
            asdict(context.hardware)
            if is_dataclass(context.hardware)
            else str(context.hardware)
        )
    try:
        import torch

        summary["cuda_available"] = torch.cuda.is_available()
        summary["mps_available"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            summary["cuda_device_count"] = torch.cuda.device_count()
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        pass
    return summary


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths and generic records associated with one run."""

    results_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_context(cls, context: RunContext) -> RunArtifacts:
        return cls(context.results_dir, context.artifacts_dir)

    @property
    def manifest_path(self) -> Path:
        return self.results_dir / MANIFEST_FILENAME

    @property
    def progress_path(self) -> Path:
        return self.results_dir / PROGRESS_FILENAME

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def tune_dir(self) -> Path:
        return self.artifacts_dir / "tune"

    def prepare(self) -> None:
        if self.results_dir == self.artifacts_dir:
            raise ValueError("result and artifact directories must be distinct")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def append_result(self, result: Mapping[str, Any]) -> None:
        """Append generic scalar metrics without assuming metric names."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        flattened = {
            "recorded_at": _utc_now(),
            **flatten_scalar_metrics(result),
        }
        with self.progress_path.open("a") as handle:
            handle.write(json.dumps(flattened, sort_keys=True) + "\n")

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        if Path(filename).name != filename:
            raise ValueError("filename must not contain directory components")
        path = self.results_dir / filename
        _write_json(path, payload)
        return path


def flatten_scalar_metrics(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten scalar leaves without assuming framework metric names."""
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{prefix}/{key}" if prefix else str(key)
                visit(child, item)
        elif value is None or isinstance(
            value, (str, bool, Integral, Real)
        ):
            flattened[prefix] = _json_value(value)

    visit("", values)
    return flattened


def start_run_manifest(
    context: RunContext,
    *,
    experiment_module: str,
    experiment_file: Path,
    command: list[str] | tuple[str, ...] | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the compact provenance record for a starting run."""
    artifacts = RunArtifacts.from_context(context)
    artifacts.prepare()
    experiment_file = Path(experiment_file).resolve()
    experiment_repository = _repository_root(experiment_file.parent)
    experiment_git = _git_state(experiment_repository)
    library_git = _library_package_info()
    lock_file = (
        experiment_repository / "uv.lock"
        if experiment_repository is not None
        else None
    )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "run_id": context.run_id,
        "status": "running",
        "started_at": _utc_now(),
        "ended_at": None,
        "experiment": {
            "module": experiment_module,
            "file": str(experiment_file),
            "source_sha256": (
                _sha256(experiment_file) if experiment_file.is_file() else None
            ),
        },
        # Dual-repo provenance: experiment science repo + shared library.
        # Top-level commit/dirty remain the experiment repo for older readers.
        "git": {
            "commit": experiment_git["commit"],
            "dirty": experiment_git["dirty"],
            "experiment_repository": experiment_git,
            "library": library_git,
        },
        "dependency_lock": {
            "file": str(lock_file) if lock_file is not None else None,
            "sha256": (
                _sha256(lock_file)
                if lock_file is not None and lock_file.is_file()
                else None
            ),
        },
        "framework_versions": _framework_versions(),
        "command": list(command if command is not None else sys.argv),
        "runtime": {
            "seed": context.seed,
            "smoke": context.smoke,
            "resume_from": context.resume_from,
            "results_dir": context.results_dir,
            "artifacts_dir": context.artifacts_dir,
            "overrides": dict(runtime_overrides or {}),
        },
        "hardware": _hardware_summary(context),
        "error": None,
    }
    _write_json(artifacts.manifest_path, manifest)
    return manifest


def finish_run_manifest(
    context: RunContext,
    *,
    status: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Mark an existing run manifest completed or failed."""
    if status not in {"completed", "failed"}:
        raise ValueError("status must be 'completed' or 'failed'")
    path = RunArtifacts.from_context(context).manifest_path
    manifest = json.loads(path.read_text())
    manifest["status"] = status
    manifest["ended_at"] = _utc_now()
    manifest["error"] = (
        None
        if error is None
        else {"type": type(error).__name__, "message": str(error)}
    )
    _write_json(path, manifest)
    return manifest


def update_run_manifest(
    context: RunContext,
    **values: Any,
) -> dict[str, Any] | None:
    """Add resolved runtime facts when a manifest is present."""
    path = RunArtifacts.from_context(context).manifest_path
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    manifest.update(values)
    _write_json(path, manifest)
    return manifest


def record_result(context: RunContext, result: Mapping[str, Any]) -> None:
    """Record one framework result using generic scalar discovery."""
    RunArtifacts.from_context(context).append_result(result)
