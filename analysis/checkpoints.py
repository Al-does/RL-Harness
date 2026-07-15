"""Generic run records and public RLlib checkpoint access."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from harness.artifacts import MANIFEST_FILENAME, PROGRESS_FILENAME, RunArtifacts
from harness.context import RunContext


def _paths(source: RunContext | RunArtifacts) -> RunArtifacts:
    return (
        RunArtifacts.from_context(source)
        if isinstance(source, RunContext)
        else source
    )


def read_manifest(source: RunContext | RunArtifacts) -> dict[str, Any]:
    return json.loads(_paths(source).manifest_path.read_text())


def read_progress(source: RunContext | RunArtifacts) -> list[dict[str, Any]]:
    path = _paths(source).progress_path
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def discover_trial_configs(
    source: RunContext | RunArtifacts,
) -> list[Path]:
    """Return Tune trial parameter records under the artifact tree."""
    return sorted(_paths(source).artifacts_dir.rglob("params.json"))


def discover_checkpoints(
    source: RunContext | RunArtifacts,
) -> list[Path]:
    """Discover complete RLlib/Tune checkpoint directories."""
    artifact_root = _paths(source).artifacts_dir
    if not artifact_root.exists():
        return []
    markers = {
        "rllib_checkpoint.json",
        "algorithm_state.pkl",
        "class_and_ctor_args.pkl",
    }
    checkpoints: set[Path] = set()
    for path in artifact_root.rglob("*"):
        if not path.is_dir():
            continue
        if path.name.startswith("checkpoint_") or any(
            (path / marker).exists() for marker in markers
        ):
            checkpoints.add(path)
    return sorted(checkpoints)


@contextmanager
def load_algorithm(checkpoint: Path) -> Iterator[Any]:
    """Restore and clean up an Algorithm through RLlib's public API."""
    from ray.rllib.algorithms.algorithm import Algorithm

    algorithm = Algorithm.from_checkpoint(str(checkpoint))
    try:
        yield algorithm
    finally:
        algorithm.stop()


@contextmanager
def load_module(
    checkpoint: Path,
    *,
    module_id: str = "default_policy",
) -> Iterator[Any]:
    """Yield a public RLModule while its restored Algorithm remains alive."""
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module(module_id)
        if module is None:
            raise KeyError(
                f"checkpoint has no RLModule with id {module_id!r}"
            )
        yield module


__all__ = [
    "MANIFEST_FILENAME",
    "PROGRESS_FILENAME",
    "discover_checkpoints",
    "discover_trial_configs",
    "load_algorithm",
    "load_module",
    "read_manifest",
    "read_progress",
]
