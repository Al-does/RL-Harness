"""Immutable runtime context shared by all experiment entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from harness.hardware import HardwareProfile


DEFAULT_SEED = 42


def new_run_id() -> str:
    """Return a sortable, collision-resistant local run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Operational inputs for one experiment run.

    Scientific choices belong in the experiment module, not in this object.
    """

    experiment_dir: Path
    results_dir: Path
    artifacts_dir: Path
    seed: int | None = DEFAULT_SEED
    run_id: str = field(default_factory=new_run_id)
    smoke: bool = False
    resume_from: Path | None = None
    hardware: HardwareProfile | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_dir", Path(self.experiment_dir))
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.results_dir == self.artifacts_dir:
            raise ValueError("results_dir and artifacts_dir must be distinct")
