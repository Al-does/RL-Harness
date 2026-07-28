"""Explicit execution phases and independent status reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    PROVISIONING = "PROVISIONING"
    BOOTSTRAP = "BOOTSTRAP"
    TRAINING = "TRAINING"
    ANALYSIS = "ANALYSIS"
    DURABLE_UPLOAD = "DURABLE_UPLOAD"
    RESULTS_PUBLICATION = "RESULTS_PUBLICATION"
    CLEANUP = "CLEANUP"


EXECUTION_PHASES: tuple[Phase, ...] = tuple(Phase)


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class TerminalReason(str, Enum):
    SUCCESS = "success"
    PREFLIGHT_REJECTED = "preflight_rejected"
    PROVISIONING_FAILED = "provisioning_failed"
    BOOTSTRAP_FAILED = "bootstrap_failed"
    TRAINING_FAILED = "training_failed"
    ANALYSIS_FAILED = "analysis_failed"
    DURABLE_UPLOAD_FAILED = "durable_upload_failed"
    PUBLICATION_FAILED = "publication_failed"
    QUEUE_TIMEOUT = "queue_timeout"
    IMAGE_INIT_TIMEOUT = "image_init_timeout"
    LIFECYCLE_TIMEOUT = "lifecycle_timeout"
    FALLBACK_EXHAUSTED = "fallback_exhausted"
    CANCELLED = "cancelled"
    CLEANUP_FAILED = "cleanup_failed"
    PROVIDER_FAILED = "provider_failed"
    UNKNOWN = "unknown"


@dataclass
class PhaseRecord:
    phase: Phase
    status: PhaseStatus = PhaseStatus.PENDING
    detail: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "detail": self.detail,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class JobReport:
    """Independent workload vs publication outcomes plus phase timeline."""

    backend: str
    run_name: str
    phases: dict[str, PhaseRecord] = field(default_factory=dict)
    workload_success: bool | None = None
    publication_status: PhaseStatus = PhaseStatus.PENDING
    publication_detail: str | None = None
    terminal_reason: TerminalReason = TerminalReason.UNKNOWN
    canonical_manifest_key: str | None = None
    recoverable_bundle_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phases:
            self.phases = {
                phase.value: PhaseRecord(phase=phase) for phase in EXECUTION_PHASES
            }

    def set_phase(
        self,
        phase: Phase,
        status: PhaseStatus,
        *,
        detail: str | None = None,
        at: str | None = None,
    ) -> None:
        record = self.phases[phase.value]
        record.status = status
        if detail is not None:
            record.detail = detail
        if status == PhaseStatus.RUNNING:
            record.started_at = at or record.started_at
        if status in {
            PhaseStatus.SUCCEEDED,
            PhaseStatus.FAILED,
            PhaseStatus.SKIPPED,
            PhaseStatus.WARNING,
        }:
            record.ended_at = at or record.ended_at

    def mark_workload(
        self,
        *,
        success: bool,
        reason: TerminalReason,
        detail: str | None = None,
    ) -> None:
        self.workload_success = success
        self.terminal_reason = reason
        if detail:
            self.metadata["workload_detail"] = detail

    def mark_publication(
        self,
        status: PhaseStatus,
        *,
        detail: str | None = None,
        recoverable_bundle_key: str | None = None,
    ) -> None:
        self.publication_status = status
        self.publication_detail = detail
        if recoverable_bundle_key:
            self.recoverable_bundle_key = recoverable_bundle_key
        self.set_phase(Phase.RESULTS_PUBLICATION, status, detail=detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "run_name": self.run_name,
            "workload_success": self.workload_success,
            "publication_status": self.publication_status.value,
            "publication_detail": self.publication_detail,
            "terminal_reason": self.terminal_reason.value,
            "canonical_manifest_key": self.canonical_manifest_key,
            "recoverable_bundle_key": self.recoverable_bundle_key,
            "phases": {
                key: value.to_dict() for key, value in self.phases.items()
            },
            "metadata": dict(self.metadata),
        }

    def launcher_exit_code(self) -> int:
        """Workload success is independent of Git publication warnings."""
        if self.workload_success is True:
            return 0
        return 1
