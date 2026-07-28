"""Phase-aware launcher/worker progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .phases import JobReport, Phase, PhaseStatus


@dataclass
class ProgressEvent:
    phase: Phase
    status: PhaseStatus
    message: str
    provider_status: str | None = None
    image_pull: bool = False
    capacity_queue: bool = False


def classify_provider_status(
    provider_status: str,
    *,
    worker_seen: bool,
    progress_message: str | None = None,
) -> ProgressEvent:
    """Distinguish capacity queueing from image initialization."""
    status = provider_status.upper()
    message = (progress_message or "").lower()
    if status == "IN_QUEUE" and not worker_seen:
        if any(
            token in message
            for token in ("pull", "extract", "download", "image", "layer")
        ):
            return ProgressEvent(
                phase=Phase.PROVISIONING,
                status=PhaseStatus.RUNNING,
                message="image initialization (provider still reports IN_QUEUE)",
                provider_status=status,
                image_pull=True,
            )
        return ProgressEvent(
            phase=Phase.PROVISIONING,
            status=PhaseStatus.RUNNING,
            message="capacity queueing (waiting for a worker slot)",
            provider_status=status,
            capacity_queue=True,
        )
    if status in {"IN_PROGRESS", "RUNNING"}:
        return ProgressEvent(
            phase=Phase.BOOTSTRAP if "bootstrap" in message or "checkout" in message else Phase.TRAINING,
            status=PhaseStatus.RUNNING,
            message=progress_message or "worker running",
            provider_status=status,
        )
    return ProgressEvent(
        phase=Phase.PROVISIONING,
        status=PhaseStatus.RUNNING,
        message=progress_message or status,
        provider_status=status,
    )


def emit_phase(
    report: JobReport,
    phase: Phase,
    status: PhaseStatus,
    message: str,
    *,
    printer: Callable[[str], Any] = print,
    at: str | None = None,
) -> None:
    report.set_phase(phase, status, detail=message, at=at)
    printer(f"  phase={phase.value} status={status.value}: {message}")
