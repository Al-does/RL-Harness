"""Automatic fallback policy from Serverless to Pods."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FallbackPolicy(str, Enum):
    NONE = "none"
    PODS = "pods"


@dataclass(frozen=True)
class FallbackDecision:
    action: str  # continue | fallback_pods | fail
    reason: str
    reuse_endpoint: bool = False


def decide_fallback(
    *,
    policy: FallbackPolicy | str,
    terminal_reason: str,
    serverless_attempts: int,
    max_serverless_attempts: int = 1,
) -> FallbackDecision:
    """Decide whether a Serverless failure should fall back to Pods.

    Fallback is reserved for provisioning/cold-start capacity problems, never
    for deterministic preflight, training, or publication failures.
    """
    normalized = FallbackPolicy(policy)
    retryable = {
        "queue_timeout",
        "image_init_timeout",
        "provisioning_failed",
    }
    reason = terminal_reason.lower()
    if reason not in retryable:
        return FallbackDecision(action="fail", reason=reason, reuse_endpoint=False)
    if normalized == FallbackPolicy.NONE:
        # Preserve a healthy endpoint for a same-backend retry when capacity
        # queueing was the only problem.
        return FallbackDecision(
            action="fail",
            reason=reason,
            reuse_endpoint=reason == "queue_timeout",
        )
    if serverless_attempts < max_serverless_attempts:
        return FallbackDecision(
            action="continue",
            reason=reason,
            reuse_endpoint=True,
        )
    return FallbackDecision(
        action="fallback_pods",
        reason=f"serverless {reason}; falling back to pods",
        reuse_endpoint=False,
    )
