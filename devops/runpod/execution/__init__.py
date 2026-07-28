"""Shared phased execution model for RunPod Pods and Serverless.

Import submodules directly from workers that may not have ``harness`` on
``sys.path`` yet (for example the baked Serverless handler before checkout).
Launcher code may use the convenience re-exports below.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CANONICAL_MANIFEST_NAME",
    "EXECUTION_PHASES",
    "FallbackPolicy",
    "JobReport",
    "Phase",
    "PhaseStatus",
    "PreflightError",
    "PreflightPlan",
    "PublicationResult",
    "TerminalReason",
    "decide_fallback",
    "print_resource_cost_plan",
    "publish_compact_results",
    "run_preflight",
    "upload_compact_results_bundle",
    "write_canonical_durability_manifest",
]


def __getattr__(name: str) -> Any:
    if name in {
        "Phase",
        "PhaseStatus",
        "TerminalReason",
        "JobReport",
        "EXECUTION_PHASES",
    }:
        from . import phases as _phases

        return getattr(_phases, name)
    if name in {"FallbackPolicy", "decide_fallback"}:
        from . import fallback as _fallback

        return getattr(_fallback, name)
    if name in {
        "PreflightError",
        "PreflightPlan",
        "run_preflight",
    }:
        from . import preflight as _preflight

        return getattr(_preflight, name)
    if name == "print_resource_cost_plan":
        from .resources_plan import print_resource_cost_plan

        return print_resource_cost_plan
    if name in {"PublicationResult", "publish_compact_results"}:
        from . import publication as _publication

        return getattr(_publication, name)
    if name in {
        "CANONICAL_MANIFEST_NAME",
        "upload_compact_results_bundle",
        "write_canonical_durability_manifest",
    }:
        from . import durability as _durability

        return getattr(_durability, name)
    raise AttributeError(name)
