"""Human-readable resource and cost plan printing for launchers."""

from __future__ import annotations

from typing import Any

from .preflight import PreflightPlan


def print_resource_cost_plan(
    plan: PreflightPlan,
    *,
    backend: str,
    printer: Any = print,
) -> None:
    contract = plan.resource_contract
    printer(f"  backend:         {backend}")
    printer(f"  experiment SHA:  {plan.experiment.resolved_sha}")
    printer(f"  library SHA:     {plan.library.resolved_sha}")
    printer(f"  image:           {plan.image}")
    printer(f"  image digest:    {plan.image_digest}")
    printer(
        "  resources:       "
        f"profile={contract.profile_name} smoke={contract.smoke} "
        f"learner_gpus={contract.learner_gpus:g} "
        f"env_runner_gpus={contract.env_runner_gpus:g} "
        f"total_gpus={contract.total_gpus:g} "
        f"runners={contract.num_env_runners} "
        f"envs/runner={contract.num_envs_per_env_runner}"
    )
    printer(
        "  capacity:        "
        f"gpus={plan.available_gpus:g}"
        + (
            f" cpus={plan.available_cpus:g}"
            if plan.available_cpus is not None
            else ""
        )
    )
    estimate = plan.estimate
    if estimate:
        printer(
            "  CONSERVATIVE ESTIMATED SPEND CEILING: "
            f"${float(estimate.get('total', 0)):.2f} "
            f"(GPU ${float(estimate.get('gpu', 0)):.2f} + "
            f"disk ${float(estimate.get('disk', 0)):.4f} + "
            f"fee reserve ${float(estimate.get('fee_reserve', 0)):.2f}); "
            "this is an estimate, not a provider-enforced hard dollar cap"
        )
        if estimate.get("assumption"):
            printer(f"  estimate assumes: {estimate['assumption']}")
    for note in plan.notes:
        printer(f"  note:            {note}")
