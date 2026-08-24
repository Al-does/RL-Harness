"""Compact training-curve records for tracked results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TRAINING_CURVES_FILENAME = "training_curves.jsonl"
PROGRESS_VERBOSE_FILENAME = "progress.jsonl"

# Legacy alias: compact curves replaced results-side progress.jsonl.
PROGRESS_FILENAME = TRAINING_CURVES_FILENAME

_COMPACT_FIELD_SOURCES: tuple[tuple[str, ...], str] = (
    (("training_iteration",), "iteration"),
    (("num_env_steps_sampled_lifetime", "timesteps_total"), "steps"),
    (("env_runners/episode_return_mean", "episode_return_mean"), "return_mean"),
    (("env_runners/episode_return_min", "episode_return_min"), "return_min"),
    (("env_runners/episode_return_max", "episode_return_max"), "return_max"),
    (("learners/default_policy/entropy", "entropy"), "entropy"),
    (("learners/default_policy/curr_entropy_coeff", "entropy_coeff"), "entropy_coeff"),
    (("learners/default_policy/policy_loss", "policy_loss"), "policy_loss"),
    (("learners/default_policy/vf_loss", "vf_loss"), "vf_loss"),
    (("learners/default_policy/total_loss", "total_loss"), "total_loss"),
    (
        ("learners/default_policy/vf_explained_var", "vf_explained_var"),
        "value_explained_variance",
    ),
    (("learners/default_policy/mean_kl_loss", "mean_kl_loss"), "mean_kl"),
    (("time_this_iter_s",), "time_iter_s"),
    (("time_total_s",), "time_total_s"),
    (("done",), "done"),
    (("trial_id",), "trial_id"),
)


def compact_training_curve_row(
    flattened: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one flattened metrics row into a short, agent-safe curve record."""
    row: dict[str, Any] = {}
    for sources, target in _COMPACT_FIELD_SOURCES:
        for key in sources:
            if key not in flattened:
                continue
            value = flattened[key]
            if value is None:
                continue
            row[target] = value
            break
    return row


def training_iteration_from_row(row: Mapping[str, Any]) -> float | None:
    """Return a positive training iteration from compact or verbose rows."""
    for key in ("iteration", "training_iteration"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                return float(value)
    for key, value in row.items():
        if (
            key.endswith("/training_iteration")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            return float(value)
    return None
