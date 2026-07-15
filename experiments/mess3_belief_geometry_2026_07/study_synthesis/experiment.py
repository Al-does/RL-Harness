"""Aggregate completed training and probe records across this study."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, pstdev

from harness.context import RunContext


TRAINING_CONDITIONS = (
    "state_guess_reward",
    "state_guess_gamma_0",
    "state_guess_no_delay",
    "state_guess_memoryless",
    "state_guess_supervised",
    "reward_only",
    "no_delay",
    "next_token_aux_0p1",
    "next_token_aux_0p5",
    "prediction_only",
    "oracle_observation",
    "belief_observation",
    "stack_02",
    "stack_04",
    "stack_08",
    "stack_16",
    "scrambled_training",
)


def _family_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_files(condition: str, filename: str) -> list[Path]:
    return sorted(
        (_family_dir() / condition / "results").glob(f"*/{filename}")
    )


def _condition_from_checkpoint(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    try:
        relative = Path(checkpoint).resolve().relative_to(_family_dir())
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def _training_records(condition: str) -> list[dict]:
    records = []
    for path in _json_files(condition, "tune_summary.json"):
        summary = json.loads(path.read_text())
        for trial in summary.get("trials", []):
            records.append(trial)
    return records


def _probe_records() -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    probe_root = _family_dir() / "checkpoint_probe" / "results"
    for manifest_path in sorted(probe_root.glob("*/run_manifest.json")):
        run_dir = manifest_path.parent
        probe_path = run_dir / "probe_result.json"
        if not probe_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        condition = _condition_from_checkpoint(
            manifest.get("runtime", {}).get("resume_from")
        )
        if condition is not None:
            records.setdefault(condition, []).append(
                json.loads(probe_path.read_text())
            )
    return records


def _aggregate(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": fmean(values),
        "standard_deviation": pstdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def run(context: RunContext):
    probes = _probe_records()
    conditions = {}
    for condition in TRAINING_CONDITIONS:
        training = _training_records(condition)
        condition_probes = probes.get(condition, [])
        return_values = [
            trial["metrics"].get("env_runners/episode_return_mean")
            for trial in training
        ]
        return_values = [
            float(value) for value in return_values if value is not None
        ]
        conditions[condition] = {
            "training_runs": len(training),
            "probe_runs": len(condition_probes),
            "episode_return": _aggregate(return_values),
            "r2_global": _aggregate(
                [
                    float(record["r2_global"])
                    for record in condition_probes
                    if record.get("r2_global") is not None
                ]
            ),
            "r2_fine": _aggregate(
                [
                    float(record["r2_fine"])
                    for record in condition_probes
                    if record.get("r2_fine") is not None
                ]
            ),
            "reward_greedy": _aggregate(
                [
                    float(record["reward_greedy"])
                    for record in condition_probes
                    if record.get("reward_greedy") is not None
                ]
            ),
        }

    payload = {"conditions": conditions}
    (context.results_dir / "study_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    lines = [
        "# MESS3 belief-geometry study summary",
        "",
        "| condition | training runs | probe runs | return | global R² | fine R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def display(metric):
        return "—" if metric is None else f"{metric['mean']:.3f}"

    for condition, values in conditions.items():
        lines.append(
            f"| {condition} | {values['training_runs']} "
            f"| {values['probe_runs']} "
            f"| {display(values['episode_return'])} "
            f"| {display(values['r2_global'])} "
            f"| {display(values['r2_fine'])} |"
        )
    (context.results_dir / "findings.md").write_text(
        "\n".join(lines) + "\n"
    )
    return payload
