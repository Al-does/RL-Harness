"""Analytic sweep that selects the study's MESS3 operating point."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter
from envs.mess3.solvers.belief_vi import solve_belief_vi
from envs.mess3.solvers.interior import interior_share
from envs.mess3.solvers.oracle import solve_oracle_box
from envs.mess3.solvers.reactive import (
    solve_constant,
    solve_reactive,
    solve_stack2,
)
from envs.mess3.solvers.stateguess_analytic import stateguess_table
from harness.context import RunContext


BETAS = (2.0, 4.0, 8.0)
W_MAX_VALUES = (2.0, 3.0, 5.0, 8.0)
DELAYS = (1, 0)
MIN_REACTIVE_PREMIUM = 0.15
MIN_STACK2_PREMIUM = 0.08


def solve_condition(
    *,
    beta: float,
    w_max: float,
    delay: int,
    grid_size: int,
    interior_steps: int,
    seed: int,
):
    started_at = time.monotonic()
    constant = solve_constant(beta, w_max)
    reactive = solve_reactive(beta, w_max, delay)
    stack2 = solve_stack2(beta, w_max, delay)
    belief = solve_belief_vi(
        beta,
        w_max,
        delay,
        n_grid=grid_size,
        polish=False,
    )
    oracle = solve_oracle_box(beta, w_max)
    row = {
        "beta": beta,
        "w_max2": w_max,
        "delay": delay,
        "constant": constant.value,
        "reactive": reactive.value,
        "stack2": stack2.value,
        "belief_ceiling": belief.rho,
        "oracle": oracle.rho,
        "oracle_boundary_any": bool(oracle.boundary.any()),
        "premium_reactive": (
            (belief.rho - reactive.value) / belief.rho
            if belief.rho > 0
            else float("nan")
        ),
        "premium_stack2": (
            (belief.rho - stack2.value) / belief.rho
            if belief.rho > 0
            else float("nan")
        ),
    }
    row["meets_premium_criteria"] = bool(
        belief.rho > 0
        and row["premium_reactive"] >= MIN_REACTIVE_PREMIUM
        and row["premium_stack2"] >= MIN_STACK2_PREMIUM
    )
    statistics = None
    if delay == 1:
        statistics = interior_share(
            belief,
            n_steps=interior_steps,
            seed=seed,
        )
        row.update(
            interior_share_w0=statistics.share_w0,
            interior_share_w1=statistics.share_w1,
            interior_share_joint=statistics.share_joint,
            interior_share_any=statistics.share_any,
            vi_policy_mc_reward=statistics.mean_reward,
            vi_policy_mc_se=statistics.mean_reward_se,
        )
    row["seconds"] = round(time.monotonic() - started_at, 1)
    return row, statistics


def _save_interior_figures(
    results_dir: Path,
    payloads: dict[tuple[float, float], object],
) -> None:
    betas = sorted({key[0] for key in payloads})
    bounds = sorted({key[1] for key in payloads})
    for metric, filename in (
        ("tse", "fig_interior_vs_time_since_s2.png"),
        ("ent", "fig_interior_vs_belief_entropy.png"),
    ):
        figure, axes = plt.subplots(
            len(betas),
            len(bounds),
            figsize=(3.4 * len(bounds), 2.7 * len(betas)),
            squeeze=False,
            sharex=True,
        )
        for row_index, beta in enumerate(betas):
            for column_index, w_max in enumerate(bounds):
                axis = axes[row_index][column_index]
                statistics = payloads[(beta, w_max)]
                counts = getattr(statistics, f"{metric}_all").astype(float)
                interior = getattr(
                    statistics,
                    f"{metric}_interior",
                ).astype(float)
                edges = getattr(statistics, f"{metric}_edges")
                centers = (
                    edges[:-1]
                    if metric == "tse"
                    else 0.5 * (edges[:-1] + edges[1:])
                )
                width = (
                    0.9
                    if metric == "tse"
                    else (edges[1] - edges[0]) * 0.9
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    fraction = np.where(
                        counts > 0,
                        interior / counts,
                        np.nan,
                    )
                axis.bar(
                    centers,
                    counts / counts.sum(),
                    width=width,
                    color="lightgray",
                )
                fraction_axis = axis.twinx()
                fraction_axis.plot(centers, fraction, "r.-", ms=3, lw=1)
                fraction_axis.set_ylim(-0.02, 1.02)
                axis.set_title(
                    f"β={beta:g}, w={w_max:g} "
                    f"(interior {statistics.share_any:.1%})",
                    fontsize=9,
                )
        figure.tight_layout()
        figure.savefig(results_dir / filename, dpi=150)
        plt.close(figure)


def _save_attractor_figures(
    results_dir: Path,
    payloads: dict[tuple[float, float], object],
) -> None:
    for statistics in payloads.values():
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        labels = ("s0", "s1", "s2")
        simplex_scatter(
            axes[0],
            statistics.beliefs_sample,
            s=1.5,
            alpha=0.4,
            title="reachable belief attractor",
            labels=labels,
        )
        span = 2 * statistics.w_max
        colors = np.stack(
            [
                (
                    statistics.actions_sample[:, 0]
                    + statistics.w_max
                )
                / span,
                (
                    statistics.actions_sample[:, 1]
                    + statistics.w_max
                )
                / span,
                np.full(len(statistics.actions_sample), 0.5),
            ],
            axis=1,
        )
        simplex_scatter(
            axes[1],
            statistics.beliefs_sample,
            colors=np.clip(colors, 0, 1),
            s=1.5,
            alpha=0.5,
            title="optimal action (R=w0, G=w1)",
            labels=labels,
        )
        figure.tight_layout()
        figure.savefig(
            results_dir
            / (
                f"fig_attractor_beta{statistics.beta:g}"
                f"_wmax{statistics.w_max:g}.png"
            ),
            dpi=150,
        )
        plt.close(figure)


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the analytic sweep requires a resolved seed")
    context.results_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = context.artifacts_dir / "interior_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    grid_size = 40 if context.smoke else 120
    interior_steps = 20_000 if context.smoke else 300_000

    rows = []
    payloads = {}
    for beta in BETAS:
        for w_max in W_MAX_VALUES:
            for delay in DELAYS:
                row, statistics = solve_condition(
                    beta=beta,
                    w_max=w_max,
                    delay=delay,
                    grid_size=grid_size,
                    interior_steps=interior_steps,
                    seed=context.seed,
                )
                rows.append(row)
                if statistics is not None:
                    payloads[(beta, w_max)] = statistics
                    np.savez_compressed(
                        payload_dir
                        / f"interior_beta{beta:g}_wmax{w_max:g}.npz",
                        **{
                            field: getattr(statistics, field)
                            for field in (
                                "tse_edges",
                                "tse_all",
                                "tse_interior",
                                "ent_edges",
                                "ent_all",
                                "ent_interior",
                                "beliefs_sample",
                                "actions_sample",
                            )
                        },
                    )

    (context.results_dir / "sweep.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    columns = list(rows[0])
    with (context.results_dir / "sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    state_guess_rows = []
    for delay in (0, 1):
        table = stateguess_table(
            delay,
            n_steps=200_000 if context.smoke else 2_000_000,
        )
        state_guess_rows.append(
            {
                "delay": delay,
                "random": table.random,
                "memoryless": table.memoryless,
                "memoryless_map": str(table.memoryless_map),
                "filter_ceiling": table.filter_ceiling,
                "filter_ceiling_se": table.filter_ceiling_se,
            }
        )
    with (
        context.results_dir / "state_guess_anchors.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(state_guess_rows[0]),
        )
        writer.writeheader()
        writer.writerows(state_guess_rows)

    _save_interior_figures(context.results_dir, payloads)
    _save_attractor_figures(context.results_dir, payloads)
    return rows
