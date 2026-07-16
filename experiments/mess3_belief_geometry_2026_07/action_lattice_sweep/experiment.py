"""Analyze belief control as the discrete action lattice is refined."""

from __future__ import annotations

import csv
import json
from itertools import product

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import to_xy
from envs.hmm import HMMEnv
from envs.mess3.model import CONTROL_TRANSITION_MATRIX
from envs.mess3.solvers.belief_vi import solve_belief_vi
from envs.mess3.solvers.reactive import chain_value
from envs.mess3.solvers.simplex_grid import nearest_index
from harness.context import RunContext


BETA = 4.0
W_MAX = 5.0
DELAY = 1
ALPHA = 0.85
LATTICE_SIDES = (2, 3, 5, 7)


def lattice(side: int) -> np.ndarray:
    coordinates = np.linspace(-W_MAX, W_MAX, side)
    return np.stack(
        np.meshgrid(coordinates, coordinates, indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)


def discrete_reactive(actions: np.ndarray) -> tuple[float, tuple[int, ...]]:
    best_value = -np.inf
    best_mapping = None
    for mapping in product(range(len(actions)), repeat=3):
        value = chain_value(
            actions[list(mapping)],
            DELAY,
            1,
            ALPHA,
            BETA,
            CONTROL_TRANSITION_MATRIX,
        )
        if value > best_value:
            best_value = value
            best_mapping = mapping
    assert best_mapping is not None
    return best_value, best_mapping


def closed_loop_cells(
    solution,
    actions: np.ndarray,
    *,
    n_steps: int,
    seed: int,
):
    env = HMMEnv(
        {
            "model": {
                "factory": "envs.mess3.model:control_model",
                "kwargs": {"alpha": ALPHA},
            },
            "task": {
                "class": (
                    "envs.mess3.tasks.occupancy_control:"
                    "OccupancyControlTask"
                ),
                "kwargs": {
                    "transition_kl_beta": BETA,
                    "action_limit": W_MAX,
                },
            },
            "delay": DELAY,
            "episode_length": 1024,
            "diagnostics": {"belief": True},
            "seed": seed,
        }
    )
    _, info = env.reset(seed=seed)
    beliefs = np.empty((n_steps, 3))
    action_indices = np.empty(n_steps, dtype=np.int64)
    rewards = np.empty(n_steps)
    try:
        for step in range(n_steps):
            belief = info["belief_current"]
            action_index = int(
                solution.greedy_k[
                    int(nearest_index(belief, solution.n_grid))
                ]
            )
            beliefs[step] = belief
            action_indices[step] = action_index
            _, reward, _, truncated, info = env.step(
                actions[action_index]
            )
            rewards[step] = reward
            if truncated:
                _, info = env.reset()
    finally:
        env.close()
    return beliefs, action_indices, rewards


def cell_diameter_statistics(
    beliefs: np.ndarray,
    actions: np.ndarray,
) -> dict:
    rng = np.random.default_rng(0)
    diameters, weights, counts = [], [], {}
    for action in np.unique(actions):
        members = actions == action
        counts[int(action)] = int(members.sum())
        points = beliefs[members]
        if len(points) < 10:
            continue
        subset = points[
            rng.choice(
                len(points),
                min(len(points), 2000),
                replace=False,
            )
        ]
        distances = np.sqrt(
            np.square(subset[:, None, :] - subset[None, :, :]).sum(-1)
        )
        diameters.append(float(np.quantile(distances, 0.975)))
        weights.append(float(members.mean()))
    diameter_array = np.asarray(diameters)
    weight_array = np.asarray(weights)
    return {
        "mean_cell_diameter": float(
            (diameter_array * weight_array).sum() / weight_array.sum()
        ),
        "max_cell_diameter": float(diameter_array.max()),
        "cells_used_closed_loop": len(counts),
        "visitation_per_cell": counts,
    }


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the lattice sweep requires a resolved seed")
    grid_size = 40 if context.smoke else 120
    rollout_steps = 20_000 if context.smoke else 200_000
    rows = []
    for side in LATTICE_SIDES:
        actions = lattice(side)
        solution = solve_belief_vi(
            BETA,
            W_MAX,
            DELAY,
            n_grid=grid_size,
            actions=actions,
            polish=False,
        )
        reactive_value, reactive_mapping = discrete_reactive(actions)
        beliefs, selected_actions, rewards = closed_loop_cells(
            solution,
            actions,
            n_steps=rollout_steps,
            seed=context.seed,
        )
        statistics = cell_diameter_statistics(
            beliefs,
            selected_actions,
        )

        figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
        color_map = plt.get_cmap("tab20", len(actions))
        grid_xy = to_xy(solution.grid)
        axes[0].scatter(
            grid_xy[:, 0],
            grid_xy[:, 1],
            c=solution.greedy_k,
            cmap=color_map,
            s=4,
            vmin=0,
            vmax=len(actions) - 1,
        )
        axes[0].set_title(
            f"grid cells ({len(np.unique(solution.greedy_k))} used)"
        )
        attractor_xy = to_xy(beliefs[::10])
        axes[1].scatter(
            attractor_xy[:, 0],
            attractor_xy[:, 1],
            c=selected_actions[::10],
            cmap=color_map,
            s=0.5,
            vmin=0,
            vmax=len(actions) - 1,
        )
        axes[1].set_title(
            f"attractor ({statistics['cells_used_closed_loop']} cells)"
        )
        for axis in axes:
            axis.set_aspect("equal")
            axis.axis("off")
        figure.tight_layout()
        action_count = side * side
        figure.savefig(
            context.results_dir / f"fig_cells_{action_count}.png",
            dpi=160,
        )
        plt.close(figure)

        rows.append(
            {
                "action_count": action_count,
                "side": side,
                "belief_ceiling": solution.rho,
                "belief_ceiling_mc": float(rewards.mean()),
                "reactive_ceiling": reactive_value,
                "premium": (
                    (solution.rho - reactive_value) / solution.rho
                ),
                "cells_on_grid": len(np.unique(solution.greedy_k)),
                **statistics,
                "reactive_mapping": str(reactive_mapping),
            }
        )

    used_cells = [row["cells_used_closed_loop"] for row in rows]
    summary = {
        "rows": rows,
        "cells_grow_with_action_count": (
            all(
                later >= earlier
                for earlier, later in zip(used_cells, used_cells[1:])
            )
            and used_cells[-1] > used_cells[0]
        ),
    }
    (context.results_dir / "lattice_sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (
        context.results_dir / "lattice_sweep.csv"
    ).open("w", newline="") as handle:
        columns = [
            key
            for key in rows[0]
            if key != "visitation_per_cell"
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key in columns
                }
                for row in rows
            ]
        )
    return summary
