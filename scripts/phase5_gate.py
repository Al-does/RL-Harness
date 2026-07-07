"""Phase-5 ANALYTIC GATE: discrete-action-count ladder, no training.

For each lattice N in {4 (2x2), 9 (3x3), 25 (5x5), 49 (7x7)} over the
operating-point box (beta=4, w_max2=5), with THE SAME exponential-tilt
dynamics and KL cost at the chosen w:

  - belief-MDP value iteration restricted to the discrete action set
    -> per-N belief ceiling;
  - discrete reactive ceiling (brute force over all N^3 token->action maps
    against the exact stationary chain);
  - the optimal-policy cell decomposition over the simplex (count of distinct
    actions used + rendering colored by argmax action), on the grid and on
    the closed-loop reachable attractor;
  - cell-resolution statistic: visitation-weighted mean cell diameter over
    the attractor (cell = maximal region of constant optimal action).

Phase-1 caution to check here: heavy box-corner saturation at (4, 5) means
small-N lattices may collapse onto corners; if the used-cell count fails to
grow with N, the lattice/box must be adjusted BEFORE any ladder training.

    uv run python scripts/phase5_gate.py

Writes results/phase5/gate_table.csv, per-N cell renderings, and
results/phase5/GATE_TABLE_READY (the review-stop artifact; training arms for
Phase 5 are to be declared only after review).
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter, to_xy  # noqa: E402
from envs.mess3.core import P0, emission_matrix  # noqa: E402
from envs.mess3.env_continuous import Mess3ContinuousConfig, Mess3ContinuousEnv  # noqa: E402
from envs.mess3.solvers.belief_vi import solve_belief_vi  # noqa: E402
from envs.mess3.solvers.reactive import _chain_value  # noqa: E402
from envs.mess3.solvers.simplex_grid import nearest_index  # noqa: E402

BETA, WMAX, DELAY, ALPHA = 4.0, 5.0, 1, 0.85
LATTICE_SIDES = [2, 3, 5, 7]
N_GRID = 120


def lattice(side: int, w_max: float) -> np.ndarray:
    g = np.linspace(-w_max, w_max, side)
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)


def discrete_reactive(actions: np.ndarray) -> tuple[float, tuple]:
    """Best token -> lattice-action map (exact stationary value, brute force)."""
    K = len(actions)
    best_v, best_map = -np.inf, None
    for combo in product(range(K), repeat=3):
        table = actions[list(combo)]
        v = _chain_value(table, DELAY, 1, ALPHA, BETA, P0)
        if v > best_v:
            best_v, best_map = v, combo
    return best_v, best_map


def closed_loop_cells(sol, actions: np.ndarray, n_steps: int = 200_000, seed: int = 0):
    """Simulate the discrete VI policy closed-loop; return visited beliefs and
    the chosen action index per step."""
    env = Mess3ContinuousEnv(Mess3ContinuousConfig(
        beta=BETA, w_max2=WMAX, delay=DELAY, alpha=ALPHA, seed=seed,
    ))
    _, info = env.reset(seed=seed)
    B = np.empty((n_steps, 3))
    A = np.empty(n_steps, dtype=np.int64)
    R = np.empty(n_steps)
    for t in range(n_steps):
        b = info["belief"]
        k = int(sol.greedy_k[int(nearest_index(b, sol.n_grid))])
        B[t] = b
        A[t] = k
        _, r, _, truncated, info = env.step(actions[k])
        R[t] = r
        if truncated:
            _, info = env.reset()
    return B, A, R


def cell_diameter_stat(B: np.ndarray, A: np.ndarray) -> dict:
    """Visitation-weighted mean cell diameter over the attractor.

    Cells are the connected structure of constant optimal action; we measure
    each used action's visited-belief set diameter (robust: 97.5th percentile
    of pairwise distances on a subsample) and weight by visitation."""
    rng = np.random.default_rng(0)
    diams, weights, counts = [], [], {}
    for k in np.unique(A):
        m = A == k
        counts[int(k)] = int(m.sum())
        pts = B[m]
        if len(pts) < 10:
            continue
        sub = pts[rng.choice(len(pts), min(len(pts), 2000), replace=False)]
        d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        diams.append(float(np.quantile(d, 0.975)))
        weights.append(m.mean())
    diams, weights = np.array(diams), np.array(weights)
    return {
        "mean_cell_diameter": float((diams * weights).sum() / weights.sum()),
        "max_cell_diameter": float(diams.max()),
        "cells_used_closed_loop": int(len(counts)),
        "visitation_per_cell": counts,
    }


def main():
    repo = Path(__file__).resolve().parents[1]
    outdir = repo / "results" / "phase5"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for side in LATTICE_SIDES:
        N = side * side
        acts = lattice(side, WMAX)
        print(f"=== N={N} ({side}x{side}) ===", flush=True)
        sol = solve_belief_vi(BETA, WMAX, DELAY, n_grid=N_GRID, actions=acts, polish=False)
        react_v, react_map = discrete_reactive(acts)

        # Cell decomposition on the grid and on the attractor.
        grid_cells = len(np.unique(sol.greedy_k))
        B, A, R = closed_loop_cells(sol, acts)
        stats = cell_diameter_stat(B, A)
        mc_value = float(R.mean())

        # Rendering: grid colored by argmax action; attractor overlay.
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
        cmap = plt.get_cmap("tab20", len(acts))
        xy = to_xy(sol.grid)
        axes[0].scatter(xy[:, 0], xy[:, 1], c=sol.greedy_k, cmap=cmap, s=4,
                        vmin=0, vmax=len(acts) - 1)
        axes[0].set_title(f"optimal-action cells on grid ({grid_cells} used)")
        axes[0].set_aspect("equal"); axes[0].axis("off")
        xyB = to_xy(B[::10])
        axes[1].scatter(xyB[:, 0], xyB[:, 1], c=A[::10], cmap=cmap, s=0.5,
                        vmin=0, vmax=len(acts) - 1)
        axes[1].set_title(
            f"attractor ({stats['cells_used_closed_loop']} cells, "
            f"mean diam {stats['mean_cell_diameter']:.3f})"
        )
        axes[1].set_aspect("equal"); axes[1].axis("off")
        fig.suptitle(f"N={N} ({side}x{side}) lattice, beta={BETA}, w_max2={WMAX}")
        fig.tight_layout()
        fig.savefig(outdir / f"fig_cells_N{N}.png", dpi=160)
        plt.close(fig)

        row = {
            "N": N, "side": side,
            "belief_ceiling": round(sol.rho, 5),
            "belief_ceiling_mc": round(mc_value, 5),
            "reactive_ceiling": round(react_v, 5),
            "premium": round((sol.rho - react_v) / sol.rho, 4) if sol.rho > 0 else float("nan"),
            "cells_on_grid": grid_cells,
            "cells_used_closed_loop": stats["cells_used_closed_loop"],
            "mean_cell_diameter": round(stats["mean_cell_diameter"], 4),
            "reactive_map": str(react_map),
        }
        rows.append(row)
        print(row, flush=True)

    with open(outdir / "gate_table.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(outdir / "gate_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Gate verdict: does the used-cell count grow with N?
    used = [r["cells_used_closed_loop"] for r in rows]
    growing = all(b >= a for a, b in zip(used, used[1:])) and used[-1] > used[0]
    verdict = (
        "cells grow with N -- ladder is viable at the operating-point box"
        if growing
        else "CELL COUNT DOES NOT GROW WITH N -- adjust lattice or box before training"
    )
    (outdir / "GATE_TABLE_READY").write_text(verdict + "\n")
    print(verdict)
    print(f"-> {outdir}/gate_table.csv (STOP for review before ladder training)")


if __name__ == "__main__":
    main()
