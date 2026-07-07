"""Phase 1 gate sweep for MESS3-Continuous.

Computes, for every (beta, w_max2, delay) in the program grid:
  constant / reactive / stack-2 ceilings (exact stationary optimization),
  the belief-MDP ceiling (VI on a 120-per-edge simplex grid), and the
  box-constrained oracle; evaluates the two gate thresholds; and, at delay=1,
  the interior-action share of the VI optimum under its own stationary
  closed-loop distribution plus the histograms the spec asks for
  (time-since-ejection-from-state-2 and belief entropy).

Also produces the Environment B analytic table for both delays.

Outputs (results/phase1/):
  gate_sweep.csv / .json     the gate + saturation table
  stateguess_table.csv       Environment B ceilings
  interior_<beta>_<wmax>.npz histogram payloads + trajectory samples
  fig_interior_histograms.png, fig_attractor_<...>.png, fig_policy_map_<...>.png

Usage:
  uv run python scripts/phase1_gate.py            # full sweep (~30-60 min)
  uv run python scripts/phase1_gate.py --quick    # coarse smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter  # noqa: E402
from envs.mess3.solvers.belief_vi import solve_belief_vi  # noqa: E402
from envs.mess3.solvers.interior import interior_share  # noqa: E402
from envs.mess3.solvers.oracle import solve_oracle_box  # noqa: E402
from envs.mess3.solvers.reactive import (  # noqa: E402
    solve_constant,
    solve_reactive,
    solve_stack2,
)
from envs.mess3.solvers.stateguess_analytic import stateguess_table  # noqa: E402

BETAS = [2.0, 4.0, 8.0]
WMAXES = [2.0, 3.0, 5.0, 8.0]
DELAYS = [1, 0]
GATE_REACTIVE = 0.15
GATE_STACK2 = 0.08


def run_config(beta, w_max, delay, n_grid, n_interior_steps, seed=0):
    t0 = time.time()
    const = solve_constant(beta, w_max)
    reac = solve_reactive(beta, w_max, delay)
    s2 = solve_stack2(beta, w_max, delay)
    vi = solve_belief_vi(beta, w_max, delay, n_grid=n_grid, polish=False)
    orc = solve_oracle_box(beta, w_max)

    row = {
        "beta": beta,
        "w_max2": w_max,
        "delay": delay,
        "constant": const.value,
        "reactive": reac.value,
        "stack2": s2.value,
        "belief_ceiling": vi.rho,
        "oracle": orc.rho,
        "oracle_boundary_any": bool(orc.boundary.any()),
        "premium_reactive": (vi.rho - reac.value) / vi.rho if vi.rho > 0 else float("nan"),
        "premium_stack2": (vi.rho - s2.value) / vi.rho if vi.rho > 0 else float("nan"),
    }
    row["gate_pass"] = bool(
        vi.rho > 0
        and row["premium_reactive"] >= GATE_REACTIVE
        and row["premium_stack2"] >= GATE_STACK2
    )

    stats = None
    if delay == 1:
        stats = interior_share(vi, n_steps=n_interior_steps, seed=seed)
        row.update(
            interior_share_w0=stats.share_w0,
            interior_share_w1=stats.share_w1,
            interior_share_joint=stats.share_joint,
            interior_share_any=stats.share_any,
            vi_policy_mc_reward=stats.mean_reward,
            vi_policy_mc_se=stats.mean_reward_se,
        )
    row["seconds"] = round(time.time() - t0, 1)
    return row, vi, stats


def save_interior_figures(outdir: Path, payloads: dict):
    """Two figures (beta rows x w_max columns): interior-action concentration
    vs time-since-leaving-s2 and vs belief entropy."""
    betas = sorted({k[0] for k in payloads})
    wmaxes = sorted({k[1] for k in payloads})

    for metric, fname in (("tse", "fig_interior_vs_time_since_s2.png"),
                          ("ent", "fig_interior_vs_belief_entropy.png")):
        fig, axes = plt.subplots(
            len(betas), len(wmaxes), figsize=(3.4 * len(wmaxes), 2.7 * len(betas)),
            squeeze=False, sharex=True,
        )
        for i, beta in enumerate(betas):
            for j, w_max in enumerate(wmaxes):
                ax = axes[i][j]
                st = payloads.get((beta, w_max))
                if st is None:
                    ax.axis("off")
                    continue
                counts = getattr(st, f"{metric}_all").astype(float)
                inter = getattr(st, f"{metric}_interior").astype(float)
                edges = getattr(st, f"{metric}_edges")
                centers = edges[:-1] if metric == "tse" else 0.5 * (edges[:-1] + edges[1:])
                width = 0.9 if metric == "tse" else (edges[1] - edges[0]) * 0.9
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(counts > 0, inter / counts, np.nan)
                ax.bar(centers, counts / counts.sum(), width=width,
                       color="lightgray", label="visitation")
                ax2 = ax.twinx()
                ax2.plot(centers, frac, "r.-", ms=3, lw=1, label="interior fraction")
                ax2.set_ylim(-0.02, 1.02)
                ax2.tick_params(labelsize=7)
                ax.set_title(
                    f"beta={beta:g}, w_max={w_max:g}  (interior-any {st.share_any:.1%})",
                    fontsize=9,
                )
                ax.tick_params(labelsize=7)
                if i == len(betas) - 1:
                    ax.set_xlabel("steps since leaving s2" if metric == "tse"
                                  else "belief entropy (nats)", fontsize=8)
                if j == 0:
                    ax.set_ylabel("visitation prob.", fontsize=8)
                if j == len(wmaxes) - 1:
                    ax2.set_ylabel("interior fraction", fontsize=8, color="r")
        fig.suptitle(
            "Where interior (off-boundary) optimal actions concentrate — "
            "VI optimum, stationary closed loop, delay=1", fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)


def save_attractor_figures(outdir: Path, payloads: dict):
    for key, st in payloads.items():
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        simplex_scatter(
            axes[0], st.beliefs_sample, s=1.5, alpha=0.4,
            title=f"reachable belief attractor (RGB=belief)\nbeta={st.beta:g}, w_max={st.w_max:g}",
        )
        span = 2 * st.w_max
        col = np.stack(
            [
                (st.actions_sample[:, 0] + st.w_max) / span,
                (st.actions_sample[:, 1] + st.w_max) / span,
                np.full(len(st.actions_sample), 0.5),
            ],
            axis=1,
        )
        simplex_scatter(
            axes[1], st.beliefs_sample, colors=np.clip(col, 0, 1), s=1.5, alpha=0.5,
            title="optimal w over the attractor (R=w0, G=w1)",
        )
        fig.tight_layout()
        fig.savefig(outdir / f"fig_attractor_beta{st.beta:g}_wmax{st.w_max:g}.png", dpi=150)
        plt.close(fig)


def replot(outdir: Path):
    """Rebuild figures from saved interior_*.npz + gate_sweep.json (no solve)."""
    from types import SimpleNamespace

    with open(outdir / "gate_sweep.json") as f:
        rows = {(r["beta"], r["w_max2"]): r for r in json.load(f) if r["delay"] == 1}
    payloads = {}
    for p in sorted(outdir.glob("interior_beta*_wmax*.npz")):
        data = np.load(p)
        stem = p.stem.replace("interior_beta", "").split("_wmax")
        beta, w_max = float(stem[0]), float(stem[1])
        row = rows[(beta, w_max)]
        payloads[(beta, w_max)] = SimpleNamespace(
            beta=beta, w_max=w_max, share_any=row["interior_share_any"],
            **{k: data[k] for k in data.files},
        )
    save_interior_figures(outdir, payloads)
    save_attractor_figures(outdir, payloads)
    print(f"replotted -> {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse smoke run")
    ap.add_argument("--replot", action="store_true", help="figures only, from saved npz")
    ap.add_argument("--out", default="results/phase1")
    args = ap.parse_args()

    if args.replot:
        replot(Path(args.out))
        return

    n_grid = 40 if args.quick else 120
    n_int = 20_000 if args.quick else 300_000
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, payloads = [], {}
    for beta in BETAS:
        for w_max in WMAXES:
            for delay in DELAYS:
                row, vi, stats = run_config(beta, w_max, delay, n_grid, n_int)
                rows.append(row)
                flag = "PASS" if row["gate_pass"] else "fail"
                extra = (
                    f" interior(any)={row['interior_share_any']:.3f}"
                    if delay == 1 else ""
                )
                print(
                    f"[{flag}] beta={beta:g} w_max={w_max:g} delay={delay}: "
                    f"react={row['reactive']:.4f} stack2={row['stack2']:.4f} "
                    f"belief={row['belief_ceiling']:.4f} oracle={row['oracle']:.4f}"
                    f"{extra} ({row['seconds']}s)",
                    flush=True,
                )
                if stats is not None:
                    key = (beta, w_max)
                    payloads[key] = stats
                    np.savez_compressed(
                        outdir / f"interior_beta{beta:g}_wmax{w_max:g}.npz",
                        **{
                            f: getattr(stats, f)
                            for f in (
                                "tse_edges", "tse_all", "tse_interior",
                                "ent_edges", "ent_all", "ent_interior",
                                "beliefs_sample", "actions_sample",
                            )
                        },
                    )

    with open(outdir / "gate_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    cols = [
        "beta", "w_max2", "delay", "constant", "reactive", "stack2",
        "belief_ceiling", "oracle", "oracle_boundary_any",
        "premium_reactive", "premium_stack2", "gate_pass",
        "interior_share_w0", "interior_share_w1", "interior_share_joint",
        "interior_share_any", "vi_policy_mc_reward", "vi_policy_mc_se", "seconds",
    ]
    with open(outdir / "gate_sweep.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # Environment B analytic table.
    with open(outdir / "stateguess_table.csv", "w") as f:
        f.write("delay,random,memoryless,memoryless_map,filter_ceiling,filter_ceiling_se\n")
        for d in (0, 1):
            t = stateguess_table(d, n_steps=200_000 if args.quick else 2_000_000)
            f.write(
                f"{d},{t.random:.6f},{t.memoryless:.6f},"
                f"\"{t.memoryless_map}\",{t.filter_ceiling:.6f},{t.filter_ceiling_se:.6f}\n"
            )
            print(f"stateguess delay={d}: memoryless={t.memoryless:.4f} "
                  f"filter={t.filter_ceiling:.5f} +/- {t.filter_ceiling_se:.5f}", flush=True)

    save_interior_figures(outdir, payloads)
    save_attractor_figures(outdir, payloads)
    print(f"done -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
