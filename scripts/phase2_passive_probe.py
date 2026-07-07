"""Probe-pipeline validation on passive MESS3(0.05, 0.85) (pre-Phase-2 gate).

Trains the canonical transformer with the supervised next-token loss on the
PASSIVE environment (actions ignored, canonical symmetric dynamics) and probes
its activations against the exact filter beliefs.  Prior round: global probe
R^2 0.994 on this check.  This validates, end to end: the env's passive mode,
the exact filter, activation collection, the affine probe, and both metrics —
before any RL training is trusted.

    uv run python scripts/phase2_passive_probe.py [--steps 3000000]

Writes results/phase2/passive_probe/{module ckpts, probe_result.json,
fig_passive_probe.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter  # noqa: E402
from analysis.probe import collect_probe_data, evaluate_probe, probe_predict  # noqa: E402
from envs.mess3.env_continuous import Mess3ContinuousEnv  # noqa: E402
from envs.mess3.supervised import train_supervised  # noqa: E402

MODEL_CONFIG = {"d_model": 96, "n_layers": 3, "n_heads": 4, "context_len": 64,
                "max_seq_len": 32}


def env_factory():
    return Mess3ContinuousEnv({"passive_mode": True, "alpha": 0.85, "episode_length": 1024})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = repo / "results" / "phase2" / "passive_probe"
    outdir.mkdir(parents=True, exist_ok=True)

    probe_env = env_factory()
    module = train_supervised(
        env_factory=env_factory,
        model_config=MODEL_CONFIG,
        obs_dim=int(probe_env.observation_space.shape[0]),
        action_space=probe_env.action_space,
        target="next_token",
        total_steps=args.steps,
        outdir=outdir,
        seed=args.seed,
    )

    # Probe: train and held-out data from DISJOINT seed ranges.
    train = collect_probe_data(
        module, env_factory, n_steps=120_000, seed=10_000, policy_mode="random"
    )
    test = collect_probe_data(
        module, env_factory, n_steps=60_000, seed=20_000, policy_mode="random"
    )
    res = evaluate_probe(train, test)
    W, b = res.pop("probe")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})

    passed = res["r2_global"] >= 0.98
    res["prior_expectation"] = 0.994
    res["passed"] = bool(passed)
    with open(outdir / "probe_result.json", "w") as f:
        json.dump(res, f, indent=2)

    pred = probe_predict(W, b, test.activations)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    simplex_scatter(axes[0], test.beliefs, s=0.5, alpha=0.4,
                    title="ground-truth beliefs (exact filter)")
    simplex_scatter(axes[1], np.clip(pred, 0, 1), s=0.5, alpha=0.4,
                    title=f"decoded from activations (global R^2={res['r2_global']:.3f})")
    fig.suptitle("Passive MESS3(0.05, 0.85): probe validation (prior 0.994)")
    fig.tight_layout()
    fig.savefig(outdir / "fig_passive_probe.png", dpi=160)
    print(("PASSED" if passed else "FAILED") + f" -> {outdir}")


if __name__ == "__main__":
    main()
