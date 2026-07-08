"""Test-time scramble evaluation: a TRAINED a_main agent fed noise tokens.

    uv run python scripts/eval_scrambled_input.py [--run results/phase3/a_main/seed0]

Contrast with the n_scramble arm (trained on noise from scratch): here the
network was trained on real tokens and only the EVALUATION input is scrambled.
The probe targets stay the true filter beliefs (the env computes them from the
real token stream regardless of what the agent is shown), so this isolates how
much of the probe R^2 flows through the input information channel vs the
flexibility of the probe itself.

Writes results/phase4/scrambled_eval.json and fig_a_main_scrambled_eval.png
(normal input vs scrambled input, decoded beliefs side by side).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.checkpoints import (  # noqa: E402
    env_factory_from_blueprint,
    load_blueprint_dict,
    load_module,
)
from analysis.plots import simplex_scatter  # noqa: E402
from analysis.probe import (  # noqa: E402
    collect_probe_data,
    evaluate_probe,
    probe_predict,
)

REPO = Path(__file__).resolve().parents[1]


def default_device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_condition(module, env_factory, *, seed_base: int, device) -> dict:
    train = collect_probe_data(module, env_factory, n_steps=120_000,
                               seed=seed_base, device=device)
    test = collect_probe_data(module, env_factory, n_steps=60_000,
                              seed=seed_base + 10_000, device=device)
    res = evaluate_probe(train, test)
    W, b = res.pop("probe")
    pred = probe_predict(W, b, test.activations)
    res["reward_mean"] = float(test.rewards.mean())
    return {"metrics": res, "pred": pred, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/phase3/a_main/seed0")
    args = ap.parse_args()

    run_dir = REPO / args.run
    bp = load_blueprint_dict(run_dir)
    module = load_module(run_dir, run_dir / "module_state_final.pt")
    device = default_device()

    normal_factory = env_factory_from_blueprint(bp)
    bp_scrambled = dict(bp, scramble_tokens=True)
    scrambled_factory = env_factory_from_blueprint(bp_scrambled)

    print(f"module: {args.run} (final); device {device}", flush=True)
    conds = {}
    for name, factory in [("normal", normal_factory), ("scrambled", scrambled_factory)]:
        conds[name] = run_condition(module, factory, seed_base=777_000, device=device)
        m = conds[name]["metrics"]
        print(f"{name:10s} global R^2 {m['r2_global']:+.3f}  "
              f"fine R^2 {m['r2_fine']:+.3f}  reward {m['reward_mean']:.4f}", flush=True)

    outdir = REPO / "results" / "phase4"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    simplex_scatter(axes[0], conds["normal"]["test"].beliefs, s=0.5, alpha=0.4,
                    title="true beliefs")
    for ax, name in [(axes[1], "normal"), (axes[2], "scrambled")]:
        pred = conds[name]["pred"]
        disp = np.clip(pred, 0, None)
        disp /= np.maximum(disp.sum(axis=1, keepdims=True), 1e-9)
        m = conds[name]["metrics"]
        simplex_scatter(ax, disp, s=0.5, alpha=0.4,
                        title=f"decoded, {name} input\n(global={m['r2_global']:.3f}, "
                              f"fine={m['r2_fine']:.3f})")
    fig.suptitle(f"{bp['name']} {run_dir.name} (trained on real tokens): "
                 "test-time input scramble")
    fig.tight_layout()
    fig.savefig(outdir / "fig_a_main_scrambled_eval.png", dpi=160)

    payload = {name: dict(c["metrics"]) for name, c in conds.items()}
    (outdir / "scrambled_eval.json").write_text(json.dumps(payload, indent=2))
    print(f"-> {outdir}/scrambled_eval.json, fig_a_main_scrambled_eval.png")


if __name__ == "__main__":
    main()
