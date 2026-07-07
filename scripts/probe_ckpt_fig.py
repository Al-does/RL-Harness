"""Render the barycentric decoded-vs-true belief figure for ONE checkpoint.

Same figure as ``probe_arm.py`` produces for the final checkpoint
(fig_probe_final.png), but for an arbitrary ``--ckpt`` so the decoded belief
geometry can be watched as it forms over training.

    uv run python scripts/probe_ckpt_fig.py \
        --run results/phase3/a_main/seed0 \
        --ckpt module_state_00524288.pt \
        --out  results/phase3/a_main/seed0/fig_probe_00524288.png

The affine probe is refit on each checkpoint's own activations (held-out
train/test seed ranges are disjoint), matching probe_arm.py exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint filename in --run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=120_000,
                    help="probe-train pairs (held-out = half); matches final fig")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run)
    bp = load_blueprint_dict(run_dir)
    env_factory = env_factory_from_blueprint(bp)
    policy_mode = "random" if bp.get("rl_loss_enabled") in (False, "False") else "policy"

    ckpt_path = run_dir / args.ckpt
    env_steps = int(torch.load(ckpt_path, map_location="cpu", weights_only=True)["env_steps"])
    module = load_module(run_dir, ckpt_path)

    # Deterministic, checkpoint-specific probe seed (disjoint train/test ranges).
    base_seed = 7_000_000 + int(bp.get("launch_seed", 0)) * 100_000 + env_steps % 1000
    train = collect_probe_data(
        module, env_factory, n_steps=args.steps, seed=base_seed,
        policy_mode=policy_mode, device=args.device, n_envs=32,
    )
    test = collect_probe_data(
        module, env_factory, n_steps=args.steps // 2, seed=base_seed + 500_000,
        policy_mode=policy_mode, device=args.device, n_envs=32,
    )
    res = evaluate_probe(train, test)
    W, b = res["probe"]
    pred = probe_predict(W, b, test.activations)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    simplex_scatter(axes[0], test.beliefs, s=0.5, alpha=0.4, title="true beliefs")
    simplex_scatter(axes[1], np.clip(pred, 0, 1), s=0.5, alpha=0.4,
                    title=f"decoded (global={res['r2_global']:.3f}, "
                          f"fine={res['r2_fine']:.3f})")
    fig.suptitle(f"{bp['name']} seed{bp.get('launch_seed')} @ {env_steps} steps")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"saved {args.out}  env_steps={env_steps}  "
          f"r2_global={res['r2_global']:.4f}  r2_fine={res['r2_fine']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
