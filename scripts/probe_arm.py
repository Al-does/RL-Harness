"""Probe every checkpoint of one (arm, seed) run: the N-init deliverable.

    uv run python scripts/probe_arm.py --run results/phase2/b_r1/seed0
    uv run python scripts/probe_arm.py --run ... --final-only --steps 120000

Per checkpoint: held-out global and fine R^2 (branch depths 1 and 2), plus —
at the final checkpoint — the barycentric decoded-vs-true figure, mean env
reward of the rollouts, within-branch action-variance fraction (Environment A
diagnostics), and belief-cluster structure (Environment B quantization check).
Writes probe_curve.json and figures into the run directory.

Probe hygiene: train pairs and held-out pairs use disjoint seed ranges
derived from (base_seed, checkpoint index); rollouts are collected under the
checkpoint's own policy (its exploration distribution), matching training.
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

from analysis.checkpoints import (  # noqa: E402
    env_factory_from_blueprint,
    list_checkpoints,
    load_blueprint_dict,
    load_module,
    read_progress,
)
from analysis.plots import simplex_scatter  # noqa: E402
from analysis.probe import (  # noqa: E402
    collect_probe_data,
    evaluate_probe,
    probe_predict,
    within_branch_action_variance_fraction,
)


def default_device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def probe_checkpoint(module, env_factory, *, steps, seed, policy_mode, device):
    train = collect_probe_data(
        module, env_factory, n_steps=steps, seed=seed, policy_mode=policy_mode,
        device=device, n_envs=32,
    )
    test = collect_probe_data(
        module, env_factory, n_steps=steps // 2, seed=seed + 500_000,
        policy_mode=policy_mode, device=device, n_envs=32,
    )
    res = evaluate_probe(train, test)
    return res, train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=60_000,
                    help="probe-train pairs per checkpoint (held-out = half)")
    ap.add_argument("--final-steps", type=int, default=120_000)
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--policy-mode", default=None,
                    help="override rollout policy (default: arm's own convention)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or default_device()

    run_dir = Path(args.run)
    bp = load_blueprint_dict(run_dir)
    env_factory = env_factory_from_blueprint(bp)
    base_seed = 7_000_000 + int(bp.get("launch_seed", 0)) * 100_000

    # Supervised arms trained on random rollouts are probed the same way.
    default_mode = "random" if bp.get("rl_loss_enabled") in (False, "False") else "policy"
    policy_mode = args.policy_mode or default_mode

    ckpts = list_checkpoints(run_dir)
    if args.final_only:
        ckpts = ckpts[-1:]

    curve = []
    for i, (env_steps, path) in enumerate(ckpts):
        module = load_module(run_dir, path)
        is_final = i == len(ckpts) - 1
        steps = args.final_steps if is_final else args.steps
        res, train, test = probe_checkpoint(
            module, env_factory, steps=steps,
            seed=base_seed + i * 1_000, policy_mode=policy_mode, device=device,
        )
        W, b = res.pop("probe")
        entry = {
            "env_steps": env_steps,
            "ckpt": path.name,
            "policy_mode": policy_mode,
            "reward_mean": float(test.rewards.mean()),
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in res.items()},
        }
        if is_final:
            # Deterministic-policy reward (comparable to the analytic ceilings,
            # which assume a deterministic optimal policy).
            greedy = collect_probe_data(
                module, env_factory, n_steps=60_000,
                seed=base_seed + 900_000, policy_mode="greedy",
                device=device, n_envs=32,
            )
            entry["reward_greedy"] = float(greedy.rewards.mean())
            entry["within_branch_action_var_frac"] = within_branch_action_variance_fraction(test)
            # State-head accuracy is defined only for models that compose the
            # corresponding supervised head (currently the B-SL arm).
            state_aux_head = getattr(module, "state_aux_head", None)
            if state_aux_head is not None:
                import torch as _torch

                with _torch.no_grad():
                    aux = state_aux_head(
                        _torch.from_numpy(test.activations).float().to(device)
                    )
                entry["aux_acc_state"] = float(
                    (aux.argmax(-1).cpu().numpy() == test.states).mean()
                )
            # Cluster structure of DECODED beliefs (Env-B quantization check):
            # fraction of decoded-belief variance explained by 3 k-means cells.
            pred = probe_predict(W, b, test.activations)
            entry["decoded_kmeans3_explained"] = kmeans_explained(pred, 3)
            entry["true_belief_kmeans3_explained"] = kmeans_explained(test.beliefs, 3)
            # Display-only simplex projection (clip negatives, renormalize).
            # Metrics above use the RAW affine outputs, which need not lie on
            # the simplex; without this projection points render outside the
            # triangle whenever predicted components do not sum to 1.
            disp = np.clip(pred, 0, None)
            disp /= np.maximum(disp.sum(axis=1, keepdims=True), 1e-9)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
            simplex_scatter(axes[0], test.beliefs, s=0.5, alpha=0.4, title="true beliefs")
            simplex_scatter(axes[1], disp, s=0.5, alpha=0.4,
                            title=f"decoded, simplex-projected for display\n"
                                  f"(global={entry['r2_global']:.3f}, "
                                  f"fine={entry['r2_fine']:.3f})")
            fig.suptitle(f"{bp['name']} seed{bp.get('launch_seed')} @ {env_steps} steps")
            fig.tight_layout()
            fig.savefig(run_dir / "fig_probe_final.png", dpi=160)
            plt.close(fig)
        curve.append(entry)
        print({k: v for k, v in entry.items() if k != "ckpt"}, flush=True)

    with open(run_dir / "probe_curve.json", "w") as f:
        json.dump(curve, f, indent=2)

    # Probe-R^2-over-training overlaid on the reward curve.
    if len(curve) > 1:
        prog = read_progress(run_dir)
        fig, ax1 = plt.subplots(figsize=(7, 4.2))
        xs = [max(c["env_steps"], 1) for c in curve]
        ax1.plot(xs, [c["r2_global"] for c in curve], "o-", label="global R^2", color="C0")
        ax1.plot(xs, [c["r2_fine"] for c in curve], "s-", label="fine R^2 (depth 2)", color="C1")
        ax1.set_xscale("log")
        ax1.set_xlabel("env steps")
        ax1.set_ylabel("held-out R^2")
        ax1.axhline(0, color="gray", lw=0.5)
        ax1.legend(loc="upper left", fontsize=8)
        if prog:
            key = "episode_return_mean" if prog[0].get("episode_return_mean") is not None else "accuracy"
            ax2 = ax1.twinx()
            ax2.plot([p["env_steps"] for p in prog if p.get(key) is not None],
                     [p[key] for p in prog if p.get(key) is not None],
                     color="C2", alpha=0.5, lw=1, label=key)
            ax2.set_ylabel(key, color="C2")
        fig.suptitle(f"{bp['name']} seed{bp.get('launch_seed')}: probe curves over training")
        fig.tight_layout()
        fig.savefig(run_dir / "fig_probe_curve.png", dpi=160)
        plt.close(fig)
    print(f"done -> {run_dir}/probe_curve.json", flush=True)


def kmeans_explained(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> float:
    """Fraction of variance explained by a k-means clustering (numpy, small k)."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    within = ((X - C[lab]) ** 2).sum()
    total = ((X - X.mean(0)) ** 2).sum()
    return float(1.0 - within / total)


if __name__ == "__main__":
    main()
