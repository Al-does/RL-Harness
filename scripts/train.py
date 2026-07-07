"""Gate-enforced training launcher: one blueprint arm + one seed per invocation.

    uv run python scripts/train.py --blueprint a_main --seed 0
    uv run python scripts/train.py --blueprint b_m1 --seed 1 --smoke

Enforcement (program rule: no training before the phase gate passes):
  - phase >= 2 arms require results/phase1/GATE_PASSED (written by the Phase-1
    scripts when the sweep reproduces the known-good values) AND
    results/phase1/REVIEW_APPROVED (written after the Phase-1 review stop).

Two training paths, dispatched on the blueprint:
  - RL arms: RLlib PPO (new API stack) with the custom RLModules
    (Mess3TransformerRLModule / Mess3MLPRLModule) and AuxPPOTorchLearner
    (auxiliary next-token CE when aux_next_token_lambda > 0).
  - Supervised arms (rl_loss_enabled=False: b_sl, a_pred): the identical
    transformer trained by envs/mess3/supervised.py on random-action rollouts.

Checkpoint format (uniform across paths, consumed by analysis/):
  outdir/module_state_<envsteps:08d>.pt   log-spaced, INCLUDING step 0 (N-init)
  outdir/module_state_final.pt
  outdir/progress.jsonl                   per-iteration metrics
  outdir/blueprint.json                   exact arm + seed provenance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueprints.base import Blueprint, get  # noqa: E402

MAX_SEQ_LEN = 32  # learner chunk length for stateful modules


def check_gate(bp: Blueprint, repo: Path):
    if bp.phase <= 1:
        return
    gate_dir = repo / "results" / bp.gate
    for artifact, hint in [
        ("GATE_PASSED", "Run scripts/phase1_gate.py and scripts/phase1_findings.py first."),
        ("REVIEW_APPROVED", "Phase 1 ends with a review stop; create after review."),
    ]:
        if not (gate_dir / artifact).exists():
            raise SystemExit(f"REFUSING to launch '{bp.name}': {gate_dir}/{artifact} missing. {hint}")


def resolve_env(bp: Blueprint):
    mod, cls = bp.env_entry.split(":")
    import importlib

    return getattr(importlib.import_module(mod), cls)


def env_kwargs_for(bp: Blueprint) -> dict:
    kw = dict(bp.env_kwargs)
    if bp.scramble_tokens:
        kw["scramble_tokens"] = True
    return kw


def model_config_for(bp: Blueprint) -> dict:
    m = bp.model
    if m.kind == "transformer":
        return {
            "d_model": m.d_model,
            "n_layers": m.n_layers,
            "n_heads": m.n_heads,
            "context_len": m.context_len,
            "max_seq_len": MAX_SEQ_LEN,
        }
    if m.kind == "mlp":
        return {"mlp_hidden": list(m.mlp_hidden)}
    raise SystemExit(f"model kind {m.kind!r} not implemented (a_lstm is optional/deferred)")


def module_class_for(bp: Blueprint):
    from envs.mess3.rlmodules import Mess3MLPRLModule, Mess3TransformerRLModule

    return Mess3TransformerRLModule if bp.model.kind == "transformer" else Mess3MLPRLModule


def build_config(bp: Blueprint, seed: int, smoke: bool):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from envs.mess3.learners import AuxPPOTorchLearner

    env_cls = resolve_env(bp)
    p = bp.ppo
    return (
        PPOConfig()
        .environment(env_cls, env_config=env_kwargs_for(bp))
        .env_runners(
            num_env_runners=0 if smoke else p.num_env_runners,
            # Batch policy inference across parallel env copies: the
            # transformer forward is ~free in batch, so this is the main
            # rollout speedup on this machine.
            num_envs_per_env_runner=1 if smoke else 24,
            # Under multi-run contention a fragment can exceed the default
            # 60s; timing out wastes the whole fragment and stalls training.
            sample_timeout_s=600.0,
        )
        .learners(learner_class=AuxPPOTorchLearner)
        .training(
            lr=p.lr,
            gamma=p.gamma,
            lambda_=p.gae_lambda,
            clip_param=p.clip,
            vf_loss_coeff=p.vf_coef,
            entropy_coeff=p.ent_coef,
            train_batch_size_per_learner=2048 if smoke else p.train_batch,
            minibatch_size=256 if smoke else p.minibatch,
            num_epochs=p.epochs,
            learner_config_dict={"aux_next_token_lambda": bp.aux_next_token_lambda},
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=module_class_for(bp),
                model_config=model_config_for(bp),
            )
        )
        .debugging(seed=seed)
    )


def save_module_state(algo, path: Path, env_steps: int):
    import torch

    module = algo.learner_group._learner.module["default_policy"].unwrapped()
    torch.save(
        {"state_dict": {k: v.cpu() for k, v in module.state_dict().items()},
         "env_steps": env_steps},
        path,
    )


def run_rl(bp: Blueprint, seed: int, smoke: bool, outdir: Path):
    import torch

    # Cap learner threads so concurrent runs share the machine cleanly.
    torch.set_num_threads(6)
    # Run the (local) learner on MPS: RLlib's get_device only knows CUDA, so
    # patch it where TorchLearner imported it.  Env runners stay on CPU.
    if torch.backends.mps.is_available():
        import ray.rllib.core.learner.torch.torch_learner as _tl

        _tl.get_device = lambda config, n=1: torch.device("mps")
    algo = build_config(bp, seed, smoke).build_algo()
    save_module_state(algo, outdir / "module_state_00000000.pt", 0)  # N-init

    target = 4096 if smoke else bp.total_steps
    sampled, it, next_ckpt = 0, 0, 1
    log = open(outdir / "progress.jsonl", "a")
    t0 = time.time()
    while sampled < target:
        if (outdir / "STOP").exists():  # graceful early stop (plateau call)
            print("STOP file found; finishing early", flush=True)
            break
        result = algo.train()
        it += 1
        er = result.get("env_runners", {})
        lr = result.get("learners", {}).get("default_policy", {})
        sampled = er.get("num_env_steps_sampled_lifetime", 0)
        rec = {
            "iter": it,
            "env_steps": int(sampled),
            "episode_return_mean": er.get("episode_return_mean"),
            "aux_ce": lr.get("aux_ce"),
            "aux_accuracy": lr.get("aux_accuracy"),
            "entropy": lr.get("entropy"),
            "wall_s": round(time.time() - t0, 1),
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if it >= next_ckpt:  # log-spaced (powers of two in iterations)
            save_module_state(algo, outdir / f"module_state_{int(sampled):08d}.pt", int(sampled))
            next_ckpt *= 2
    save_module_state(algo, outdir / "module_state_final.pt", int(sampled))
    log.close()
    algo.stop()
    print(f"done -> {outdir}", flush=True)


def run_supervised(bp: Blueprint, seed: int, smoke: bool, outdir: Path):
    import gymnasium as gym
    import numpy as np

    from envs.mess3.supervised import train_supervised

    env_cls = resolve_env(bp)
    kw = env_kwargs_for(bp)
    env_factory = lambda: env_cls(dict(kw))  # noqa: E731
    probe_env = env_factory()
    obs_dim = int(probe_env.observation_space.shape[0])
    action_space = probe_env.action_space
    # Target: state classification for Environment B's twin; next visible
    # token for Environment A's prediction-only arm.
    target = "state" if isinstance(action_space, gym.spaces.Discrete) else "next_token"
    train_supervised(
        env_factory=env_factory,
        model_config=model_config_for(bp),
        obs_dim=obs_dim,
        action_space=action_space,
        target=target,
        total_steps=8192 if smoke else bp.total_steps,
        outdir=outdir,
        seed=seed,
        fresh_data_episodes=8 if smoke else 256,
    )
    print(f"done -> {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    bp = get(args.blueprint)
    if not args.smoke:
        check_gate(bp, repo)

    outdir = (
        Path(args.out)
        if args.out
        else repo / "results" / f"phase{bp.phase}" / bp.name / f"seed{args.seed}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "blueprint.json", "w") as f:
        json.dump(
            {**bp.__dict__, "model": bp.model.__dict__, "ppo": bp.ppo.__dict__,
             "launch_seed": args.seed},
            f, indent=2, default=str,
        )

    if bp.rl_loss_enabled:
        run_rl(bp, args.seed, args.smoke, outdir)
    else:
        run_supervised(bp, args.seed, args.smoke, outdir)


if __name__ == "__main__":
    main()
