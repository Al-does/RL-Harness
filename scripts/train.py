"""Gate-enforced training launcher: one blueprint arm + one seed per invocation.

    uv run python scripts/train.py --blueprint a_main --seed 0
    uv run python scripts/train.py --blueprint b_m1 --seed 1 --smoke

Enforcement (program rule: no training before the phase gate passes):
  - phase >= 2 arms require results/phase1/GATE_PASSED (written by the Phase-1
    scripts when the sweep reproduces the known-good values) AND
    results/phase1/REVIEW_APPROVED (written after the Phase-1 review stop).

Two training paths, dispatched on the blueprint:
  - RL arms: RLlib PPO (new API stack) with the custom RLModules
    and Learner composition selected directly by each blueprint.
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
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueprints.base import Blueprint, get  # noqa: E402
from scripts.hardware import (  # noqa: E402
    PROFILES,
    HardwareProfile,
    available_cpus,
    configure_hardware,
    detect_profile,
    resolve_env_runners,
)

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
    return bp.model.to_model_config()


def build_config(bp: Blueprint, seed: int, smoke: bool, prof: HardwareProfile):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    env_cls = resolve_env(bp)
    p = bp.ppo
    return (
        PPOConfig()
        .environment(env_cls, env_config=env_kwargs_for(bp))
        .env_runners(
            num_env_runners=0 if smoke else resolve_env_runners(prof, p.num_env_runners),
            # Batch policy inference across parallel env copies: the
            # transformer forward is ~free in batch, so this is the main
            # rollout speedup on this machine.
            num_envs_per_env_runner=1 if smoke else prof.num_envs_per_env_runner,
            num_gpus_per_env_runner=0 if smoke else prof.num_gpus_per_env_runner,
            # Under multi-run contention a fragment can exceed the default
            # 60s; timing out wastes the whole fragment and stalls training.
            sample_timeout_s=600.0,
        )
        .learners(
            learner_class=bp.learner_class,
            learner_config_dict=dict(bp.aux_config),
            # RLlib defaults to 0 GPUs; without this the learner runs on CPU
            # even when CUDA is present. (MPS is handled by the patch in
            # run_rl -- RLlib's get_device only knows CUDA.)
            num_gpus_per_learner=1 if prof.learner_device == "cuda" else 0,
        )
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
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=bp.model.model_class,
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


def run_rl(bp: Blueprint, seed: int, smoke: bool, outdir: Path,
           prof: HardwareProfile, max_steps: int | None = None):
    configure_hardware(prof)
    algo = build_config(bp, seed, smoke, prof).build_algo()
    save_module_state(algo, outdir / "module_state_00000000.pt", 0)  # N-init

    target = 4096 if smoke else (max_steps or bp.total_steps)
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
        timers = result.get("timers", {})
        sampled = er.get("num_env_steps_sampled_lifetime", 0)
        rec = {
            "iter": it,
            "env_steps": int(sampled),
            "episode_return_mean": er.get("episode_return_mean"),
            "next_token_aux/ce": lr.get("next_token_aux/ce"),
            "next_token_aux/accuracy": lr.get(
                "next_token_aux/accuracy"
            ),
            "entropy": lr.get("entropy"),
            "wall_s": round(time.time() - t0, 1),
            # Where the iteration's time went (sampling vs learner update):
            # makes remote benchmark runs diagnosable from progress.jsonl alone.
            "sample_s": round(float(timers.get("env_runner_sampling_timer") or 0.0), 2),
            "learn_s": round(float(timers.get("learner_update_timer") or 0.0), 2),
        }
        try:
            load1, _, load15 = os.getloadavg()
            rec.update(host_load1=round(load1, 2), host_load15=round(load15, 2))
        except OSError:
            pass
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
        model_class=bp.model.model_class,
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


def maybe_vast_teardown(args, success: bool):
    """Opt-in vast box self-destruct: push results/ then destroy the instance.

    Inert unless the box was provisioned with self-destruct (VAST_SELF_DESTRUCT=1)
    or --vast-teardown was passed. A crashed run stays up for debugging unless
    --teardown-on-error / VAST_TEARDOWN_ON_ERROR=1 is set. The hook lives here in
    the launcher so every blueprint benefits with no per-blueprint edits.
    """
    import os

    enabled = os.environ.get("VAST_SELF_DESTRUCT") == "1" or args.vast_teardown
    if not enabled:
        return
    if not success and not (args.teardown_on_error or os.environ.get("VAST_TEARDOWN_ON_ERROR") == "1"):
        print("run failed; leaving vast box up for debugging "
              "(pass --teardown-on-error to push+destroy anyway)", flush=True)
        return
    from devops.vast.self_destruct import push_results_and_destroy

    push_results_and_destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default=None, choices=sorted(PROFILES),
                    help="hardware profile (default: auto-detect by accelerator)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the blueprint's total_steps (diagnostic/benchmark runs)")
    ap.add_argument("--env-runners", type=int, default=None,
                    help="override the profile's env-runner count (benchmark runs)")
    ap.add_argument("--envs-per-runner", type=int, default=None,
                    help="override the profile's envs-per-runner (benchmark runs)")
    ap.add_argument("--gpu-per-runner", type=float, default=None,
                    help="override the profile's GPU fraction per env runner (benchmark runs)")
    ap.add_argument("--vast-teardown", action="store_true",
                    help="on completion, push results/ to the remote and destroy this vast box")
    ap.add_argument("--teardown-on-error", action="store_true",
                    help="with --vast-teardown/self-destruct, also tear down if the run raises")
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
    prof = PROFILES[args.profile or os.environ.get("TRAIN_PROFILE") or detect_profile()]
    from dataclasses import replace

    if args.env_runners is not None:
        prof = replace(prof, num_env_runners=args.env_runners)
    if args.envs_per_runner is not None:
        prof = replace(prof, num_envs_per_env_runner=args.envs_per_runner)
    if args.gpu_per_runner is not None:
        prof = replace(prof, num_gpus_per_env_runner=args.gpu_per_runner)
    print(f"hardware profile: {prof}", flush=True)
    cpu_count = available_cpus()
    planned_runners = 0 if args.smoke else resolve_env_runners(prof, bp.ppo.num_env_runners)
    try:
        startup_load = tuple(round(x, 2) for x in os.getloadavg())
    except OSError:
        startup_load = None
    print(
        f"hardware runtime: available_cpus={cpu_count:.2f}, "
        f"env_runners={planned_runners}, host_load={startup_load}",
        flush=True,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "blueprint.json", "w") as f:
        learner_class = bp.learner_class
        json.dump(
            {**bp.__dict__, "model": bp.model.to_dict(), "ppo": bp.ppo.__dict__,
             "learner_class": (
                 f"{learner_class.__module__}:{learner_class.__qualname__}"
             ),
             "launch_seed": args.seed, "hardware_profile": prof.__dict__,
             "hardware_runtime": {
                 "available_cpus": cpu_count,
                 "env_runners": planned_runners,
                 "host_load": startup_load,
             }},
            f, indent=2, default=str,
        )

    success = False
    try:
        if bp.rl_loss_enabled:
            run_rl(bp, args.seed, args.smoke, outdir, prof, args.max_steps)
        else:
            run_supervised(bp, args.seed, args.smoke, outdir)
        success = True
    finally:
        maybe_vast_teardown(args, success)


if __name__ == "__main__":
    main()
