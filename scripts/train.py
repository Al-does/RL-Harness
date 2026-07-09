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
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueprints.base import Blueprint, get  # noqa: E402

MAX_SEQ_LEN = 32  # learner chunk length for stateful modules

# Ray's `uv run` hook (default on in ray>=2.56) makes every worker re-create
# the uv env; on a fresh box each env-runner actor then downloads/builds the
# whole venv (hundreds of MB) and actor startup times out. The driver's venv
# is already correct everywhere we run, so opt out before ray.init().
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


# ---------------------------------------------------------------------------
# Hardware profiles: infra/throughput knobs only, NEVER learning hyperparams
# (lr / batch / epochs stay in the blueprint so results are comparable across
# machines). Selected with --profile; "auto" picks by available accelerator.
# ---------------------------------------------------------------------------


AUTO_RUNNERS = -1  # sentinel: all available CPUs minus one for the learner/driver


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    learner_device: str            # "cuda" | "mps" | "cpu"
    num_env_runners: int | None    # None -> blueprint's PPOSpec value
    num_envs_per_env_runner: int
    torch_threads: int | None      # None -> torch default
    # Fraction of a GPU per env runner for ROLLOUT inference (0 -> CPU
    # forward). The rollout forward recomputes the transformer's full
    # 193-obs lookback window every env step, so on CPU it dominates
    # sampling (~1.7ms/env-step/thread vs 16us for env.step itself).
    num_gpus_per_env_runner: float = 0.0


PROFILES = {
    # This Mac: MPS learner (via the get_device patch below), 4 remote
    # runners x 24 envs, threads capped so concurrent runs share the machine.
    "mac": HardwareProfile("mac", "mps", None, 24, 6),
    # vast.ai RTX 4090 box: learner on CUDA (num_gpus_per_learner=1 -- without
    # it RLlib silently trains the learner on CPU, measured 152s vs 2.7s per
    # update), and one env runner per available CPU core since sampling is the
    # bottleneck (benchmarked 2026-07: 4 runners 0.8k steps/s -> 14 runners
    # 2.2k steps/s on a 15.4-core box; 48 envs/runner was worse than 24).
    "cuda4090": HardwareProfile("cuda4090", "cuda", AUTO_RUNNERS, 24, None),
    # Rollout inference on the GPU too, shared with the learner (the model is
    # ~330K params; both fit easily). Layout swept on a 15.4-core box
    # (163,840-step slices, wall_s): CPU-infer baseline 80s; GPU-infer
    # 1x384 56s, 2x192 38s, 4x96 31s, 8x48 27s (best), 14x24 28s. Fat
    # runners lose to per-episode Python bookkeeping (doesn't amortize with
    # batch); many thin runners lose to GPU contention.
    "cuda4090_gpuinfer": HardwareProfile(
        "cuda4090_gpuinfer", "cuda", 8, 48, None, num_gpus_per_env_runner=0.1,
    ),
    "cpu": HardwareProfile("cpu", "cpu", None, 24, None),
}


def available_cpus() -> float:
    """Container-aware CPU count: vast boxes report the host's cores via
    os.cpu_count() (e.g. 128) but are capped by a docker cgroup quota
    (e.g. 15.4); sizing env runners off the wrong number oversubscribes badly.
    """
    try:  # cgroup v2
        quota_str, period_str = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_str != "max":
            return float(quota_str) / float(period_str)
    except (FileNotFoundError, ValueError):
        pass
    try:  # cgroup v1
        quota = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return quota / period
    except (FileNotFoundError, ValueError):
        pass
    return float(os.cpu_count() or 1)


def resolve_env_runners(prof: HardwareProfile, default: int) -> int:
    if prof.num_env_runners == AUTO_RUNNERS:
        return max(1, min(int(available_cpus()) - 1, 16))
    return prof.num_env_runners or default


def detect_profile() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda4090_gpuinfer"  # verified reward-parity vs cuda4090/mac
    if torch.backends.mps.is_available():
        return "mac"
    return "cpu"


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


def build_config(bp: Blueprint, seed: int, smoke: bool, prof: HardwareProfile):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from envs.mess3.learners import AuxPPOTorchLearner

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
            learner_class=AuxPPOTorchLearner,
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


def run_rl(bp: Blueprint, seed: int, smoke: bool, outdir: Path,
           prof: HardwareProfile, max_steps: int | None = None):
    import torch

    # Cap learner threads so concurrent runs share the machine cleanly.
    if prof.torch_threads:
        torch.set_num_threads(prof.torch_threads)
    # Run the (local) learner on MPS: RLlib's get_device only knows CUDA, so
    # patch it where TorchLearner imported it.  Env runners stay on CPU.
    if prof.learner_device == "mps" and torch.backends.mps.is_available():
        import ray.rllib.core.learner.torch.torch_learner as _tl

        _tl.get_device = lambda config, n=1: torch.device("mps")
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
            "aux_ce": lr.get("aux_ce"),
            "aux_accuracy": lr.get("aux_accuracy"),
            "entropy": lr.get("entropy"),
            "wall_s": round(time.time() - t0, 1),
            # Where the iteration's time went (sampling vs learner update):
            # makes remote benchmark runs diagnosable from progress.jsonl alone.
            "sample_s": round(float(timers.get("env_runner_sampling_timer") or 0.0), 2),
            "learn_s": round(float(timers.get("learner_update_timer") or 0.0), 2),
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

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "blueprint.json", "w") as f:
        json.dump(
            {**bp.__dict__, "model": bp.model.__dict__, "ppo": bp.ppo.__dict__,
             "launch_seed": args.seed, "hardware_profile": prof.__dict__},
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
