"""Hardware profiles and runtime configuration for training launchers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


AUTO_RUNNERS = -1  # all available CPUs minus one for the learner/driver


@dataclass(frozen=True)
class HardwareProfile:
    """Infrastructure and throughput settings, never learning hyperparameters."""

    name: str
    learner_device: str  # "cuda" | "mps" | "cpu"
    num_env_runners: int | None  # None -> blueprint's PPOSpec value
    num_envs_per_env_runner: int
    torch_threads: int | None  # None -> torch default
    # Fraction of a GPU per env runner for rollout inference (0 -> CPU
    # forward). The rollout forward recomputes the transformer's full
    # 193-observation lookback window every env step, so CPU inference can
    # dominate sampling.
    num_gpus_per_env_runner: float = 0.0


PROFILES = {
    # MPS learner with threads capped so concurrent runs share the machine.
    "mac": HardwareProfile("mac", "mps", None, 24, 6),
    # RTX 4090 learner on CUDA with one runner per available CPU, up to the
    # benchmarked ceiling.
    "cuda4090": HardwareProfile("cuda4090", "cuda", AUTO_RUNNERS, 24, None),
    # GPU rollout layout swept on a 15.4-core RTX 4090 box: 8 runners x 48
    # environments gave the best measured throughput.
    "cuda4090_gpuinfer": HardwareProfile(
        "cuda4090_gpuinfer",
        "cuda",
        8,
        48,
        None,
        num_gpus_per_env_runner=0.1,
    ),
    "cpu": HardwareProfile("cpu", "cpu", None, 24, None),
}


def available_cpus() -> float:
    """Return the smallest host, affinity, or container CPU limit.

    Vast boxes can report the host's cores through ``os.cpu_count()`` while
    being capped by a much smaller Docker cgroup quota. Using the host count
    would oversubscribe the box.
    """
    limits = [float(os.cpu_count() or 1)]
    try:
        limits.append(float(len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    try:  # cgroup v2
        quota_str, period_str = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_str != "max":
            limits.append(float(quota_str) / float(period_str))
    except (FileNotFoundError, PermissionError, ValueError, ZeroDivisionError):
        pass
    try:  # cgroup v1
        quota = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            limits.append(quota / period)
    except (FileNotFoundError, PermissionError, ValueError, ZeroDivisionError):
        pass
    return max(1.0, min(limits))


def resolve_env_runners(profile: HardwareProfile, default: int) -> int:
    """Resolve runner count without exceeding the schedulable CPU budget."""
    # Reserve one CPU for the driver/learner. Ray silently queues actors that
    # exceed its logical pool, which can otherwise hang before iteration one.
    cap = max(1, int(available_cpus()) - 1)
    if profile.num_env_runners == AUTO_RUNNERS:
        return min(cap, 16)
    return min(profile.num_env_runners or default, cap)


def ensure_ray_initialized() -> None:
    """Make Ray's logical CPU pool match the container's actual CPU budget."""
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=max(1, int(available_cpus())))


def detect_profile() -> str:
    """Select the default profile from the available accelerator backend."""
    import torch

    if torch.cuda.is_available():
        return "cuda4090_gpuinfer"  # verified reward parity vs CUDA CPU-infer/MPS
    if torch.backends.mps.is_available():
        return "mac"
    return "cpu"


def configure_hardware(profile: HardwareProfile) -> None:
    """Validate and configure Ray and Torch for a hardware profile."""
    import torch

    if profile.learner_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA hardware profile selected, but torch cannot use CUDA. "
            "Check the host driver and torch CUDA build."
        )

    # Ray's uv hook can make each worker recreate the already-correct driver
    # environment, downloading and building hundreds of MB per actor.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ensure_ray_initialized()

    # Cap learner threads so concurrent runs share the machine cleanly.
    if profile.torch_threads:
        torch.set_num_threads(profile.torch_threads)

    # RLlib's get_device does not support MPS. Patch the symbol imported by the
    # local TorchLearner; environment runners remain on CPU.
    if profile.learner_device == "mps" and torch.backends.mps.is_available():
        import ray.rllib.core.learner.torch.torch_learner as torch_learner

        torch_learner.get_device = lambda config, n=1: torch.device("mps")
