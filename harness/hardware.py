"""Operational hardware profiles and Ray/Torch runtime setup."""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


AUTO_RUNNERS = -1


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Infrastructure settings that do not change an experiment's science."""

    name: str
    learner_device: str
    num_env_runners: int | None
    num_envs_per_env_runner: int
    torch_threads: int | None
    num_gpus_per_env_runner: float = 0.0


PROFILES: dict[str, HardwareProfile] = {
    "mac": HardwareProfile("mac", "mps", None, 24, 6),
    "cuda4090": HardwareProfile("cuda4090", "cuda", AUTO_RUNNERS, 24, None),
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
    """Return the smallest host, affinity, or container CPU limit."""
    limits = [float(os.cpu_count() or 1)]
    try:
        limits.append(float(len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    try:
        quota_str, period_str = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_str != "max":
            limits.append(float(quota_str) / float(period_str))
    except (FileNotFoundError, PermissionError, ValueError, ZeroDivisionError):
        pass
    try:
        quota = float(
            Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text()
        )
        period = float(
            Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text()
        )
        if quota > 0:
            limits.append(quota / period)
    except (FileNotFoundError, PermissionError, ValueError, ZeroDivisionError):
        pass
    return max(1.0, min(limits))


def resolve_env_runners(profile: HardwareProfile, default: int) -> int:
    """Resolve a requested runner count within the schedulable CPU budget."""
    cap = max(1, int(available_cpus()) - 1)
    if profile.num_env_runners == AUTO_RUNNERS:
        return min(cap, 16)
    return min(profile.num_env_runners or default, cap)


def ensure_ray_initialized() -> bool:
    """Initialize Ray to the actual CPU budget; return whether we started it."""
    import ray

    if ray.is_initialized():
        return False
    # Ray executes runtime-env worker commands through a shell. Quote the
    # interpreter explicitly so project paths containing spaces remain valid.
    ray.init(
        num_cpus=max(1, int(available_cpus())),
        runtime_env={"py_executable": shlex.quote(sys.executable)},
    )
    return True


def detect_profile() -> str:
    """Select the default profile from the available accelerator backend."""
    import torch

    if torch.cuda.is_available():
        return "cuda4090_gpuinfer"
    if torch.backends.mps.is_available():
        return "mac"
    return "cpu"


def configure_hardware(profile: HardwareProfile) -> bool:
    """Validate and configure Ray and Torch for a hardware profile.

    Returns whether this call initialized Ray, allowing its caller to clean up
    only the runtime it owns.
    """
    import torch

    if profile.learner_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA hardware profile selected, but torch cannot use CUDA. "
            "Check the host driver and torch CUDA build."
        )

    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    started_ray = ensure_ray_initialized()

    if profile.torch_threads:
        torch.set_num_threads(profile.torch_threads)

    # RLlib 2.56 has no public MPS resource setting. Keep this compatibility
    # patch isolated here; environment runners remain on CPU.
    if profile.learner_device == "mps" and torch.backends.mps.is_available():
        import ray.rllib.core.learner.torch.torch_learner as torch_learner

        torch_learner.get_device = lambda config, n=1: torch.device("mps")

    return started_ray


def shutdown_ray_if_owned(started_ray: bool) -> None:
    """Shut down Ray only when the corresponding setup call started it."""
    if not started_ray:
        return
    import ray

    ray.shutdown()
