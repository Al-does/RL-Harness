"""Declarative experiment resource and budget contracts.

These contracts are pure data derived from ``HardwareProfile`` and CLI smoke
flags. They can be inspected by launchers without constructing a Ray cluster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from harness.hardware import AUTO_RUNNERS, PROFILES, HardwareProfile


@dataclass(frozen=True, slots=True)
class ResourceContract:
    """RLlib placement demand implied by a hardware profile and smoke mode."""

    profile_name: str
    smoke: bool
    learner_device: str
    learner_gpus: float
    num_env_runners: int
    num_envs_per_env_runner: int
    gpus_per_env_runner: float
    env_runner_gpus: float
    total_gpus: float
    estimated_cpus: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fits(
        self,
        *,
        available_gpus: float,
        available_cpus: float | None = None,
    ) -> bool:
        if self.total_gpus > available_gpus + 1e-9:
            return False
        if available_cpus is not None and self.estimated_cpus > available_cpus + 1e-9:
            return False
        return True

    def rejection_reason(
        self,
        *,
        available_gpus: float,
        available_cpus: float | None = None,
    ) -> str | None:
        if self.total_gpus > available_gpus + 1e-9:
            return (
                f"resource contract requests {self.total_gpus:g} GPU(s) but "
                f"endpoint capacity is {available_gpus:g} GPU(s); "
                f"profile={self.profile_name!r} smoke={self.smoke}"
            )
        if available_cpus is not None and self.estimated_cpus > available_cpus + 1e-9:
            return (
                f"resource contract requests {self.estimated_cpus:g} CPU(s) but "
                f"endpoint capacity is {available_cpus:g} CPU(s); "
                f"profile={self.profile_name!r} smoke={self.smoke}"
            )
        return None


def resolve_contract_runners(
    profile: HardwareProfile,
    *,
    smoke: bool,
    default_env_runners: int = 8,
) -> int:
    """Mirror experiment ``apply_runtime_resources`` without Ray init."""
    if smoke:
        return 0
    if profile.num_env_runners == AUTO_RUNNERS:
        return min(default_env_runners, 16)
    return int(profile.num_env_runners or default_env_runners)


def resource_contract_from_profile(
    profile: HardwareProfile | str,
    *,
    smoke: bool = False,
    default_env_runners: int = 8,
) -> ResourceContract:
    """Compute placement demand from a profile without starting Ray."""
    resolved = PROFILES[profile] if isinstance(profile, str) else profile
    runners = resolve_contract_runners(
        resolved,
        smoke=smoke,
        default_env_runners=default_env_runners,
    )
    envs_per = 1 if smoke else resolved.num_envs_per_env_runner
    gpus_per_runner = 0.0 if smoke else float(resolved.num_gpus_per_env_runner)
    learner_gpus = 1.0 if resolved.learner_device == "cuda" else 0.0
    env_runner_gpus = runners * gpus_per_runner
    notes: list[str] = []
    if smoke:
        notes.append("smoke disables env runners; only the learner GPU counts")
    if resolved.num_env_runners == AUTO_RUNNERS and not smoke:
        notes.append(
            f"AUTO_RUNNERS preflight assumes {runners} env runners "
            "(remote hosts may resolve fewer from cgroup CPU limits)"
        )
    # One CPU for the driver/learner plus one per env runner is a lower bound
    # used only for capacity rejection, not for Ray init.
    estimated_cpus = 1.0 + float(runners)
    return ResourceContract(
        profile_name=resolved.name,
        smoke=smoke,
        learner_device=resolved.learner_device,
        learner_gpus=learner_gpus,
        num_env_runners=runners,
        num_envs_per_env_runner=envs_per,
        gpus_per_env_runner=gpus_per_runner,
        env_runner_gpus=env_runner_gpus,
        total_gpus=learner_gpus + env_runner_gpus,
        estimated_cpus=estimated_cpus,
        notes=tuple(notes),
    )


def parse_hardware_from_argv(
    argv: list[str],
    *,
    default_profile: str,
) -> tuple[str, bool]:
    """Extract ``--hardware`` / ``--smoke`` from an ``rl-harness`` argv list."""
    smoke = "--smoke" in argv
    profile = default_profile
    for index, part in enumerate(argv):
        if part == "--hardware":
            if index + 1 >= len(argv):
                raise ValueError("--hardware requires a value")
            profile = argv[index + 1]
        elif part.startswith("--hardware="):
            profile = part.partition("=")[2]
    if not profile or profile == "auto":
        profile = default_profile
    if profile not in PROFILES:
        raise ValueError(
            f"unknown hardware profile {profile!r}; "
            f"known={sorted(PROFILES)}"
        )
    return profile, smoke


__all__ = [
    "ResourceContract",
    "parse_hardware_from_argv",
    "resolve_contract_runners",
    "resource_contract_from_profile",
]
