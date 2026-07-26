"""Conservative defaults for RunPod Pod experiment jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunPodConfig:
    # GraphQL creates/verifies on-demand placement; v1 REST handles lifecycle
    # and billing until v2 exposes equivalent rental-safety fields.
    API_BASE: str = "https://rest.runpod.io/v1"
    V2_API_BASE: str = "https://api.runpod.io/v2"
    API_VERSION: str = "v1"
    GRAPHQL_URL: str = "https://api.runpod.io/graphql"
    CLOUD_TYPE: str = "COMMUNITY"
    INTERRUPTIBLE: bool = False
    GPU_TYPE_IDS: tuple[str, ...] = (
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA RTX 6000 Ada Generation",
    )
    GPU_COUNT: int = 1
    DISK_GB: int = 30
    VOLUME_GB: int = 10
    VOLUME_MOUNT_PATH: str = "/workspace"
    MIN_CUDA: str = "13.0"

    # Public image built from the adjacent Dockerfile. Keep this digest-pinned:
    # it bakes CUDA, torch, Ray/RLlib, Gymnasium, B2, and MLflow dependencies.
    IMAGE: str = (
        "ghcr.io/al-does/rl-harness-runpod"
        "@sha256:1257ac0a0f2b57b80022849a96fa1d9a5bfffff69fabbf233b3cf45dc665fb3c"
    )
    RAY_VERSION: str = "2.56.0"
    TORCH_VERSION: str = "2.12.1"
    GYMNASIUM_VERSION: str = "1.2.2"
    BOTO3_VERSION: str = "1.43.51"
    MLFLOW_VERSION: str = "3.14.0"

    # Public on-demand Community price shown on RunPod's pricing page. The
    # create response is authoritative and is re-checked against --max-price.
    PUBLIC_4090_PRICE_PER_HOUR: float = 0.34

    LIBRARY_REPO_URL: str = "https://github.com/Al-does/RL-Harness.git"
    LIBRARY_REPO_SLUG: str = "Al-does/RL-Harness"
    LIBRARY_DEFAULT_REF: str = "main"
    EXPERIMENT_REPO_URL: str = (
        "https://github.com/Al-does/alex-rl-experiments.git"
    )
    EXPERIMENT_REPO_SLUG: str = "Al-does/alex-rl-experiments"
    DEFAULT_RESULTS_BRANCH: str = "results"
    GIT_USER_NAME: str = "runpod-bot"
    GIT_USER_EMAIL: str = "runpod-bot@users.noreply.github.com"
    EXPERIMENT_REPO_LOCAL: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3].parent
        / "alex-rl-experiments"
    )

    RUNNING_TIMEOUT_S: float = 900.0
    POLL_INTERVAL_S: float = 10.0
    # Mirrors Vast's low default. RunPod never permits disabling this cap.
    MAX_AGE_HOURS: float = 5.0
    INTERACTIVE_MAX_AGE_HOURS: float = 2.0
    MANAGED_NAME_PREFIX: str = "rlh-runpod-"

    STATE_PATH: Path = _HERE / "state.json"
    COST_HISTORY_PATH: Path = _HERE / "cost_history.json"


CONFIG = RunPodConfig()
