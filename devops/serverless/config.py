"""Conservative policy defaults for disposable RunPod Serverless jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ServerlessConfig:
    API_BASE: str = "https://api.runpod.io"
    QUEUE_BASE: str = "https://api.runpod.ai/v2"
    USER_AGENT: str = "rl-harness-serverless/1.0"

    LIBRARY_REPO_URL: str = "https://github.com/Al-does/RL-Harness.git"
    EXPERIMENT_REPO_URL: str = (
        "https://github.com/Al-does/alex-rl-experiments.git"
    )
    DEFAULT_RESULTS_BRANCH: str = "results"

    # Set only after the adjacent Dockerfile has been published to a registry
    # that Serverless can pull anonymously. Overrides must remain digest-pinned.
    IMAGE: str | None = None
    IMAGE_BASE: str = (
        "ghcr.io/al-does/rl-harness-runpod"
        "@sha256:1257ac0a0f2b57b80022849a96fa1d9a5bfffff69fabbf233b3cf45dc665fb3c"
    )
    RAY_VERSION: str = "2.56.0"
    TORCH_VERSION: str = "2.12.1"
    GYMNASIUM_VERSION: str = "1.2.2"

    MANAGED_NAME_PREFIX: str = "rlh-serverless-"
    GPU_POOLS: tuple[str, ...] = ("ADA_24",)
    GPU_COUNT: int = 1
    DISK_GB: int = 30
    WORKERS_MIN: int = 0
    WORKERS_MAX: int = 1
    SCALER_TYPE: str = "QUEUE_DELAY"
    SCALER_VALUE: float = 4.0
    IDLE_TIMEOUT_S: int = 5
    FLASHBOOT: str = "FLASHBOOT"

    DEFAULT_MAX_AGE_HOURS: float = 5.0
    DEFAULT_QUEUE_TIMEOUT_MINUTES: float = 60.0
    MAX_JOB_HOURS: float = 7 * 24
    MIN_EXECUTION_SECONDS: int = 5
    MIN_TTL_SECONDS: int = 10
    POLL_INTERVAL_SECONDS: float = 5.0
    BILLING_SETTLEMENT_DELAY_SECONDS: float = 15 * 60

    # Current public ADA_24 flex rate is $0.00031/s. A 15-minute startup
    # reserve covers image pull/boot. Disk is documented at about $0.10/GB/mo.
    GPU_RATE_PER_SECOND: float = 0.00031
    DISK_RATE_PER_GB_MONTH: float = 0.10
    STARTUP_RESERVE_SECONDS: int = 15 * 60
    ESTIMATED_FEE_RESERVE_FRACTION: float = 0.20

    STATE_PATH: Path = field(default_factory=lambda: _HERE / "state.json")
    COST_HISTORY_PATH: Path = field(
        default_factory=lambda: _HERE / "cost_history.json"
    )


CONFIG = ServerlessConfig()
