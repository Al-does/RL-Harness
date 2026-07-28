"""Policy defaults for the no-Docker RunPod Flash backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FlashConfig:
    LIBRARY_REPO_URL: str = "https://github.com/Al-does/RL-Harness.git"
    EXPERIMENT_REPO_URL: str = (
        "https://github.com/Al-does/alex-rl-experiments.git"
    )
    DEFAULT_RESULTS_BRANCH: str = "results"
    RAY_VERSION: str = "2.56.0"
    TORCH_VERSION: str = "2.12.1"
    GYMNASIUM_VERSION: str = "1.2.2"
    PYTHON_VERSION: str = "3.12"
    # Prefer 4090-class Ada workers, then accept slower 24 GB Ampere-class
    # equivalents (L4/A5000/3090) when Ada capacity is unavailable.
    GPU_POOLS: tuple[str, ...] = ("ADA_24", "AMPERE_24")
    GPU_COUNT: int = 1
    WORKERS_MIN: int = 0
    IDLE_TIMEOUT_S: int = 5
    STARTUP_RESERVE_SECONDS: int = 10 * 60
    GPU_RATE_PER_SECOND: float = 0.00031
    ESTIMATED_FEE_RESERVE_FRACTION: float = 0.20
    MAX_JOB_HOURS: float = 7 * 24
    POLL_INTERVAL_SECONDS: float = 5.0
    PROGRESS_INTERVAL_SECONDS: float = 30.0
    NO_PROGRESS_TIMEOUT_SECONDS: float = 10 * 60
    STATE_PATH: Path = Path(__file__).resolve().with_name("state.json")


CONFIG = FlashConfig()
