"""Defaults for the vast.ai provisioning toolkit.

Everything tunable lives here so the client / scoring / CLI stay declarative.
Values are deliberately conservative for an RTX 4090 training box; override the
common ones from the CLI (``--disk``, ``--image``, ``--max-price``, ``--regions``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VastConfig:
    # --- what to rent ---------------------------------------------------
    GPU_NAME: str = "RTX_4090"
    NUM_GPUS: int = 1
    DISK_GB: float = 30.0
    # Minimal Ubuntu+CUDA base. torch's PyPI wheels bundle their own CUDA
    # runtime, so `uv sync` only needs a compatible host NVIDIA driver — not a
    # CUDA-matched container. @vastai-automatic-tag is resolved server-side.
    IMAGE: str = "vastai/base-image:@vastai-automatic-tag"

    # --- proximity (see scoring.py) -------------------------------------
    # Ordered region preference by 2-letter country code. Offers only expose a
    # coarse geolocation string ("California, US"), so this is a tiebreak, not
    # a true geodistance.
    HOME_REGIONS: tuple[str, ...] = ("US", "CA")

    # --- hard gates -----------------------------------------------------
    MIN_RELIABILITY: float = 0.98        # reliability2 >= this
    MIN_DAYS: float = 2.0                # offer max rental duration >= this
    DISK_HEADROOM_GB: float = 5.0        # require disk_space >= DISK_GB + this
    # Host driver must support this CUDA version: the pinned torch==2.12.1
    # PyPI wheels are CUDA 13.0 builds and refuse to run on older drivers
    # (hit in practice: driver 570 / CUDA 12.8 box -> torch.cuda unusable).
    MIN_CUDA: float = 13.0
    # Sampling is CPU-bound (parallel env runners); a box with a tiny docker
    # CPU quota starves rollouts no matter the GPU (hit in practice: a
    # 5.76-core box was ~3x slower than a 15.4-core one).
    MIN_CPU_CORES: float = 12.0

    # --- ranking --------------------------------------------------------
    # Prices within this $/hr band are treated as equal, so proximity breaks
    # near-ties (see scoring.rank_offers). 0.02 == 2 cents/hr.
    PRICE_TOLERANCE: float = 0.02
    # Auto bid for interruptible = min_bid * this margin (headroom over the
    # market floor so the box actually starts).
    BID_MARGIN: float = 1.5

    # --- code delivery / git -------------------------------------------
    REPO_URL: str = "https://github.com/Al-does/RLLibHarnesBeta.git"
    REPO_SLUG: str = "Al-does/RLLibHarnesBeta"
    DEFAULT_RESULTS_BRANCH: str = "results"
    GIT_USER_NAME: str = "vast-bot"
    GIT_USER_EMAIL: str = "vast-bot@users.noreply.github.com"
    # push_results retry loop (survives concurrent boxes racing the branch tip).
    RESULT_PUSH_ATTEMPTS: int = 6

    # --- local machine paths -------------------------------------------
    SSH_KEY_PATH: Path = field(default_factory=lambda: Path("~/.ssh/id_rsa.pub").expanduser())
    API_KEY_FILE: Path = field(default_factory=lambda: Path("~/.vast_api_key").expanduser())
    SSH_CONFIG_PATH: Path = field(default_factory=lambda: Path("~/.ssh/config.d/vast.conf").expanduser())
    STATE_PATH: Path = _HERE / "state.json"

    # --- readiness polling ----------------------------------------------
    RUNNING_TIMEOUT_S: float = 900.0     # wait for actual_status == running
    READY_TIMEOUT_S: float = 1200.0      # additional wait for /root/.vast_ready
    POLL_INTERVAL_S: float = 10.0

    # --- max-age cost cap -----------------------------------------------
    # Hard wall-clock lifetime cap. Each box arms an on-box watchdog that
    # destroys it after this many hours (independent of the local machine — it
    # fires even if this Mac is off, or if the run never finished). Set 0 (or
    # --max-age 0) to disable. The local `reap` subcommand is the backstop.
    MAX_AGE_HOURS: float = 5.0


CONFIG = VastConfig()
