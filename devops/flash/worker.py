"""No-Docker RunPod Flash GPU capability probe.

Flash packages this source as an artifact and mounts it into a provider-managed
runtime.  It deliberately does not refer to a container image.
"""

from __future__ import annotations

import os
import platform
from typing import Any

from runpod_flash import Endpoint, GpuGroup


def _endpoint_name() -> str:
    # Flash injects this name into the deployed worker.  Prefer it at runtime:
    # deployment-only environment variables are not retained by the worker.
    value = os.environ.get(
        "FLASH_RESOURCE_NAME",
        os.environ.get("RL_HARNESS_FLASH_ENDPOINT", "rlh-flash-probe"),
    ).strip()
    if not value:
        raise ValueError("RL_HARNESS_FLASH_ENDPOINT must not be empty")
    return value


def _workers() -> tuple[int, int]:
    maximum = int(os.environ.get("RL_HARNESS_FLASH_MAX_WORKERS", "1"))
    if maximum < 1:
        raise ValueError("RL_HARNESS_FLASH_MAX_WORKERS must be at least one")
    return (0, maximum)


@Endpoint(
    name=_endpoint_name(),
    gpu=GpuGroup.ADA_24,
    workers=_workers(),
    idle_timeout=5,
    dependencies=["torch"],
    min_cuda_version="13.0",
)
async def probe(input_data: dict[str, Any]) -> dict[str, Any]:
    """Return GPU runtime evidence without downloading a user Docker image."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Flash worker has no CUDA device")
    return {
        "probe": str(input_data.get("probe", "ok")),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": str(torch.version.cuda or ""),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "workers_min": 0,
        "workers_max": _workers()[1],
        "delivery": "runpod-flash-artifact",
    }
