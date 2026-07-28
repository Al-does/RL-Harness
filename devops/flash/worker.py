"""No-Docker RunPod Flash RL experiment worker.

Flash packages this source as an artifact and mounts it into a provider-managed
runtime.  It deliberately does not refer to a container image.
"""

from __future__ import annotations

import asyncio
import os
import platform
import uuid
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


def _endpoint_env() -> dict[str, str]:
    """Forward only credentials required by the immutable experiment worker."""
    names = (
        "GH_TOKEN",
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
        "B2_PREFIX",
        "RL_HARNESS_SOURCE_SHA256",
    )
    return {name: os.environ[name] for name in names if os.environ.get(name)}


@Endpoint(
    name=_endpoint_name(),
    gpu=GpuGroup.ADA_24,
    workers=_workers(),
    idle_timeout=5,
    dependencies=[
        "ray[rllib]==2.56.0",
        "gymnasium==1.2.2",
        "matplotlib==3.11.1",
        "scipy==1.18.0",
        "boto3==1.43.51",
        "mlflow-skinny==3.14.0",
    ],
    system_dependencies=["git"],
    env=_endpoint_env(),
    min_cuda_version="12.8",
)
async def run_experiment(input_data: dict[str, Any]) -> dict[str, Any]:
    """Run one immutable experiment or a lightweight runtime probe."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Flash worker has no CUDA device")
    if input_data.get("_probe_only") is True:
        sleep_seconds = float(input_data.get("sleep_seconds", 0))
        if not 0 <= sleep_seconds <= 60:
            raise ValueError("sleep_seconds must be between zero and 60")
        if sleep_seconds:
            await asyncio.sleep(sleep_seconds)
        return {
            "probe": str(input_data.get("probe", "ok")),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": str(torch.version.cuda or ""),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "workers_min": 0,
            "workers_max": _workers()[1],
            "sleep_seconds": sleep_seconds,
            "delivery": "runpod-flash-artifact",
        }

    # The shared handler is staged under a harness-specific name. Flash reserves
    # handler.py for its generated entrypoint, so importing that name can load a
    # stale provider file instead of the source artifact.
    import rlh_experiment_handler as experiment_handler

    def emit(_job: dict[str, Any], message: str) -> None:
        experiment_handler.log(message)

    experiment_handler.progress = emit
    job_id = str(input_data.get("launcher_job_id") or uuid.uuid4())
    return await asyncio.to_thread(
        experiment_handler.handler,
        {"id": job_id, "input": input_data},
    )
