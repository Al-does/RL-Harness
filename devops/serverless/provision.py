"""Dry-run-first lifecycle CLI for disposable RunPod Serverless jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.storage.b2 import b2_env_for_remote

from .client import ServerlessClient, ServerlessClientError, resolve_api_key
from .config import CONFIG, ServerlessConfig
from .redaction import redact_metadata, redact_sensitive
from .retrieve import load_manifest, retrieve_manifest_artifacts

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "EXPIRED"}
_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_DIGEST_IMAGE = re.compile(r"^.+@sha256:([0-9a-fA-F]{64})$")
_EXPERIMENT_MODULE = re.compile(
    r"^experiments(?:\.[A-Za-z_][A-Za-z0-9_]*)+\.experiment$"
)
_SAFE_OUTPUT_FIELDS = {
    "status",
    "run_name",
    "experiment_sha",
    "library_sha",
    "image_digest",
    "elapsed_seconds",
    "endpoint_id",
    "job_id",
    "gpu_name",
    "cuda_version",
    "ray_version",
    "torch_version",
    "gymnasium_version",
    "training_iteration",
    "artifact_file_count",
    "checkpoint_keys",
    "remote_manifest_key",
    "serverless_result_key",
    "mlflow_prefix",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _require_sha(value: str, label: str) -> str:
    if not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be an explicit 40- or 64-hex commit SHA")
    return value.lower()


def resolve_image(image: str | None) -> tuple[str, str]:
    if not image:
        raise ValueError(
            "--image is required until the Serverless worker image is "
            "published; pass registry/repository@sha256:..."
        )
    match = _DIGEST_IMAGE.fullmatch(image)
    if not match:
        raise ValueError("--image must be immutable and digest-pinned")
    return image, f"sha256:{match.group(1).lower()}"


def parse_run_command(run_cmd: str, run_name: str) -> list[str]:
    """Validate the only supported secret-free experiment command."""
    try:
        argv = shlex.split(run_cmd)
    except ValueError as error:
        raise ValueError(f"--run is not valid shell-style quoting: {error}") from None
    if len(argv) < 2 or argv[0] != "rl-harness":
        raise ValueError("--run must be an rl-harness invocation")
    if not _EXPERIMENT_MODULE.fullmatch(argv[1]):
        raise ValueError(
            "--run must invoke an experiments.*.experiment module"
        )
    if argv.count("--upload-artifacts") != 1:
        raise ValueError("--run must include --upload-artifacts exactly once")
    run_ids: list[str] = []
    for index, value in enumerate(argv):
        if value == "--run-id":
            if index + 1 >= len(argv):
                raise ValueError("--run-id requires a value")
            run_ids.append(argv[index + 1])
        elif value.startswith("--run-id="):
            run_ids.append(value.partition("=")[2])
    if run_ids != [run_name]:
        raise ValueError("--run must contain exactly one --run-id equal to --run-name")
    return argv


def validate_endpoint_response(
    endpoint: object,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Require the create response to prove every requested safety field."""
    if not isinstance(endpoint, dict) or not endpoint.get("id"):
        raise ValueError("endpoint create response omitted id")
    checks = (
        ("image", endpoint.get("image"), request["image"]),
        ("gpu.pools", (endpoint.get("gpu") or {}).get("pools"), request["gpu"]["pools"]),
        ("gpu.count", (endpoint.get("gpu") or {}).get("count"), request["gpu"]["count"]),
        (
            "workers.min",
            (endpoint.get("workers") or {}).get("min"),
            request["workers"]["min"],
        ),
        (
            "workers.max",
            (endpoint.get("workers") or {}).get("max"),
            request["workers"]["max"],
        ),
        (
            "scaling.idleTimeout",
            (endpoint.get("scaling") or {}).get("idleTimeout"),
            request["scaling"]["idleTimeout"],
        ),
        ("timeout", endpoint.get("timeout"), request["timeout"]),
        ("flashboot", endpoint.get("flashboot"), request["flashboot"]),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"endpoint create response did not prove requested {label}"
            )
    return endpoint


def validate_cuda_policy_response(
    endpoint: object,
    required_version: str,
) -> dict[str, Any]:
    """Require the compatibility response to prove exact CUDA placement."""
    if not isinstance(endpoint, dict):
        raise ValueError("CUDA policy update response omitted endpoint metadata")
    if endpoint.get("minCudaVersion") != required_version:
        raise ValueError(
            "CUDA policy update response did not prove requested minCudaVersion"
        )
    if endpoint.get("allowedCudaVersions") != [required_version]:
        raise ValueError(
            "CUDA policy update response did not prove exact allowedCudaVersions"
        )
    return endpoint


def sanitize_terminal_output(output: object) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    safe = {
        key: value
        for key, value in output.items()
        if key in _SAFE_OUTPUT_FIELDS
        and (
            value is None
            or isinstance(value, (str, bool, int, float))
            or (
                key == "checkpoint_keys"
                and isinstance(value, list)
                and all(isinstance(item, str) for item in value)
            )
        )
    }
    return redact_metadata(safe)  # type: ignore[return-value]


def terminal_output_proves_success(
    output: object,
    entry: dict[str, Any] | None = None,
) -> bool:
    safe = sanitize_terminal_output(output)
    required_strings = (
        "experiment_sha",
        "library_sha",
        "image_digest",
        "endpoint_id",
        "job_id",
        "remote_manifest_key",
        "serverless_result_key",
    )
    valid = bool(
        safe.get("status") == "completed"
        and all(isinstance(safe.get(key), str) and safe[key] for key in required_strings)
        and float(safe.get("training_iteration") or 0) > 0
        and int(safe.get("artifact_file_count") or 0) > 0
        and isinstance(safe.get("checkpoint_keys"), list)
        and bool(safe["checkpoint_keys"])
    )
    if not valid or entry is None:
        return valid
    return all(
        safe.get(output_key) == entry.get(entry_key)
        for output_key, entry_key in (
            ("experiment_sha", "experiment_ref"),
            ("library_sha", "library_ref"),
            ("image_digest", "image_digest"),
            ("endpoint_id", "endpoint_id"),
            ("job_id", "job_id"),
            ("run_name", "run_name"),
        )
    )


def resolve_github_token() -> str | None:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def load_state(cfg: ServerlessConfig = CONFIG) -> dict[str, Any]:
    if cfg.STATE_PATH.exists():
        try:
            payload = json.loads(cfg.STATE_PATH.read_text())
            if isinstance(payload, dict):
                payload.setdefault("runs", [])
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    return {"runs": []}


def save_state(state: dict[str, Any], cfg: ServerlessConfig = CONFIG) -> None:
    cfg.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _record(state: dict[str, Any], entry: dict[str, Any]) -> None:
    endpoint_id = entry["endpoint_id"]
    state["runs"] = [
        row
        for row in state.get("runs", [])
        if row.get("endpoint_id") != endpoint_id
    ]
    state["runs"].append(entry)


def _unrecord(state: dict[str, Any], endpoint_id: str) -> None:
    state["runs"] = [
        row
        for row in state.get("runs", [])
        if row.get("endpoint_id") != endpoint_id
    ]


def _cost_history(cfg: ServerlessConfig) -> list[dict[str, Any]]:
    if cfg.COST_HISTORY_PATH.exists():
        try:
            value = json.loads(cfg.COST_HISTORY_PATH.read_text())
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            pass
    return []


def record_cost(
    entry: dict[str, Any],
    cost: dict[str, Any],
    *,
    settled: bool,
    cfg: ServerlessConfig = CONFIG,
) -> None:
    history = _cost_history(cfg)
    revision = (
        sum(
            1
            for row in history
            if row.get("endpoint_id") == entry["endpoint_id"]
        )
        + 1
    )
    payload = {
        "endpoint_id": entry["endpoint_id"],
        "job_id": entry.get("job_id"),
        "name": entry.get("name"),
        "run_name": entry.get("run_name"),
        "experiment_ref": entry.get("experiment_ref"),
        "library_ref": entry.get("library_ref"),
        "image_digest": entry.get("image_digest"),
        "created_at": entry.get("created_at_iso"),
        "deleted_at": entry.get("deleted_at_iso"),
        "revision": revision,
        "settled": settled,
        **cost,
    }
    history.append(payload)
    cfg.COST_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.COST_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )


def build_endpoint_env(
    *,
    github_token: str,
    forward_b2: bool,
) -> dict[str, str]:
    """Build endpoint env; all worker secrets live here, never in job input."""
    env = {"GH_TOKEN": github_token}
    if forward_b2:
        b2 = b2_env_for_remote()
        required = {
            "B2_BUCKET",
            "B2_ENDPOINT",
            "B2_APPLICATION_KEY_ID",
            "B2_APPLICATION_KEY",
        }
        if not required.issubset(b2):
            raise ValueError(
                "--forward-b2 requires B2_BUCKET, B2_ENDPOINT, "
                "B2_APPLICATION_KEY_ID, and B2_APPLICATION_KEY"
            )
        allowed = (*required, "B2_PREFIX")
        env.update({key: b2[key] for key in allowed if b2.get(key)})
    return env


def build_create_request(
    cfg: ServerlessConfig,
    *,
    name: str,
    image: str,
    env: dict[str, str],
    execution_timeout_ms: int,
    disk_gb: int | None = None,
    gpu_pools: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    pools = gpu_pools or cfg.GPU_POOLS
    if not pools:
        raise ValueError("at least one Serverless GPU pool is required")
    if execution_timeout_ms < cfg.MIN_EXECUTION_SECONDS * 1000:
        raise ValueError("execution timeout is below RunPod's minimum")
    return {
        "name": name,
        "image": image,
        "disk": int(disk_gb or cfg.DISK_GB),
        "env": env,
        "gpu": {"pools": list(pools), "count": cfg.GPU_COUNT},
        "workers": {"min": cfg.WORKERS_MIN, "max": cfg.WORKERS_MAX},
        "scaling": {
            "type": cfg.SCALER_TYPE,
            "value": cfg.SCALER_VALUE,
            "idleTimeout": cfg.IDLE_TIMEOUT_S,
        },
        "timeout": execution_timeout_ms,
        "flashboot": cfg.FLASHBOOT,
    }


def build_job_request(
    cfg: ServerlessConfig,
    *,
    run_argv: list[str],
    run_name: str,
    experiment_ref: str,
    library_ref: str,
    image_digest: str,
    execution_timeout_ms: int,
    ttl_ms: int,
    push_results: bool,
    results_branch: str,
) -> dict[str, Any]:
    if ttl_ms <= execution_timeout_ms or ttl_ms > cfg.MAX_JOB_HOURS * 3_600_000:
        raise ValueError("job TTL must exceed execution timeout and be <= 7 days")
    job_input = {
        "run_argv": run_argv,
        "run_name": run_name,
        "experiment_repo_url": cfg.EXPERIMENT_REPO_URL,
        "experiment_ref": experiment_ref,
        "library_repo_url": cfg.LIBRARY_REPO_URL,
        "library_ref": library_ref,
        "image_digest": image_digest,
        "ray_version": cfg.RAY_VERSION,
        "torch_version": cfg.TORCH_VERSION,
        "gymnasium_version": cfg.GYMNASIUM_VERSION,
        "push_results": push_results,
        "results_branch": results_branch,
    }
    return {
        "input": job_input,
        "policy": {
            "executionTimeout": execution_timeout_ms,
            "ttl": ttl_ms,
            "lowPriority": False,
        },
    }


def estimate_spend(
    cfg: ServerlessConfig,
    *,
    ttl_seconds: float,
    disk_gb: int,
) -> dict[str, Any]:
    """Assume one worker can be billed continuously over the full TTL."""
    billed_seconds = ttl_seconds + cfg.IDLE_TIMEOUT_S
    gpu = billed_seconds * cfg.GPU_RATE_PER_SECOND
    disk_hourly = disk_gb * cfg.DISK_RATE_PER_GB_MONTH / (730 * 24)
    disk = disk_hourly * billed_seconds / 3600
    subtotal = gpu + disk
    fee_reserve = subtotal * cfg.ESTIMATED_FEE_RESERVE_FRACTION
    return {
        "gpu": gpu,
        "disk": disk,
        "fee_reserve": fee_reserve,
        "total": subtotal + fee_reserve,
        "gpu_hourly": cfg.GPU_RATE_PER_SECOND * 3600,
        "reserved_seconds": billed_seconds,
        "assumption": (
            "one active worker billed continuously for the full provider TTL "
            "plus idle timeout, covering sequential cold starts and retries"
        ),
    }


def _managed(endpoint: dict[str, Any], cfg: ServerlessConfig) -> bool:
    return str(endpoint.get("name") or "").startswith(cfg.MANAGED_NAME_PREFIX)


def _endpoint_created(endpoint: dict[str, Any]) -> float | None:
    return _parse_time(str(endpoint.get("createdAt") or ""))


def _delete_and_mark(
    client: ServerlessClient,
    entry: dict[str, Any],
    state: dict[str, Any],
    cfg: ServerlessConfig,
    *,
    reason: str,
) -> None:
    client.delete_endpoint(entry["endpoint_id"])
    entry["deleted_at_iso"] = entry.get("deleted_at_iso") or _utc_now()
    entry["cleanup_reason"] = reason
    _record(state, entry)
    save_state(state, cfg)


def cmd_up(args, cfg: ServerlessConfig) -> int:
    api_key = resolve_api_key()
    github_token = resolve_github_token()
    if not api_key:
        print("RUNPOD_API_KEY is required")
        return 2
    if not github_token:
        print("GH_TOKEN is required")
        return 2
    if not args.run or not args.run.strip():
        print("--run is required; idle Serverless endpoints are not permitted")
        return 2
    if not args.run_name:
        print("--run-name is required for deterministic durable artifacts")
        return 2
    if not args.forward_b2:
        print("--forward-b2 is required before any Serverless spend")
        return 2
    if args.max_age <= 0 or args.max_age > cfg.MAX_JOB_HOURS:
        print("--max-age must be positive and no more than 168 hours")
        return 2
    if args.queue_timeout <= 0:
        print("--queue-timeout must be positive")
        return 2
    if args.max_price is None or args.max_estimated_cost is None:
        print("--max-price and --max-estimated-cost are required safety gates")
        return 2
    try:
        experiment_ref = _require_sha(args.experiment_ref, "--experiment-ref")
        library_ref = _require_sha(args.library_ref, "--library-ref")
        image, image_digest = resolve_image(args.image or cfg.IMAGE)
        run_argv = parse_run_command(args.run, args.run_name)
        env = build_endpoint_env(
            github_token=github_token,
            forward_b2=True,
        )
    except ValueError as error:
        print(error)
        return 2

    execution_ms = int(args.max_age * 3_600_000)
    minimum_ttl_hours = (
        args.max_age
        + args.queue_timeout / 60
        + cfg.STARTUP_RESERVE_SECONDS / 3600
    )
    ttl_hours = args.ttl if args.ttl is not None else minimum_ttl_hours
    if ttl_hours < minimum_ttl_hours or ttl_hours > cfg.MAX_JOB_HOURS:
        print(
            "--ttl must cover max-age plus queue-timeout plus the startup "
            "reserve, and be no more than 168 hours"
        )
        return 2
    ttl_ms = int(ttl_hours * 3_600_000)
    disk_gb = int(args.disk or cfg.DISK_GB)
    estimate = estimate_spend(
        cfg,
        ttl_seconds=ttl_ms / 1000,
        disk_gb=disk_gb,
    )
    if estimate["gpu_hourly"] > args.max_price:
        print(
            f"conservative GPU rate ${estimate['gpu_hourly']:.3f}/h exceeds "
            f"--max-price ${args.max_price:.3f}/h"
        )
        return 2
    if estimate["total"] > args.max_estimated_cost:
        print(
            f"conservative estimated spend ${estimate['total']:.2f} exceeds "
            f"--max-estimated-cost ${args.max_estimated_cost:.2f}"
        )
        return 2
    run_name = args.run_name
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")
    if not safe_run_name:
        print("--run-name must contain at least one safe character")
        return 2
    suffix = uuid.uuid4().hex[:8]
    name = f"{cfg.MANAGED_NAME_PREFIX}{safe_run_name}-{suffix}"[:191]
    create_request = build_create_request(
        cfg,
        name=name,
        image=image,
        env=env,
        execution_timeout_ms=execution_ms,
        disk_gb=disk_gb,
    )
    job_request = build_job_request(
        cfg,
        run_argv=run_argv,
        run_name=run_name,
        experiment_ref=experiment_ref,
        library_ref=library_ref,
        image_digest=image_digest,
        execution_timeout_ms=execution_ms,
        ttl_ms=ttl_ms,
        push_results=bool(args.self_destruct),
        results_branch=args.results_branch or cfg.DEFAULT_RESULTS_BRANCH,
    )

    print(f"  experiment ref: {experiment_ref}")
    print(f"  library ref:    {library_ref}")
    print(f"  image:          {image}")
    print(
        f"  policy:         {cfg.GPU_POOLS[0]} x1, workers 0..1, "
        f"idle {cfg.IDLE_TIMEOUT_S}s, execution {args.max_age:g}h, "
        f"TTL {ttl_hours:g}h"
    )
    cuda_update_url = (
        f"{cfg.LEGACY_ENDPOINT_API_BASE.rstrip('/')}"
        "/endpoints/{endpointId}/update"
    )
    cuda_update_body = {
        "allowedCudaVersions": [cfg.REQUIRED_CUDA_VERSION],
        "minCudaVersion": cfg.REQUIRED_CUDA_VERSION,
    }
    print(
        f"  CUDA policy:    POST {cuda_update_url} "
        f"{json.dumps(cuda_update_body, separators=(',', ':'))}; "
        "response must prove the exact policy before job submission"
    )
    print(
        "  CONSERVATIVE ESTIMATED SPEND CEILING: "
        f"${estimate['total']:.2f} "
        f"(GPU ${estimate['gpu']:.2f} + disk ${estimate['disk']:.4f} + "
        f"fee reserve ${estimate['fee_reserve']:.2f}); this is an estimate, "
        "not a provider-enforced hard dollar cap"
    )
    print(f"  estimate assumes: {estimate['assumption']}")
    if args.dry_run:
        print("--dry-run: requests validated; no endpoint or job created.")
        return 0
    if not args.yes:
        answer = input("Create one billing endpoint and submit one job? [y/N] ")
        if answer.lower() not in {"y", "yes"}:
            print("aborted.")
            return 1

    client = ServerlessClient(cfg, api_key=api_key)
    state = load_state(cfg)
    entry: dict[str, Any] | None = None
    endpoint_id: str | None = None
    job_id: str | None = None
    return_code = 1
    cleanup_reason = "launch failure"
    current_status = "NOT_SUBMITTED"
    try:
        endpoint = client.create_endpoint(create_request)
        endpoint_id = str(endpoint["id"])
        created_at = time.time()
        entry = {
            "endpoint_id": endpoint_id,
            "name": name,
            "run_name": run_name,
            "experiment_ref": experiment_ref,
            "library_ref": library_ref,
            "image": image,
            "image_digest": image_digest,
            "created_at": created_at,
            "created_at_iso": _utc_now(),
            "max_age_s": args.max_age * 3600,
            "queue_timeout_s": args.queue_timeout * 60,
            "ttl_ms": ttl_ms,
            "estimated_cost": estimate,
        }
        _record(state, entry)
        save_state(state, cfg)
        validate_endpoint_response(endpoint, create_request)
        entry["endpoint_policy_verified"] = True
        _record(state, entry)
        save_state(state, cfg)
        cuda_endpoint = client.update_endpoint_cuda_policy(endpoint_id)
        validate_cuda_policy_response(
            cuda_endpoint,
            cfg.REQUIRED_CUDA_VERSION,
        )
        entry["cuda_policy_verified"] = True
        entry["required_cuda_version"] = cfg.REQUIRED_CUDA_VERSION
        entry["cuda_policy_verified_at_iso"] = _utc_now()
        _record(state, entry)
        save_state(state, cfg)
        job = client.run_job(endpoint_id, job_request)
        job_id = str(job["id"])
        submitted_at = time.time()
        current_status = str(job.get("status") or "IN_QUEUE").upper()
        entry["job_id"] = job_id
        entry["job_status"] = current_status
        entry["submitted_at"] = submitted_at
        entry["submitted_at_iso"] = _utc_now()
        entry["queue_deadline"] = submitted_at + args.queue_timeout * 60
        # Reaping bounds endpoint wall lifetime from creation, while the
        # provider TTL independently bounds the job from submission.
        entry["lifecycle_deadline"] = created_at + ttl_ms / 1000
        entry["lifecycle_deadline_iso"] = datetime.fromtimestamp(
            entry["lifecycle_deadline"], tz=timezone.utc
        ).isoformat()
        _record(state, entry)
        save_state(state, cfg)
        print(
            f"created endpoint {endpoint_id}; submitted exactly one async job "
            f"{job_id}; monitoring until terminal"
        )
        last_status: str | None = None
        while True:
            now = time.time()
            observed = client.job_status(endpoint_id, job_id)
            if observed is None:
                current_status = "EXPIRED"
                terminal_output: dict[str, Any] = {}
            else:
                current_status = str(
                    observed.get("status") or "UNKNOWN"
                ).upper()
                terminal_output = sanitize_terminal_output(
                    observed.get("output")
                )
            if current_status != last_status:
                print(f"  job {job_id}: status={current_status}")
                last_status = current_status
            entry["job_status"] = current_status
            entry["last_polled_at_iso"] = _utc_now()
            _record(state, entry)
            save_state(state, cfg)
            if current_status in _TERMINAL:
                entry["terminal_output"] = terminal_output
                entry["terminal_at_iso"] = _utc_now()
                cleanup_reason = f"terminal job {current_status}"
                return_code = (
                    0
                    if current_status == "COMPLETED"
                    and terminal_output_proves_success(terminal_output, entry)
                    else 1
                )
                if current_status == "COMPLETED" and return_code:
                    print(
                        "job reported COMPLETED but durable success output "
                        "did not pass validation"
                    )
                break
            queue_timed_out = (
                current_status == "IN_QUEUE"
                and now >= float(entry["queue_deadline"])
            )
            lifecycle_timed_out = now >= float(entry["lifecycle_deadline"])
            if queue_timed_out or lifecycle_timed_out:
                reason = (
                    "queue timeout" if queue_timed_out else "provider TTL deadline"
                )
                cancelled = client.cancel_job(endpoint_id, job_id)
                refreshed = (
                    cancelled
                    if isinstance(cancelled, dict) and cancelled.get("status")
                    else client.job_status(endpoint_id, job_id)
                )
                current_status = str(
                    (refreshed or {}).get("status") or "CANCEL_REQUESTED"
                ).upper()
                entry["job_status"] = current_status
                entry["timeout_reason"] = reason
                entry["terminal_output"] = sanitize_terminal_output(
                    (refreshed or {}).get("output")
                )
                cleanup_reason = reason
                print(f"  {reason}; cancel returned status={current_status}")
                break
            time.sleep(cfg.POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        cleanup_reason = "KeyboardInterrupt"
        return_code = 130
        print("interrupted; cancelling job and deleting endpoint")
    except Exception as error:  # noqa: BLE001
        detail = redact_sensitive(error, (api_key, github_token, *env.values()))
        print(f"RunPod Serverless launch failed: {detail}")
        if entry is not None:
            entry["launch_error_type"] = type(error).__name__
        return_code = 1
    finally:
        if entry is not None and endpoint_id is not None:
            if job_id is not None and current_status not in _TERMINAL:
                try:
                    cancelled = client.cancel_job(endpoint_id, job_id)
                    if isinstance(cancelled, dict):
                        current_status = str(
                            cancelled.get("status") or current_status
                        ).upper()
                    entry["job_status"] = current_status
                    entry["cancel_requested_at_iso"] = _utc_now()
                except Exception as error:  # noqa: BLE001
                    entry["cancel_error_type"] = type(error).__name__
            try:
                _delete_and_mark(
                    client,
                    entry,
                    state,
                    cfg,
                    reason=cleanup_reason,
                )
            except Exception as error:  # noqa: BLE001
                entry["cleanup_failed"] = True
                entry["cleanup_error_type"] = type(error).__name__
                _record(state, entry)
                save_state(state, cfg)
                return_code = 1
            if entry.get("deleted_at_iso"):
                try:
                    _observe_billing(client, entry, state, cfg)
                except Exception as error:  # noqa: BLE001
                    print(
                        "  provisional actual billing pending "
                        f"({type(error).__name__})"
                    )
                save_state(state, cfg)
    return return_code


def _collect_billing(
    client: ServerlessClient,
    entry: dict[str, Any],
) -> dict[str, Any]:
    start = entry.get("created_at_iso") or _utc_now()
    end = entry.get("deleted_at_iso") or _utc_now()
    return client.serverless_billing(
        entry["endpoint_id"],
        start_time=start,
        end_time=end,
    )


def _observe_billing(
    client: ServerlessClient,
    entry: dict[str, Any],
    state: dict[str, Any],
    cfg: ServerlessConfig,
) -> bool:
    """Record a provisional revision and settle only after stable delayed polls."""
    cost = _collect_billing(client, entry)
    if not cost["record_count"]:
        print("  actual billing not posted yet; retaining state")
        return False
    total = round(float(cost["actual_cost_usd"]), 10)
    if entry.get("billing_last_total") == total:
        identical = int(entry.get("billing_identical_polls") or 1) + 1
    else:
        identical = 1
    entry["billing_last_total"] = total
    entry["billing_identical_polls"] = identical
    entry["billing_last_observed_at_iso"] = _utc_now()
    deleted_at = _parse_time(entry.get("deleted_at_iso"))
    delay_elapsed = bool(
        deleted_at is not None
        and time.time() - deleted_at >= cfg.BILLING_SETTLEMENT_DELAY_SECONDS
    )
    settled = delay_elapsed and identical >= 2
    record_cost(entry, cost, settled=settled, cfg=cfg)
    if settled:
        _unrecord(state, entry["endpoint_id"])
        print(
            f"  settled actual billed cost: ${total:.4f} "
            f"after {identical} identical polls"
        )
    else:
        _record(state, entry)
        print(
            f"  provisional actual billed cost: ${total:.4f}; "
            f"identical polls={identical}/2, settlement delay "
            f"elapsed={delay_elapsed}"
        )
    return settled


def cmd_status(args, cfg: ServerlessConfig) -> int:
    client = ServerlessClient(cfg)
    state = load_state(cfg)
    if not state.get("runs"):
        print("No tracked RunPod Serverless jobs.")
        return 0
    for entry in list(state["runs"]):
        endpoint_id = entry["endpoint_id"]
        job_id = entry.get("job_id")
        status = str(entry.get("job_status") or "UNKNOWN").upper()
        cleanup_reason: str | None = None
        if job_id and not entry.get("deleted_at_iso"):
            job = client.job_status(endpoint_id, job_id)
            status = (
                str(job.get("status") or "UNKNOWN").upper()
                if job is not None
                else "EXPIRED"
            )
            submitted = (
                float(entry["submitted_at"])
                if entry.get("submitted_at") is not None
                else _parse_time(entry.get("submitted_at_iso"))
            )
            queue_timeout_s = float(entry.get("queue_timeout_s") or 0)
            queue_timed_out = (
                status == "IN_QUEUE"
                and submitted is not None
                and queue_timeout_s > 0
                and time.time() - submitted > queue_timeout_s
            )
            lifecycle_deadline = (
                float(entry["lifecycle_deadline"])
                if entry.get("lifecycle_deadline") is not None
                else (
                    float(submitted) + float(entry["ttl_ms"]) / 1000
                    if submitted is not None and entry.get("ttl_ms")
                    else None
                )
            )
            lifecycle_timed_out = bool(
                status not in _TERMINAL
                and lifecycle_deadline is not None
                and time.time() >= lifecycle_deadline
            )
            if queue_timed_out or lifecycle_timed_out:
                cancelled = client.cancel_job(endpoint_id, job_id)
                refreshed = (
                    cancelled
                    if isinstance(cancelled, dict) and cancelled.get("status")
                    else client.job_status(endpoint_id, job_id)
                )
                status = str(
                    (refreshed or {}).get("status") or "CANCEL_REQUESTED"
                ).upper()
                cleanup_reason = (
                    "queue timeout" if queue_timed_out else "lifecycle deadline"
                )
                entry["timeout_reason"] = cleanup_reason
                job = refreshed
            entry["job_status"] = status
            if status in _TERMINAL and isinstance(job, dict):
                entry["terminal_output"] = sanitize_terminal_output(
                    job.get("output")
                )
            _record(state, entry)
        if (
            (status in _TERMINAL or cleanup_reason is not None)
            and not entry.get("deleted_at_iso")
        ):
            _delete_and_mark(
                client,
                entry,
                state,
                cfg,
                reason=cleanup_reason or f"terminal job {status}",
            )
        endpoint_state = (
            "DELETED"
            if entry.get("deleted_at_iso")
            else "PRESENT"
            if client.get_endpoint(endpoint_id)
            else "GONE"
        )
        print(
            f"{endpoint_id} job={job_id or '-'} status={status} "
            f"endpoint={endpoint_state}"
        )
        if entry.get("deleted_at_iso"):
            try:
                _observe_billing(client, entry, state, cfg)
            except ServerlessClientError as error:
                print(f"  billing pending: {error}")
                continue
    save_state(state, cfg)
    return 0


def _targets(args, state: dict[str, Any]) -> list[dict[str, Any]]:
    tracked = list(state.get("runs", []))
    if args.all:
        return tracked
    if not args.id:
        return []
    by_id = {str(row["endpoint_id"]): row for row in tracked}
    return [
        by_id.get(str(endpoint_id), {"endpoint_id": str(endpoint_id)})
        for endpoint_id in args.id
    ]


def cmd_destroy(args, cfg: ServerlessConfig) -> int:
    client = ServerlessClient(cfg)
    state = load_state(cfg)
    targets = _targets(args, state)
    if not targets:
        print("Specify --all or --id ENDPOINT_ID [ENDPOINT_ID ...]")
        return 2
    if not args.yes:
        answer = input(f"Cancel and delete {len(targets)} endpoint(s)? [y/N] ")
        if answer.lower() not in {"y", "yes"}:
            print("aborted.")
            return 1
    failures = 0
    for entry in targets:
        endpoint_id = entry["endpoint_id"]
        try:
            if entry.get("job_id"):
                client.cancel_job(endpoint_id, entry["job_id"])
            _delete_and_mark(
                client, entry, state, cfg, reason="manual destroy"
            )
            print(f"deleted {endpoint_id}")
        except ServerlessClientError as error:
            failures += 1
            print(f"failed to delete {endpoint_id}: {error}")
    return 1 if failures else 0


def cmd_reap(args, cfg: ServerlessConfig) -> int:
    client = ServerlessClient(cfg)
    state = load_state(cfg)
    tracked = {
        str(row["endpoint_id"]): row for row in state.get("runs", [])
    }
    now = time.time()
    untracked_lifecycle_s = (
        args.max_age * 3600
        if args.max_age is not None
        else (
            cfg.DEFAULT_MAX_AGE_HOURS * 3600
            + cfg.DEFAULT_QUEUE_TIMEOUT_MINUTES * 60
            + cfg.STARTUP_RESERVE_SECONDS
        )
    )
    targets: list[dict[str, Any]] = []
    for endpoint in client.list_endpoints():
        if not _managed(endpoint, cfg):
            continue
        endpoint_id = str(endpoint.get("id") or "")
        entry = tracked.get(endpoint_id)
        created = (entry or {}).get("created_at") or _endpoint_created(endpoint)
        deadline = (
            float(entry["lifecycle_deadline"])
            if entry and entry.get("lifecycle_deadline") is not None
            else (
                float(created)
                + (
                    float(entry.get("ttl_ms") or 0) / 1000
                    if entry and entry.get("ttl_ms")
                    else untracked_lifecycle_s
                )
                if created
                else None
            )
        )
        if deadline is not None and now > deadline:
            targets.append(
                entry
                or {
                    "endpoint_id": endpoint_id,
                    "name": endpoint.get("name"),
                    "created_at": created,
                    "created_at_iso": (
                        datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                        if created
                        else _utc_now()
                    ),
                    "ttl_ms": int(untracked_lifecycle_s * 1000),
                    "lifecycle_deadline": deadline,
                }
            )
    if not targets:
        print("No managed over-age Serverless endpoints.")
        return 0
    for entry in targets:
        print(f"will reap {entry['endpoint_id']} ({entry.get('name', '?')})")
    if not args.yes:
        answer = input(f"Delete {len(targets)} endpoint(s)? [y/N] ")
        if answer.lower() not in {"y", "yes"}:
            print("aborted.")
            return 1
    for entry in targets:
        if entry.get("job_id"):
            client.cancel_job(entry["endpoint_id"], entry["job_id"])
        _delete_and_mark(client, entry, state, cfg, reason="over-age reap")
        print(f"reaped {entry['endpoint_id']}")
    return 0


def cmd_inspect(args, cfg: ServerlessConfig) -> int:
    client = ServerlessClient(cfg)
    endpoint = client.get_endpoint(args.id)
    if endpoint is None:
        print(f"endpoint {args.id} not found")
        return 1
    payload: dict[str, Any] = {
        "endpoint": endpoint,
        "workers": client.list_workers(args.id),
        "health": client.health(args.id),
    }
    state = load_state(cfg)
    entry = next(
        (
            row
            for row in state.get("runs", [])
            if row.get("endpoint_id") == args.id
        ),
        None,
    )
    if entry and entry.get("job_id"):
        payload["job"] = client.job_status(args.id, entry["job_id"])
    print(json.dumps(redact_metadata(payload), indent=2, sort_keys=True))
    return 0


def cmd_logs(args, cfg: ServerlessConfig) -> int:
    client = ServerlessClient(cfg)
    worker_ids = [args.worker] if args.worker else [
        str(row["id"])
        for row in client.list_workers(args.id).get("workers", [])
        if isinstance(row, dict) and row.get("id")
    ]
    if not worker_ids:
        print("No active workers; worker logs are unavailable after scale-down.")
        return 0
    secrets = (
        resolve_api_key(),
        resolve_github_token(),
        os.environ.get("B2_APPLICATION_KEY_ID"),
        os.environ.get("B2_APPLICATION_KEY"),
    )

    def emit(event: dict[str, str]) -> None:
        print(
            f"{event.get('ts', '')} [{event.get('source', '?')}] "
            f"{redact_sensitive(event.get('line', ''), secrets)}".strip(),
            flush=True,
        )

    for worker_id in worker_ids:
        print(f"== worker {worker_id} ==")
        client.worker_logs(
            args.id,
            worker_id,
            source=args.source,
            tail=args.tail,
            since=args.since,
            follow=args.follow,
            emit=emit,
        )
    return 0


def cmd_retrieve(args, cfg: ServerlessConfig) -> int:
    manifest = load_manifest(
        path=Path(args.manifest).expanduser() if args.manifest else None,
        key=args.manifest_key,
        bucket=args.bucket,
    )
    destination = Path(args.destination).expanduser().resolve()
    files = retrieve_manifest_artifacts(manifest, destination)
    print(f"retrieved and SHA-256 verified {len(files)} file(s) to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devops.serverless.provision",
        description="Run one experiment on one disposable Serverless endpoint.",
    )
    sub = parser.add_subparsers(dest="command")
    up = sub.add_parser(
        "up", help="create, monitor, verify, and clean one disposable job"
    )
    up.add_argument("--run", required=True, metavar="CMD")
    up.add_argument(
        "--experiment-ref", "--commit", dest="experiment_ref", required=True
    )
    up.add_argument(
        "--library-ref", "--library-commit", dest="library_ref", required=True
    )
    up.add_argument("--image")
    up.add_argument("--run-name")
    up.add_argument("--results-branch")
    up.add_argument(
        "--max-age",
        type=float,
        default=CONFIG.DEFAULT_MAX_AGE_HOURS,
        metavar="HOURS",
    )
    up.add_argument(
        "--queue-timeout",
        type=float,
        default=CONFIG.DEFAULT_QUEUE_TIMEOUT_MINUTES,
        metavar="MINUTES",
    )
    up.add_argument("--ttl", type=float, metavar="HOURS")
    up.add_argument("--max-price", type=float, required=True, metavar="USD_PER_H")
    up.add_argument(
        "--max-estimated-cost", type=float, required=True, metavar="USD"
    )
    up.add_argument("--disk", type=int)
    up.add_argument("--forward-b2", action="store_true")
    up.add_argument(
        "--self-destruct",
        action="store_true",
        help="push compact results; endpoint cleanup is always external",
    )
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--yes", action="store_true")

    status = sub.add_parser(
        "status", help="recover interrupted jobs and settle delayed billing"
    )
    status.set_defaults(command="status")
    inspect = sub.add_parser("inspect", help="show redacted endpoint/job metadata")
    inspect.add_argument("id")
    destroy = sub.add_parser("destroy", help="cancel jobs and delete endpoints")
    destroy.add_argument("--id", nargs="+")
    destroy.add_argument("--all", action="store_true")
    destroy.add_argument("--yes", action="store_true")
    reap = sub.add_parser("reap", help="delete managed over-lifecycle endpoints")
    reap.add_argument(
        "--max-age",
        type=float,
        metavar="HOURS",
        help="untracked endpoint lifecycle override only",
    )
    reap.add_argument("--yes", action="store_true")
    logs = sub.add_parser("logs", help="stream active worker logs")
    logs.add_argument("id", help="endpoint ID")
    logs.add_argument("--worker")
    logs.add_argument("--source", choices=["container", "system"])
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--since")
    logs.add_argument("--follow", action="store_true")
    retrieve = sub.add_parser("retrieve", help="download and verify B2 artifacts")
    source = retrieve.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--manifest-key")
    retrieve.add_argument("--bucket")
    retrieve.add_argument("--destination", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {
        "up",
        "status",
        "inspect",
        "destroy",
        "reap",
        "logs",
        "retrieve",
        "-h",
        "--help",
    }
    if not argv or argv[0] not in known:
        argv = ["up", *argv]
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return cmd_status(args, CONFIG)
        if args.command == "inspect":
            return cmd_inspect(args, CONFIG)
        if args.command == "destroy":
            return cmd_destroy(args, CONFIG)
        if args.command == "reap":
            return cmd_reap(args, CONFIG)
        if args.command == "logs":
            return cmd_logs(args, CONFIG)
        if args.command == "retrieve":
            return cmd_retrieve(args, CONFIG)
        return cmd_up(args, CONFIG)
    except (ServerlessClientError, ValueError, OSError, json.JSONDecodeError) as error:
        print(
            redact_sensitive(
                error,
                (
                    resolve_api_key(),
                    resolve_github_token(),
                    os.environ.get("B2_APPLICATION_KEY_ID"),
                    os.environ.get("B2_APPLICATION_KEY"),
                ),
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
