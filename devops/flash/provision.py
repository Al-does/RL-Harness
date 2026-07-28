"""Dry-run-first deployment and job launcher for RunPod Flash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devops.runpod.execution.preflight import (
    PreflightError,
    build_resource_contract_for_run,
    require_sha,
    verify_remote_sha_fetchable,
)
from devops.serverless.client import (
    ServerlessClient,
    ServerlessClientError,
    resolve_api_key,
)
from devops.serverless.provision import (
    build_job_request,
    parse_run_command,
    provider_failure_summary,
    resolve_github_token,
    sanitize_terminal_output,
    terminal_output_proves_success,
)
from devops.serverless.redaction import redact_sensitive
from devops.serverless.retrieve import load_manifest, retrieve_manifest_artifacts

from .config import CONFIG, FlashConfig

_ROOT = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).resolve().with_name("worker.py")
_HANDLER = _ROOT / "devops" / "serverless" / "handler.py"
_STAGED_HANDLER_NAME = "rlh_experiment_handler.py"
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "EXPIRED"}
_FINGERPRINT = re.compile(r"^[0-9a-fA-F]{64}$")
_PHASE = re.compile(r"\bphase=([A-Z_]+)\b")
_EXPERIMENT_MODULE = re.compile(
    r"^experiments(?:\.[A-Za-z_][A-Za-z0-9_]*)+\.experiment$"
)


@dataclass
class MonitorResult:
    status: str
    observed: dict[str, Any] | None
    output: dict[str, Any]
    timed_out: bool = False
    stalled: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deadline_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _remaining_seconds(value: object, fallback: float) -> float:
    if isinstance(value, str):
        try:
            deadline = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).timestamp()
            return max(1.0, deadline - time.time())
        except ValueError:
            pass
    return fallback


def ensure_one_gpu_hardware(run_argv: list[str]) -> list[str]:
    """Inject the generic one-GPU layout unless the operator chose a profile."""
    if any(
        part in {"--hardware", "--hardware-profile"}
        or part.startswith("--hardware=")
        or part.startswith("--hardware-profile=")
        for part in run_argv
    ):
        return list(run_argv)
    return [*run_argv, "--hardware", "cuda4090"]


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def _launch_app_name(module: str) -> str:
    study = module.split(".")[1]
    slug = re.sub(r"[^a-z0-9-]+", "-", study.lower().replace("_", "-")).strip("-")
    return f"rlh-flash-{slug}"[:63].rstrip("-")


def load_state(cfg: FlashConfig = CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(cfg.STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"jobs": []}
    if not isinstance(value, dict):
        return {"jobs": []}
    value.setdefault("jobs", [])
    return value


def save_state(state: dict[str, Any], cfg: FlashConfig = CONFIG) -> None:
    cfg.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _record_job(
    state: dict[str, Any],
    entry: dict[str, Any],
    cfg: FlashConfig = CONFIG,
) -> None:
    provider_job_id = entry["provider_job_id"]
    state["jobs"] = [
        row
        for row in state.get("jobs", [])
        if row.get("provider_job_id") != provider_job_id
    ]
    state["jobs"].append(entry)
    save_state(state, cfg)


def _flash_executable() -> str:
    candidate = Path(sys.executable).with_name("flash")
    executable = shutil.which("flash") or (
        str(candidate) if candidate.exists() else None
    )
    if not executable:
        raise RuntimeError("runpod-flash is missing; run uv sync --group flash")
    return executable


def _run_flash(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_flash_executable(), *args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def _stage_source(destination: Path) -> str:
    worker = _WORKER.read_bytes()
    handler = _HANDLER.read_bytes()
    (destination / "worker.py").write_bytes(worker)
    (destination / _STAGED_HANDLER_NAME).write_bytes(handler)
    digest = hashlib.sha256()
    digest.update(worker)
    digest.update(b"\0")
    digest.update(handler)
    return digest.hexdigest()


def _ensure_flash_app(app: str, environment: str, cwd: Path) -> None:
    if _run_flash(["app", "get", app], cwd=cwd, check=False).returncode:
        _run_flash(["app", "create", app], cwd=cwd)
    if _run_flash(
        ["env", "get", environment, "--app", app],
        cwd=cwd,
        check=False,
    ).returncode:
        _run_flash(["env", "create", environment, "--app", app], cwd=cwd)


def _validate_endpoint(
    endpoint: object,
    *,
    app: str,
    max_workers: int,
    cfg: FlashConfig,
    source_digest: str | None = None,
) -> dict[str, Any]:
    if not isinstance(endpoint, dict) or not endpoint.get("id"):
        raise ValueError("Flash deployment did not produce an endpoint")
    checks = (
        ("name", endpoint.get("name"), app),
        ("workers.min", (endpoint.get("workers") or {}).get("min"), 0),
        (
            "workers.max",
            (endpoint.get("workers") or {}).get("max"),
            max_workers,
        ),
        (
            "scaling.idleTimeout",
            (endpoint.get("scaling") or {}).get("idleTimeout"),
            cfg.IDLE_TIMEOUT_S,
        ),
        ("flashboot", endpoint.get("flashboot"), "FLASHBOOT"),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"Flash endpoint did not prove {label}={expected!r}; got {actual!r}"
            )
    image = str(endpoint.get("image") or "")
    if not image.startswith("runpod/flash:"):
        raise ValueError("Flash endpoint is not using the provider-managed runtime")
    if source_digest is not None:
        env = endpoint.get("env") if isinstance(endpoint.get("env"), dict) else {}
        if env.get("RL_HARNESS_SOURCE_SHA256") != source_digest:
            raise ValueError(
                "Flash endpoint did not activate the staged source revision"
            )
    return endpoint


def _find_endpoint(
    client: ServerlessClient,
    *,
    app: str,
    max_workers: int,
    cfg: FlashConfig,
    source_digest: str | None = None,
) -> dict[str, Any]:
    matches = [
        endpoint
        for endpoint in client.list_endpoints()
        if endpoint.get("name") == app
    ]
    if not matches:
        raise ValueError(
            f"expected an endpoint named {app!r}, found none"
        )
    if source_digest is not None:
        matching_source = [
            row
            for row in matches
            if isinstance(row.get("env"), dict)
            and row["env"].get("RL_HARNESS_SOURCE_SHA256") == source_digest
        ]
        if not matching_source:
            raise ValueError(
                "Flash deployment has not activated the staged source revision"
            )
        matches.sort(key=lambda row: str(row.get("createdAt") or ""))
        selected = max(
            matching_source,
            key=lambda row: str(row.get("createdAt") or ""),
        )
        matches.remove(selected)
        matches.append(selected)
    else:
        matches.sort(key=lambda row: str(row.get("createdAt") or ""))
    newest_id = str(matches[-1].get("id") or "")
    newest = _validate_endpoint(
        client.get_endpoint(newest_id),
        app=app,
        max_workers=max_workers,
        cfg=cfg,
        source_digest=source_digest,
    )
    # Flash deploy currently creates a replacement endpoint rather than updating
    # in place. Remove only idle superseded endpoints after the replacement has
    # passed every policy check, otherwise repeated deploys leak billable workers.
    for stale in matches[:-1]:
        stale_id = str(stale.get("id") or "")
        workers = client.list_workers(stale_id)
        summary = workers.get("summary") if isinstance(workers, dict) else {}
        if int((summary or {}).get("running") or 0) > 0:
            raise ValueError(
                f"superseded endpoint {stale_id} still has running workers; "
                "refusing automatic deletion"
            )
        client.delete_endpoint(stale_id)
        print(f"deleted idle superseded endpoint {stale_id}")
    return newest


def estimate_spend(
    cfg: FlashConfig,
    *,
    execution_seconds: float,
) -> dict[str, float]:
    billed_seconds = execution_seconds + cfg.IDLE_TIMEOUT_S
    subtotal = billed_seconds * cfg.GPU_RATE_PER_SECOND
    return {
        "gpu_hourly": cfg.GPU_RATE_PER_SECOND * 3600,
        "reserved_seconds": billed_seconds,
        "gpu": subtotal,
        "fee_reserve": subtotal * cfg.ESTIMATED_FEE_RESERVE_FRACTION,
        "total": subtotal * (1 + cfg.ESTIMATED_FEE_RESERVE_FRACTION),
    }


def _require_worker_secrets(endpoint: dict[str, Any]) -> None:
    env = endpoint.get("env") if isinstance(endpoint.get("env"), dict) else {}
    required = {
        "GH_TOKEN",
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
    }
    missing = sorted(required - set(env))
    if missing:
        raise ValueError(
            "Flash endpoint is missing worker credentials; redeploy with: "
            + ", ".join(missing)
        )


def cmd_deploy(args: argparse.Namespace, cfg: FlashConfig) -> int:
    if not resolve_api_key():
        raise ValueError("RUNPOD_API_KEY is required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", args.app):
        raise ValueError("--app must be 3-63 lowercase letters, digits, or hyphens")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", args.environment):
        raise ValueError("--environment contains unsafe characters")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least one")
    if not resolve_github_token():
        raise ValueError("GH_TOKEN is required")
    for name in (
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
    ):
        if not os.environ.get(name):
            raise ValueError(f"{name} is required")

    with tempfile.TemporaryDirectory(prefix="rlh-flash-") as raw:
        stage = Path(raw)
        source_digest = _stage_source(stage)
        print(
            f"Flash deploy plan: app={args.app} env={args.environment} "
            f"workers=0..{args.max_workers} idle={cfg.IDLE_TIMEOUT_S}s "
            f"python={cfg.PYTHON_VERSION} source_sha256={source_digest}"
        )
        print("  delivery: provider-managed runpod/flash runtime; no user image")
        if args.dry_run:
            print("--dry-run: no app, environment, endpoint, or job changed.")
            return 0
        if not args.yes:
            raise ValueError("--yes is required for a live Flash deployment")
        _ensure_flash_app(args.app, args.environment, stage)
        env = dict(os.environ)
        env["RL_HARNESS_FLASH_ENDPOINT"] = args.app
        env["RL_HARNESS_FLASH_MAX_WORKERS"] = str(args.max_workers)
        env["RL_HARNESS_SOURCE_SHA256"] = source_digest
        client = ServerlessClient()
        endpoint = None
        for attempt in range(2):
            try:
                completed = subprocess.run(
                    [
                        _flash_executable(),
                        "deploy",
                        "--app",
                        args.app,
                        "--env",
                        args.environment,
                        "--python-version",
                        cfg.PYTHON_VERSION,
                    ],
                    cwd=stage,
                    env=env,
                    check=True,
                    text=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()[-2000:]
                raise RuntimeError(f"Flash deploy command failed: {detail}") from None
            print(completed.stdout.strip())
            try:
                endpoint = _find_endpoint(
                    client,
                    app=args.app,
                    max_workers=args.max_workers,
                    cfg=cfg,
                    source_digest=source_digest,
                )
                break
            except ValueError as error:
                if attempt or "source revision" not in str(error):
                    raise
                print(
                    "Flash returned the previous artifact revision; "
                    "redeploying once to activate the uploaded source"
                )
        if endpoint is None:
            raise ValueError("Flash did not activate the staged source revision")
    _require_worker_secrets(endpoint)
    print(
        f"verified endpoint {endpoint['id']}: workers=0..{args.max_workers}, "
        f"idle={cfg.IDLE_TIMEOUT_S}s, image={endpoint['image']}"
    )
    return 0


def _runtime_digest(endpoint: dict[str, Any]) -> str:
    env = endpoint.get("env") if isinstance(endpoint.get("env"), dict) else {}
    fingerprint = str(env.get("_FLASH_SOURCE_FINGERPRINT") or "")
    if not _FINGERPRINT.fullmatch(fingerprint):
        raise ValueError("Flash endpoint omitted a valid source fingerprint")
    return f"sha256:{fingerprint.lower()}"


def _monitor_secrets() -> tuple[str | None, ...]:
    return (
        resolve_api_key(),
        resolve_github_token(),
        os.environ.get("B2_APPLICATION_KEY_ID"),
        os.environ.get("B2_APPLICATION_KEY"),
    )


def _worker_progress(
    client: ServerlessClient,
    endpoint_id: str,
    *,
    preferred_worker_id: str | None,
    seen: set[tuple[str, str, str, str]],
    current_phase: str,
) -> tuple[dict[str, Any], int, str]:
    payload = client.list_workers(endpoint_id)
    rows = payload.get("workers", []) if isinstance(payload, dict) else []
    rows = [row for row in rows if isinstance(row, dict) and row.get("id")]
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    worker_ids: list[str] = []
    if preferred_worker_id:
        worker_ids.append(preferred_worker_id)
    worker_ids.extend(
        str(row["id"])
        for row in rows
        if str(row.get("status") or "").upper()
        in {"RUNNING", "INITIALIZING", "IDLE"}
    )
    worker_ids = list(dict.fromkeys(worker_ids))[:4]
    new_events = 0
    phase = current_phase
    secrets = _monitor_secrets()
    for worker_id in worker_ids:
        events = client.worker_logs(
            endpoint_id,
            worker_id,
            tail=200,
            idle_timeout_s=0.5,
        )
        for event in events:
            key = (
                worker_id,
                str(event.get("ts") or ""),
                str(event.get("source") or ""),
                str(event.get("line") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            new_events += 1
            line = redact_sensitive(event.get("line", ""), secrets)
            match = _PHASE.search(line)
            if match:
                phase = match.group(1)
            print(
                f"  worker={worker_id} {event.get('ts', '')} "
                f"[{event.get('source', '?')}] {line}".strip(),
                flush=True,
            )
    return (summary if isinstance(summary, dict) else {}), new_events, phase


def monitor_job(
    client: ServerlessClient,
    *,
    endpoint_id: str,
    provider_job_id: str,
    queue_timeout_s: float,
    lifecycle_timeout_s: float,
    progress_interval_s: float,
    no_progress_timeout_s: float,
    entry: dict[str, Any] | None = None,
    cfg: FlashConfig = CONFIG,
) -> MonitorResult:
    """Monitor provider status and worker evidence until terminal or stalled."""
    if min(
        queue_timeout_s,
        lifecycle_timeout_s,
        progress_interval_s,
        no_progress_timeout_s,
    ) <= 0:
        raise ValueError("monitoring timeouts and intervals must be positive")
    started = time.monotonic()
    queue_deadline = started + queue_timeout_s
    lifecycle_deadline = started + lifecycle_timeout_s
    next_progress = started
    last_progress = started
    last_status = ""
    current_phase = "PROVISIONING"
    seen: set[tuple[str, str, str, str]] = set()
    state = load_state(cfg) if entry is not None else None
    observed: dict[str, Any] | None = None
    while True:
        now = time.monotonic()
        observed = client.job_status(endpoint_id, provider_job_id)
        status = str((observed or {}).get("status") or "EXPIRED").upper()
        preferred_worker_id = (
            str((observed or {}).get("workerId") or "") or None
        )
        if status != last_status:
            print(f"  job {provider_job_id}: {status}", flush=True)
            if status == "IN_PROGRESS":
                last_progress = now
            last_status = status
        if status in _TERMINAL:
            output = sanitize_terminal_output((observed or {}).get("output"))
            if entry is not None and state is not None:
                entry.update(
                    {
                        "status": status,
                        "phase": current_phase,
                        "terminal_at_iso": _utc_now(),
                        "terminal_output": output,
                    }
                )
                _record_job(state, entry, cfg)
            return MonitorResult(status=status, observed=observed, output=output)

        if now >= next_progress:
            monitor_error = None
            summary: dict[str, Any] = {}
            try:
                summary, event_count, current_phase = _worker_progress(
                    client,
                    endpoint_id,
                    preferred_worker_id=preferred_worker_id,
                    seen=seen,
                    current_phase=current_phase,
                )
                # On shared parallel endpoints, logs from an unrelated running
                # worker must not keep this job alive. Once RunPod identifies
                # the assigned worker, its fresh logs are attributable.
                if event_count and (
                    preferred_worker_id is not None or status == "IN_QUEUE"
                ):
                    last_progress = now
            except Exception as error:  # noqa: BLE001
                monitor_error = type(error).__name__
            elapsed = now - started
            progress_age = now - last_progress
            print(
                "  heartbeat "
                f"job={provider_job_id} provider={status} "
                f"phase={current_phase} elapsed={elapsed:.0f}s "
                f"last_progress={progress_age:.0f}s "
                f"workers={json.dumps(summary, sort_keys=True)}"
                + (
                    f" monitor_error={monitor_error}"
                    if monitor_error
                    else ""
                ),
                flush=True,
            )
            if entry is not None and state is not None:
                entry.update(
                    {
                        "status": status,
                        "phase": current_phase,
                        "last_heartbeat_at_iso": _utc_now(),
                        "last_progress_age_seconds": round(progress_age, 3),
                        "worker_summary": summary,
                        "monitor_error": monitor_error,
                    }
                )
                _record_job(state, entry, cfg)
            next_progress = now + progress_interval_s

        if status == "IN_PROGRESS" and now - last_progress >= no_progress_timeout_s:
            client.cancel_job(endpoint_id, provider_job_id)
            print(
                f"cancelled stalled Flash job {provider_job_id}: no worker "
                f"progress for {no_progress_timeout_s:.0f}s",
                flush=True,
            )
            if entry is not None and state is not None:
                entry.update(
                    {
                        "status": "CANCEL_REQUESTED",
                        "phase": current_phase,
                        "stalled": True,
                        "terminal_at_iso": _utc_now(),
                    }
                )
                _record_job(state, entry, cfg)
            return MonitorResult(
                status="CANCEL_REQUESTED",
                observed=observed,
                output={},
                stalled=True,
            )
        queue_timed_out = status == "IN_QUEUE" and now >= queue_deadline
        lifecycle_timed_out = now >= lifecycle_deadline
        if queue_timed_out or lifecycle_timed_out:
            client.cancel_job(endpoint_id, provider_job_id)
            reason = "queue timeout" if queue_timed_out else "lifecycle timeout"
            print(f"cancelled Flash job {provider_job_id}: {reason}", flush=True)
            if entry is not None and state is not None:
                entry.update(
                    {
                        "status": "CANCEL_REQUESTED",
                        "phase": current_phase,
                        "timeout_reason": reason,
                        "terminal_at_iso": _utc_now(),
                    }
                )
                _record_job(state, entry, cfg)
            return MonitorResult(
                status="CANCEL_REQUESTED",
                observed=observed,
                output={},
                timed_out=True,
            )
        time.sleep(cfg.POLL_INTERVAL_SECONDS)


def cmd_up(args: argparse.Namespace, cfg: FlashConfig) -> int:
    if not resolve_api_key():
        raise ValueError("RUNPOD_API_KEY is required")
    token = resolve_github_token()
    if not token:
        raise ValueError("GH_TOKEN is required")
    if not args.forward_b2:
        raise ValueError("--forward-b2 is required")
    if args.max_age <= 0 or args.max_age > cfg.MAX_JOB_HOURS:
        raise ValueError("--max-age must be positive and no more than 168 hours")
    if args.queue_timeout <= 0:
        raise ValueError("--queue-timeout must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.no_progress_timeout <= 0:
        raise ValueError("--no-progress-timeout must be positive")

    experiment_ref = require_sha(args.experiment_ref, "--experiment-ref")
    library_ref = require_sha(args.library_ref, "--library-ref")
    run_argv = ensure_one_gpu_hardware(
        parse_run_command(args.run, args.run_name)
    )
    verify_remote_sha_fetchable(
        cfg.EXPERIMENT_REPO_URL,
        experiment_ref,
        label="experiment",
        github_token=token,
    )
    verify_remote_sha_fetchable(
        cfg.LIBRARY_REPO_URL,
        library_ref,
        label="library",
        github_token=token,
    )
    contract = build_resource_contract_for_run(
        run_argv,
        default_profile="cuda4090_gpuinfer",
        available_gpus=1,
    )
    estimate = estimate_spend(cfg, execution_seconds=args.max_age * 3600)
    if estimate["gpu_hourly"] > args.max_price:
        raise ValueError("conservative GPU hourly rate exceeds --max-price")
    if estimate["total"] > args.max_estimated_cost:
        raise ValueError("conservative estimated spend exceeds --max-estimated-cost")

    client = ServerlessClient()
    endpoint = client.get_endpoint(args.endpoint_id)
    if not isinstance(endpoint, dict):
        raise ValueError(f"endpoint {args.endpoint_id!r} was not found")
    workers = endpoint.get("workers") or {}
    max_workers = int(workers.get("max") or 0)
    _validate_endpoint(
        endpoint,
        app=str(endpoint.get("name") or ""),
        max_workers=max_workers,
        cfg=cfg,
    )
    _require_worker_secrets(endpoint)
    runtime_digest = _runtime_digest(endpoint)
    launcher_job_id = f"flash-{uuid.uuid4().hex}"
    execution_ms = int(args.max_age * 3_600_000)
    ttl_seconds = (
        args.max_age * 3600
        + args.queue_timeout * 60
        + cfg.STARTUP_RESERVE_SECONDS
    )
    request = build_job_request(
        cfg,  # type: ignore[arg-type]
        run_argv=run_argv,
        run_name=args.run_name,
        experiment_ref=experiment_ref,
        library_ref=library_ref,
        image_digest=runtime_digest,
        execution_timeout_ms=execution_ms,
        ttl_ms=int(ttl_seconds * 1000),
        push_results=bool(args.self_destruct),
        results_branch=args.results_branch or cfg.DEFAULT_RESULTS_BRANCH,
    )
    spec = request["input"]
    spec["delivery"] = "runpod-flash-artifact"
    spec["launcher_job_id"] = launcher_job_id
    request["input"] = {"input_data": spec}

    print(
        f"Flash job plan: endpoint={args.endpoint_id} workers=0..{max_workers} "
        f"run={args.run_name} estimated_max=${estimate['total']:.2f}"
    )
    print(f"  resource_contract={json.dumps(contract.to_dict(), sort_keys=True)}")
    print(
        f"  experiment_sha={experiment_ref} library_sha={library_ref} "
        f"source={runtime_digest}"
    )
    if args.dry_run:
        print("--dry-run: preflight passed; no job submitted.")
        return 0
    if not args.yes:
        raise ValueError("--yes is required to submit a live Flash job")

    submitted = client.run_job(args.endpoint_id, request)
    provider_job_id = str(submitted["id"])
    print(
        f"submitted Flash job {provider_job_id}; endpoint remains reusable and "
        "scales to zero after completion"
    )
    submitted_epoch = time.time()
    entry = {
        "provider_job_id": provider_job_id,
        "launcher_job_id": launcher_job_id,
        "endpoint_id": args.endpoint_id,
        "run_name": args.run_name,
        "experiment_ref": experiment_ref,
        "library_ref": library_ref,
        "image_digest": runtime_digest,
        "status": str(submitted.get("status") or "IN_QUEUE").upper(),
        "phase": "PROVISIONING",
        "submitted_at_iso": _deadline_iso(submitted_epoch),
        "queue_timeout_seconds": args.queue_timeout * 60,
        "lifecycle_timeout_seconds": ttl_seconds,
        "queue_deadline_iso": _deadline_iso(
            submitted_epoch + args.queue_timeout * 60
        ),
        "lifecycle_deadline_iso": _deadline_iso(
            submitted_epoch + ttl_seconds
        ),
        "progress_interval_seconds": args.progress_interval,
        "no_progress_timeout_seconds": args.no_progress_timeout * 60,
    }
    state = load_state(cfg)
    _record_job(state, entry, cfg)
    result = monitor_job(
        client,
        endpoint_id=args.endpoint_id,
        provider_job_id=provider_job_id,
        queue_timeout_s=args.queue_timeout * 60,
        lifecycle_timeout_s=ttl_seconds,
        progress_interval_s=args.progress_interval,
        no_progress_timeout_s=args.no_progress_timeout * 60,
        entry=entry,
        cfg=cfg,
    )
    output = result.output
    print(json.dumps(output, indent=2, sort_keys=True))
    validation_entry = {
        "experiment_ref": experiment_ref,
        "library_ref": library_ref,
        "image_digest": runtime_digest,
        "endpoint_id": args.endpoint_id,
        "job_id": launcher_job_id,
        "run_name": args.run_name,
    }
    if result.status == "COMPLETED" and terminal_output_proves_success(
        output, validation_entry
    ):
        if not output.get("canonical_manifest_key"):
            raise RuntimeError("Flash output omitted canonical_manifest_key")
        print("workload_success=true; durable Flash training verified")
        return 0
    failure = provider_failure_summary(result.observed, secrets=())
    if failure:
        print(json.dumps(failure, indent=2, sort_keys=True))
    return 1


def _launch_run_argv(args: argparse.Namespace, run_name: str) -> list[str]:
    if not _EXPERIMENT_MODULE.fullmatch(args.experiment):
        raise ValueError(
            "experiment must be an experiments.*.experiment module"
        )
    extra = shlex.split(args.run_args) if args.run_args else []
    forbidden = {
        "--run-id",
        "--upload-artifacts",
        "--smoke",
    }
    if any(
        part in forbidden
        or any(part.startswith(f"{name}=") for name in forbidden)
        for part in extra
    ):
        raise ValueError(
            "--run-args must not override --run-id, --upload-artifacts, or --smoke"
        )
    argv = [
        "rl-harness",
        args.experiment,
        *extra,
        "--seed",
        str(args.seed),
        "--upload-artifacts",
        "--run-id",
        run_name,
    ]
    if args.smoke:
        argv.append("--smoke")
    return ensure_one_gpu_hardware(argv)


def _launch_preflight(
    *,
    cfg: FlashConfig,
    run_argv: list[str],
    experiment_ref: str,
    library_ref: str,
    max_age: float,
    max_price: float,
    max_estimated_cost: float,
) -> None:
    token = resolve_github_token()
    if not token:
        raise ValueError("GH_TOKEN is required")
    verify_remote_sha_fetchable(
        cfg.EXPERIMENT_REPO_URL,
        experiment_ref,
        label="experiment",
        github_token=token,
    )
    verify_remote_sha_fetchable(
        cfg.LIBRARY_REPO_URL,
        library_ref,
        label="library",
        github_token=token,
    )
    contract = build_resource_contract_for_run(
        run_argv,
        default_profile="cuda4090",
        available_gpus=1,
    )
    estimate = estimate_spend(cfg, execution_seconds=max_age * 3600)
    if estimate["gpu_hourly"] > max_price:
        raise ValueError("conservative GPU hourly rate exceeds --max-price")
    if estimate["total"] > max_estimated_cost:
        raise ValueError("conservative estimated spend exceeds --max-estimated-cost")
    print(
        "Flash launch preflight: "
        f"resources={json.dumps(contract.to_dict(), sort_keys=True)} "
        f"estimated_max=${estimate['total']:.2f}",
        flush=True,
    )


def _current_launch_endpoint(
    *,
    cfg: FlashConfig,
    app: str,
    max_workers: int,
) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix="rlh-flash-launch-") as raw:
        source_digest = _stage_source(Path(raw))
    try:
        return _find_endpoint(
            ServerlessClient(),
            app=app,
            max_workers=max_workers,
            cfg=cfg,
            source_digest=source_digest,
        )
    except ValueError:
        return None


def cmd_launch(args: argparse.Namespace, cfg: FlashConfig) -> int:
    """Deploy/reuse Flash and run one experiment from a compact generic CLI."""
    experiment_repo = args.experiment_dir.expanduser().resolve()
    if not experiment_repo.is_dir():
        raise ValueError(f"experiment repository not found: {experiment_repo}")
    experiment_ref = args.experiment_ref or _git_sha(experiment_repo)
    library_ref = args.library_ref or _git_sha(_ROOT)
    condition = args.experiment.removeprefix("experiments.").removesuffix(
        ".experiment"
    )
    run_name = args.run_name or (
        f"{condition.replace('.', '-')}-seed{args.seed}-"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    run_argv = _launch_run_argv(args, run_name)
    max_age = (
        args.max_age
        if args.max_age is not None
        else (
            cfg.DEFAULT_SMOKE_MAX_AGE_HOURS
            if args.smoke
            else cfg.DEFAULT_MAX_AGE_HOURS
        )
    )
    _launch_preflight(
        cfg=cfg,
        run_argv=run_argv,
        experiment_ref=experiment_ref,
        library_ref=library_ref,
        max_age=max_age,
        max_price=args.max_price,
        max_estimated_cost=args.max_estimated_cost,
    )
    app = args.app or _launch_app_name(args.experiment)
    endpoint = (
        ServerlessClient().get_endpoint(args.endpoint_id)
        if args.endpoint_id
        else _current_launch_endpoint(cfg=cfg, app=app, max_workers=args.max_workers)
    )
    if args.dry_run and endpoint is None:
        deploy_args = argparse.Namespace(
            app=app,
            environment=args.environment,
            max_workers=args.max_workers,
            dry_run=True,
            yes=False,
        )
        result = cmd_deploy(deploy_args, cfg)
        if result:
            return result
        print(
            f"Flash launch plan: app={app} run_name={run_name} "
            f"run={shlex.join(run_argv)} max_age={max_age:g}h; "
            "live launch will deploy and monitor automatically.",
            flush=True,
        )
        return 0
    if endpoint is None:
        deploy_args = argparse.Namespace(
            app=app,
            environment=args.environment,
            max_workers=args.max_workers,
            dry_run=False,
            yes=True,
        )
        result = cmd_deploy(deploy_args, cfg)
        if result:
            return result
        endpoint = _current_launch_endpoint(
            cfg=cfg,
            app=app,
            max_workers=args.max_workers,
        )
    if not isinstance(endpoint, dict) or not endpoint.get("id"):
        raise RuntimeError("Flash launch could not resolve a verified endpoint")

    up_args = argparse.Namespace(
        endpoint_id=str(endpoint["id"]),
        run=shlex.join(run_argv),
        run_name=run_name,
        experiment_ref=experiment_ref,
        library_ref=library_ref,
        results_branch=args.results_branch,
        max_age=max_age,
        queue_timeout=args.queue_timeout,
        max_price=args.max_price,
        max_estimated_cost=args.max_estimated_cost,
        forward_b2=True,
        self_destruct=args.publish_results,
        progress_interval=args.progress_interval,
        no_progress_timeout=args.no_progress_timeout,
        dry_run=args.dry_run,
        yes=args.yes,
    )
    return cmd_up(up_args, cfg)


def cmd_status(args: argparse.Namespace, cfg: FlashConfig) -> int:
    client = ServerlessClient()
    state = load_state(cfg)
    jobs = state.get("jobs", [])
    if not jobs:
        print("No tracked RunPod Flash jobs.")
        return 0
    payload = []
    for entry in jobs:
        row = dict(entry)
        if not row.get("terminal_at_iso"):
            observed = client.job_status(
                str(row["endpoint_id"]),
                str(row["provider_job_id"]),
            )
            row["provider_status"] = (
                str((observed or {}).get("status") or "EXPIRED").upper()
            )
            row["worker_id"] = (observed or {}).get("workerId")
        payload.append(row)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_watch(args: argparse.Namespace, cfg: FlashConfig) -> int:
    state = load_state(cfg)
    entry = next(
        (
            row
            for row in state.get("jobs", [])
            if row.get("endpoint_id") == args.endpoint_id
            and row.get("provider_job_id") == args.job_id
        ),
        None,
    )
    queue_timeout = (
        args.queue_timeout * 60
        if args.queue_timeout is not None
        else _remaining_seconds(
            (entry or {}).get("queue_deadline_iso"),
            float((entry or {}).get("queue_timeout_seconds") or 20 * 60),
        )
    )
    lifecycle_timeout = (
        args.max_age * 3600
        if args.max_age is not None
        else _remaining_seconds(
            (entry or {}).get("lifecycle_deadline_iso"),
            float((entry or {}).get("lifecycle_timeout_seconds") or 3600),
        )
    )
    progress_interval = (
        args.progress_interval
        if args.progress_interval is not None
        else float(
            (entry or {}).get("progress_interval_seconds")
            or cfg.PROGRESS_INTERVAL_SECONDS
        )
    )
    no_progress_timeout = (
        args.no_progress_timeout * 60
        if args.no_progress_timeout is not None
        else float(
            (entry or {}).get("no_progress_timeout_seconds")
            or cfg.NO_PROGRESS_TIMEOUT_SECONDS
        )
    )
    result = monitor_job(
        ServerlessClient(),
        endpoint_id=args.endpoint_id,
        provider_job_id=args.job_id,
        queue_timeout_s=queue_timeout,
        lifecycle_timeout_s=lifecycle_timeout,
        progress_interval_s=progress_interval,
        no_progress_timeout_s=no_progress_timeout,
        entry=entry,
        cfg=cfg,
    )
    print(json.dumps(result.output, indent=2, sort_keys=True))
    if result.status != "COMPLETED":
        return 1
    if entry is None:
        return 0
    validation_entry = {
        "experiment_ref": entry.get("experiment_ref"),
        "library_ref": entry.get("library_ref"),
        "image_digest": entry.get("image_digest"),
        "endpoint_id": entry.get("endpoint_id"),
        "job_id": entry.get("launcher_job_id"),
        "run_name": entry.get("run_name"),
    }
    valid = terminal_output_proves_success(result.output, validation_entry)
    if valid and result.output.get("canonical_manifest_key"):
        print("workload_success=true; recovered durable Flash training verified")
        return 0
    print("completed provider job did not prove durable workload success")
    return 1


def cmd_logs(args: argparse.Namespace, cfg: FlashConfig) -> int:
    client = ServerlessClient()
    workers = client.list_workers(args.endpoint_id).get("workers", [])
    worker_ids = [args.worker] if args.worker else [
        str(row["id"])
        for row in workers
        if isinstance(row, dict) and row.get("id")
    ]
    if not worker_ids:
        print("No active workers; logs are unavailable after scale-down.")
        return 0
    secrets = _monitor_secrets()

    def emit(event: dict[str, str]) -> None:
        print(
            f"{event.get('ts', '')} [{event.get('source', '?')}] "
            f"{redact_sensitive(event.get('line', ''), secrets)}".strip(),
            flush=True,
        )

    for worker_id in worker_ids:
        print(f"worker {worker_id}:", flush=True)
        client.worker_logs(
            args.endpoint_id,
            worker_id,
            source=args.source,
            tail=args.tail,
            since=args.since,
            follow=args.follow,
            emit=emit,
        )
    return 0


def cmd_inspect(args: argparse.Namespace, cfg: FlashConfig) -> int:
    client = ServerlessClient()
    endpoint = client.get_endpoint(args.endpoint_id)
    if endpoint is None:
        print("null")
        return 1
    safe = dict(endpoint)
    if isinstance(safe.get("env"), dict):
        safe["env"] = {key: "<configured>" for key in safe["env"]}
    payload = {
        "endpoint": safe,
        "workers": client.list_workers(args.endpoint_id),
        "health": client.health(args.endpoint_id),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_destroy(args: argparse.Namespace, cfg: FlashConfig) -> int:
    if not args.yes:
        raise ValueError("--yes is required to delete a Flash endpoint")
    ServerlessClient().delete_endpoint(args.endpoint_id)
    print(f"deleted endpoint {args.endpoint_id}")
    return 0


def cmd_retrieve(args: argparse.Namespace, cfg: FlashConfig) -> int:
    manifest = load_manifest(
        path=Path(args.manifest) if args.manifest else None,
        key=args.manifest_key,
        bucket=args.bucket,
    )
    destination = Path(args.destination).expanduser().resolve()
    files = retrieve_manifest_artifacts(manifest, destination)
    print(f"retrieved and SHA-256 verified {len(files)} file(s) to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m devops.flash.provision")
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser(
        "launch",
        help="deploy/reuse an endpoint and run one experiment with safe defaults",
    )
    launch.add_argument("experiment", help="experiments.*.experiment module")
    launch.add_argument("--experiment-dir", type=Path, default=CONFIG.EXPERIMENT_REPO_LOCAL)
    launch.add_argument("--experiment-ref")
    launch.add_argument("--library-ref")
    launch.add_argument("--run-name")
    launch.add_argument("--seed", type=int, default=42)
    launch.add_argument(
        "--run-args",
        default="",
        help="additional quoted operational rl-harness arguments",
    )
    launch.add_argument("--smoke", action="store_true")
    launch.add_argument("--app")
    launch.add_argument("--environment", default="production")
    launch.add_argument("--endpoint-id")
    launch.add_argument("--max-workers", type=int, default=1)
    launch.add_argument("--max-age", type=float)
    launch.add_argument(
        "--queue-timeout",
        type=float,
        default=CONFIG.DEFAULT_QUEUE_TIMEOUT_MINUTES,
    )
    launch.add_argument(
        "--progress-interval",
        type=float,
        default=CONFIG.PROGRESS_INTERVAL_SECONDS,
    )
    launch.add_argument(
        "--no-progress-timeout",
        type=float,
        default=CONFIG.NO_PROGRESS_TIMEOUT_SECONDS / 60,
    )
    launch.add_argument("--max-price", type=float, default=CONFIG.DEFAULT_MAX_PRICE)
    launch.add_argument(
        "--max-estimated-cost",
        type=float,
        default=CONFIG.DEFAULT_MAX_ESTIMATED_COST,
    )
    launch.add_argument("--results-branch")
    launch.add_argument("--publish-results", action="store_true")
    launch_action = launch.add_mutually_exclusive_group(required=True)
    launch_action.add_argument("--dry-run", action="store_true")
    launch_action.add_argument("--yes", action="store_true")

    deploy = sub.add_parser("deploy", help="deploy source artifact without a user image")
    deploy.add_argument("--app", required=True)
    deploy.add_argument("--environment", default="production")
    deploy.add_argument("--max-workers", type=int, default=1)
    deploy.add_argument("--dry-run", action="store_true")
    deploy.add_argument("--yes", action="store_true")

    up = sub.add_parser("up", help="submit and verify one experiment job")
    up.add_argument("--endpoint-id", required=True)
    up.add_argument("--run", required=True)
    up.add_argument("--run-name", required=True)
    up.add_argument("--experiment-ref", required=True)
    up.add_argument("--library-ref", required=True)
    up.add_argument("--results-branch")
    up.add_argument("--max-age", type=float, default=0.5)
    up.add_argument("--queue-timeout", type=float, default=20)
    up.add_argument("--max-price", type=float, required=True)
    up.add_argument("--max-estimated-cost", type=float, required=True)
    up.add_argument("--forward-b2", action="store_true")
    up.add_argument("--self-destruct", action="store_true")
    up.add_argument(
        "--progress-interval",
        type=float,
        default=CONFIG.PROGRESS_INTERVAL_SECONDS,
        metavar="SECONDS",
        help="emit worker/log heartbeat at this interval",
    )
    up.add_argument(
        "--no-progress-timeout",
        type=float,
        default=CONFIG.NO_PROGRESS_TIMEOUT_SECONDS / 60,
        metavar="MINUTES",
        help="cancel IN_PROGRESS jobs with no new worker evidence",
    )
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--yes", action="store_true")

    sub.add_parser("status", help="show tracked jobs and current provider status")
    watch = sub.add_parser("watch", help="recover monitoring for a submitted job")
    watch.add_argument("endpoint_id")
    watch.add_argument("job_id")
    watch.add_argument("--queue-timeout", type=float, metavar="MINUTES")
    watch.add_argument("--max-age", type=float, metavar="HOURS")
    watch.add_argument("--progress-interval", type=float, metavar="SECONDS")
    watch.add_argument("--no-progress-timeout", type=float, metavar="MINUTES")
    logs = sub.add_parser("logs", help="stream redacted Flash worker logs")
    logs.add_argument("endpoint_id")
    logs.add_argument("--worker")
    logs.add_argument("--source", choices=["container", "system"])
    logs.add_argument("--tail", type=int, default=200)
    logs.add_argument("--since")
    logs.add_argument("--follow", action="store_true")
    inspect = sub.add_parser("inspect", help="show redacted endpoint configuration")
    inspect.add_argument("endpoint_id")
    destroy = sub.add_parser("destroy", help="delete a Flash-created endpoint")
    destroy.add_argument("endpoint_id")
    destroy.add_argument("--yes", action="store_true")
    retrieve = sub.add_parser("retrieve", help="download and verify B2 artifacts")
    source = retrieve.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--manifest-key")
    retrieve.add_argument("--bucket")
    retrieve.add_argument("--destination", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "launch": cmd_launch,
        "deploy": cmd_deploy,
        "up": cmd_up,
        "status": cmd_status,
        "watch": cmd_watch,
        "logs": cmd_logs,
        "inspect": cmd_inspect,
        "destroy": cmd_destroy,
        "retrieve": cmd_retrieve,
    }
    try:
        return commands[args.command](args, CONFIG)
    except (
        OSError,
        PreflightError,
        ServerlessClientError,
        subprocess.CalledProcessError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"RunPod Flash {args.command} failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
