"""Dry-run-first deployment and job launcher for RunPod Flash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
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
from devops.serverless.retrieve import load_manifest, retrieve_manifest_artifacts

from .config import CONFIG, FlashConfig

_ROOT = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).resolve().with_name("worker.py")
_HANDLER = _ROOT / "devops" / "serverless" / "handler.py"
_STAGED_HANDLER_NAME = "rlh_experiment_handler.py"
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "EXPIRED"}
_FINGERPRINT = re.compile(r"^[0-9a-fA-F]{64}$")


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
    return endpoint


def _find_endpoint(
    client: ServerlessClient,
    *,
    app: str,
    max_workers: int,
    cfg: FlashConfig,
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
    matches.sort(key=lambda row: str(row.get("createdAt") or ""))
    newest_id = str(matches[-1].get("id") or "")
    newest = _validate_endpoint(
        client.get_endpoint(newest_id),
        app=app,
        max_workers=max_workers,
        cfg=cfg,
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
        print(completed.stdout.strip())
    endpoint = _find_endpoint(
        ServerlessClient(),
        app=args.app,
        max_workers=args.max_workers,
        cfg=cfg,
    )
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

    experiment_ref = require_sha(args.experiment_ref, "--experiment-ref")
    library_ref = require_sha(args.library_ref, "--library-ref")
    run_argv = parse_run_command(args.run, args.run_name)
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
    queue_deadline = time.monotonic() + args.queue_timeout * 60
    lifecycle_deadline = time.monotonic() + ttl_seconds
    last_status = ""
    while True:
        observed = client.job_status(args.endpoint_id, provider_job_id)
        status = (
            str((observed or {}).get("status") or "EXPIRED").upper()
        )
        if status != last_status:
            print(f"  job {provider_job_id}: {status}")
            last_status = status
        if status in _TERMINAL:
            output = sanitize_terminal_output((observed or {}).get("output"))
            print(json.dumps(output, indent=2, sort_keys=True))
            entry = {
                "experiment_ref": experiment_ref,
                "library_ref": library_ref,
                "image_digest": runtime_digest,
                "endpoint_id": args.endpoint_id,
                "job_id": launcher_job_id,
                "run_name": args.run_name,
            }
            if status == "COMPLETED" and terminal_output_proves_success(output, entry):
                if not output.get("canonical_manifest_key"):
                    raise RuntimeError("Flash output omitted canonical_manifest_key")
                print("workload_success=true; durable Flash training verified")
                return 0
            failure = provider_failure_summary(observed, secrets=())
            if failure:
                print(json.dumps(failure, indent=2, sort_keys=True))
            return 1
        now = time.monotonic()
        if (status == "IN_QUEUE" and now >= queue_deadline) or now >= lifecycle_deadline:
            client.cancel_job(args.endpoint_id, provider_job_id)
            print(f"cancelled Flash job {provider_job_id} after timeout")
            return 1
        time.sleep(5)


def cmd_inspect(args: argparse.Namespace, cfg: FlashConfig) -> int:
    endpoint = ServerlessClient().get_endpoint(args.endpoint_id)
    if endpoint is None:
        print("null")
        return 1
    safe = dict(endpoint)
    if isinstance(safe.get("env"), dict):
        safe["env"] = {key: "<configured>" for key in safe["env"]}
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


def cmd_destroy(args: argparse.Namespace, cfg: FlashConfig) -> int:
    if not args.yes:
        raise ValueError("--yes is required to delete a Flash endpoint")
    ServerlessClient().delete_endpoint(args.endpoint_id)
    print(f"deleted endpoint {args.endpoint_id}")
    return 0


def cmd_retrieve(args: argparse.Namespace, cfg: FlashConfig) -> int:
    manifest = load_manifest(
        manifest_path=args.manifest,
        manifest_key=args.manifest_key,
        bucket=args.bucket,
    )
    destination = Path(args.destination).expanduser().resolve()
    files = retrieve_manifest_artifacts(manifest, destination)
    print(f"retrieved and SHA-256 verified {len(files)} file(s) to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m devops.flash.provision")
    sub = parser.add_subparsers(dest="command", required=True)

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
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--yes", action="store_true")

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
        "deploy": cmd_deploy,
        "up": cmd_up,
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
        ValueError,
    ) as error:
        print(f"RunPod Flash {args.command} failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
