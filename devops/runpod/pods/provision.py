"""RunPod Pods provisioning CLI mirroring the Vast lifecycle surface."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from harness.storage.b2 import b2_env_for_remote

from .client import (
    RunPodClient,
    RunPodClientError,
    assert_safe_pod,
    reject_explicitly_unsafe_pod,
    resolve_api_key,
)
from .config import CONFIG, RunPodConfig
from .images import ImageResolutionError, resolve_image_digest
from .redaction import redact_pod_metadata, redact_sensitive

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(cfg: RunPodConfig = CONFIG) -> dict[str, Any]:
    if cfg.STATE_PATH.exists():
        try:
            payload = json.loads(cfg.STATE_PATH.read_text())
            if isinstance(payload, dict):
                payload.setdefault("pods", [])
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    return {"pods": []}


def save_state(state: dict[str, Any], cfg: RunPodConfig = CONFIG) -> None:
    cfg.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )


def _record(state: dict[str, Any], entry: dict[str, Any]) -> None:
    state["pods"] = [
        row for row in state.get("pods", []) if row.get("id") != entry["id"]
    ]
    state["pods"].append(entry)


def _unrecord(state: dict[str, Any], pod_id: str) -> None:
    state["pods"] = [
        row for row in state.get("pods", []) if row.get("id") != pod_id
    ]


def load_cost_history(cfg: RunPodConfig = CONFIG) -> list[dict[str, Any]]:
    if cfg.COST_HISTORY_PATH.exists():
        try:
            payload = json.loads(cfg.COST_HISTORY_PATH.read_text())
            return payload if isinstance(payload, list) else []
        except (OSError, json.JSONDecodeError):
            pass
    return []


def record_cost(
    entry: dict[str, Any],
    cost: dict[str, Any],
    cfg: RunPodConfig = CONFIG,
) -> None:
    history = load_cost_history(cfg)
    payload = {
        "pod_id": entry["id"],
        "name": entry.get("name"),
        "run_name": entry.get("run_name"),
        "experiment_ref": entry.get("ref"),
        "library_ref": entry.get("library_ref"),
        "image_digest": entry.get("image_digest"),
        "created_at": entry.get("created_at_iso"),
        **cost,
    }
    history = [row for row in history if row.get("pod_id") != entry["id"]]
    history.append(payload)
    cfg.COST_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )


def _git_head(repo: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0 and completed.stdout.strip()
        else None
    )


def resolve_experiment_repo(args, cfg: RunPodConfig) -> Path:
    if args.experiment_repo:
        return Path(args.experiment_repo).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "experiments").is_dir() and (cwd / "pyproject.toml").is_file():
        return cwd
    sibling = cfg.EXPERIMENT_REPO_LOCAL.expanduser().resolve()
    return sibling if sibling.is_dir() else cwd


_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


def resolve_ref(args, cfg: RunPodConfig, log=print) -> str:
    from devops.runpod.execution.preflight import (
        PreflightError,
        resolve_ref_with_rev_parse,
    )

    repo = resolve_experiment_repo(args, cfg)
    if args.commit:
        if _FULL_SHA.fullmatch(args.commit):
            return args.commit.lower()
        try:
            return resolve_ref_with_rev_parse(
                args.commit,
                label="--commit",
                repository=str(repo),
            )
        except PreflightError:
            return args.commit
    if args.branch:
        try:
            return resolve_ref_with_rev_parse(
                args.branch,
                label="--branch",
                repository=str(repo),
            )
        except PreflightError:
            return args.branch
    sha = _git_head(repo)
    if not sha:
        log(f"could not resolve experiment HEAD at {repo}")
        raise ValueError("pass --branch or --commit explicitly")
    return sha


def resolve_library_ref(args, cfg: RunPodConfig) -> str:
    from devops.runpod.execution.preflight import (
        PreflightError,
        resolve_ref_with_rev_parse,
    )

    value = (
        args.library_commit
        or args.library_branch
        or cfg.LIBRARY_DEFAULT_REF
    )
    if _FULL_SHA.fullmatch(value):
        return value.lower()
    library_repo = Path(__file__).resolve().parents[3]
    try:
        return resolve_ref_with_rev_parse(
            value,
            label="library-ref",
            repository=str(library_repo),
        )
    except PreflightError:
        # Allow remote branch names that are not present locally; the Pod
        # checkout remains the source of truth for non-SHA refs.
        return value


def resolve_github_token() -> str | None:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_ssh_key(path: str | None = None) -> tuple[Path, str]:
    """Resolve a local private key and its matching public key."""
    if path:
        candidates = [Path(path).expanduser()]
    else:
        configured = os.environ.get("RUNPOD_SSH_KEY_PATH")
        candidates = (
            [Path(configured).expanduser()]
            if configured
            else [
                Path("~/.ssh/id_ed25519").expanduser(),
                Path("~/.ssh/id_rsa").expanduser(),
            ]
        )
    for candidate in candidates:
        if candidate.suffix == ".pub":
            public_path = candidate
            private_path = Path(str(candidate)[: -len(".pub")])
        else:
            private_path = candidate
            public_path = Path(f"{candidate}.pub")
        if not private_path.is_file() or not public_path.is_file():
            continue
        public_key = public_path.read_text().strip()
        if "\n" in public_key or public_key.startswith("-----BEGIN"):
            raise ValueError(f"invalid SSH public key file: {public_path}")
        key_type = public_key.partition(" ")[0]
        if key_type not in {
            "ssh-ed25519",
            "ssh-rsa",
            "ecdsa-sha2-nistp256",
            "ecdsa-sha2-nistp384",
            "ecdsa-sha2-nistp521",
        }:
            raise ValueError(f"unsupported SSH public key type in {public_path}")
        return private_path, public_key
    raise ValueError(
        "interactive mode requires a local SSH keypair; pass --ssh-key or "
        "create ~/.ssh/id_ed25519(.pub) or ~/.ssh/id_rsa(.pub)"
    )


def build_env(
    cfg: RunPodConfig,
    *,
    experiment_ref: str,
    library_ref: str,
    run_cmd: str,
    run_name: str,
    results_branch: str,
    github_token: str,
    image_digest: str,
    max_age_s: float,
    estimated_price: float,
    push_results: bool,
    forward_b2: bool,
    interactive: bool = False,
    ssh_public_key: str | None = None,
    gpu_type_id: str | None = None,
) -> dict[str, str]:
    if max_age_s <= 0:
        raise ValueError("RunPod max-age must be positive")
    env = {
        "RUNPOD_EXPERIMENT_REPO_URL": cfg.EXPERIMENT_REPO_URL,
        "RUNPOD_EXPERIMENT_REPO_SLUG": cfg.EXPERIMENT_REPO_SLUG,
        "RUNPOD_EXPERIMENT_GIT_REF": experiment_ref,
        "RUNPOD_LIBRARY_REPO_URL": cfg.LIBRARY_REPO_URL,
        "RUNPOD_LIBRARY_GIT_REF": library_ref,
        "RUNPOD_RUN_CMD": run_cmd,
        "RUNPOD_RUN_NAME": run_name,
        "RUNPOD_RESULTS_BRANCH": results_branch,
        "RUNPOD_MAX_AGE_S": str(int(max_age_s)),
        "RUNPOD_ESTIMATED_PRICE_PER_HOUR": f"{estimated_price:.6f}",
        "RUNPOD_IMAGE_DIGEST": image_digest,
        "RUNPOD_GPU_TYPE_IDS": gpu_type_id or cfg.GPU_TYPE_IDS[0],
        "RUNPOD_RAY_VERSION": cfg.RAY_VERSION,
        "RUNPOD_TORCH_VERSION": cfg.TORCH_VERSION,
        "RUNPOD_GYMNASIUM_VERSION": cfg.GYMNASIUM_VERSION,
        "RUNPOD_BOTO3_VERSION": cfg.BOTO3_VERSION,
        "RUNPOD_MLFLOW_VERSION": cfg.MLFLOW_VERSION,
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "GH_TOKEN": github_token,
    }
    if push_results:
        env["RUNPOD_PUSH_RESULTS"] = "1"
    if interactive:
        if not ssh_public_key:
            raise ValueError("interactive mode requires an SSH public key")
        env["RUNPOD_INTERACTIVE"] = "1"
        # Documented per-Pod override; the corresponding private key never
        # leaves the launching machine or Cursor Cloud agent.
        env["SSH_PUBLIC_KEY"] = ssh_public_key
    if forward_b2:
        b2 = b2_env_for_remote()
        required_b2 = {
            "B2_BUCKET",
            "B2_ENDPOINT",
            "B2_APPLICATION_KEY_ID",
            "B2_APPLICATION_KEY",
        }
        if not required_b2.issubset(b2):
            raise ValueError(
                "--forward-b2 requires B2_BUCKET, B2_ENDPOINT, "
                "B2_APPLICATION_KEY_ID, and B2_APPLICATION_KEY"
            )
        env.update(b2)
    return env


def build_create_request(
    cfg: RunPodConfig,
    *,
    name: str,
    image: str,
    disk_gb: int,
    regions: list[str],
    env: dict[str, str],
    terminate_after: str,
    interactive: bool = False,
    gpu_type_id: str | None = None,
) -> dict[str, Any]:
    """Build the official on-demand GraphQL Pod request."""
    request: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "cloudType": cfg.CLOUD_TYPE,
        # Internal assertion consumed by RunPodClient. The API operation itself
        # is podFindAndDeployOnDemand, never the interruptible mutation.
        "interruptible": False,
        "gpuTypeId": gpu_type_id or cfg.GPU_TYPE_IDS[0],
        "gpuCount": cfg.GPU_COUNT,
        "minCudaVersion": cfg.MIN_CUDA,
        "containerDiskInGb": int(disk_gb),
        "volumeInGb": cfg.VOLUME_GB,
        "volumeMountPath": cfg.VOLUME_MOUNT_PATH,
        "supportPublicIp": interactive,
        "startSsh": interactive,
        "terminateAfter": terminate_after,
        "env": env,
    }
    if interactive:
        request["ports"] = "22/tcp"
    if regions:
        request["countryCode"] = regions[0]
    return request


def _pod_price(pod: dict[str, Any], fallback: float) -> float:
    return float(
        pod.get("adjustedCostPerHr")
        or pod.get("costPerHr")
        or pod.get("cost")
        or fallback
    )


def _pod_status(pod: dict[str, Any] | None) -> str:
    if not pod:
        return "GONE"
    return str(
        pod.get("desiredStatus") or pod.get("status") or "UNKNOWN"
    ).upper()


def _created_timestamp(pod: dict[str, Any]) -> float | None:
    value = pod.get("createdAt")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            return None
    return None


def _managed(pod: dict[str, Any], cfg: RunPodConfig) -> bool:
    return str(pod.get("name") or "").startswith(cfg.MANAGED_NAME_PREFIX)


def _ssh_command(pod: dict[str, Any], private_key: Path) -> list[str]:
    public_ip = str(pod.get("publicIp") or "").strip()
    mappings = pod.get("portMappings")
    if public_ip and isinstance(mappings, dict):
        port = mappings.get("22") or mappings.get(22)
        if port:
            return [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-i",
                str(private_key),
                "-p",
                str(port),
                f"root@{public_ip}",
            ]
    machine = pod.get("machine")
    host_id = (
        machine.get("podHostId") if isinstance(machine, dict) else None
    )
    if host_id:
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-i",
            str(private_key),
            f"{host_id}@ssh.runpod.io",
        ]
    raise RunPodClientError(
        "RunPod has not published SSH connection details yet; retry shortly "
        "or use the Pod Connect tab"
    )


def _reject_unsupported_vast_flags(args) -> str | None:
    if args.mode != "ondemand":
        return "RunPod Pods are restricted to --mode ondemand"
    if args.bid is not None:
        return "--bid is incompatible with mandatory on-demand RunPod Pods"
    if args.offer_id is not None or args.machine_id is not None:
        return (
            "RunPod schedules by GPU type and does not expose Vast offer or "
            "machine selection; --offer-id/--machine-id are unsupported"
        )
    if args.exclude_machine:
        return (
            "RunPod does not expose a pre-create host exclusion primitive; "
            "--exclude-machine is unsupported"
        )
    return None


def cmd_up(args, cfg: RunPodConfig) -> int:
    log = print
    incompatibility = _reject_unsupported_vast_flags(args)
    if incompatibility:
        log(incompatibility)
        return 2
    if args.count < 1:
        log("--count must be at least 1")
        return 2
    if args.interactive and args.run:
        log("--interactive does not accept --run; connect over SSH and run it")
        return 2
    if not args.interactive and not args.run:
        log("--run is required; idle Pods are not permitted")
        return 2

    api_key = resolve_api_key()
    github_token = resolve_github_token()
    if not api_key:
        log("RUNPOD_API_KEY is required")
        return 2
    if not github_token:
        log("GH_TOKEN is required")
        return 2

    default_max_age = (
        cfg.INTERACTIVE_MAX_AGE_HOURS
        if args.interactive
        else cfg.MAX_AGE_HOURS
    )
    max_age_hours = (
        default_max_age if args.max_age is None else float(args.max_age)
    )
    if max_age_hours <= 0:
        log(
            "--max-age must be positive for RunPod; disabling the hard "
            "wall-clock ceiling is not permitted"
        )
        return 2
    max_age_s = max_age_hours * 3600.0
    experiment_ref = resolve_ref(args, cfg, log)
    library_ref = resolve_library_ref(args, cfg)
    # Fail closed on nonexistent exact SHAs before any Pod spend.
    if _FULL_SHA.fullmatch(experiment_ref) or _FULL_SHA.fullmatch(library_ref):
        from devops.runpod.execution.preflight import (
            PreflightError,
            verify_remote_sha_fetchable,
        )

        try:
            if _FULL_SHA.fullmatch(experiment_ref):
                verify_remote_sha_fetchable(
                    cfg.EXPERIMENT_REPO_URL,
                    experiment_ref.lower(),
                    label="experiment",
                    github_token=github_token,
                )
            if _FULL_SHA.fullmatch(library_ref):
                verify_remote_sha_fetchable(
                    cfg.LIBRARY_REPO_URL,
                    library_ref.lower(),
                    label="library",
                    github_token=github_token,
                )
        except PreflightError as error:
            log(f"PREFLIGHT rejected before provisioning: {error}")
            return 2
    regions = (
        [part.strip().upper() for part in args.regions.split(",") if part.strip()]
        if args.regions
        else []
    )
    gpu_type_id = args.gpu_type or cfg.GPU_TYPE_IDS[0]
    disk_gb = int(args.disk or cfg.DISK_GB)
    ssh_private_key: Path | None = None
    ssh_public_key: str | None = None
    if args.interactive:
        try:
            ssh_private_key, ssh_public_key = resolve_ssh_key(args.ssh_key)
        except ValueError as error:
            log(str(error))
            return 2
    try:
        image, image_digest = resolve_image_digest(args.image or cfg.IMAGE)
    except ImageResolutionError as error:
        log(str(error))
        return 2

    estimated_price = cfg.PUBLIC_4090_PRICE_PER_HOUR
    if args.max_price is not None and estimated_price > args.max_price:
        log(
            f"public estimate ${estimated_price:.3f}/h exceeds "
            f"--max-price ${args.max_price:.3f}/h"
        )
        return 2
    max_compute = estimated_price * max_age_hours * args.count
    log(f"  experiment ref: {experiment_ref}")
    log(f"  library ref:    {library_ref}")
    log(f"  image:          {image}")
    log(
        f"  request: {args.count} on-demand, non-interruptible "
        f"{gpu_type_id} Pod(s), Community Cloud"
    )
    if args.interactive:
        log(
            f"  access:         interactive SSH using "
            f"{ssh_private_key}; manual destroy or {max_age_hours:g}h ceiling"
        )
    log(
        f"  estimate: ${estimated_price:.3f}/h each; hard-ceiling compute "
        f"estimate ${max_compute:.2f} at {max_age_hours:g}h"
    )
    if (
        args.run
        and "--smoke" in args.run
        and "--upload-artifacts" not in args.run
    ):
        log(
            "WARNING: smoke runs disable automatic B2 upload; add "
            "--upload-artifacts to persist checkpoints"
        )

    run_name = args.run_name or f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    results_branch = args.results_branch or cfg.DEFAULT_RESULTS_BRANCH
    try:
        env = build_env(
            cfg,
            experiment_ref=experiment_ref,
            library_ref=library_ref,
            run_cmd=args.run or "",
            run_name=run_name,
            results_branch=results_branch,
            github_token=github_token,
            image_digest=image_digest,
            max_age_s=max_age_s,
            estimated_price=estimated_price,
            push_results=bool(args.self_destruct),
            forward_b2=bool(args.forward_b2),
            interactive=bool(args.interactive),
            ssh_public_key=ssh_public_key,
            gpu_type_id=gpu_type_id,
        )
    except ValueError as error:
        log(str(error))
        return 2

    if args.dry_run:
        log("--dry-run: request validated; no Pods created.")
        return 0
    if not args.yes:
        answer = input(f"Create {args.count} billing Pod(s)? [y/N] ").lower()
        if answer not in {"y", "yes"}:
            log("aborted.")
            return 1

    client = RunPodClient(cfg, api_key=api_key)
    state = load_state(cfg)
    created: list[dict[str, Any]] = []
    try:
        for index in range(args.count):
            suffix = uuid.uuid4().hex[:8]
            name = f"{cfg.MANAGED_NAME_PREFIX}{run_name}-{index + 1}-{suffix}"
            terminate_after = (
                datetime.now(timezone.utc) + timedelta(seconds=max_age_s)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            request = build_create_request(
                cfg,
                name=name,
                image=image,
                disk_gb=disk_gb,
                regions=regions,
                env=env,
                terminate_after=terminate_after,
                interactive=bool(args.interactive),
                gpu_type_id=gpu_type_id,
            )
            log(f"creating {name}")
            pod = client.create_pod(request)
            pod_id = str(pod["id"])
            entry = {
                "id": pod_id,
                "name": name,
                "run_name": run_name,
                "ref": experiment_ref,
                "library_ref": library_ref,
                "image": image,
                "image_digest": image_digest,
                "price": _pod_price(pod, estimated_price),
                "mode": "ondemand",
                "cloud": cfg.CLOUD_TYPE,
                "created_at": time.time(),
                "created_at_iso": _utc_now(),
                "max_age_s": max_age_s,
                "interactive": bool(args.interactive),
            }
            _record(state, entry)
            save_state(state, cfg)
            created.append(entry)

            inspected = client.get_pod(pod_id) or pod
            try:
                # During allocation RunPod can omit placement fields. Reject
                # any explicit contradiction now, then require complete proof
                # once the Pod reaches RUNNING.
                reject_explicitly_unsafe_pod(inspected, cfg)
            except RunPodClientError:
                client.terminate_pod(pod_id)
                entry["terminated_after_launch_error"] = True
                entry["terminated_at_iso"] = _utc_now()
                _record(state, entry)
                save_state(state, cfg)
                raise
            price = _pod_price(inspected, entry["price"])
            entry["price"] = price
            if args.max_price is not None and price > args.max_price:
                client.terminate_pod(pod_id)
                entry["terminated_after_launch_error"] = True
                entry["terminated_at_iso"] = _utc_now()
                _record(state, entry)
                save_state(state, cfg)
                raise RunPodClientError(
                    f"Pod price ${price:.3f}/h exceeds "
                    f"--max-price ${args.max_price:.3f}/h; terminated"
                )
            _record(state, entry)
            save_state(state, cfg)
            log(
                f"  created Pod {pod_id}: ${price:.3f}/h; pending full "
                "placement verification at RUNNING"
            )

        for entry in created:
            pod = client.wait_until_running(entry["id"], log=log)
            verification = client.get_pod_safety_fields(entry["id"])
            pod["podType"] = verification["podType"]
            pod["machine"] = {
                **(
                    pod.get("machine")
                    if isinstance(pod.get("machine"), dict)
                    else {}
                ),
                **verification["machine"],
            }
            assert_safe_pod(pod, cfg)
            log(
                f"  Pod {entry['id']} RUNNING; podType={pod['podType']}, "
                f"secureCloud={pod['machine']['secureCloud']}, "
                f"gpu={pod['machine']['gpuTypeId']}; runner owns completion, "
                "failure, and max-age termination"
            )
            if args.interactive and ssh_private_key is not None:
                connection_pod = client.get_pod(entry["id"]) or pod
                connection_pod["machine"] = {
                    **(
                        connection_pod.get("machine")
                        if isinstance(connection_pod.get("machine"), dict)
                        else {}
                    ),
                    **verification["machine"],
                }
                try:
                    command = _ssh_command(connection_pod, ssh_private_key)
                    log(f"  SSH: {shlex.join(command)}")
                except RunPodClientError as error:
                    log(f"  SSH pending: {error}")
    except Exception as error:  # noqa: BLE001
        detail = redact_sensitive(error, (api_key, github_token))
        log(f"RunPod launch failed: {detail}")
        for entry in created:
            try:
                client.terminate_pod(entry["id"])
            except Exception:
                pass
            entry["terminated_after_launch_error"] = True
            entry["terminated_at_iso"] = _utc_now()
            entry["launch_error"] = detail
            _record(state, entry)
        save_state(state, cfg)
        return 1

    if args.interactive:
        log(
            "Interactive Pods are live and billing. Destroy them when done; "
            "`terminateAfter` and `reap` are backstops."
        )
    else:
        log("Pods are live and billing. Check `status`; `reap` is the backstop.")
    return 0


def _collect_actual_cost(
    client: RunPodClient,
    entry: dict[str, Any],
    cfg: RunPodConfig,
) -> dict[str, Any]:
    return client.pod_cost(
        entry["id"],
        start_time=entry.get("created_at_iso"),
        end_time=entry.get("terminated_at_iso") or _utc_now(),
    )


def cmd_status(args, cfg: RunPodConfig) -> int:
    client = RunPodClient(cfg)
    state = load_state(cfg)
    tracked = {str(row["id"]): row for row in state.get("pods", [])}
    live = client.list_pods()
    live_by_id = {str(pod.get("id")): pod for pod in live if pod.get("id")}
    managed = [pod for pod in live if _managed(pod, cfg)]
    print(
        f"  {'id':<18}{'status':<13}{'$/hr':<8}{'age_h':<8}"
        f"{'tracked':<9}name"
    )
    print("  " + "-" * 92)
    now = time.time()
    for pod in managed:
        pod_id = str(pod["id"])
        entry = tracked.get(pod_id)
        created = (entry or {}).get("created_at") or _created_timestamp(pod)
        age_h = (now - created) / 3600.0 if created else 0.0
        price = _pod_price(pod, (entry or {}).get("price") or 0.0)
        print(
            f"  {pod_id:<18}{_pod_status(pod):<13}{price:<8.3f}"
            f"{age_h:<8.2f}{str(bool(entry)):<9}{pod.get('name', '?')}"
        )

    pending = list(state.get("pods", []))
    for entry in list(pending):
        if entry["id"] in live_by_id:
            continue
        if not entry.get("terminated_at_iso"):
            # RunPod removes self-terminated Pods from the live list without a
            # terminal record. Bound estimates at first observed absence
            # instead of letting them grow until billing aggregation posts.
            entry["terminated_at_iso"] = datetime.fromtimestamp(
                now, tz=timezone.utc
            ).isoformat()
            _record(state, entry)
        try:
            cost = _collect_actual_cost(client, entry, cfg)
        except RunPodClientError as error:
            print(f"  billing pending for {entry['id']}: {error}")
            continue
        if cost["record_count"]:
            record_cost(entry, cost, cfg)
            _unrecord(state, entry["id"])
            print(
                f"  completed {entry['id']}: actual cost "
                f"${cost['actual_cost_usd']:.4f} "
                f"({cost['time_billed_ms'] / 3_600_000:.3f} billed h)"
            )
        else:
            ended = now
            terminated_at = entry.get("terminated_at_iso")
            if isinstance(terminated_at, str):
                try:
                    ended = datetime.fromisoformat(
                        terminated_at.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    pass
            elapsed_h = max(
                0.0,
                (ended - float(entry.get("created_at") or ended)) / 3600.0,
            )
            estimate = elapsed_h * float(entry.get("price") or 0.0)
            print(
                f"  completed {entry['id']}: billing not posted yet; "
                f"current estimate ${estimate:.4f}"
            )
    save_state(state, cfg)
    if not managed and not pending:
        print("No managed RunPod Pods are active or pending billing.")
    return 0


def cmd_destroy(args, cfg: RunPodConfig) -> int:
    client = RunPodClient(cfg)
    state = load_state(cfg)
    tracked = list(state.get("pods", []))
    tracked_by_id = {str(row["id"]): row for row in tracked}
    if args.id:
        wanted = {str(value) for value in args.id}
        targets = [
            row for row in tracked if str(row.get("id")) in wanted
        ]
        known = {str(row.get("id")) for row in targets}
        targets.extend(
            {"id": pod_id, "name": "(explicit untracked)"}
            for pod_id in wanted - known
        )
    elif args.all:
        targets = tracked
    else:
        print("Specify --all or --id <id> [<id> ...]")
        return 2
    if not targets:
        print("Nothing to destroy.")
        return 0
    for target in targets:
        print(f"  Pod {target['id']} ({target.get('name', '?')})")
    if not args.yes:
        answer = input(f"Terminate {len(targets)} Pod(s)? [y/N] ").lower()
        if answer not in {"y", "yes"}:
            print("aborted.")
            return 1
    for target in targets:
        try:
            client.terminate_pod(str(target["id"]))
            print(f"terminated {target['id']}")
            tracked_entry = tracked_by_id.get(str(target["id"]))
            if tracked_entry:
                tracked_entry["terminated_at_iso"] = _utc_now()
                tracked_entry["terminated_manually"] = True
                _record(state, tracked_entry)
        except RunPodClientError as error:
            print(f"failed to terminate {target['id']}: {error}")
    save_state(state, cfg)
    return 0


def cmd_reap(args, cfg: RunPodConfig) -> int:
    client = RunPodClient(cfg)
    state = load_state(cfg)
    tracked = {str(row["id"]): row for row in state.get("pods", [])}
    now = time.time()
    override_s = (
        float(args.max_age) * 3600.0 if args.max_age is not None else None
    )
    targets: list[dict[str, Any]] = []
    for pod in client.list_pods():
        if not _managed(pod, cfg):
            continue
        pod_id = str(pod["id"])
        entry = tracked.get(pod_id)
        created = (entry or {}).get("created_at") or _created_timestamp(pod)
        cap_s = (
            override_s
            if override_s is not None
            else (entry or {}).get("max_age_s")
            or cfg.MAX_AGE_HOURS * 3600.0
        )
        age_s = now - created if created else 0.0
        status = _pod_status(pod)
        if status in {"EXITED", "ERROR", "TERMINATED"} or (
            cap_s > 0 and age_s > cap_s
        ):
            targets.append(
                {
                    "id": pod_id,
                    "name": pod.get("name"),
                    "status": status,
                    "age_h": age_s / 3600.0,
                    "orphan": entry is None,
                }
            )
    if not targets:
        print("No managed orphaned, exited, failed, or over-age Pods.")
        return 0
    print("Will reap:")
    for target in targets:
        print(
            f"  {target['id']} status={target['status']} "
            f"age={target['age_h']:.2f}h orphan={target['orphan']}"
        )
    if not args.yes:
        answer = input(f"Terminate {len(targets)} Pod(s)? [y/N] ").lower()
        if answer not in {"y", "yes"}:
            print("aborted.")
            return 1
    for target in targets:
        client.terminate_pod(target["id"])
        tracked_entry = tracked.get(target["id"])
        if tracked_entry:
            tracked_entry["terminated_at_iso"] = _utc_now()
            tracked_entry["terminated_by_reap"] = True
            _record(state, tracked_entry)
        print(f"reaped {target['id']}")
    save_state(state, cfg)
    return 0


def cmd_inspect(args, cfg: RunPodConfig) -> int:
    client = RunPodClient(cfg)
    pod = client.get_pod(args.id)
    if pod is None:
        print(f"Pod {args.id} not found")
        return 1
    print(json.dumps(redact_pod_metadata(pod), indent=2, sort_keys=True))
    return 0


def cmd_logs(args, cfg: RunPodConfig) -> int:
    client = RunPodClient(cfg)
    secrets = (
        resolve_api_key(),
        resolve_github_token(),
        os.environ.get("B2_APPLICATION_KEY"),
        os.environ.get("B2_APPLICATION_KEY_ID"),
    )

    def emit(event: dict[str, str]) -> None:
        prefix = " ".join(
            part
            for part in (event.get("ts"), f"[{event.get('source', '?')}]")
            if part
        )
        line = redact_sensitive(event.get("line", ""), secrets)
        print(f"{prefix} {line}".lstrip(), flush=True)

    try:
        client.pod_logs(
            args.id,
            source=args.source,
            tail=args.tail,
            since=args.since,
            follow=args.follow,
            emit=emit,
        )
    except KeyboardInterrupt:
        return 130
    return 0


def cmd_ssh(args, cfg: RunPodConfig) -> int:
    private_key, _ = resolve_ssh_key(args.ssh_key)
    client = RunPodClient(cfg)
    pod = client.get_pod(args.id)
    if pod is None:
        print(f"Pod {args.id} not found")
        return 1
    try:
        verification = client.get_pod_safety_fields(args.id)
        pod["machine"] = {
            **(
                pod.get("machine")
                if isinstance(pod.get("machine"), dict)
                else {}
            ),
            **verification["machine"],
        }
    except RunPodClientError:
        pass
    command = _ssh_command(pod, private_key)
    print(shlex.join(command), flush=True)
    if args.print_only:
        return 0
    return subprocess.run(command).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devops.runpod.pods.provision",
        description=(
            "Create on-demand Community Cloud RunPod Pods for experiment jobs."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    up = sub.add_parser(
        "up", help="validate and create Pods (default command)"
    )
    up.add_argument("-n", "--count", type=int, default=1)
    up.add_argument(
        "--mode",
        choices=["ondemand", "interruptible"],
        default="ondemand",
    )
    up.add_argument("--bid", type=float)
    up.add_argument("--disk", type=float)
    up.add_argument("--image")
    up.add_argument("--branch")
    up.add_argument("--commit")
    up.add_argument("--experiment-repo")
    up.add_argument("--library-branch")
    up.add_argument("--library-commit")
    up.add_argument("--run", metavar="CMD")
    up.add_argument("--max-price", type=float)
    up.add_argument(
        "--gpu-type",
        choices=CONFIG.GPU_TYPE_IDS,
        help=(
            "exact compatible Pod GPU (default: RTX 4090); unlike Serverless "
            "pools, Pod placement accepts one type per request"
        ),
    )
    up.add_argument("--regions")
    up.add_argument("--offer-id", type=int)
    up.add_argument("--machine-id", type=int)
    up.add_argument(
        "--exclude-machine",
        type=int,
        nargs="+",
        action="extend",
        default=[],
    )
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--yes", action="store_true")
    up.add_argument("--no-open", action="store_true")
    up.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "prepare an SSH-accessible CUDA workspace and keep it alive until "
            "manual destroy or the hard max-age ceiling"
        ),
    )
    up.add_argument(
        "--ssh-key",
        help=(
            "private key path for interactive access (default: "
            "~/.ssh/id_ed25519 or ~/.ssh/id_rsa)"
        ),
    )
    up.add_argument(
        "--self-destruct",
        action="store_true",
        help=(
            "push compact results before teardown; all RunPod jobs terminate "
            "on success and failure regardless of this flag"
        ),
    )
    up.add_argument("--run-name")
    up.add_argument("--results-branch")
    up.add_argument(
        "--teardown-on-error",
        action="store_true",
        help="compatibility flag; RunPod always tears down on error",
    )
    up.add_argument("--max-age", type=float, metavar="HOURS")
    up.add_argument("--forward-b2", action="store_true")

    destroy = sub.add_parser("destroy")
    destroy.add_argument("--all", action="store_true")
    destroy.add_argument("--id", nargs="+")
    destroy.add_argument("--yes", action="store_true")

    reap = sub.add_parser(
        "reap", help="discover and terminate managed orphaned/over-age Pods"
    )
    reap.add_argument("--max-age", type=float, metavar="HOURS")
    reap.add_argument("--yes", action="store_true")

    sub.add_parser("status", help="list managed Pods and collect actual costs")
    inspect = sub.add_parser("inspect", help="show redacted Pod metadata")
    inspect.add_argument("id")
    logs = sub.add_parser("logs", help="read the v2 Pod SSE log stream")
    logs.add_argument("id")
    logs.add_argument("--source", choices=["container", "system"])
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--since")
    logs.add_argument("--follow", action="store_true")
    ssh = sub.add_parser("ssh", help="connect to an interactive Pod")
    ssh.add_argument("id")
    ssh.add_argument("--ssh-key")
    ssh.add_argument("--print-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {
        "up",
        "destroy",
        "reap",
        "status",
        "inspect",
        "logs",
        "ssh",
        "-h",
        "--help",
    }
    if not argv or argv[0] not in known:
        argv = ["up", *argv]
    args = build_parser().parse_args(argv)
    try:
        if args.command == "destroy":
            return cmd_destroy(args, CONFIG)
        if args.command == "reap":
            return cmd_reap(args, CONFIG)
        if args.command == "status":
            return cmd_status(args, CONFIG)
        if args.command == "inspect":
            return cmd_inspect(args, CONFIG)
        if args.command == "logs":
            return cmd_logs(args, CONFIG)
        if args.command == "ssh":
            return cmd_ssh(args, CONFIG)
        return cmd_up(args, CONFIG)
    except (RunPodClientError, ValueError) as error:
        print(redact_sensitive(error, (resolve_api_key(), resolve_github_token())))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
