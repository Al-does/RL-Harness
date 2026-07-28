"""RunPod Serverless worker: clone exact refs, run once, persist, refresh."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import runpod
except ImportError:  # Local validation helpers do not require the worker SDK.
    runpod = None

WORK_ROOT = Path("/workspace/serverless-job")
LIBRARY_DIR = WORK_ROOT / "rl-harness"
EXPERIMENT_DIR = WORK_ROOT / "experiments"
MLFLOW_DIR = WORK_ROOT / "mlruns"
PYTHON = Path("/opt/venv/bin/python")
_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_REQUIRED = {
    "run_argv": list,
    "run_name": str,
    "experiment_repo_url": str,
    "experiment_ref": str,
    "library_repo_url": str,
    "library_ref": str,
    "image_digest": str,
    "ray_version": str,
    "torch_version": str,
    "gymnasium_version": str,
    "push_results": bool,
    "results_branch": str,
}
_SECRET_NAME = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key|access[_-]?key|"
    r"application[_-]?key|private[_-]?key)"
)
_EXPERIMENT_MODULE = re.compile(
    r"^experiments(?:\.[A-Za-z_][A-Za-z0-9_]*)+\.experiment$"
)


def log(message: str) -> None:
    print(
        f"[serverless {time.strftime('%H:%M:%S', time.gmtime())}] {message}",
        flush=True,
    )


def progress(job: dict[str, Any], message: str) -> None:
    log(message)
    if runpod is None:
        raise RuntimeError("runpod worker SDK is unavailable")
    runpod.serverless.progress_update(job, message)


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _validate_repo_url(value: str, name: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or not parsed.path.endswith(".git")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be a credential-free GitHub HTTPS .git URL")


def validate_input(job: object) -> dict[str, Any]:
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise ValueError("job must contain an input object")
    value = job["input"]
    if set(value) != set(_REQUIRED):
        missing = sorted(set(_REQUIRED) - set(value))
        extra = sorted(set(value) - set(_REQUIRED))
        raise ValueError(f"invalid input fields; missing={missing}, extra={extra}")
    for key, expected in _REQUIRED.items():
        if type(value[key]) is not expected:
            raise ValueError(f"{key} has the wrong type")
        if _SECRET_NAME.search(key):
            raise ValueError("secrets are forbidden in job input")
    for key in ("experiment_ref", "library_ref"):
        if not _SHA.fullmatch(value[key]):
            raise ValueError(f"{key} must be an explicit commit SHA")
    if not _DIGEST.fullmatch(value["image_digest"]):
        raise ValueError("image_digest must be sha256:<64 hex>")
    _validate_repo_url(value["experiment_repo_url"], "experiment_repo_url")
    _validate_repo_url(value["library_repo_url"], "library_repo_url")
    argv = value["run_argv"]
    if (
        len(argv) < 2
        or not all(isinstance(part, str) and part for part in argv)
        or argv[0] != "rl-harness"
        or not _EXPERIMENT_MODULE.fullmatch(argv[1])
    ):
        raise ValueError(
            "run_argv must invoke an experiments.*.experiment module"
        )
    if argv.count("--upload-artifacts") != 1:
        raise ValueError("run_argv must include --upload-artifacts exactly once")
    run_ids: list[str] = []
    for index, part in enumerate(argv):
        if part == "--run-id":
            if index + 1 >= len(argv):
                raise ValueError("--run-id requires a value")
            run_ids.append(argv[index + 1])
        elif part.startswith("--run-id="):
            run_ids.append(part.partition("=")[2])
    if run_ids != [value["run_name"]]:
        raise ValueError("run_argv --run-id must equal run_name")
    if not value["run_name"].strip() or len(value["run_name"]) > 200:
        raise ValueError("run_name must be non-empty and at most 200 characters")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value["results_branch"]):
        raise ValueError("results_branch contains unsafe characters")
    return dict(value)


def clean_workspace() -> None:
    """Start each job from an empty ephemeral workspace."""
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True, exist_ok=False)


def git_auth_env() -> dict[str, str]:
    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GH_TOKEN is required")
    encoded = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def experiment_env(mlflow_run_id: str) -> dict[str, str]:
    """Keep B2 durability credentials but remove GitHub/control credentials."""
    env = dict(os.environ)
    for key in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "RUNPOD_API_KEY",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    ):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"/opt/venv/bin:{env.get('PATH', '')}",
            "VIRTUAL_ENV": "/opt/venv",
            "MLFLOW_ALLOW_FILE_STORE": "true",
            "MLFLOW_TRACKING_URI": f"file:{MLFLOW_DIR}",
            "MLFLOW_RUN_ID": mlflow_run_id,
        }
    )
    return env


def checkout(url: str, ref: str, target: Path) -> str:
    env = git_auth_env()
    run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)], env=env)
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=target, env=env)
    run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=target)
    result = run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        capture_output=True,
    )
    actual = result.stdout.strip().lower()
    if actual != ref.lower():
        raise RuntimeError("checkout did not resolve to the requested commit SHA")
    # Ensure credentials were never embedded in the persisted remote.
    remote = run(
        ["git", "remote", "get-url", "origin"],
        cwd=target,
        capture_output=True,
    ).stdout.strip()
    if "@" in urlparse(remote).netloc:
        raise RuntimeError("credential-bearing git remote refused")
    return actual


def install_sources() -> None:
    run(
        [
            str(Path("/root/.local/bin/uv")),
            "pip",
            "install",
            "--python",
            str(PYTHON),
            "--no-deps",
            "-e",
            str(LIBRARY_DIR),
            "-e",
            str(EXPERIMENT_DIR),
        ]
    )


def validate_runtime(spec: dict[str, Any]) -> dict[str, str]:
    import gymnasium
    import ray
    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    if ray.__version__ != spec["ray_version"]:
        raise RuntimeError("Ray version mismatch")
    if torch_version != spec["torch_version"]:
        raise RuntimeError("Torch version mismatch")
    if gymnasium.__version__ != spec["gymnasium_version"]:
        raise RuntimeError("Gymnasium version mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    cuda_version = str(torch.version.cuda or "")
    if not cuda_version.startswith("13."):
        raise RuntimeError("CUDA 13 runtime is required")
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": cuda_version,
        "ray_version": ray.__version__,
        "torch_version": torch_version,
        "gymnasium_version": gymnasium.__version__,
    }


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is missing or invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _results_directory(
    experiment_dir: Path,
    run_argv: list[str],
    run_name: str,
) -> Path:
    module_dir = experiment_dir.joinpath(*run_argv[1].split(".")[:-1])
    if "--smoke" in run_argv:
        return module_dir / ".smoke" / run_name / "results"
    return module_dir / "results" / run_name


def _training_iterations(path: Path) -> list[float]:
    """Read valid positive training iterations from a JSON-lines file."""
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    iterations: list[float] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if (
                (key == "training_iteration" or key.endswith("/training_iteration"))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
            ):
                iterations.append(float(value))
    return iterations


def _runtime_artifacts_directory(
    experiment_dir: Path,
    run_manifest: dict[str, Any],
) -> Path:
    runtime = run_manifest.get("runtime")
    raw_path = runtime.get("artifacts_dir") if isinstance(runtime, dict) else None
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("run manifest runtime omitted artifacts_dir")
    repository = experiment_dir.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    try:
        artifacts_dir = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            "runtime artifacts_dir is not an existing directory"
        ) from error
    if not artifacts_dir.is_dir():
        raise RuntimeError("runtime artifacts_dir is not an existing directory")
    try:
        artifacts_dir.relative_to(repository)
    except ValueError as error:
        raise RuntimeError(
            "runtime artifacts_dir escapes the experiment repository"
        ) from error
    return artifacts_dir


def _manifest_represents_path(
    files: list[object],
    artifact_relative_path: str,
) -> bool:
    for row in files:
        if not isinstance(row, dict):
            continue
        relative_path = row.get("relative_path")
        key = row.get("key")
        if relative_path == artifact_relative_path:
            return True
        if isinstance(key, str) and (
            key == artifact_relative_path
            or key.endswith(f"/{artifact_relative_path}")
        ):
            return True
    return False


def validate_run_outputs(
    experiment_dir: Path,
    run_argv: list[str],
    run_name: str,
) -> dict[str, Any]:
    """Validate local completion, durable artifacts, and training evidence."""
    results_dir = _results_directory(experiment_dir, run_argv, run_name)
    run_manifest_path = results_dir / "run_manifest.json"
    remote_manifest_path = results_dir / "remote_artifacts.json"
    progress_path = results_dir / "progress.jsonl"
    run_manifest = _json_object(run_manifest_path, "run_manifest.json")
    remote_manifest = _json_object(
        remote_manifest_path, "remote_artifacts.json"
    )
    if run_manifest.get("status") != "completed":
        raise RuntimeError("run manifest does not prove completed status")
    if run_manifest.get("run_id") != run_name:
        raise RuntimeError("run manifest run_id does not match run_name")
    if remote_manifest.get("status") != "completed":
        raise RuntimeError("remote artifact manifest is not completed")
    files = remote_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("remote artifact manifest has no uploaded files")
    checkpoint_keys: list[str] = []
    for row in files:
        if not isinstance(row, dict):
            raise RuntimeError("remote artifact manifest file is not an object")
        key = row.get("key")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise RuntimeError("remote artifact manifest file is incomplete")
        relative = str(row.get("relative_path") or key).lower()
        if "checkpoint" in relative or relative.endswith(
            (".pt", ".pth", ".ckpt", ".pkl")
        ):
            checkpoint_keys.append(key)
    if not checkpoint_keys:
        raise RuntimeError("no checkpoint-like uploaded artifact was recorded")
    training_iterations = _training_iterations(progress_path)
    unrepresented_tune_evidence = False
    if not training_iterations:
        artifacts_dir = _runtime_artifacts_directory(experiment_dir, run_manifest)
        for result_path in sorted(artifacts_dir.glob("tune/**/result.json")):
            try:
                resolved_result = result_path.resolve(strict=True)
                artifact_relative_path = resolved_result.relative_to(
                    artifacts_dir
                ).as_posix()
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "Tune result.json escapes runtime artifacts_dir"
                ) from error
            result_iterations = _training_iterations(resolved_result)
            if not result_iterations:
                continue
            if _manifest_represents_path(files, artifact_relative_path):
                training_iterations.extend(result_iterations)
            else:
                unrepresented_tune_evidence = True
    if not training_iterations:
        if unrepresented_tune_evidence:
            raise RuntimeError(
                "Tune result.json with positive training_iteration is missing "
                "from remote artifact manifest"
            )
        raise RuntimeError(
            "no valid positive training_iteration in progress.jsonl or "
            "uploaded Tune result.json"
        )
    bucket = remote_manifest.get("bucket")
    prefix = remote_manifest.get("prefix")
    if not isinstance(bucket, str) or not bucket:
        raise RuntimeError("remote artifact manifest omitted bucket")
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError("remote artifact manifest omitted prefix")
    return {
        "results_dir": results_dir,
        "run_manifest_path": run_manifest_path,
        "remote_manifest_path": remote_manifest_path,
        "bucket": bucket,
        "prefix": prefix.strip("/"),
        "artifact_file_count": len(files),
        # Keep provider output compact; the uploaded manifest retains all keys.
        "checkpoint_keys": checkpoint_keys[:20],
        "training_iteration": max(training_iterations),
    }


def _b2_client():
    required = (
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
    )
    if not all(os.environ.get(key) for key in required):
        raise RuntimeError("B2 is required for Serverless jobs")
    import boto3
    from botocore.config import Config

    endpoint = os.environ["B2_ENDPOINT"]
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=Config(signature_version="s3v4"),
    )


def _ensure_library_on_sys_path() -> None:
    """Editable installs can omit namespace packages in some worker envs."""
    root = str(LIBRARY_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def write_and_upload_serverless_result(
    evidence: dict[str, Any],
    result: dict[str, Any],
    *,
    client=None,
) -> tuple[str, str, str | None]:
    """Upload manifests and compact result; return durable metadata keys."""
    _ensure_library_on_sys_path()
    from devops.runpod.execution.durability import (
        CANONICAL_MANIFEST_NAME,
        upload_compact_results_bundle,
        write_canonical_durability_manifest,
    )

    results_dir = Path(evidence["results_dir"])
    result_path = results_dir / "serverless_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    prefix = str(evidence["prefix"]).strip("/")
    remote_manifest_key = f"{prefix}/metadata/remote_artifacts.json"
    serverless_result_key = f"{prefix}/metadata/serverless_result.json"
    s3 = client or _b2_client()
    bucket = str(evidence["bucket"])
    remote_manifest = _json_object(
        Path(evidence["remote_manifest_path"]),
        "remote_artifacts.json",
    )
    # Keep only training/artifact rows from the harness manifest. Compact
    # results are re-uploaded below so the canonical manifest has one final
    # hash per object (avoids stale size/sha mismatches on retrieve).
    artifact_files = [
        row
        for row in remote_manifest.get("files", [])
        if isinstance(row, dict) and row.get("kind") != "compact_result"
    ]
    compact_files = upload_compact_results_bundle(
        results_dir=results_dir,
        bucket=bucket,
        artifact_prefix=prefix,
        client=s3,
    )
    _, canonical_key, _ = write_canonical_durability_manifest(
        results_dir=results_dir,
        bucket=bucket,
        artifact_prefix=prefix,
        artifact_files=artifact_files,
        compact_files=compact_files,
        provenance={
            "run_name": result.get("run_name"),
            "experiment_sha": result.get("experiment_sha"),
            "library_sha": result.get("library_sha"),
            "image_digest": result.get("image_digest"),
        },
        client=s3,
    )
    s3.upload_file(
        str(evidence["remote_manifest_path"]),
        bucket,
        remote_manifest_key,
    )
    s3.upload_file(str(result_path), bucket, serverless_result_key)
    # Prefer the canonical key name from the durability helper.
    if not canonical_key:
        canonical_key = f"{prefix}/metadata/{CANONICAL_MANIFEST_NAME}"
    return remote_manifest_key, serverless_result_key, canonical_key


def start_mlflow(
    spec: dict[str, Any],
    *,
    job_id: str,
    experiment_sha: str,
    library_sha: str,
    runtime: dict[str, str],
) -> str:
    import mlflow

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("runpod-serverless")
    active = mlflow.start_run(run_name=spec["run_name"])
    mlflow.set_tags(
        {
            "git.commit": experiment_sha,
            "git.experiment_commit": experiment_sha,
            "git.library_commit": library_sha,
            "container.image.digest": spec["image_digest"],
            "runpod.serverless.endpoint_id": os.environ.get(
                "RUNPOD_ENDPOINT_ID", ""
            ),
            "runpod.serverless.job_id": job_id,
            "runpod.serverless.worker_id": os.environ.get("RUNPOD_POD_ID", ""),
            "runpod.gpu.actual": runtime["gpu_name"],
            "runtime.cuda": runtime["cuda_version"],
            "runtime.ray": runtime["ray_version"],
            "runtime.torch": runtime["torch_version"],
            "runtime.gymnasium": runtime["gymnasium_version"],
        }
    )
    return active.info.run_id


def finish_mlflow(run_id: str, status: str, *, strict: bool = False) -> None:
    try:
        import mlflow

        mlflow.tracking.MlflowClient().set_terminated(run_id, status=status)
    except Exception as error:  # noqa: BLE001
        if strict:
            raise RuntimeError("MLflow finalization failed") from error
        log(f"WARNING: MLflow finalization failed ({type(error).__name__})")


def upload_mlflow(run_name: str) -> str:
    if not MLFLOW_DIR.exists() or not any(MLFLOW_DIR.rglob("*")):
        raise RuntimeError("MLflow directory is missing or empty")
    client = _b2_client()
    prefix = "/".join(
        part
        for part in (
            os.environ.get("B2_PREFIX", "").strip("/"),
            "serverless",
            "mlflow",
            run_name,
        )
        if part
    )
    for path in sorted(MLFLOW_DIR.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(MLFLOW_DIR).as_posix()}"
            client.upload_file(str(path), os.environ["B2_BUCKET"], key)
    log(f"MLflow metadata uploaded under {prefix}/")
    return prefix


def push_results(spec: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Publish compact results without affecting workload success.

    Uses a clean results-branch worktree overlay. Never rebases experiment
    history onto the results branch. Returns a status payload; failures are
    warnings with a recoverable bundle path.
    """
    if not spec["push_results"]:
        return {
            "publication_status": "skipped",
            "publication_detail": "push_results disabled",
            "recoverable_bundle_key": None,
        }
    _ensure_library_on_sys_path()
    from devops.runpod.execution.publication import publish_compact_results

    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        return {
            "publication_status": "failed",
            "publication_detail": "GH_TOKEN missing for results publication",
            "recoverable_bundle_key": None,
        }
    result = publish_compact_results(
        experiment_repo=EXPERIMENT_DIR,
        remote_url=spec["experiment_repo_url"],
        branch=spec["results_branch"],
        commit_message=f"results: {spec['run_name']} (Serverless {job_id})",
        github_token=token,
        bot_name="runpod-serverless-bot",
        bot_email="runpod-serverless-bot@users.noreply.github.com",
    )
    log(f"results publication status={result.status}: {result.detail}")
    recoverable_key = None
    if result.recoverable_bundle and result.status in {"failed", "warning"}:
        # Keep a durable copy under the run metadata prefix when possible.
        try:
            prefix = os.environ.get("B2_PREFIX", "").strip("/")
            key = "/".join(
                part
                for part in (
                    prefix,
                    "serverless",
                    "recoverable-results",
                    spec["run_name"],
                    "bundle-marker.txt",
                )
                if part
            )
            marker = Path(result.recoverable_bundle) / "BUNDLE_PATH.txt"
            marker.write_text(result.recoverable_bundle + "\n")
            _b2_client().upload_file(
                str(marker),
                os.environ["B2_BUCKET"],
                key,
            )
            recoverable_key = key
        except Exception as error:  # noqa: BLE001
            log(
                "WARNING: could not upload recoverable publication bundle "
                f"({type(error).__name__})"
            )
    return {
        "publication_status": result.status,
        "publication_detail": result.detail,
        "recoverable_bundle_key": recoverable_key,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Process one validated experiment job; endpoint policy allows only one."""
    started = time.monotonic()
    spec = validate_input(job)
    job_id = str(job.get("id") or "")
    if not job_id:
        raise ValueError("job id is required")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if not endpoint_id:
        raise RuntimeError("RUNPOD_ENDPOINT_ID is required")
    if not all(
        os.environ.get(key)
        for key in (
            "B2_BUCKET",
            "B2_ENDPOINT",
            "B2_APPLICATION_KEY_ID",
            "B2_APPLICATION_KEY",
        )
    ):
        raise RuntimeError("B2 durability environment is required")
    clean_workspace()
    progress(job, "phase=BOOTSTRAP: workspace cleaned; checking out exact commits")
    experiment_sha = ""
    library_sha = ""
    mlflow_run_id: str | None = None
    runtime: dict[str, str] = {}
    try:
        library_sha = checkout(
            spec["library_repo_url"], spec["library_ref"], LIBRARY_DIR
        )
        experiment_sha = checkout(
            spec["experiment_repo_url"], spec["experiment_ref"], EXPERIMENT_DIR
        )
        progress(job, "phase=BOOTSTRAP: exact commits checked out; installing")
        install_sources()
        runtime = validate_runtime(spec)
        progress(
            job,
            "phase=TRAINING: "
            f"runtime validated on {runtime['gpu_name']}; starting experiment",
        )
        mlflow_run_id = start_mlflow(
            spec,
            job_id=job_id,
            experiment_sha=experiment_sha,
            library_sha=library_sha,
            runtime=runtime,
        )
        completed = run(
            spec["run_argv"],
            cwd=EXPERIMENT_DIR,
            env=experiment_env(mlflow_run_id),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"experiment command exited with status {completed.returncode}"
            )
        progress(job, "phase=DURABLE_UPLOAD: validating and uploading evidence")
        evidence = validate_run_outputs(
            EXPERIMENT_DIR,
            spec["run_argv"],
            spec["run_name"],
        )
        finish_mlflow(mlflow_run_id, "FINISHED", strict=True)
        mlflow_prefix = upload_mlflow(spec["run_name"])
        prefix = str(evidence["prefix"])
        remote_manifest_key = f"{prefix}/metadata/remote_artifacts.json"
        serverless_result_key = f"{prefix}/metadata/serverless_result.json"
        result = {
            "validation_status": "completed",
            "run_name": spec["run_name"],
            "experiment_sha": experiment_sha,
            "library_sha": library_sha,
            "image_digest": spec["image_digest"],
            "endpoint_id": endpoint_id,
            "job_id": job_id,
            "gpu_name": runtime["gpu_name"],
            "cuda_version": runtime["cuda_version"],
            "ray_version": runtime["ray_version"],
            "torch_version": runtime["torch_version"],
            "gymnasium_version": runtime["gymnasium_version"],
            "training_iteration": evidence["training_iteration"],
            "artifact_file_count": evidence["artifact_file_count"],
            "checkpoint_keys": evidence["checkpoint_keys"],
            "remote_manifest_key": remote_manifest_key,
            "serverless_result_key": serverless_result_key,
            "mlflow_prefix": mlflow_prefix,
            "workload_success": True,
            "terminal_reason": "success",
        }
        (
            remote_manifest_key,
            serverless_result_key,
            canonical_manifest_key,
        ) = write_and_upload_serverless_result(evidence, result)
        result["remote_manifest_key"] = remote_manifest_key
        result["serverless_result_key"] = serverless_result_key
        result["canonical_manifest_key"] = canonical_manifest_key
        # Publication is best-effort and never flips workload_success.
        progress(job, "phase=RESULTS_PUBLICATION: publishing compact results")
        publication = push_results(spec, job_id)
        result.update(publication)
        if publication["publication_status"] in {"failed", "warning"}:
            # Re-upload result JSON so publication failure is durable too.
            write_and_upload_serverless_result(evidence, result)
        elapsed = time.monotonic() - started
        progress(
            job,
            "phase=CLEANUP: durable evidence verified; requesting worker refresh",
        )
        return {
            "refresh_worker": True,
            "status": "completed",
            **result,
            "elapsed_seconds": round(elapsed, 3),
        }
    except BaseException:
        if mlflow_run_id:
            finish_mlflow(mlflow_run_id, "FAILED")
            try:
                upload_mlflow(spec["run_name"])
            except Exception as error:  # noqa: BLE001
                log(
                    "WARNING: failure-state MLflow upload failed "
                    f"({type(error).__name__})"
                )
        raise


def main() -> None:
    if runpod is None:
        raise RuntimeError("runpod worker SDK is unavailable")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
