"""Durable RunPod Pod execution loaded from the exact harness checkout."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from devops.runpod.execution.durability import (
    CANONICAL_MANIFEST_NAME,
    REMOTE_ARTIFACTS_FILENAME,
    upload_compact_results_bundle,
    upload_tree,
    write_canonical_durability_manifest,
)
from devops.runpod.execution.phases import (
    JobReport,
    Phase,
    PhaseStatus,
    TerminalReason,
)
from devops.runpod.execution.publication import publish_compact_results
from harness.storage.b2 import B2StorageConfig


@dataclass(frozen=True)
class BatchOutcome:
    exit_code: int
    cleanup_allowed: bool
    reason: str
    canonical_manifest_key: str | None
    publication_status: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(message: str, log: Callable[[str], None]) -> None:
    log(f"phase={message}")


def _run_argv() -> list[str]:
    raw = os.environ.get("RUNPOD_RUN_ARGV", "")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("RUNPOD_RUN_ARGV must be valid JSON") from error
    if (
        not isinstance(value, list)
        or len(value) < 2
        or not all(isinstance(part, str) and part for part in value)
        or value[0] != "rl-harness"
        or not value[1].startswith("experiments.")
        or not value[1].endswith(".experiment")
    ):
        raise RuntimeError("RUNPOD_RUN_ARGV is not a supported experiment command")
    return value


def _run_paths(
    experiment_repo: Path,
    run_argv: list[str],
    run_name: str,
) -> tuple[Path, Path, Path]:
    experiment_dir = experiment_repo.joinpath(*run_argv[1].split(".")[:-1])
    results_dir = experiment_dir / "results" / run_name
    artifacts_dir = experiment_dir / "artifacts" / run_name
    return experiment_dir, results_dir, artifacts_dir


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is missing or invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _validate_run_outputs(
    results_dir: Path,
    run_name: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    run_manifest = _json_object(results_dir / "run_manifest.json", "run manifest")
    if run_manifest.get("run_id") != run_name:
        raise RuntimeError("run manifest id does not match RUNPOD_RUN_NAME")
    if run_manifest.get("status") not in {"completed", "failed"}:
        raise RuntimeError("run manifest is not terminal")
    remote_summary = run_manifest.get("remote_artifacts")
    if (
        not isinstance(remote_summary, dict)
        or remote_summary.get("status") != "completed"
    ):
        raise RuntimeError("run manifest does not prove completed B2 upload")
    remote_manifest = _json_object(
        results_dir / REMOTE_ARTIFACTS_FILENAME,
        REMOTE_ARTIFACTS_FILENAME,
    )
    if remote_manifest.get("status") != "completed":
        raise RuntimeError("remote artifact manifest is not completed")
    bucket = remote_manifest.get("bucket")
    prefix = remote_manifest.get("prefix")
    files = remote_manifest.get("files")
    if not isinstance(bucket, str) or not bucket:
        raise RuntimeError("remote artifact manifest omitted bucket")
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError("remote artifact manifest omitted prefix")
    if not isinstance(files, list):
        raise RuntimeError("remote artifact manifest omitted files")
    artifact_files = [
        row
        for row in files
        if isinstance(row, dict) and row.get("kind") == "artifact"
    ]
    return run_manifest, remote_manifest, artifact_files


def _write_report(results_dir: Path, report: JobReport) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "runpod_result.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def _compact_run_files(
    experiment_repo: Path,
    results_dir: Path,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name in {CANONICAL_MANIFEST_NAME, REMOTE_ARTIFACTS_FILENAME}:
            continue
        relative = path.relative_to(experiment_repo).as_posix()
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative],
            cwd=experiment_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if ignored.returncode != 0:
            files.append(path)
    return files


def _finalize_durability(
    *,
    results_dir: Path,
    run_name: str,
    report: JobReport,
    provenance: dict[str, Any],
    client: Any,
) -> str:
    run_manifest, remote_manifest, artifact_files = _validate_run_outputs(
        results_dir,
        run_name,
    )
    run_manifest["runpod"] = {
        "pod_id": provenance.get("pod_id"),
        "image_digest": provenance.get("image_digest"),
        "publication_status": report.publication_status.value,
        "terminal_reason": report.terminal_reason.value,
        "workload_success": report.workload_success,
    }
    run_manifest["remote_artifacts"]["canonical_manifest_key"] = (
        report.canonical_manifest_key
    )
    (results_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    _write_report(results_dir, report)
    bucket = str(remote_manifest["bucket"])
    prefix = str(remote_manifest["prefix"]).strip("/")
    compact_files = upload_compact_results_bundle(
        results_dir=results_dir,
        bucket=bucket,
        artifact_prefix=prefix,
        client=client,
    )
    _, canonical_key, _ = write_canonical_durability_manifest(
        results_dir=results_dir,
        bucket=bucket,
        artifact_prefix=prefix,
        artifact_files=artifact_files,
        compact_files=compact_files,
        provenance=provenance,
        client=client,
    )
    return canonical_key


def _upload_mlflow(
    *,
    mlflow_dir: Path,
    run_name: str,
    client: Any,
    bucket: str,
    log: Callable[[str], None],
) -> str | None:
    if not mlflow_dir.exists():
        return None
    prefix = "/".join(
        part
        for part in (
            os.environ.get("B2_PREFIX", "").strip("/"),
            "runpod",
            "mlflow",
            run_name,
        )
        if part
    )
    files = upload_tree(
        local_root=mlflow_dir,
        bucket=bucket,
        key_prefix=prefix,
        client=client,
        kind="mlflow",
    )
    log(f"MLflow metadata uploaded under {prefix}/ ({len(files)} files)")
    return prefix


def _upload_recoverable_bundle(
    *,
    bundle: str | None,
    run_name: str,
    bucket: str,
    artifact_prefix: str,
    client: Any,
) -> str | None:
    if not bundle:
        return None
    root = Path(bundle)
    if not root.is_dir():
        return None
    prefix = f"{artifact_prefix.strip('/')}/recoverable-results/{run_name}"
    upload_tree(
        local_root=root,
        bucket=bucket,
        key_prefix=prefix,
        client=client,
        kind="recoverable_result",
    )
    return prefix


def _start_mlflow(
    *,
    mlflow_dir: Path,
    run_name: str,
    tags: dict[str, str],
) -> str:
    import mlflow

    mlflow_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlflow_dir.as_uri()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("runpod-pods")
    active = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(tags)
    return active.info.run_id


def _finish_mlflow(run_id: str | None, status: str) -> None:
    if not run_id:
        return
    import mlflow

    mlflow.tracking.MlflowClient().set_terminated(run_id, status=status)


def run_batch_job(
    *,
    experiment_repo: Path,
    python: Path,
    mlflow_dir: Path,
    experiment_sha: str,
    library_sha: str,
    log: Callable[[str], None] = print,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> BatchOutcome:
    """Run, durably finalize, publish, and authorize cleanup for one Pod."""
    run_name = os.environ["RUNPOD_RUN_NAME"]
    run_argv = _run_argv()
    _, results_dir, _ = _run_paths(experiment_repo, run_argv, run_name)
    report = JobReport(backend="runpod-pods", run_name=run_name)
    now = _utc_now()
    for phase in (Phase.PREFLIGHT, Phase.PROVISIONING, Phase.BOOTSTRAP):
        report.set_phase(phase, PhaseStatus.SUCCEEDED, at=now)
    report.set_phase(
        Phase.ANALYSIS,
        PhaseStatus.SKIPPED,
        detail="analysis is owned by the experiment command",
        at=now,
    )
    provenance = {
        "run_name": run_name,
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "experiment_sha": experiment_sha,
        "library_sha": library_sha,
        "image_digest": os.environ.get("RUNPOD_IMAGE_DIGEST"),
    }
    report.metadata.update(provenance)
    b2 = B2StorageConfig.from_env()
    if b2 is None:
        raise RuntimeError("B2 durability environment is required")
    client = b2.s3_client()
    mlflow_run_id: str | None = None
    exit_code = 1
    try:
        _log("TRAINING: starting experiment", log)
        report.set_phase(Phase.TRAINING, PhaseStatus.RUNNING, at=_utc_now())
        mlflow_run_id = _start_mlflow(
            mlflow_dir=mlflow_dir,
            run_name=run_name,
            tags={
                "git.commit": experiment_sha,
                "git.experiment_commit": experiment_sha,
                "git.library_commit": library_sha,
                "container.image.digest": str(provenance["image_digest"] or ""),
                "runpod.pod_id": str(provenance["pod_id"] or ""),
                "runpod.cloud": "COMMUNITY",
                "runpod.interruptible": "false",
                "runpod.gpu.requested": os.environ.get("RUNPOD_GPU_TYPE_IDS", ""),
            },
        )
        env = dict(os.environ)
        for key in ("GH_TOKEN", "RUNPOD_API_KEY"):
            env.pop(key, None)
        env["PATH"] = f"{python.parent}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(python.parent.parent)
        env["MLFLOW_ALLOW_FILE_STORE"] = "true"
        env["MLFLOW_TRACKING_URI"] = mlflow_dir.as_uri()
        env["MLFLOW_RUN_ID"] = mlflow_run_id
        completed = runner(
            run_argv,
            cwd=str(experiment_repo),
            env=env,
            check=False,
            text=True,
        )
        exit_code = completed.returncode
        workload_success = exit_code == 0
        report.set_phase(
            Phase.TRAINING,
            PhaseStatus.SUCCEEDED if workload_success else PhaseStatus.FAILED,
            detail=f"experiment exited with status {exit_code}",
            at=_utc_now(),
        )
        report.mark_workload(
            success=workload_success,
            reason=(
                TerminalReason.SUCCESS
                if workload_success
                else TerminalReason.TRAINING_FAILED
            ),
        )
    except Exception as error:
        report.set_phase(
            Phase.TRAINING,
            PhaseStatus.FAILED,
            detail=f"{type(error).__name__}",
            at=_utc_now(),
        )
        report.mark_workload(
            success=False,
            reason=TerminalReason.TRAINING_FAILED,
            detail=type(error).__name__,
        )
    finally:
        try:
            _finish_mlflow(
                mlflow_run_id,
                "FINISHED" if report.workload_success else "FAILED",
            )
        except Exception as error:
            log(f"WARNING: MLflow finalization failed ({type(error).__name__})")

    _log("DURABLE_UPLOAD: validating and refreshing B2 evidence", log)
    report.set_phase(Phase.DURABLE_UPLOAD, PhaseStatus.RUNNING, at=_utc_now())
    try:
        canonical_key = _finalize_durability(
            results_dir=results_dir,
            run_name=run_name,
            report=report,
            provenance=provenance,
            client=client,
        )
        report.canonical_manifest_key = canonical_key
        _upload_mlflow(
            mlflow_dir=mlflow_dir,
            run_name=run_name,
            client=client,
            bucket=b2.bucket,
            log=log,
        )
        report.set_phase(
            Phase.DURABLE_UPLOAD,
            PhaseStatus.SUCCEEDED,
            detail=canonical_key,
            at=_utc_now(),
        )
    except Exception as error:
        report.set_phase(
            Phase.DURABLE_UPLOAD,
            PhaseStatus.FAILED,
            detail=type(error).__name__,
            at=_utc_now(),
        )
        report.terminal_reason = TerminalReason.DURABLE_UPLOAD_FAILED
        _write_report(results_dir, report)
        log(
            "ERROR: required B2 durability failed; preserving Pod until "
            "the hard max-age ceiling"
        )
        return BatchOutcome(
            exit_code=exit_code or 1,
            cleanup_allowed=False,
            reason="durable upload failed",
            canonical_manifest_key=None,
            publication_status="pending",
        )

    _log("RESULTS_PUBLICATION: publishing compact GitHub results", log)
    token = os.environ.get("GH_TOKEN", "").strip()
    if os.environ.get("RUNPOD_PUSH_RESULTS") != "1":
        publication_status = PhaseStatus.FAILED
        publication_detail = "results publication disabled"
        recoverable_bundle = None
    elif not token:
        publication_status = PhaseStatus.FAILED
        publication_detail = "GH_TOKEN missing"
        recoverable_bundle = None
    else:
        try:
            publication = publish_compact_results(
                experiment_repo=experiment_repo,
                remote_url=os.environ["RUNPOD_EXPERIMENT_REPO_URL"],
                branch=os.environ["RUNPOD_RESULTS_BRANCH"],
                commit_message=(
                    f"results: {run_name} "
                    f"(RunPod {os.environ.get('RUNPOD_POD_ID', '?')})"
                ),
                github_token=token,
                bot_name="runpod-bot",
                bot_email="runpod-bot@users.noreply.github.com",
                bundle_files=_compact_run_files(experiment_repo, results_dir),
            )
            publication_status = {
                "succeeded": PhaseStatus.SUCCEEDED,
                "skipped": PhaseStatus.SKIPPED,
                "warning": PhaseStatus.WARNING,
                "failed": PhaseStatus.FAILED,
            }[publication.status]
            publication_detail = publication.detail
            recoverable_bundle = publication.recoverable_bundle
        except Exception as error:
            publication_status = PhaseStatus.FAILED
            publication_detail = f"publication raised {type(error).__name__}"
            recoverable_bundle = None
    remote_manifest = _json_object(
        results_dir / REMOTE_ARTIFACTS_FILENAME,
        REMOTE_ARTIFACTS_FILENAME,
    )
    try:
        recoverable_key = _upload_recoverable_bundle(
            bundle=recoverable_bundle,
            run_name=run_name,
            bucket=str(remote_manifest["bucket"]),
            artifact_prefix=str(remote_manifest["prefix"]),
            client=client,
        )
    except Exception as error:
        publication_status = PhaseStatus.FAILED
        publication_detail = (
            f"{publication_detail}; recoverable upload raised "
            f"{type(error).__name__}"
        )
        recoverable_key = None
    report.mark_publication(
        publication_status,
        detail=publication_detail,
        recoverable_bundle_key=recoverable_key,
    )
    publication_ok = publication_status in {
        PhaseStatus.SUCCEEDED,
        PhaseStatus.SKIPPED,
    }
    if publication_ok:
        report.set_phase(
            Phase.CLEANUP,
            PhaseStatus.RUNNING,
            detail="durability and publication verified",
            at=_utc_now(),
        )
    else:
        report.terminal_reason = TerminalReason.PUBLICATION_FAILED
        report.set_phase(
            Phase.CLEANUP,
            PhaseStatus.SKIPPED,
            detail="Pod preserved because GitHub publication failed",
            at=_utc_now(),
        )
    try:
        report.canonical_manifest_key = _finalize_durability(
            results_dir=results_dir,
            run_name=run_name,
            report=report,
            provenance=provenance,
            client=client,
        )
    except Exception as error:
        report.set_phase(
            Phase.DURABLE_UPLOAD,
            PhaseStatus.FAILED,
            detail=f"final evidence refresh failed: {type(error).__name__}",
            at=_utc_now(),
        )
        report.terminal_reason = TerminalReason.DURABLE_UPLOAD_FAILED
        _write_report(results_dir, report)
        publication_ok = False
    return BatchOutcome(
        exit_code=exit_code,
        cleanup_allowed=publication_ok,
        reason=(
            "job completed" if report.workload_success else "job failed durably"
        ),
        canonical_manifest_key=report.canonical_manifest_key,
        publication_status=report.publication_status.value,
    )
