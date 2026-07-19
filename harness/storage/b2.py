"""Backblaze B2 artifact upload via the S3-compatible API."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.context import RunContext

REMOTE_ARTIFACTS_FILENAME = "remote_artifacts.json"
B2_ENV_KEYS = (
    "B2_BUCKET",
    "B2_ENDPOINT",
    "B2_APPLICATION_KEY_ID",
    "B2_APPLICATION_KEY",
    "B2_PREFIX",
)
DEFAULT_B2_ENV_FILE = Path("~/.rl_harness_b2_env")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _object_prefix(
    context: RunContext,
    *,
    base_prefix: str,
    experiment_module: str | None,
) -> str:
    """Build a stable, hierarchical object key prefix for one run."""
    segments: list[str] = []
    if base_prefix:
        segments.append(base_prefix.strip("/"))
    repository = _repository_root(context.experiment_dir)
    if repository is not None:
        try:
            relative = context.experiment_dir.resolve().relative_to(
                repository.resolve()
            )
            segments.append(relative.as_posix())
        except ValueError:
            segments.append(context.experiment_dir.name)
    elif experiment_module:
        segments.append(experiment_module.replace(".", "/"))
    else:
        segments.append(context.experiment_dir.name)
    segments.append(context.run_id)
    return "/".join(segment for segment in segments if segment)


def _iter_artifact_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    files.sort()
    return files


def _normalize_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    return normalized


def b2_env_file_path() -> Path:
    return Path(
        os.environ.get("RL_HARNESS_B2_ENV_FILE", str(DEFAULT_B2_ENV_FILE))
    ).expanduser()


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a shell-style KEY=VALUE secrets file."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if key in B2_ENV_KEYS:
            values[key] = value
    return values


def load_b2_settings(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load B2 settings from ``~/.rl_harness_b2_env`` plus process env."""
    settings = parse_env_file(b2_env_file_path())
    source = dict(environ) if environ is not None else dict(os.environ)
    for key in B2_ENV_KEYS:
        value = source.get(key)
        if value:
            settings[key] = value
    access_key_id = settings.get("B2_APPLICATION_KEY_ID") or source.get(
        "AWS_ACCESS_KEY_ID"
    )
    secret_access_key = settings.get("B2_APPLICATION_KEY") or source.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    if access_key_id:
        settings["B2_APPLICATION_KEY_ID"] = access_key_id
    if secret_access_key:
        settings["B2_APPLICATION_KEY"] = secret_access_key
    return settings


def b2_env_for_remote(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return B2 env vars to inject onto remote training boxes."""
    settings = load_b2_settings(environ)
    required = (
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
    )
    if not all(settings.get(key) for key in required):
        return {}
    return {key: settings[key] for key in B2_ENV_KEYS if settings.get(key)}


@dataclass(frozen=True, slots=True)
class B2StorageConfig:
    """Credentials and bucket settings for B2's S3-compatible endpoint."""

    bucket: str
    endpoint: str
    access_key_id: str
    secret_access_key: str
    prefix: str = ""

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, str] | None = None,
    ) -> B2StorageConfig | None:
        resolved = load_b2_settings() if settings is None else dict(settings)
        bucket = resolved.get("B2_BUCKET")
        endpoint = resolved.get("B2_ENDPOINT")
        access_key_id = resolved.get("B2_APPLICATION_KEY_ID")
        secret_access_key = resolved.get("B2_APPLICATION_KEY")
        if not all([bucket, endpoint, access_key_id, secret_access_key]):
            return None
        return cls(
            bucket=bucket,
            endpoint=_normalize_endpoint(endpoint),
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            prefix=resolved.get("B2_PREFIX", "").strip("/"),
        )

    @classmethod
    def from_env(cls) -> B2StorageConfig | None:
        return cls.from_settings()

    def s3_client(self):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError(
                "boto3 is required for artifact upload; install with "
                "`uv sync --extra storage` in rl-harness or add boto3 to your "
                "experiment environment."
            ) from error
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version="s3v4"),
        )


def is_b2_configured() -> bool:
    return B2StorageConfig.from_env() is not None


def upload_run_artifacts(
    context: RunContext,
    *,
    config: B2StorageConfig | None = None,
    experiment_module: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Upload ``context.artifacts_dir`` to B2 and return a compact manifest."""
    resolved = config or B2StorageConfig.from_env()
    if resolved is None:
        raise RuntimeError(
            "B2 artifact upload is not configured. Set B2_BUCKET, B2_ENDPOINT, "
            "B2_APPLICATION_KEY_ID, and B2_APPLICATION_KEY."
        )

    prefix = _object_prefix(
        context,
        base_prefix=resolved.prefix,
        experiment_module=experiment_module,
    )
    base_uri = f"s3://{resolved.bucket}/{prefix}/"
    started_at = _utc_now()
    files: list[dict[str, Any]] = []
    total_bytes = 0
    s3 = client or resolved.s3_client()

    for path in _iter_artifact_files(context.artifacts_dir):
        relative_path = path.relative_to(context.artifacts_dir).as_posix()
        key = f"{prefix}/{relative_path}"
        size_bytes = path.stat().st_size
        digest = _file_sha256(path)
        s3.upload_file(str(path), resolved.bucket, key)
        files.append(
            {
                "relative_path": relative_path,
                "key": key,
                "uri": f"s3://{resolved.bucket}/{key}",
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        total_bytes += size_bytes

    finished_at = _utc_now()
    payload: dict[str, Any] = {
        "backend": "b2-s3",
        "bucket": resolved.bucket,
        "endpoint": resolved.endpoint,
        "prefix": prefix,
        "base_uri": base_uri,
        "status": "completed",
        "started_at": started_at,
        "uploaded_at": finished_at,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    context.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = context.results_dir / REMOTE_ARTIFACTS_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "backend": payload["backend"],
        "bucket": payload["bucket"],
        "endpoint": payload["endpoint"],
        "prefix": payload["prefix"],
        "base_uri": payload["base_uri"],
        "status": payload["status"],
        "started_at": payload["started_at"],
        "uploaded_at": payload["uploaded_at"],
        "file_count": payload["file_count"],
        "total_bytes": payload["total_bytes"],
        "manifest_file": REMOTE_ARTIFACTS_FILENAME,
    }
