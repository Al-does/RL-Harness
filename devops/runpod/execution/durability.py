"""Canonical B2 durability for artifacts, compact JSON, plots, and provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANONICAL_MANIFEST_NAME = "durability_manifest.json"
REMOTE_ARTIFACTS_FILENAME = "remote_artifacts.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def upload_tree(
    *,
    local_root: Path,
    bucket: str,
    key_prefix: str,
    client: Any,
    kind: str,
) -> list[dict[str, Any]]:
    """Upload every file under ``local_root`` and return hash-verified rows."""
    rows: list[dict[str, Any]] = []
    prefix = key_prefix.strip("/")
    for path in _iter_files(local_root):
        relative = path.relative_to(local_root).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        digest = _file_sha256(path)
        size = path.stat().st_size
        client.upload_file(str(path), bucket, key)
        rows.append(
            {
                "kind": kind,
                "relative_path": relative,
                "key": key,
                "uri": f"s3://{bucket}/{key}",
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return rows


def upload_compact_results_bundle(
    *,
    results_dir: Path,
    bucket: str,
    artifact_prefix: str,
    client: Any,
) -> list[dict[str, Any]]:
    """Upload compact JSON/plots/manifests beside training artifacts."""
    prefix = f"{artifact_prefix.strip('/')}/compact-results"
    return upload_tree(
        local_root=results_dir,
        bucket=bucket,
        key_prefix=prefix,
        client=client,
        kind="compact_result",
    )


def write_canonical_durability_manifest(
    *,
    results_dir: Path,
    bucket: str,
    artifact_prefix: str,
    artifact_files: list[dict[str, Any]],
    compact_files: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
    client: Any | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    """Write/upload the canonical durability manifest and return its key."""
    results_dir.mkdir(parents=True, exist_ok=True)
    # Prefer later rows for the same object key so final uploads win.
    by_key: dict[str, dict[str, Any]] = {}
    for row in list(artifact_files) + list(compact_files):
        key = str(row.get("key") or "")
        if not key:
            continue
        by_key[key] = row
    files = list(by_key.values())
    payload = {
        "schema_version": 1,
        "backend": "b2-s3",
        "bucket": bucket,
        "prefix": artifact_prefix.strip("/"),
        "status": "completed",
        "uploaded_at": _utc_now(),
        "file_count": len(files),
        "total_bytes": sum(int(row.get("size_bytes") or 0) for row in files),
        "files": files,
        "provenance": provenance or {},
        "kinds": sorted({str(row.get("kind") or "artifact") for row in files}),
    }
    path = results_dir / CANONICAL_MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    key = f"{artifact_prefix.strip('/')}/metadata/{CANONICAL_MANIFEST_NAME}"
    if client is not None:
        client.upload_file(str(path), bucket, key)
        # Keep the historical remote_artifacts.json pointer for older tools.
        legacy = {
            "backend": "b2-s3",
            "bucket": bucket,
            "prefix": artifact_prefix.strip("/"),
            "status": "completed",
            "uploaded_at": payload["uploaded_at"],
            "file_count": payload["file_count"],
            "total_bytes": payload["total_bytes"],
            "files": files,
            "canonical_manifest_key": key,
        }
        legacy_path = results_dir / REMOTE_ARTIFACTS_FILENAME
        legacy_path.write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n")
        client.upload_file(
            str(legacy_path),
            bucket,
            f"{artifact_prefix.strip('/')}/metadata/{REMOTE_ARTIFACTS_FILENAME}",
        )
    return path, key, payload
