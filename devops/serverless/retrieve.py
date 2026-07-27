"""Retrieve and verify durable artifacts from B2 manifests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from harness.storage.b2 import B2StorageConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe manifest relative_path: {relative!r}")
    target = root.joinpath(*path.parts).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"manifest path escapes destination: {relative!r}")
    return target


def retrieve_manifest_artifacts(
    manifest: dict[str, Any],
    destination: Path,
    *,
    client=None,
    config: B2StorageConfig | None = None,
) -> list[Path]:
    """Download every manifest entry and verify size and SHA-256."""
    if manifest.get("status") != "completed":
        raise ValueError("artifact manifest is not completed")
    bucket = str(manifest.get("bucket") or "")
    files = manifest.get("files")
    if not bucket or not isinstance(files, list):
        raise ValueError("artifact manifest requires bucket and files")
    resolved = config or B2StorageConfig.from_env()
    if client is None:
        if resolved is None:
            raise ValueError("B2 credentials are not configured")
        client = resolved.s3_client()
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("artifact manifest contains a non-object file")
        relative = str(row.get("relative_path") or "")
        key = str(row.get("key") or "")
        expected_hash = str(row.get("sha256") or "").lower()
        expected_size = int(row.get("size_bytes", -1))
        if not key or len(expected_hash) != 64:
            raise ValueError(f"incomplete manifest entry for {relative!r}")
        target = _safe_target(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(target))
        if target.stat().st_size != expected_size:
            target.unlink(missing_ok=True)
            raise ValueError(f"size mismatch for {relative}")
        actual_hash = _sha256(target)
        if actual_hash != expected_hash:
            target.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {relative}")
        downloaded.append(target)
    return downloaded


def load_manifest(
    *,
    path: Path | None = None,
    key: str | None = None,
    bucket: str | None = None,
    client=None,
    config: B2StorageConfig | None = None,
) -> dict[str, Any]:
    if (path is None) == (key is None):
        raise ValueError("specify exactly one of --manifest or --manifest-key")
    if path is not None:
        payload = json.loads(path.read_text())
    else:
        resolved = config or B2StorageConfig.from_env()
        if client is None:
            if resolved is None:
                raise ValueError("B2 credentials are not configured")
            client = resolved.s3_client()
        resolved_bucket = bucket or (resolved.bucket if resolved else None)
        if not resolved_bucket:
            raise ValueError("--bucket or B2_BUCKET is required")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "remote_artifacts.json"
            client.download_file(resolved_bucket, str(key), str(target))
            payload = json.loads(target.read_text())
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest must be a JSON object")
    return payload
