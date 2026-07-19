"""Tests for optional Backblaze B2 artifact upload."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from harness.artifacts import maybe_upload_run_artifacts, start_run_manifest
from harness.context import RunContext
from harness.storage.b2 import (
    B2StorageConfig,
    B2_ENV_KEYS,
    REMOTE_ARTIFACTS_FILENAME,
    b2_env_for_remote,
    is_b2_configured,
    load_b2_settings,
    parse_env_file,
    upload_run_artifacts,
)


@pytest.fixture
def isolated_b2_env(monkeypatch, tmp_path):
    """Ignore developer ~/.rl_harness_b2_env during tests."""
    monkeypatch.setenv("RL_HARNESS_B2_ENV_FILE", str(tmp_path / "missing-b2.env"))
    for key in (*B2_ENV_KEYS, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key, raising=False)


def make_context(tmp_path: Path, **overrides) -> RunContext:
    values = {
        "experiment_dir": tmp_path / "experiments" / "study" / "condition",
        "results_dir": tmp_path / "experiments" / "study" / "condition" / "results" / "run",
        "artifacts_dir": tmp_path / "experiments" / "study" / "condition" / "artifacts" / "run",
        "run_id": "run",
    }
    values.update(overrides)
    return RunContext(**values)


def test_is_b2_configured_reads_env(isolated_b2_env, monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    assert is_b2_configured() is False

    monkeypatch.setenv("B2_BUCKET", "bucket")
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    assert is_b2_configured() is True


def test_upload_run_artifacts_writes_manifest_and_uploads(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    experiment_dir = repo / "experiments" / "study" / "condition"
    results_dir = experiment_dir / "results" / "run-id"
    artifacts_dir = experiment_dir / "artifacts" / "run-id"
    checkpoint = artifacts_dir / "checkpoints" / "module_state_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    context = make_context(
        tmp_path,
        experiment_dir=experiment_dir,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id="run-id",
    )

    client = MagicMock()
    monkeypatch.setattr(
        "harness.storage.b2.B2StorageConfig.s3_client",
        lambda self: client,
    )

    config = B2StorageConfig(
        bucket="alex-rl-artifacts",
        endpoint="https://s3.us-west-004.backblazeb2.com",
        access_key_id="key-id",
        secret_access_key="secret",
        prefix="dev",
    )
    summary = upload_run_artifacts(
        context,
        config=config,
        experiment_module="experiments.study.condition.experiment",
        client=client,
    )

    client.upload_file.assert_called_once()
    uploaded_path, bucket, key = client.upload_file.call_args.args
    assert uploaded_path == str(checkpoint)
    assert bucket == "alex-rl-artifacts"
    assert key == (
        "dev/experiments/study/condition/run-id/checkpoints/module_state_final.pt"
    )
    assert summary["status"] == "completed"
    assert summary["file_count"] == 1
    assert summary["manifest_file"] == REMOTE_ARTIFACTS_FILENAME

    remote_manifest = json.loads(
        (results_dir / REMOTE_ARTIFACTS_FILENAME).read_text()
    )
    assert remote_manifest["files"][0]["uri"].startswith("s3://alex-rl-artifacts/")
    assert remote_manifest["files"][0]["sha256"]


def test_maybe_upload_run_artifacts_skips_when_not_configured(
    isolated_b2_env, tmp_path, monkeypatch
):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    context = make_context(tmp_path)
    source = context.experiment_dir / "experiment.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run(context):\n    return None\n")
    start_run_manifest(
        context,
        experiment_module="experiments.study.condition.experiment",
        experiment_file=source,
    )

    assert maybe_upload_run_artifacts(context) is None
    manifest = json.loads((context.results_dir / "run_manifest.json").read_text())
    assert "remote_artifacts" not in manifest


def test_maybe_upload_run_artifacts_records_manifest_summary(
    isolated_b2_env, tmp_path, monkeypatch
):
    monkeypatch.setenv("B2_BUCKET", "bucket")
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")

    context = make_context(tmp_path)
    source = context.experiment_dir / "experiment.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run(context):\n    return None\n")
    start_run_manifest(
        context,
        experiment_module="experiments.study.condition.experiment",
        experiment_file=source,
    )

    monkeypatch.setattr(
        "harness.storage.b2.upload_run_artifacts",
        lambda *args, **kwargs: {
            "backend": "b2-s3",
            "bucket": "bucket",
            "endpoint": "https://s3.us-west-004.backblazeb2.com",
            "prefix": "experiments/study/condition/run",
            "base_uri": "s3://bucket/experiments/study/condition/run/",
            "status": "completed",
            "started_at": "t0",
            "uploaded_at": "t1",
            "file_count": 1,
            "total_bytes": 7,
            "manifest_file": REMOTE_ARTIFACTS_FILENAME,
        },
    )

    summary = maybe_upload_run_artifacts(context)
    assert summary is not None
    manifest = json.loads((context.results_dir / "run_manifest.json").read_text())
    assert manifest["remote_artifacts"]["base_uri"].startswith("s3://bucket/")


def test_maybe_upload_run_artifacts_requires_config_when_forced(
    isolated_b2_env, tmp_path
):
    context = make_context(tmp_path)
    with pytest.raises(RuntimeError, match="not configured"):
        maybe_upload_run_artifacts(context, upload=True)


def test_parse_env_file_reads_exported_values(tmp_path):
    path = tmp_path / "b2.env"
    path.write_text(
        "\n".join(
            [
                "# comment",
                'export B2_BUCKET="my-bucket"',
                "export B2_ENDPOINT=https://s3.us-west-004.backblazeb2.com",
                "export B2_APPLICATION_KEY_ID=004abc",
                "export B2_APPLICATION_KEY=K004secret",
                "export B2_PREFIX=dev",
            ]
        )
    )
    values = parse_env_file(path)
    assert values["B2_BUCKET"] == "my-bucket"
    assert values["B2_PREFIX"] == "dev"


def test_b2_storage_config_normalizes_endpoint_without_scheme():
    config = B2StorageConfig.from_settings(
        {
            "B2_BUCKET": "bucket",
            "B2_ENDPOINT": "s3.us-west-004.backblazeb2.com",
            "B2_APPLICATION_KEY_ID": "key-id",
            "B2_APPLICATION_KEY": "secret",
        }
    )
    assert config is not None
    assert config.endpoint == "https://s3.us-west-004.backblazeb2.com"


def test_load_b2_settings_normalizes_endpoint_from_secrets_file(
    isolated_b2_env, tmp_path, monkeypatch
):
    env_file = tmp_path / "b2.env"
    env_file.write_text(
        "\n".join(
            [
                "export B2_BUCKET=bucket",
                "export B2_ENDPOINT=s3.us-west-004.backblazeb2.com",
                "export B2_APPLICATION_KEY_ID=key-id",
                "export B2_APPLICATION_KEY=secret",
            ]
        )
    )
    monkeypatch.setenv("RL_HARNESS_B2_ENV_FILE", str(env_file))

    settings = load_b2_settings()
    assert settings["B2_ENDPOINT"] == "https://s3.us-west-004.backblazeb2.com"
    assert b2_env_for_remote()["B2_ENDPOINT"] == (
        "https://s3.us-west-004.backblazeb2.com"
    )


def test_b2_env_for_remote_requires_all_required_keys(isolated_b2_env, monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    monkeypatch.delenv("B2_ENDPOINT", raising=False)
    monkeypatch.delenv("B2_APPLICATION_KEY_ID", raising=False)
    monkeypatch.delenv("B2_APPLICATION_KEY", raising=False)
    assert b2_env_for_remote() == {}

    monkeypatch.setenv("B2_BUCKET", "bucket")
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("B2_PREFIX", "alex")
    assert b2_env_for_remote() == {
        "B2_BUCKET": "bucket",
        "B2_ENDPOINT": "https://s3.us-west-004.backblazeb2.com",
        "B2_APPLICATION_KEY_ID": "key-id",
        "B2_APPLICATION_KEY": "secret",
        "B2_PREFIX": "alex",
    }
