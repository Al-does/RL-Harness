from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import devops.serverless.handler as handler_module
from devops.flash.config import FlashConfig
from devops.flash.probe import main, run_probe, validate_args
from devops.flash.provision import (
    _find_endpoint,
    _stage_source,
    _validate_endpoint,
    estimate_spend,
)


def _args(**overrides):
    values = {
        "endpoint_id": "flash-endpoint",
        "jobs": 2,
        "max_workers": 2,
        "timeout": 5,
        "sleep_seconds": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_probe_submits_bounded_parallel_jobs(monkeypatch):
    submitted = []

    class Client:
        def run_job(self, endpoint_id, body):
            submitted.append((endpoint_id, body))
            return {"id": f"job-{len(submitted)}"}

        def job_status(self, endpoint_id, job_id):
            return {
                "id": job_id,
                "status": "COMPLETED",
                "output": {"delivery": "runpod-flash-artifact"},
            }

    monkeypatch.setattr("devops.flash.probe.time.sleep", lambda _: None)
    result = run_probe(_args(), Client())

    assert [body["input"]["input_data"]["probe"] for _, body in submitted] == [
        "probe-1",
        "probe-2",
    ]
    assert all(endpoint_id == "flash-endpoint" for endpoint_id, _ in submitted)
    assert [job["id"] for job in result] == ["job-1", "job-2"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_args(jobs=0), "--jobs"),
        (_args(max_workers=0), "--max-workers"),
        (_args(jobs=2, max_workers=1), "--jobs cannot exceed"),
        (_args(timeout=0), "--timeout"),
        (_args(sleep_seconds=61), "--sleep-seconds"),
    ],
)
def test_probe_rejects_unsafe_bounds(args, message):
    with pytest.raises(ValueError, match=message):
        validate_args(args)


def test_delete_endpoint_does_not_submit_a_job(monkeypatch, capsys):
    deleted = []

    class Client:
        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

    monkeypatch.setattr("devops.flash.probe.ServerlessClient", Client)

    assert main(["--endpoint-id", "ep-1", "--delete-endpoint"]) == 0
    assert deleted == ["ep-1"]
    assert "deleted endpoint ep-1" in capsys.readouterr().out


def test_flash_endpoint_requires_zero_minimum_workers():
    endpoint = {
        "id": "ep-1",
        "name": "flash-app",
        "image": "runpod/flash:py3.12-latest",
        "workers": {"min": 0, "max": 3},
        "scaling": {"idleTimeout": 5},
        "flashboot": "FLASHBOOT",
    }
    assert (
        _validate_endpoint(
            endpoint,
            app="flash-app",
            max_workers=3,
            cfg=FlashConfig(),
        )
        == endpoint
    )
    endpoint["workers"]["min"] = 1
    with pytest.raises(ValueError, match="workers.min"):
        _validate_endpoint(
            endpoint,
            app="flash-app",
            max_workers=3,
            cfg=FlashConfig(),
        )


def test_flash_endpoint_proves_staged_source_revision():
    endpoint = {
        "id": "ep-1",
        "name": "flash-app",
        "image": "runpod/flash:py3.12-latest",
        "workers": {"min": 0, "max": 1},
        "scaling": {"idleTimeout": 5},
        "flashboot": "FLASHBOOT",
        "env": {"RL_HARNESS_SOURCE_SHA256": "a" * 64},
    }
    _validate_endpoint(
        endpoint,
        app="flash-app",
        max_workers=1,
        cfg=FlashConfig(),
        source_digest="a" * 64,
    )
    with pytest.raises(ValueError, match="source revision"):
        _validate_endpoint(
            endpoint,
            app="flash-app",
            max_workers=1,
            cfg=FlashConfig(),
            source_digest="b" * 64,
        )


def test_flash_cost_estimate_has_no_idle_worker_reservation():
    estimate = estimate_spend(FlashConfig(), execution_seconds=60)
    assert estimate["reserved_seconds"] == 65
    assert estimate["total"] < 0.03


def test_redeploy_keeps_newest_endpoint_and_deletes_idle_superseded():
    deleted = []
    endpoints = [
        {
            "id": "old",
            "createdAt": "2026-01-01T00:00:00Z",
            "name": "flash-app",
            "image": "runpod/flash:py3.12-latest",
            "workers": {"min": 0, "max": 1},
            "scaling": {"idleTimeout": 5},
            "flashboot": "FLASHBOOT",
        },
        {
            "id": "new",
            "createdAt": "2026-01-02T00:00:00Z",
            "name": "flash-app",
            "image": "runpod/flash:py3.12-latest",
            "workers": {"min": 0, "max": 1},
            "scaling": {"idleTimeout": 5},
            "flashboot": "FLASHBOOT",
        },
    ]

    class Client:
        def list_endpoints(self):
            return endpoints

        def get_endpoint(self, endpoint_id):
            return next(row for row in endpoints if row["id"] == endpoint_id)

        def list_workers(self, endpoint_id):
            return {"summary": {"running": 0}}

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

    result = _find_endpoint(
        Client(),
        app="flash-app",
        max_workers=1,
        cfg=FlashConfig(),
    )
    assert result["id"] == "new"
    assert deleted == ["old"]


def test_flash_worker_declares_managed_torch_compatible_dependencies():
    source = (
        Path(__file__).resolve().parents[1] / "devops" / "flash" / "worker.py"
    ).read_text()
    assert '"ray[rllib]==2.56.0"' in source
    assert 'min_cuda_version="12.8"' in source
    assert '"torch==' not in source
    assert "workers=_workers()" in source


def test_flash_stages_handler_under_harness_specific_module_name(tmp_path):
    root = Path(__file__).resolve().parents[1]
    worker = (root / "devops" / "flash" / "worker.py").read_bytes()
    handler = (root / "devops" / "serverless" / "handler.py").read_bytes()

    digest = _stage_source(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "worker.py",
        "rlh_experiment_handler.py",
    }
    assert (tmp_path / "rlh_experiment_handler.py").read_bytes() == handler
    assert b"import rlh_experiment_handler as experiment_handler" in worker
    expected = hashlib.sha256(worker + b"\0" + handler).hexdigest()
    assert digest == expected


def test_flash_experiment_env_preserves_staged_artifact_pythonpath(
    tmp_path, monkeypatch
):
    experiment_dir = tmp_path / "experiments"
    monkeypatch.setattr(handler_module, "_FLASH_DELIVERY", True)
    monkeypatch.setattr(handler_module, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setenv("PYTHONPATH", "inherited-path")

    env = handler_module.experiment_env("mlflow-run")

    assert env["PYTHONPATH"].split(handler_module.os.pathsep) == [
        str(experiment_dir),
        str(Path(handler_module.__file__).resolve().parent),
        "inherited-path",
    ]


def test_flash_runtime_accepts_provider_torch_and_cuda(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.9.1+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "RTX 4090",
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "ray",
        SimpleNamespace(__version__="2.56.0"),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(
        __import__("sys").modules,
        "gymnasium",
        SimpleNamespace(__version__="1.2.2"),
    )
    runtime = handler_module.validate_runtime(
        {
            "delivery": "runpod-flash-artifact",
            "ray_version": "2.56.0",
            "torch_version": "2.12.1",
            "gymnasium_version": "1.2.2",
        }
    )
    assert runtime["torch_version"] == "2.9.1"
    assert runtime["cuda_version"] == "12.8"
