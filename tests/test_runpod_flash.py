from __future__ import annotations

from types import SimpleNamespace

import pytest

from devops.flash.probe import run_probe, validate_args


def _args(**overrides):
    values = {
        "endpoint_id": "flash-endpoint",
        "jobs": 2,
        "max_workers": 2,
        "timeout": 5,
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

    assert [body["input"]["probe"] for _, body in submitted] == [
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
    ],
)
def test_probe_rejects_unsafe_bounds(args, message):
    with pytest.raises(ValueError, match=message):
        validate_args(args)


def test_flash_worker_uses_zero_minimum_workers(monkeypatch):
    monkeypatch.setenv("RL_HARNESS_FLASH_MAX_WORKERS", "3")
    from devops.flash.worker import _workers

    assert _workers() == (0, 3)
