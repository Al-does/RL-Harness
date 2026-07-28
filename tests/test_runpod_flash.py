from __future__ import annotations

from types import SimpleNamespace

import pytest

from devops.flash.probe import main, run_probe, validate_args


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


def test_flash_worker_uses_zero_minimum_workers(monkeypatch):
    monkeypatch.setenv("RL_HARNESS_FLASH_MAX_WORKERS", "3")
    from devops.flash.worker import _workers

    assert _workers() == (0, 3)


def test_delete_endpoint_does_not_submit_a_job(monkeypatch, capsys):
    deleted = []

    class Client:
        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

    monkeypatch.setattr("devops.flash.probe.ServerlessClient", Client)

    assert main(["--endpoint-id", "ep-1", "--delete-endpoint"]) == 0
    assert deleted == ["ep-1"]
    assert "deleted endpoint ep-1" in capsys.readouterr().out
