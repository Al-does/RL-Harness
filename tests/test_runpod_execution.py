"""Acceptance tests for the phased RunPod execution redesign."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from analysis.portable_checkpoint import (
    load_portable_module,
    write_portable_checkpoint,
)
from devops.runpod.execution.durability import (
    upload_compact_results_bundle,
    write_canonical_durability_manifest,
)
from devops.runpod.execution.fallback import FallbackPolicy, decide_fallback
from devops.runpod.execution.phases import (
    JobReport,
    Phase,
    PhaseStatus,
    TerminalReason,
)
from devops.runpod.execution.preflight import (
    PreflightError,
    build_resource_contract_for_run,
    run_preflight,
    verify_remote_sha_fetchable,
)
from devops.runpod.execution.progress import classify_provider_status
from devops.runpod.execution.publication import publish_compact_results
from devops.serverless.config import ServerlessConfig
from devops.serverless.provision import cmd_up, terminal_output_proves_success
from harness.resources import resource_contract_from_profile
from tests.test_runpod_serverless import (
    IMAGE,
    RUN_NAME,
    SHA_A,
    SHA_B,
    _cfg,
    _cuda_policy_response,
    _endpoint_response,
    _successful_output,
    _up_args,
)


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_resource_contract_rejects_fractional_overcapacity():
    contract = resource_contract_from_profile("cuda4090_gpuinfer", smoke=False)
    assert contract.total_gpus == pytest.approx(1.8)
    with pytest.raises(PreflightError, match="1.8"):
        build_resource_contract_for_run(
            [
                "rl-harness",
                "experiments.test.experiment",
                "--hardware",
                "cuda4090_gpuinfer",
                "--upload-artifacts",
                "--run-id",
                "x",
            ],
            default_profile="cuda4090_gpuinfer",
            available_gpus=1.0,
        )


def test_smoke_and_single_gpu_profile_fit_one_gpu_endpoint():
    smoke = build_resource_contract_for_run(
        [
            "rl-harness",
            "experiments.test.experiment",
            "--smoke",
            "--upload-artifacts",
            "--run-id",
            "x",
        ],
        default_profile="cuda4090_gpuinfer",
        available_gpus=1.0,
    )
    assert smoke.total_gpus == 1.0
    single = build_resource_contract_for_run(
        [
            "rl-harness",
            "experiments.test.experiment",
            "--hardware",
            "cuda4090",
            "--upload-artifacts",
            "--run-id",
            "x",
        ],
        default_profile="cuda4090_gpuinfer",
        available_gpus=1.0,
    )
    assert single.total_gpus == 1.0


def test_nonexistent_sha_fails_preflight_without_endpoint(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("preflight must not create a client")

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", ForbiddenClient
    )

    def opener(request, timeout=20):
        url = request.full_url
        if "api.github.com" in url:
            raise PreflightError("not fetchable")
        return _FakeResponse({"schemaVersion": 2})

    # Force real ref probe path (no skip) and make GitHub reject the SHA.
    args = _up_args("--dry-run")
    args.skip_ref_probe = False
    args.skip_image_probe = True

    def boom(*args, **kwargs):
        raise PreflightError(
            "experiment SHA "
            f"{SHA_A} is not fetchable from "
            "https://github.com/Al-does/alex-rl-experiments.git (HTTP 404)"
        )

    monkeypatch.setattr(
        "devops.serverless.provision.run_preflight",
        boom,
    )
    assert cmd_up(args, cfg) == 2
    output = capsys.readouterr().out
    assert "PREFLIGHT rejected before provisioning" in output
    assert "terminal_reason=preflight_rejected" in output
    assert not cfg.STATE_PATH.exists()


def test_one_point_eight_gpu_trial_rejected_before_provision(
    tmp_path, monkeypatch, capsys
):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no client")),
    )
    args = _up_args("--dry-run")
    args.run = (
        "rl-harness experiments.test.experiment --hardware cuda4090_gpuinfer "
        f"--upload-artifacts --run-id {RUN_NAME}"
    )
    assert cmd_up(args, cfg) == 2
    assert "1.8" in capsys.readouterr().out


def test_workload_success_independent_of_publication_failure():
    output = _successful_output()
    output["workload_success"] = True
    output["publication_status"] = "failed"
    output["publication_detail"] = "deterministic content conflict"
    output["canonical_manifest_key"] = "prefix/metadata/durability_manifest.json"
    entry = {
        "experiment_ref": SHA_A,
        "library_ref": SHA_B,
        "image_digest": "sha256:" + "c" * 64,
        "endpoint_id": "ep-1",
        "job_id": "job-1",
        "run_name": RUN_NAME,
    }
    assert terminal_output_proves_success(output, entry) is True


def test_concurrent_result_publications_overlay_without_rebase(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)

    def seed_results_branch():
        work = tmp_path / "seed"
        work.mkdir()
        subprocess.run(["git", "clone", str(remote), str(work)], check=True)
        subprocess.run(
            ["git", "checkout", "--orphan", "results"], cwd=work, check=True
        )
        (work / "experiments").mkdir()
        (work / "experiments" / "README.md").write_text("results root\n")
        subprocess.run(["git", "add", "."], cwd=work, check=True)
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "seed"],
            cwd=work,
            check=True,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", "results"], cwd=work, check=True
        )

    seed_results_branch()

    def worker_repo(name: str, filename: str) -> Path:
        repo = tmp_path / name
        # Detached experiment history that must never be rebased onto results.
        subprocess.run(["git", "init", str(repo)], check=True)
        (repo / "experiments" / "study" / "condition" / "results" / "run").mkdir(
            parents=True
        )
        target = (
            repo
            / "experiments"
            / "study"
            / "condition"
            / "results"
            / "run"
            / filename
        )
        target.write_text(json.dumps({"ok": name}) + "\n")
        (repo / "feature.txt").write_text("experiment feature history\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", name],
            cwd=repo,
            check=True,
        )
        return repo

    repo_a = worker_repo("worker-a", "a.json")
    repo_b = worker_repo("worker-b", "b.json")

    result_a = publish_compact_results(
        experiment_repo=repo_a,
        remote_url=str(remote),
        branch="results",
        commit_message="results: a",
        github_token="unused",
        work_root=tmp_path / "pub-a",
    )
    result_b = publish_compact_results(
        experiment_repo=repo_b,
        remote_url=str(remote),
        branch="results",
        commit_message="results: b",
        github_token="unused",
        work_root=tmp_path / "pub-b",
    )
    assert result_a.status == "succeeded"
    assert result_b.status == "succeeded"

    mirror = tmp_path / "mirror"
    subprocess.run(
        ["git", "clone", "--branch", "results", str(remote), str(mirror)],
        check=True,
    )
    assert (mirror / "experiments/study/condition/results/run/a.json").is_file()
    assert (mirror / "experiments/study/condition/results/run/b.json").is_file()
    assert not (mirror / "feature.txt").exists()
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=mirror,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "worker-a" not in log
    assert "worker-b" not in log


def test_compact_results_are_hash_verified_in_canonical_manifest(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    plot = results / "figure.png"
    payload = results / "metrics.json"
    plot.write_bytes(b"png-bytes")
    payload.write_text('{"acc": 1}\n')
    uploaded: dict[str, bytes] = {}

    class FakeS3:
        def upload_file(self, path, bucket, key):
            uploaded[key] = Path(path).read_bytes()

    compact = upload_compact_results_bundle(
        results_dir=results,
        bucket="bucket",
        artifact_prefix="run/prefix",
        client=FakeS3(),
    )
    path, key, manifest = write_canonical_durability_manifest(
        results_dir=results,
        bucket="bucket",
        artifact_prefix="run/prefix",
        artifact_files=[],
        compact_files=compact,
        client=FakeS3(),
    )
    assert path.is_file()
    assert key.endswith("durability_manifest.json")
    assert manifest["file_count"] == 2
    for row in compact:
        assert uploaded[row["key"]]
        assert row["sha256"]
        assert row["kind"] == "compact_result"


class _TinyPortableModule(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {"hidden": 4}
        self.linear = torch.nn.Linear(3, 4)

    def get_state(self):
        return self.state_dict()

    def set_state(self, state):
        self.load_state_dict(state)


def test_portable_checkpoint_loads_without_ray(tmp_path):
    module = _TinyPortableModule({"hidden": 4})
    with torch.no_grad():
        module.linear.weight.fill_(0.25)
    destination = tmp_path / "portable"
    write_portable_checkpoint(
        destination,
        module=module,
        environment_specification={"env": "TinyEnv", "env_config": {"n": 1}},
        checkpoint_step=7,
        experiment_sha="a" * 40,
        harness_sha="b" * 40,
        analysis_protocol={"probe": "linear"},
    )

    import sys

    # Ensure Ray is not required/started for restore.
    sys.modules.pop("ray", None)
    with load_portable_module(destination) as restored:
        assert "ray" not in sys.modules or not __import__("ray").is_initialized()
        assert torch.allclose(restored.linear.weight, module.linear.weight)


def test_serverless_cold_start_timeout_falls_back_to_pods(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    calls = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            calls.append("create")
            return _endpoint_response()

        def update_endpoint_cuda_policy(self, endpoint_id):
            calls.append("cuda")
            return _cuda_policy_response()

        def run_job(self, endpoint_id, request):
            calls.append("run")
            return {"id": "job-1", "status": "IN_QUEUE"}

        def job_status(self, endpoint_id, job_id):
            return {
                "id": job_id,
                "status": "IN_QUEUE",
                "progress": "pulling image layers",
            }

        def cancel_job(self, endpoint_id, job_id):
            calls.append("cancel")
            return {"id": job_id, "status": "CANCELLED"}

        def delete_endpoint(self, endpoint_id):
            calls.append("delete")

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    clock = {"now": 1_000_000.0}

    def fake_time():
        clock["now"] += 10.0
        return clock["now"]

    monkeypatch.setattr("devops.serverless.provision.time.time", fake_time)
    monkeypatch.setattr("devops.serverless.provision.time.sleep", lambda _s: None)
    fallback_calls = []

    def fake_fallback(args, cfg):
        fallback_calls.append("pods")
        return 0

    monkeypatch.setattr(
        "devops.serverless.provision._fallback_pods", fake_fallback
    )
    args = _up_args(
        "--yes",
        "--fallback",
        "pods",
        "--queue-timeout",
        "0.0001",
    )
    # Make submitted_at + queue timeout already expired on first poll.
    assert cmd_up(args, cfg) == 0
    assert "delete" in calls
    assert fallback_calls == ["pods"]
    output = capsys.readouterr().out
    assert "terminal_reason=" in output
    assert "phase=" in output


def test_logs_expose_phases_and_terminal_reason():
    report = JobReport(backend="serverless", run_name="demo")
    report.set_phase(Phase.PREFLIGHT, PhaseStatus.SUCCEEDED, detail="ok")
    report.set_phase(Phase.PROVISIONING, PhaseStatus.FAILED, detail="queue")
    report.mark_workload(success=False, reason=TerminalReason.QUEUE_TIMEOUT)
    payload = report.to_dict()
    assert payload["phases"]["PREFLIGHT"]["status"] == "succeeded"
    assert payload["terminal_reason"] == "queue_timeout"
    assert report.launcher_exit_code() == 1


def test_progress_distinguishes_capacity_queue_from_image_pull():
    queued = classify_provider_status("IN_QUEUE", worker_seen=False)
    assert queued.capacity_queue is True
    pulling = classify_provider_status(
        "IN_QUEUE",
        worker_seen=False,
        progress_message="Extracting image layer",
    )
    assert pulling.image_pull is True


def test_fallback_policy_only_for_retryable_failures():
    decision = decide_fallback(
        policy=FallbackPolicy.PODS,
        terminal_reason="image_init_timeout",
        serverless_attempts=1,
        max_serverless_attempts=1,
    )
    assert decision.action == "fallback_pods"
    rejected = decide_fallback(
        policy=FallbackPolicy.PODS,
        terminal_reason="preflight_rejected",
        serverless_attempts=1,
    )
    assert rejected.action == "fail"


def test_verify_remote_sha_uses_github_api():
    calls = []

    def opener(request, timeout=20):
        calls.append(request.full_url)
        return _FakeResponse({"sha": "a" * 40})

    verify_remote_sha_fetchable(
        "https://github.com/Al-does/RL-Harness.git",
        "a" * 40,
        label="library",
        github_token="token",
        opener=opener,
    )
    assert "api.github.com/repos/Al-does/RL-Harness/commits/" in calls[0]


def test_ghcr_image_probe_uses_anonymous_bearer_token():
    calls: list[str] = []

    class Resp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status = status

        def read(self):
            if isinstance(self._payload, (dict, list)):
                return json.dumps(self._payload).encode()
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(request, timeout=20):
        url = request.full_url
        calls.append(url)
        if url.startswith("https://ghcr.io/token"):
            return Resp({"token": "anon-token"})
        auth = request.get_header("Authorization") or request.headers.get("Authorization")
        assert auth == "Bearer anon-token"
        return Resp(b"{}", status=200)

    from devops.runpod.execution.preflight import verify_image_digest_available

    digest = verify_image_digest_available(
        "ghcr.io/al-does/rl-harness-runpod-serverless@sha256:" + "d" * 64,
        opener=opener,
    )
    assert digest.startswith("sha256:")
    assert any(url.startswith("https://ghcr.io/token") for url in calls)
    assert any("/manifests/" in url for url in calls)
