from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from devops.runpod.pods.client import (
    RunPodClient,
    RunPodClientError,
    assert_safe_pod,
    reject_explicitly_unsafe_pod,
)
from devops.runpod.pods.config import RunPodConfig
from devops.runpod.pods import remote_runner
from devops.runpod.pods.provision import (
    _ssh_command,
    build_create_request,
    build_env,
    build_parser,
    cmd_reap,
    cmd_up,
    normalize_run_argv,
    resolve_ssh_key,
)
from devops.runpod.execution.publication import PublicationResult


def _cfg(tmp_path: Path) -> RunPodConfig:
    return RunPodConfig(
        STATE_PATH=tmp_path / "state.json",
        COST_HISTORY_PATH=tmp_path / "cost_history.json",
        EXPERIMENT_REPO_LOCAL=tmp_path / "experiments",
    )


def _safe_pod(**overrides):
    pod = {
        "id": "pod-1",
        "interruptible": False,
        "gpuTypeId": "NVIDIA GeForce RTX 4090",
        "machine": {
            "secureCloud": False,
            "gpuDisplayName": "NVIDIA GeForce RTX 4090",
        },
    }
    pod.update(overrides)
    return pod


def test_create_request_explicitly_requires_community_on_demand(tmp_path):
    cfg = _cfg(tmp_path)
    request = build_create_request(
        cfg,
        name="test",
        image="pytorch/pytorch@sha256:abc",
        disk_gb=30,
        regions=["US"],
        env={"RUNPOD_RUN_CMD": "rl-harness test.experiment"},
        terminate_after="2026-07-25T23:30:00Z",
    )

    assert request["cloudType"] == "COMMUNITY"
    assert request["interruptible"] is False
    assert request["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert request["countryCode"] == "US"
    assert request["startSsh"] is False
    assert request["terminateAfter"] == "2026-07-25T23:30:00Z"


def test_pod_request_accepts_explicit_compatible_gpu(tmp_path):
    cfg = _cfg(tmp_path)
    gpu = "NVIDIA RTX A5000"
    request = build_create_request(
        cfg,
        name="compatible",
        image="image@sha256:abc",
        disk_gb=30,
        regions=[],
        env={},
        terminate_after="2026-07-25T23:30:00Z",
        gpu_type_id=gpu,
    )
    args = build_parser().parse_args(["up", "--gpu-type", gpu])

    assert request["gpuTypeId"] == gpu
    assert args.gpu_type == gpu


def test_interactive_request_enables_only_ssh_port(tmp_path):
    cfg = _cfg(tmp_path)
    request = build_create_request(
        cfg,
        name="interactive",
        image="image@sha256:abc",
        disk_gb=30,
        regions=[],
        env={"RUNPOD_INTERACTIVE": "1"},
        terminate_after="2026-07-25T23:30:00Z",
        interactive=True,
    )

    assert request["supportPublicIp"] is True
    assert request["startSsh"] is True
    assert request["ports"] == "22/tcp"


def test_client_uses_on_demand_graphql_mutation_and_bearer_header(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": {
                        "podFindAndDeployOnDemand": {
                            "id": "pod-1",
                            "podType": "RESERVED",
                        }
                    }
                }
            ).encode()

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(
        "devops.runpod.pods.client.urllib.request.urlopen",
        urlopen,
    )
    client = RunPodClient(api_key="account-secret")
    pod = client.create_pod(
        {
            "name": "test",
            "imageName": "image@sha256:abc",
            "cloudType": "COMMUNITY",
            "interruptible": False,
            "gpuTypeId": "NVIDIA GeForce RTX 4090",
            "gpuCount": 1,
            "terminateAfter": "2026-07-25T23:30:00Z",
            "env": {"GH_TOKEN": "github-secret"},
        }
    )

    assert pod["id"] == "pod-1"
    assert seen["url"] == "https://api.runpod.io/graphql"
    assert seen["authorization"] == "Bearer account-secret"
    assert "podFindAndDeployOnDemand" in seen["body"]["query"]
    assert seen["body"]["variables"]["input"]["terminateAfter"].endswith("Z")
    assert seen["body"]["variables"]["input"]["env"] == [
        {"key": "GH_TOKEN", "value": "github-secret"}
    ]


def test_graphql_errors_are_actionable_and_redacted(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "errors": [
                        {
                            "message": (
                                "No GPU available for github-secret "
                                "using account-secret"
                            )
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(
        "devops.runpod.pods.client.urllib.request.urlopen",
        lambda request, timeout: Response(),
    )

    with pytest.raises(RunPodClientError) as caught:
        RunPodClient(api_key="account-secret").create_pod(
            {
                "interruptible": False,
                "env": {"GH_TOKEN": "github-secret"},
            }
        )

    assert "No GPU available" in str(caught.value)
    assert "account-secret" not in str(caught.value)
    assert "github-secret" not in str(caught.value)


def test_safety_assertion_rejects_interruptible_secure_or_unknown_gpu():
    assert_safe_pod(_safe_pod())
    reserved = _safe_pod(interruptible=None, podType="RESERVED")
    assert_safe_pod(reserved)

    with pytest.raises(RunPodClientError, match="interruptible=false"):
        assert_safe_pod(_safe_pod(interruptible=True))
    with pytest.raises(RunPodClientError, match="interruptible=false"):
        assert_safe_pod(_safe_pod(interruptible=None, podType="BID"))
    with pytest.raises(RunPodClientError, match="Community Cloud"):
        assert_safe_pod(
            _safe_pod(machine={"secureCloud": True, "gpuDisplayName": "NVIDIA GeForce RTX 4090"})
        )
    with pytest.raises(RunPodClientError, match="unexpected GPU"):
        assert_safe_pod(
            _safe_pod(
                gpuTypeId="Tesla T4",
                machine={"secureCloud": False, "gpuDisplayName": "Tesla T4"},
            )
        )


def test_pending_safety_check_allows_omitted_fields_but_rejects_contradictions():
    reject_explicitly_unsafe_pod({"id": "pending-pod"})

    with pytest.raises(RunPodClientError, match="interruptible"):
        reject_explicitly_unsafe_pod(
            {"id": "bad-pod", "interruptible": True}
        )


def test_remote_env_never_forwards_account_runpod_key(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "account-secret")
    env = build_env(
        cfg,
        experiment_ref="exp-sha",
        library_ref="lib-sha",
        run_cmd="rl-harness experiments.test.experiment",
        run_name="test",
        results_branch="results",
        github_token="github-secret",
        image_digest="sha256:abc",
        max_age_s=3600,
        estimated_price=0.34,
        push_results=True,
        forward_b2=False,
    )

    assert "RUNPOD_API_KEY" not in env
    assert env["GH_TOKEN"] == "github-secret"
    assert env["RUNPOD_MAX_AGE_S"] == "3600"
    assert env["RUNPOD_PUSH_RESULTS"] == "1"
    assert env["RUNPOD_AUTO_TERMINATE"] == "1"
    assert json.loads(env["RUNPOD_RUN_ARGV"]) == [
        "rl-harness",
        "experiments.test.experiment",
        "--run-id",
        "test",
        "--upload-artifacts",
    ]


def test_run_argv_normalizes_durable_smoke_execution():
    argv = normalize_run_argv(
        "rl-harness experiments.study.condition.experiment --smoke",
        "smoke-run",
    )

    assert argv[-4:] == [
        "--run-id",
        "smoke-run",
        "--upload-artifacts",
        "--publish-smoke",
    ]
    assert "--smoke" in argv


def test_run_argv_rejects_disabled_artifact_upload():
    with pytest.raises(ValueError, match="cannot disable"):
        normalize_run_argv(
            "rl-harness experiments.study.condition.experiment "
            "--no-upload-artifacts",
            "run",
        )


def test_interactive_env_injects_only_public_ssh_key(tmp_path):
    cfg = _cfg(tmp_path)
    env = build_env(
        cfg,
        experiment_ref="exp-sha",
        library_ref="lib-sha",
        run_cmd="",
        run_name="interactive",
        results_branch="results",
        github_token="github-secret",
        image_digest="sha256:abc",
        max_age_s=3600,
        estimated_price=0.34,
        push_results=False,
        forward_b2=False,
        interactive=True,
        ssh_public_key="ssh-ed25519 AAAA test",
    )

    assert env["RUNPOD_INTERACTIVE"] == "1"
    assert env["SSH_PUBLIC_KEY"] == "ssh-ed25519 AAAA test"
    assert "RUNPOD_SSH_PRIVATE_KEY" not in env


def test_resolve_ssh_key_requires_matching_pair(tmp_path):
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("private material")
    public_key = tmp_path / "id_ed25519.pub"
    public_key.write_text("ssh-ed25519 AAAA test\n")

    resolved_path, resolved_public = resolve_ssh_key(str(private_key))

    assert resolved_path == private_key
    assert resolved_public == "ssh-ed25519 AAAA test"


def test_ssh_command_prefers_public_ip_mapping(tmp_path):
    command = _ssh_command(
        {
            "publicIp": "203.0.113.10",
            "portMappings": {"22": 32022},
            "machine": {"podHostId": "pod-host"},
        },
        tmp_path / "id_ed25519",
    )

    assert command[-3:] == ["-p", "32022", "root@203.0.113.10"]
    assert "-p" in command


def test_cli_preserves_vast_common_surface_and_rejects_spot():
    parser = build_parser()
    args = parser.parse_args(
        [
            "up",
            "--mode",
            "ondemand",
            "--branch",
            "feature",
            "--library-branch",
            "main",
            "--run",
            "rl-harness test.experiment",
            "--max-age",
            "1",
            "--dry-run",
        ]
    )
    assert args.mode == "ondemand"
    assert args.branch == "feature"
    assert args.library_branch == "main"
    assert args.dry_run is True


def test_dry_run_validates_without_creating_pod(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "runpod-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    monkeypatch.setenv("B2_BUCKET", "bucket")
    monkeypatch.setenv("B2_ENDPOINT", "s3.example.com")
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "key")
    monkeypatch.setattr(
        "devops.runpod.pods.provision.resolve_image_digest",
        lambda image: ("pytorch/pytorch@sha256:abc", "sha256:abc"),
    )
    args = build_parser().parse_args(
        [
            "up",
            "--commit",
            "exp-sha",
            "--library-commit",
            "lib-sha",
            "--run",
            "rl-harness experiments.test.experiment",
            "--dry-run",
        ]
    )

    assert cmd_up(args, cfg) == 0
    output = capsys.readouterr().out
    assert "interruptible" in output
    assert "no Pods created" in output
    assert not cfg.STATE_PATH.exists()


def test_runpod_disallows_disabling_max_age(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "runpod-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    args = build_parser().parse_args(
        [
            "up",
            "--commit",
            "exp-sha",
            "--run",
            "rl-harness test.experiment",
            "--max-age",
            "0",
            "--dry-run",
        ]
    )
    assert cmd_up(args, cfg) == 2


def test_reap_discovers_untracked_managed_orphans(tmp_path, monkeypatch):
    terminated = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def list_pods(self):
            return [
                {
                    "id": "orphan-1",
                    "name": "rlh-runpod-old",
                    "desiredStatus": "ERROR",
                    "createdAt": "2026-07-25T00:00:00Z",
                },
                {
                    "id": "someone-else",
                    "name": "manual-pod",
                    "desiredStatus": "ERROR",
                    "createdAt": "2026-07-25T00:00:00Z",
                },
            ]

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        "devops.runpod.pods.provision.RunPodClient",
        FakeClient,
    )
    args = SimpleNamespace(max_age=None, yes=True)

    assert cmd_reap(args, cfg) == 0
    assert terminated == ["orphan-1"]


def test_client_errors_do_not_expose_api_key(monkeypatch):
    secret = "runpod-secret-do-not-log"

    def fail(*args, **kwargs):
        raise RuntimeError(
            f"https://rest.runpod.io/v1/pods?api_key={secret}"
        )

    monkeypatch.setattr(
        "devops.runpod.pods.client.urllib.request.urlopen",
        fail,
    )
    client = RunPodClient(api_key=secret)
    with pytest.raises(RunPodClientError) as caught:
        client.list_pods()
    assert secret not in str(caught.value)
    assert "api_key=<REDACTED>" in str(caught.value)


def test_client_sends_user_agent_to_rest_api(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"[]"

    def urlopen(request, timeout):
        seen["user_agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr(
        "devops.runpod.pods.client.urllib.request.urlopen",
        urlopen,
    )

    assert RunPodClient(api_key="secret").list_pods() == []
    assert seen["user_agent"] == "rl-harness-runpod/1.0 (RunPod Pods client)"


def test_client_reads_v2_sse_pod_logs(monkeypatch):
    seen = {}

    class Response:
        def __iter__(self):
            return iter(
                [
                    b"id: cursor-1\n",
                    b'data: {\"ts\":\"2026-07-26T00:00:00Z\",'
                    b'\"source\":\"container\",\"line\":\"ready\"}\n',
                    b"\n",
                ]
            )

        def close(self):
            seen["closed"] = True

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["accept"] = request.get_header("Accept")
        return Response()

    monkeypatch.setattr(
        "devops.runpod.pods.client.urllib.request.urlopen",
        urlopen,
    )
    events = RunPodClient(api_key="secret").pod_logs(
        "pod-1", source="container", tail=25
    )

    assert seen["url"].startswith("https://api.runpod.io/v2/pods/pod-1/logs?")
    assert "source=container" in seen["url"]
    assert "tail=25" in seen["url"]
    assert seen["accept"] == "text/event-stream"
    assert seen["closed"] is True
    assert events == [
        {
            "ts": "2026-07-26T00:00:00Z",
            "source": "container",
            "line": "ready",
        }
    ]


def test_pod_cost_filters_unreliable_billing_endpoint_locally(monkeypatch):
    client = RunPodClient(api_key="runpod-secret")
    seen_query = {}

    def request(method, path, *, query=None, **kwargs):
        seen_query.update(query or {})
        return [
            {"podId": "pod-1", "amount": 0.12, "timeBilledMs": 1_000},
            {"podId": "pod-2", "amount": 0.34, "timeBilledMs": 2_000},
        ]

    monkeypatch.setattr(client, "_request", request)
    cost = client.pod_cost("pod-2")

    assert "podId" not in seen_query
    assert cost["actual_cost_usd"] == pytest.approx(0.34)
    assert cost["time_billed_ms"] == 2_000
    assert cost["record_count"] == 1


def test_container_runner_terminates_in_finally_and_has_watchdog():
    source = (
        Path(__file__).resolve().parents[1]
        / "devops"
        / "runpod"
        / "pods"
        / "container_entrypoint.py"
    ).read_text()
    assert "start_watchdog(max_age_s)" in source
    assert "finally:" in source
    assert "run_batch_job(" in source
    assert "if cleanup_allowed" in source
    assert "terminate_self(cleanup_reason)" in source
    assert "automatic teardown withheld" in source
    assert "start_ssh_server()" in source
    assert "interactive CUDA workspace ready" in source


class _FakeS3:
    def __init__(self):
        self.uploaded: dict[str, bytes] = {}

    def upload_file(self, path, bucket, key):
        assert bucket == "bucket"
        self.uploaded[key] = Path(path).read_bytes()


def _prepare_remote_run(
    tmp_path: Path,
    monkeypatch,
    *,
    run_status: str = "completed",
    remote_status: str = "completed",
):
    repo = tmp_path / "experiment-repo"
    results = repo / "experiments" / "study" / "condition" / "results" / "run-1"
    results.mkdir(parents=True)
    (repo / ".gitignore").write_text(
        "experiments/**/results/**/remote_artifacts.json\n"
        "experiments/**/results/**/durability_manifest.json\n"
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    remote_summary = {
        "status": remote_status,
        "bucket": "bucket",
        "prefix": "runs/run-1",
    }
    (results / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": run_status,
                "remote_artifacts": remote_summary,
            }
        )
    )
    (results / "summary.json").write_text('{"score": 1}\n')
    (results / "remote_artifacts.json").write_text(
        json.dumps(
            {
                "status": remote_status,
                "bucket": "bucket",
                "prefix": "runs/run-1",
                "files": [
                    {
                        "kind": "artifact",
                        "relative_path": "checkpoint/file",
                        "key": "runs/run-1/checkpoint/file",
                        "sha256": "a" * 64,
                        "size_bytes": 12,
                    }
                ],
            }
        )
    )
    fake_s3 = _FakeS3()
    fake_b2 = SimpleNamespace(bucket="bucket", s3_client=lambda: fake_s3)
    monkeypatch.setattr(
        remote_runner.B2StorageConfig,
        "from_env",
        classmethod(lambda cls: fake_b2),
    )
    monkeypatch.setattr(remote_runner, "_start_mlflow", lambda **kwargs: "mlflow-1")
    monkeypatch.setattr(remote_runner, "_finish_mlflow", lambda *args: None)
    monkeypatch.setattr(remote_runner, "_upload_mlflow", lambda **kwargs: None)
    monkeypatch.setenv(
        "RUNPOD_RUN_ARGV",
        json.dumps(
            [
                "rl-harness",
                "experiments.study.condition.experiment",
                "--run-id",
                "run-1",
                "--upload-artifacts",
            ]
        ),
    )
    monkeypatch.setenv("RUNPOD_RUN_NAME", "run-1")
    monkeypatch.setenv("RUNPOD_PUSH_RESULTS", "1")
    monkeypatch.setenv("RUNPOD_EXPERIMENT_REPO_URL", "https://example.test/repo.git")
    monkeypatch.setenv("RUNPOD_RESULTS_BRANCH", "results")
    monkeypatch.setenv("RUNPOD_IMAGE_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    monkeypatch.setenv("GH_TOKEN", "token")
    return repo, results, fake_s3


@pytest.mark.parametrize(
    ("returncode", "run_status", "workload_success"),
    [(0, "completed", True), (9, "failed", False)],
)
def test_remote_runner_persists_and_publishes_before_cleanup(
    tmp_path,
    monkeypatch,
    returncode,
    run_status,
    workload_success,
):
    repo, results, fake_s3 = _prepare_remote_run(
        tmp_path,
        monkeypatch,
        run_status=run_status,
    )
    monkeypatch.setattr(
        remote_runner,
        "publish_compact_results",
        lambda **kwargs: PublicationResult(
            status="succeeded",
            detail="pushed",
            branch="results",
            commit="abc",
            attempts=1,
        ),
    )

    outcome = remote_runner.run_batch_job(
        experiment_repo=repo,
        python=Path("/opt/venv/bin/python"),
        mlflow_dir=tmp_path / "mlruns",
        experiment_sha="b" * 40,
        library_sha="c" * 40,
        runner=lambda *args, **kwargs: SimpleNamespace(returncode=returncode),
    )

    report = json.loads((results / "runpod_result.json").read_text())
    assert outcome.cleanup_allowed is True
    assert outcome.exit_code == returncode
    assert report["workload_success"] is workload_success
    assert report["publication_status"] == "succeeded"
    canonical = json.loads(
        fake_s3.uploaded["runs/run-1/metadata/durability_manifest.json"]
    )
    assert canonical["status"] == "completed"
    assert {row["kind"] for row in canonical["files"]} == {
        "artifact",
        "compact_result",
    }


def test_remote_runner_withholds_cleanup_when_publication_fails(
    tmp_path,
    monkeypatch,
):
    repo, results, _ = _prepare_remote_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        remote_runner,
        "publish_compact_results",
        lambda **kwargs: PublicationResult(
            status="failed",
            detail="push rejected",
            branch="results",
        ),
    )

    outcome = remote_runner.run_batch_job(
        experiment_repo=repo,
        python=Path("/opt/venv/bin/python"),
        mlflow_dir=tmp_path / "mlruns",
        experiment_sha="b" * 40,
        library_sha="c" * 40,
        runner=lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    report = json.loads((results / "runpod_result.json").read_text())
    assert outcome.cleanup_allowed is False
    assert report["workload_success"] is True
    assert report["publication_status"] == "failed"
    assert report["terminal_reason"] == "publication_failed"
    assert report["phases"]["CLEANUP"]["status"] == "skipped"


def test_remote_runner_withholds_cleanup_when_b2_is_not_durable(
    tmp_path,
    monkeypatch,
):
    repo, results, _ = _prepare_remote_run(
        tmp_path,
        monkeypatch,
        remote_status="failed",
    )
    monkeypatch.setattr(
        remote_runner,
        "publish_compact_results",
        lambda **kwargs: pytest.fail("publication must not run"),
    )

    outcome = remote_runner.run_batch_job(
        experiment_repo=repo,
        python=Path("/opt/venv/bin/python"),
        mlflow_dir=tmp_path / "mlruns",
        experiment_sha="b" * 40,
        library_sha="c" * 40,
        runner=lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    report = json.loads((results / "runpod_result.json").read_text())
    assert outcome.cleanup_allowed is False
    assert report["terminal_reason"] == "durable_upload_failed"
    assert report["phases"]["DURABLE_UPLOAD"]["status"] == "failed"
