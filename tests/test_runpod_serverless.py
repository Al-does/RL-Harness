from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

import devops.serverless.handler as handler_module
from devops.serverless.client import (
    ServerlessClient,
    ServerlessClientError,
)
from devops.serverless.config import ServerlessConfig
from devops.serverless.provision import (
    build_create_request,
    build_endpoint_env,
    build_job_request,
    build_parser,
    cmd_reap,
    cmd_status,
    cmd_up,
    estimate_spend,
    load_state,
    parse_run_command,
    provider_failure_summary,
    resolve_image,
    terminal_output_proves_success,
    validate_cuda_policy_response,
    validate_endpoint_response,
)
from devops.serverless.handler import (
    experiment_env,
    validate_input,
    validate_run_outputs,
    write_and_upload_serverless_result,
)
from devops.serverless.redaction import REDACTED, redact_metadata
from devops.serverless.retrieve import retrieve_manifest_artifacts

SHA_A = "a" * 40
SHA_B = "b" * 40
IMAGE = "ghcr.io/example/worker@sha256:" + "c" * 64
RUN_NAME = "serverless-test"


@pytest.fixture(autouse=True)
def configured_b2(monkeypatch):
    values = {
        "B2_BUCKET": "bucket",
        "B2_ENDPOINT": "https://b2.example",
        "B2_APPLICATION_KEY_ID": "key-id",
        "B2_APPLICATION_KEY": "application-key",
        "B2_PREFIX": "prefix",
    }
    monkeypatch.setattr(
        "devops.serverless.provision.b2_env_for_remote",
        lambda: dict(values),
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _cfg(tmp_path: Path, **overrides) -> ServerlessConfig:
    return ServerlessConfig(
        STATE_PATH=tmp_path / "state.json",
        COST_HISTORY_PATH=tmp_path / "cost_history.json",
        **overrides,
    )


def _up_args(*extra: str):
    return build_parser().parse_args(
        [
            "up",
            "--experiment-ref",
            SHA_A,
            "--library-ref",
            SHA_B,
            "--image",
            IMAGE,
            "--run",
            "rl-harness experiments.test.experiment --hardware cuda4090 "
            f"--upload-artifacts --run-id {RUN_NAME}",
            "--run-name",
            RUN_NAME,
            "--max-age",
            "1",
            "--queue-timeout",
            "30",
            "--ttl",
            "1.75",
            "--max-price",
            "2",
            "--max-estimated-cost",
            "10",
            "--forward-b2",
            "--skip-ref-probe",
            "--skip-image-probe",
            *extra,
        ]
    )


def _endpoint_response(endpoint_id: str = "ep-1"):
    return {
        "id": endpoint_id,
        "image": IMAGE,
        "gpu": {"pools": ["ADA_24"], "count": 1},
        "workers": {"min": 0, "max": 1},
        "scaling": {"type": "QUEUE_DELAY", "value": 4.0, "idleTimeout": 5},
        "timeout": 3_600_000,
        "flashboot": "FLASHBOOT",
    }


def _cuda_policy_response():
    return {
        "id": "ep-1",
        "allowedCudaVersions": ["13.0"],
        "minCudaVersion": "13.0",
    }


def _successful_output(endpoint_id="ep-1", job_id="job-1"):
    return {
        "status": "completed",
        "run_name": RUN_NAME,
        "experiment_sha": SHA_A,
        "library_sha": SHA_B,
        "image_digest": "sha256:" + "c" * 64,
        "endpoint_id": endpoint_id,
        "job_id": job_id,
        "training_iteration": 1,
        "artifact_file_count": 2,
        "checkpoint_keys": ["prefix/checkpoints/model.pt"],
        "remote_manifest_key": "prefix/metadata/remote_artifacts.json",
        "serverless_result_key": "prefix/metadata/serverless_result.json",
    }


def test_client_endpoint_request_shapes_bearer_and_user_agent(monkeypatch):
    seen: list[dict] = []

    class Response:
        def __init__(self, payload=None):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"" if self.payload is None else json.dumps(self.payload).encode()

    responses = iter(
        [
            Response({"endpoints": [{"id": "ep-1"}]}),
            Response({"id": "ep-1"}),
            Response({"id": "ep-2"}),
            Response(),
        ]
    )

    def urlopen(request, timeout):
        seen.append(
            {
                "method": request.method,
                "url": request.full_url,
                "auth": request.get_header("Authorization"),
                "ua": request.get_header("User-agent"),
                "body": json.loads(request.data) if request.data else None,
            }
        )
        return next(responses)

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", urlopen
    )
    client = ServerlessClient(api_key="account-secret")
    assert client.list_endpoints() == [{"id": "ep-1"}]
    assert client.get_endpoint("ep-1") == {"id": "ep-1"}
    assert client.create_endpoint({"name": "test"})["id"] == "ep-2"
    client.delete_endpoint("ep-2")

    assert [row["method"] for row in seen] == ["GET", "GET", "POST", "DELETE"]
    assert seen[0]["url"] == "https://api.runpod.io/v2/serverless"
    assert seen[1]["url"].endswith("/v2/serverless/ep-1")
    assert seen[2]["body"] == {"name": "test"}
    assert all(row["auth"] == "Bearer account-secret" for row in seen)
    assert all(row["ua"] == "rl-harness-serverless/1.0" for row in seen)


def test_client_cuda_policy_update_url_body_auth_and_redaction(monkeypatch):
    seen = {}
    api_secret = "account-secret-never-print"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(_cuda_policy_response()).encode()

    def succeed(request, timeout):
        seen["method"] = request.method
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", succeed
    )
    client = ServerlessClient(api_key=api_secret)
    assert client.update_endpoint_cuda_policy("ep/1") == _cuda_policy_response()
    assert seen == {
        "method": "POST",
        "url": "https://rest.runpod.io/v1/endpoints/ep%2F1/update",
        "auth": f"Bearer {api_secret}",
        "body": {
            "allowedCudaVersions": ["13.0"],
            "minCudaVersion": "13.0",
        },
    }

    def fail(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            {},
            io.BytesIO(f"authorization: Bearer {api_secret}".encode()),
        )

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", fail
    )
    with pytest.raises(ServerlessClientError) as caught:
        client.update_endpoint_cuda_policy("ep-1")
    assert api_secret not in str(caught.value)
    assert REDACTED in str(caught.value)


def test_client_queue_job_status_cancel_and_health_shapes(monkeypatch):
    seen = []
    payloads = iter(
        [
            {"id": "job-1", "status": "IN_QUEUE"},
            {"id": "job-1", "status": "RUNNING"},
            {"id": "job-1", "status": "CANCELLED"},
            {"jobs": {"inQueue": 0}},
        ]
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(next(payloads)).encode()

    def urlopen(request, timeout):
        seen.append(
            (
                request.method,
                request.full_url,
                json.loads(request.data) if request.data else None,
            )
        )
        return Response()

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", urlopen
    )
    client = ServerlessClient(api_key="secret")
    body = {"input": {"run": "x"}, "policy": {"ttl": 10_000}}
    assert client.run_job("ep/1", body)["id"] == "job-1"
    assert client.job_status("ep/1", "job/1")["status"] == "RUNNING"
    assert client.cancel_job("ep/1", "job/1")["status"] == "CANCELLED"
    assert client.health("ep/1")["jobs"]["inQueue"] == 0

    assert seen[0] == (
        "POST",
        "https://api.runpod.ai/v2/ep%2F1/run",
        body,
    )
    assert seen[1][1].endswith("/ep%2F1/status/job%2F1")
    assert seen[2][0] == "POST"
    assert seen[2][1].endswith("/ep%2F1/cancel/job%2F1")
    assert seen[3][1].endswith("/ep%2F1/health")


def test_client_worker_list_and_sse_logs(monkeypatch):
    client = ServerlessClient(api_key="secret")
    monkeypatch.setattr(
        client,
        "_management",
        lambda method, path: {
            "workers": [{"id": "worker-1", "status": "RUNNING"}],
            "summary": {"running": 1, "total": 1},
        },
    )
    assert client.list_workers("ep-1")["workers"][0]["id"] == "worker-1"

    seen = {}

    class Response:
        def __iter__(self):
            return iter(
                [
                    b"id: cursor\n",
                    b'data: {"ts":"now","source":"container","line":"ready"}\n',
                    b"\n",
                ]
            )

        def close(self):
            seen["closed"] = True

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["accept"] = request.get_header("Accept")
        seen["auth"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", urlopen
    )
    events = client.worker_logs(
        "ep-1", "worker-1", source="container", tail=25
    )
    assert seen["url"].startswith(
        "https://api.runpod.io/v2/serverless/ep-1/workers/worker-1/logs?"
    )
    assert "source=container" in seen["url"]
    assert "tail=25" in seen["url"]
    assert seen["accept"] == "text/event-stream"
    assert seen["auth"] == "Bearer secret"
    assert seen["closed"] is True
    assert events == [{"ts": "now", "source": "container", "line": "ready"}]


def test_client_404_support_and_error_redaction(monkeypatch):
    secret = "never-print-this"

    def not_found(*args, **kwargs):
        raise urllib.error.HTTPError(
            f"https://api.runpod.io/v2/serverless/missing?token={secret}",
            404,
            "missing",
            {},
            io.BytesIO(f"authorization: Bearer {secret}".encode()),
        )

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", not_found
    )
    client = ServerlessClient(api_key=secret)
    assert client.get_endpoint("missing") is None

    with pytest.raises(ServerlessClientError) as caught:
        client.list_endpoints()
    assert secret not in str(caught.value)
    assert REDACTED in str(caught.value)


def test_client_redacts_endpoint_env_if_api_echoes_failed_body(monkeypatch):
    github_secret = "github-secret-never-print"
    b2_secret = "b2-secret-never-print"

    def fail(request, timeout):
        echoed = request.data or b""
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            {},
            io.BytesIO(echoed),
        )

    monkeypatch.setattr(
        "devops.serverless.client.urllib.request.urlopen", fail
    )
    client = ServerlessClient(api_key="account-secret")
    with pytest.raises(ServerlessClientError) as caught:
        client.create_endpoint(
            {
                "name": "test",
                "env": {
                    "GH_TOKEN": github_secret,
                    "B2_APPLICATION_KEY": b2_secret,
                },
            }
        )
    assert github_secret not in str(caught.value)
    assert b2_secret not in str(caught.value)


def test_endpoint_policy_and_job_timeouts_are_explicit(tmp_path):
    cfg = _cfg(tmp_path)
    env = {"GH_TOKEN": "github-secret"}
    request = build_create_request(
        cfg,
        name="rlh-serverless-test",
        image=IMAGE,
        env=env,
        execution_timeout_ms=3_600_000,
    )
    assert request == {
        "name": "rlh-serverless-test",
        "image": IMAGE,
        "disk": 30,
        "env": env,
        "gpu": {"pools": ["ADA_24"], "count": 1},
        "workers": {"min": 0, "max": 1},
        "scaling": {
            "type": "QUEUE_DELAY",
            "value": 4.0,
            "idleTimeout": 5,
        },
        "timeout": 3_600_000,
        "flashboot": "FLASHBOOT",
    }
    job = build_job_request(
        cfg,
        run_argv=[
            "rl-harness",
            "experiments.test.experiment",
            "--upload-artifacts",
            "--run-id",
            "test",
        ],
        run_name="test",
        experiment_ref=SHA_A,
        library_ref=SHA_B,
        image_digest="sha256:" + "c" * 64,
        execution_timeout_ms=3_600_000,
        ttl_ms=5_400_000,
        push_results=True,
        results_branch="results",
    )
    assert job["policy"] == {
        "executionTimeout": 3_600_000,
        "ttl": 5_400_000,
        "lowPriority": False,
    }
    assert job["input"]["image_digest"] == "sha256:" + "c" * 64
    assert not any("token" in key.lower() for key in job["input"])
    with pytest.raises(ValueError, match="<= 7 days"):
        build_job_request(
            cfg,
            run_argv=[
                "rl-harness",
                "experiments.test.experiment",
                "--upload-artifacts",
                "--run-id",
                "test",
            ],
            run_name="test",
            experiment_ref=SHA_A,
            library_ref=SHA_B,
            image_digest="sha256:" + "c" * 64,
            execution_timeout_ms=3_600_000,
            ttl_ms=7 * 24 * 3_600_000 + 1,
            push_results=False,
            results_branch="results",
        )


def test_run_command_is_restricted_and_run_id_matches():
    command = (
        "rl-harness experiments.study.condition.experiment --smoke "
        f"--upload-artifacts --run-id {RUN_NAME}"
    )
    assert parse_run_command(command, RUN_NAME) == [
        "rl-harness",
        "experiments.study.condition.experiment",
        "--smoke",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    with pytest.raises(ValueError, match="rl-harness"):
        parse_run_command("bash -lc 'echo unsafe'", RUN_NAME)
    with pytest.raises(ValueError, match="experiments"):
        parse_run_command(
            f"rl-harness harness.cli --upload-artifacts --run-id {RUN_NAME}",
            RUN_NAME,
        )
    with pytest.raises(ValueError, match="upload-artifacts"):
        parse_run_command(
            f"rl-harness experiments.x.experiment --run-id {RUN_NAME}",
            RUN_NAME,
        )
    with pytest.raises(ValueError, match="equal"):
        parse_run_command(
            "rl-harness experiments.x.experiment --upload-artifacts "
            "--run-id other",
            RUN_NAME,
        )


def test_terminal_output_requires_durable_success_and_matching_provenance():
    output = _successful_output()
    entry = {
        "run_name": RUN_NAME,
        "experiment_ref": SHA_A,
        "library_ref": SHA_B,
        "image_digest": "sha256:" + "c" * 64,
        "endpoint_id": "ep-1",
        "job_id": "job-1",
    }
    assert terminal_output_proves_success(output, entry)
    broken = dict(output, training_iteration=0)
    assert not terminal_output_proves_success(broken, entry)
    wrong_ref = dict(output, experiment_sha="d" * 40)
    assert not terminal_output_proves_success(wrong_ref, entry)


def test_endpoint_create_response_must_prove_complete_policy(tmp_path):
    cfg = _cfg(tmp_path)
    request = build_create_request(
        cfg,
        name="rlh-serverless-test",
        image=IMAGE,
        env={"GH_TOKEN": "secret"},
        execution_timeout_ms=3_600_000,
    )
    endpoint = _endpoint_response()
    assert validate_endpoint_response(endpoint, request) is endpoint
    for path in (
        "image",
        "gpu",
        "workers",
        "scaling",
        "timeout",
        "flashboot",
    ):
        broken = json.loads(json.dumps(endpoint))
        broken.pop(path)
        with pytest.raises(ValueError, match="did not prove"):
            validate_endpoint_response(broken, request)
    contradicted = json.loads(json.dumps(endpoint))
    contradicted["workers"]["max"] = 2
    with pytest.raises(ValueError, match="workers.max"):
        validate_endpoint_response(contradicted, request)


def test_cuda_policy_response_must_prove_exact_version():
    expected = _cuda_policy_response()
    assert validate_cuda_policy_response(expected, "13.0") is expected
    for broken in (
        {"allowedCudaVersions": ["13.0"]},
        {"minCudaVersion": "13.0"},
        {"allowedCudaVersions": ["13.0"], "minCudaVersion": "12.8"},
        {
            "allowedCudaVersions": ["13.0", "12.8"],
            "minCudaVersion": "13.0",
        },
    ):
        with pytest.raises(ValueError, match="did not prove|omitted"):
            validate_cuda_policy_response(broken, "13.0")


def test_digest_is_mandatory_and_provenance_is_extracted():
    assert resolve_image(IMAGE) == (IMAGE, "sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="digest-pinned"):
        resolve_image("ghcr.io/example/worker:latest")
    with pytest.raises(ValueError, match="required"):
        resolve_image(None)


def test_endpoint_env_forwards_only_approved_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "account-secret")
    monkeypatch.setattr(
        "devops.serverless.provision.b2_env_for_remote",
        lambda: {
            "B2_BUCKET": "bucket",
            "B2_ENDPOINT": "https://b2.example",
            "B2_APPLICATION_KEY_ID": "id",
            "B2_APPLICATION_KEY": "key",
            "B2_PREFIX": "prefix",
            "RUNPOD_API_KEY": "bad",
        },
    )
    env = build_endpoint_env(github_token="github-secret", forward_b2=True)
    assert env["GH_TOKEN"] == "github-secret"
    assert env["B2_APPLICATION_KEY"] == "key"
    assert "RUNPOD_API_KEY" not in env
    assert set(env) == {
        "GH_TOKEN",
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
        "B2_PREFIX",
    }


def test_estimated_cost_includes_reserves_and_both_gates(
    tmp_path, monkeypatch, capsys
):
    cfg = _cfg(tmp_path)
    estimate = estimate_spend(cfg, ttl_seconds=7200, disk_gb=30)
    assert estimate["reserved_seconds"] == 7200 + 5
    assert estimate["total"] > estimate["gpu"] + estimate["disk"]
    assert estimate["gpu_hourly"] == pytest.approx(1.116)
    assert "full provider TTL" in estimate["assumption"]

    monkeypatch.setenv("RUNPOD_API_KEY", "api")
    monkeypatch.setenv("GH_TOKEN", "gh")
    low_hourly = _up_args(
        "--max-price",
        "1.00",
        "--dry-run",
    )
    assert cmd_up(low_hourly, cfg) == 2
    assert "exceeds --max-price" in capsys.readouterr().out

    low_total = _up_args(
        "--max-estimated-cost",
        "0.01",
        "--dry-run",
    )
    assert cmd_up(low_total, cfg) == 2
    assert "exceeds --max-estimated-cost" in capsys.readouterr().out


def test_dry_run_has_no_api_mutation_or_state(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry run must not instantiate the API client")

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", ForbiddenClient
    )
    assert cmd_up(_up_args("--dry-run"), cfg) == 0
    output = capsys.readouterr().out
    assert "CONSERVATIVE ESTIMATED SPEND CEILING" in output
    assert "not a provider-enforced hard dollar cap" in output
    assert (
        "POST https://rest.runpod.io/v1/endpoints/{endpointId}/update" in output
    )
    assert (
        '{"allowedCudaVersions":["13.0"],"minCudaVersion":"13.0"}'
        in output
    )
    assert "preflight passed; no endpoint or job created" in output
    assert "resources:" in output
    assert not cfg.STATE_PATH.exists()


def test_up_requires_b2_durability_contract_before_spend(
    tmp_path, monkeypatch, capsys
):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    args = _up_args("--dry-run")
    args.forward_b2 = False
    assert cmd_up(args, cfg) == 2
    assert "--forward-b2 is required" in capsys.readouterr().out

    args = _up_args("--dry-run")
    args.run = (
        f"rl-harness experiments.test.experiment --run-id {RUN_NAME}"
    )
    assert cmd_up(args, cfg) == 2
    assert "--upload-artifacts" in capsys.readouterr().out


def test_real_launch_submits_exactly_one_job(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    calls = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            calls.append("create")
            assert request["env"]["GH_TOKEN"] == "github-secret"
            assert request["env"]["B2_BUCKET"] == "bucket"
            return _endpoint_response()

        def update_endpoint_cuda_policy(self, endpoint_id):
            calls.append("cuda-policy")
            assert endpoint_id == "ep-1"
            return _cuda_policy_response()

        def run_job(self, endpoint_id, request):
            calls.append("run")
            assert endpoint_id == "ep-1"
            assert "GH_TOKEN" not in json.dumps(request)
            return {"id": "job-1", "status": "IN_QUEUE"}

        def job_status(self, endpoint_id, job_id):
            return {
                "id": job_id,
                "status": "COMPLETED",
                "output": _successful_output(),
            }

        def delete_endpoint(self, endpoint_id):
            calls.append("delete")

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_up(_up_args("--yes"), cfg) == 0
    assert calls == ["create", "cuda-policy", "run", "delete"]
    entry = load_state(cfg)["runs"][0]
    assert entry["endpoint_id"] == "ep-1"
    assert entry["cuda_policy_verified"] is True
    assert entry["required_cuda_version"] == "13.0"
    assert entry["job_id"] == "job-1"
    assert entry["job_status"] == "COMPLETED"
    assert entry["deleted_at_iso"]
    assert entry["terminal_output"]["serverless_result_key"]
    assert "secret" not in json.dumps(entry)


def test_launch_failure_deletes_created_endpoint(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    deleted = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            return _endpoint_response("ep-failed")

        def update_endpoint_cuda_policy(self, endpoint_id):
            return _cuda_policy_response()

        def run_job(self, endpoint_id, request):
            raise ServerlessClientError("submission failed")

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_up(_up_args("--yes"), cfg) == 1
    assert deleted == ["ep-failed"]
    entry = load_state(cfg)["runs"][0]
    assert entry["cleanup_reason"] == "launch failure"
    assert entry["deleted_at_iso"]


def test_launch_deletes_endpoint_when_create_response_omits_policy(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    deleted = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            return {"id": "ep-unsafe", "image": IMAGE}

        def run_job(self, endpoint_id, request):
            raise AssertionError("unsafe endpoint must not receive a job")

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_up(_up_args("--yes"), cfg) == 1
    assert deleted == ["ep-unsafe"]
    assert "endpoint_policy_verified" not in load_state(cfg)["runs"][0]


def test_launch_deletes_endpoint_and_skips_job_when_cuda_policy_unproven(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    calls = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            calls.append("create")
            return _endpoint_response("ep-unsafe-cuda")

        def update_endpoint_cuda_policy(self, endpoint_id):
            calls.append("cuda-policy")
            return {
                "id": endpoint_id,
                "allowedCudaVersions": ["13.0", "12.8"],
                "minCudaVersion": "13.0",
            }

        def run_job(self, endpoint_id, request):
            raise AssertionError("unverified CUDA endpoint must not receive a job")

        def delete_endpoint(self, endpoint_id):
            calls.append(("delete", endpoint_id))

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_up(_up_args("--yes"), cfg) == 1
    assert calls == ["create", "cuda-policy", ("delete", "ep-unsafe-cuda")]
    entry = load_state(cfg)["runs"][0]
    assert entry["endpoint_policy_verified"] is True
    assert "cuda_policy_verified" not in entry
    assert entry["deleted_at_iso"]


def test_blocking_up_queue_timeout_uses_cancel_status_and_finally_deletes(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path, POLL_INTERVAL_SECONDS=0)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    cancelled = []
    deleted = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            return _endpoint_response("ep-race")

        def update_endpoint_cuda_policy(self, endpoint_id):
            return _cuda_policy_response()

        def run_job(self, endpoint_id, request):
            return {"id": "job-race", "status": "IN_QUEUE"}

        def job_status(self, endpoint_id, job_id):
            return {"id": job_id, "status": "IN_QUEUE"}

        def cancel_job(self, endpoint_id, job_id):
            cancelled.append((endpoint_id, job_id))
            return {"id": job_id, "status": "RUNNING"}

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    args = _up_args("--yes", "--queue-timeout", "0.0000001")
    assert cmd_up(args, cfg) == 1
    assert cancelled
    assert deleted == ["ep-race"]
    entry = load_state(cfg)["runs"][0]
    assert entry["timeout_reason"] == "queue timeout"
    assert entry["job_status"] == "RUNNING"


def test_failed_job_retains_only_bounded_redacted_provider_detail(
    tmp_path, monkeypatch, capsys
):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    deleted = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            pass

        def create_endpoint(self, request):
            return _endpoint_response("ep-provider-failure")

        def update_endpoint_cuda_policy(self, endpoint_id):
            return _cuda_policy_response()

        def run_job(self, endpoint_id, request):
            return {"id": "job-provider-failure", "status": "IN_QUEUE"}

        def job_status(self, endpoint_id, job_id):
            return {
                "id": job_id,
                "status": "FAILED",
                "error": (
                    "worker failed with api-secret github-secret "
                    "application-key " + "x" * 1000
                ),
                "output": {"raw": "must-not-persist"},
                "workerId": "worker-github-secret",
                "arbitrary": {"metadata": "must-not-persist"},
            }

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {"record_count": 0, "actual_cost_usd": 0}

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_up(_up_args("--yes"), cfg) == 1
    output = capsys.readouterr().out
    assert "provider failure: worker=worker-<REDACTED>" in output
    for secret in ("api-secret", "github-secret", "application-key"):
        assert secret not in output
    entry = load_state(cfg)["runs"][0]
    assert entry["provider_failure_source"] == "error"
    assert len(entry["provider_failure_detail"]) <= 512
    assert entry["worker_id"] == "worker-<REDACTED>"
    assert entry["terminal_output"] == {}
    persisted = json.dumps(entry)
    assert "must-not-persist" not in persisted
    assert deleted == ["ep-provider-failure"]


def test_provider_failure_uses_string_output_and_ignores_raw_metadata():
    summary = provider_failure_summary(
        {
            "error": {"nested": "not documented string detail"},
            "output": "plain failure secret-value",
            "workerId": "worker-1",
            "arbitrary": "not retained",
        },
        secrets=("secret-value",),
    )
    assert summary == {
        "provider_failure_source": "output",
        "provider_failure_detail": f"plain failure {REDACTED}",
        "worker_id": "worker-1",
    }


def test_terminal_status_deletes_endpoint_and_records_actual_billing(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path, BILLING_SETTLEMENT_DELAY_SECONDS=0)
    cfg.STATE_PATH.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "endpoint_id": "ep-1",
                        "job_id": "job-1",
                        "name": "rlh-serverless-test",
                        "created_at": 1,
                        "created_at_iso": "2026-07-26T00:00:00+00:00",
                    }
                ]
            }
        )
    )
    deleted = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def job_status(self, endpoint_id, job_id):
            return {"id": job_id, "status": "COMPLETED", "output": {"ok": True}}

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {
                "endpoint_id": endpoint_id,
                "actual_cost_usd": 0.42,
                "components": {
                    "totalAmount": 0.42,
                    "gpuAmount": 0.4,
                    "cpuAmount": 0,
                    "diskAmount": 0.01,
                    "feeAmount": 0.01,
                },
                "record_count": 1,
                "records": [{"serverlessId": endpoint_id}],
                "queried_at": "now",
            }

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_status(SimpleNamespace(), cfg) == 0
    assert deleted == ["ep-1"]
    assert len(load_state(cfg)["runs"]) == 1
    assert cmd_status(SimpleNamespace(), cfg) == 0
    assert load_state(cfg)["runs"] == []
    revisions = json.loads(cfg.COST_HISTORY_PATH.read_text())
    assert [row["revision"] for row in revisions] == [1, 2]
    assert [row["settled"] for row in revisions] == [False, True]
    assert revisions[-1]["actual_cost_usd"] == pytest.approx(0.42)
    assert revisions[-1]["components"]["gpuAmount"] == pytest.approx(0.4)
    assert "job_result" not in revisions[-1]


def test_status_cancels_jobs_that_exceed_queue_timeout(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.STATE_PATH.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "endpoint_id": "ep-queued",
                        "job_id": "job-queued",
                        "name": "rlh-serverless-queued",
                        "created_at": 1,
                        "created_at_iso": "2026-07-26T00:00:00+00:00",
                        "submitted_at_iso": "2026-07-26T00:00:00+00:00",
                        "queue_timeout_s": 60,
                    }
                ]
            }
        )
    )
    cancelled = []
    deleted = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def job_status(self, endpoint_id, job_id):
            return {"id": job_id, "status": "IN_QUEUE"}

        def cancel_job(self, endpoint_id, job_id):
            cancelled.append((endpoint_id, job_id))

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

        def serverless_billing(self, endpoint_id, *, start_time, end_time):
            return {
                "actual_cost_usd": 0,
                "record_count": 0,
                "records": [],
            }

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    monkeypatch.setattr("devops.serverless.provision.time.time", lambda: 2_000_000_000)
    assert cmd_status(SimpleNamespace(), cfg) == 0
    assert cancelled == [("ep-queued", "job-queued")]
    assert deleted == ["ep-queued"]
    entry = load_state(cfg)["runs"][0]
    assert entry["timeout_reason"] == "queue timeout"
    assert entry["job_status"] == "IN_QUEUE"


def test_reap_discovers_only_managed_over_age_endpoints(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    deleted = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def list_endpoints(self):
            return [
                {
                    "id": "managed",
                    "name": "rlh-serverless-old",
                    "createdAt": "2020-01-01T00:00:00Z",
                },
                {
                    "id": "foreign",
                    "name": "production-endpoint",
                    "createdAt": "2020-01-01T00:00:00Z",
                },
            ]

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    assert cmd_reap(SimpleNamespace(max_age=1, yes=True), cfg) == 0
    assert deleted == ["managed"]


def test_reap_uses_recorded_lifecycle_deadline_not_execution_age(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    cfg.STATE_PATH.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "endpoint_id": "future",
                        "name": "rlh-serverless-future",
                        "created_at": 1,
                        "max_age_s": 1,
                        "lifecycle_deadline": 3_000_000_000,
                    },
                    {
                        "endpoint_id": "expired",
                        "name": "rlh-serverless-expired",
                        "created_at": 1,
                        "max_age_s": 99_999_999,
                        "lifecycle_deadline": 2,
                    },
                ]
            }
        )
    )
    deleted = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def list_endpoints(self):
            return [
                {
                    "id": name,
                    "name": f"rlh-serverless-{name}",
                    "createdAt": "2020-01-01T00:00:00Z",
                }
                for name in ("future", "expired")
            ]

        def delete_endpoint(self, endpoint_id):
            deleted.append(endpoint_id)

    monkeypatch.setattr(
        "devops.serverless.provision.ServerlessClient", FakeClient
    )
    monkeypatch.setattr("devops.serverless.provision.time.time", lambda: 100)
    assert cmd_reap(SimpleNamespace(max_age=None, yes=True), cfg) == 0
    assert deleted == ["expired"]


def test_serverless_billing_uses_v2_fields_and_aggregates(monkeypatch):
    client = ServerlessClient(api_key="secret")
    seen = {}

    def management(method, path, *, query=None, **kwargs):
        seen.update({"method": method, "path": path, "query": query})
        return {
            "records": [
                {
                    "serverlessId": "ep-1",
                    "totalAmount": 0.3,
                    "gpuAmount": 0.2,
                    "cpuAmount": 0,
                    "diskAmount": 0.05,
                    "feeAmount": 0.05,
                },
                {
                    "serverlessId": "ep-1",
                    "totalAmount": 0.4,
                    "gpuAmount": 0.3,
                    "cpuAmount": 0,
                    "diskAmount": 0.05,
                    "feeAmount": 0.05,
                },
            ]
        }

    monkeypatch.setattr(client, "_management", management)
    cost = client.serverless_billing(
        "ep-1", start_time="start", end_time="end"
    )
    assert seen["path"] == "billing/serverless"
    assert seen["query"] == {
        "serverlessId": "ep-1",
        "startTime": "start",
        "endTime": "end",
        "bucketSize": "hour",
    }
    assert cost["actual_cost_usd"] == pytest.approx(0.7)
    assert cost["components"]["gpuAmount"] == pytest.approx(0.5)


def test_retrieval_downloads_and_validates_hashes(tmp_path):
    content = b"durable checkpoint"
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "status": "completed",
        "bucket": "bucket",
        "files": [
            {
                "relative_path": "checkpoints/model.pt",
                "key": "prefix/checkpoints/model.pt",
                "size_bytes": len(content),
                "sha256": digest,
            }
        ],
    }

    class FakeS3:
        def __init__(self, value):
            self.value = value

        def download_file(self, bucket, key, target):
            assert bucket == "bucket"
            assert key == "prefix/checkpoints/model.pt"
            Path(target).write_bytes(self.value)

    files = retrieve_manifest_artifacts(
        manifest, tmp_path / "good", client=FakeS3(content)
    )
    assert files[0].read_bytes() == content

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        retrieve_manifest_artifacts(
            manifest,
            tmp_path / "bad",
            client=FakeS3(b"x" * len(content)),
        )
    manifest["files"][0]["relative_path"] = "../escape"
    with pytest.raises(ValueError, match="unsafe"):
        retrieve_manifest_artifacts(
            manifest, tmp_path / "escape", client=FakeS3(content)
        )


def test_metadata_redaction_recurses_through_endpoint_data():
    value = {
        "env": {
            "GH_TOKEN": "github-secret",
            "B2_APPLICATION_KEY": "b2-secret",
            "NORMAL": "visible",
        },
        "workers": [{"password": "worker-secret"}],
    }
    safe = redact_metadata(value)
    assert safe["env"]["GH_TOKEN"] == REDACTED
    assert safe["env"]["B2_APPLICATION_KEY"] == REDACTED
    assert safe["env"]["NORMAL"] == "visible"
    assert safe["workers"][0]["password"] == REDACTED


def _write_handler_evidence(tmp_path: Path):
    results = (
        tmp_path
        / "experiments"
        / "test"
        / "results"
        / RUN_NAME
    )
    results.mkdir(parents=True)
    artifacts = (
        tmp_path
        / "experiments"
        / "test"
        / "artifacts"
        / RUN_NAME
    )
    artifacts.mkdir(parents=True)
    (results / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": RUN_NAME,
                "runtime": {"artifacts_dir": str(artifacts)},
            }
        )
    )
    remote = {
        "status": "completed",
        "bucket": "bucket",
        "prefix": "experiments/test/" + RUN_NAME,
        "files": [
            {
                "relative_path": "checkpoints/model.pt",
                "key": "experiments/test/run/checkpoints/model.pt",
                "sha256": "d" * 64,
                "size_bytes": 123,
            }
        ],
    }
    (results / "remote_artifacts.json").write_text(json.dumps(remote))
    (results / "progress.jsonl").write_text(
        json.dumps({"training_iteration": 1}) + "\n"
    )
    return results


def _write_tune_result(
    results: Path,
    lines: list[str],
    *,
    include_in_remote_manifest: bool = True,
) -> Path:
    run_manifest = json.loads((results / "run_manifest.json").read_text())
    artifacts = Path(run_manifest["runtime"]["artifacts_dir"])
    result = artifacts / "tune" / "PPO_test_00000_0_2026-07-26_00-00-00" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text("\n".join(lines) + "\n")
    if include_in_remote_manifest:
        remote_path = results / "remote_artifacts.json"
        remote = json.loads(remote_path.read_text())
        relative = result.relative_to(artifacts).as_posix()
        remote["files"].append(
            {
                "relative_path": relative,
                "key": f"{remote['prefix']}/{relative}",
                "sha256": "e" * 64,
                "size_bytes": result.stat().st_size,
            }
        )
        remote_path.write_text(json.dumps(remote))
    return result


def test_handler_validates_actual_local_manifests_and_training_evidence(tmp_path):
    results = _write_handler_evidence(tmp_path)
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    evidence = validate_run_outputs(tmp_path, argv, RUN_NAME)
    assert evidence["results_dir"] == results
    assert evidence["training_iteration"] == 1
    assert evidence["artifact_file_count"] == 1
    assert evidence["checkpoint_keys"] == [
        "experiments/test/run/checkpoints/model.pt"
    ]

    (results / "progress.jsonl").write_text(
        json.dumps({"training_iteration": 0}) + "\n"
    )
    with pytest.raises(RuntimeError, match="positive training_iteration"):
        validate_run_outputs(tmp_path, argv, RUN_NAME)


def test_handler_accepts_uploaded_tune_only_training_evidence(tmp_path):
    results = _write_handler_evidence(tmp_path)
    (results / "progress.jsonl").unlink()
    _write_tune_result(
        results,
        [
            json.dumps(
                {
                    "trial_id": "00000",
                    "training_iteration": 1,
                    "episode_return_mean": 0.5,
                }
            ),
            json.dumps({"evaluation/training_iteration": 2}),
        ],
    )
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    evidence = validate_run_outputs(tmp_path, argv, RUN_NAME)
    assert evidence["training_iteration"] == 2
    assert evidence["artifact_file_count"] == 2


def test_handler_rejects_tune_artifacts_dir_path_escape(tmp_path):
    results = _write_handler_evidence(tmp_path)
    (results / "progress.jsonl").unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    manifest_path = results / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"]["artifacts_dir"] = str(outside)
    manifest_path.write_text(json.dumps(manifest))
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    with pytest.raises(RuntimeError, match="escapes the experiment repository"):
        validate_run_outputs(tmp_path, argv, RUN_NAME)


def test_handler_tune_json_lines_tolerate_malformed_records_but_require_positive(
    tmp_path,
):
    results = _write_handler_evidence(tmp_path)
    (results / "progress.jsonl").unlink()
    tune_result = _write_tune_result(
        results,
        ["{truncated", json.dumps(["not", "an", "object"])],
    )
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    with pytest.raises(RuntimeError, match="no valid positive"):
        validate_run_outputs(tmp_path, argv, RUN_NAME)
    tune_result.write_text(
        "{truncated\n"
        + json.dumps({"metrics/training_iteration": 3})
        + "\n"
    )
    assert validate_run_outputs(tmp_path, argv, RUN_NAME)[
        "training_iteration"
    ] == 3


def test_handler_rejects_unuploaded_tune_training_evidence(tmp_path):
    results = _write_handler_evidence(tmp_path)
    (results / "progress.jsonl").unlink()
    _write_tune_result(
        results,
        [json.dumps({"training_iteration": 1})],
        include_in_remote_manifest=False,
    )
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    with pytest.raises(RuntimeError, match="missing from remote artifact manifest"):
        validate_run_outputs(tmp_path, argv, RUN_NAME)


def test_handler_rejects_missing_checkpoint_and_bad_remote_hash(tmp_path):
    results = _write_handler_evidence(tmp_path)
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    remote_path = results / "remote_artifacts.json"
    remote = json.loads(remote_path.read_text())
    remote["files"][0]["relative_path"] = "metrics/data.json"
    remote["files"][0]["key"] = "metrics/data.json"
    remote_path.write_text(json.dumps(remote))
    with pytest.raises(RuntimeError, match="checkpoint-like"):
        validate_run_outputs(tmp_path, argv, RUN_NAME)
    remote["files"][0]["relative_path"] = "checkpoints/model.pt"
    remote["files"][0]["sha256"] = "bad"
    remote_path.write_text(json.dumps(remote))
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_run_outputs(tmp_path, argv, RUN_NAME)


def test_handler_uploads_deterministic_metadata_keys(tmp_path):
    _write_handler_evidence(tmp_path)
    argv = [
        "rl-harness",
        "experiments.test.experiment",
        "--upload-artifacts",
        "--run-id",
        RUN_NAME,
    ]
    evidence = validate_run_outputs(tmp_path, argv, RUN_NAME)
    uploaded = []

    class FakeS3:
        def upload_file(self, path, bucket, key):
            uploaded.append((Path(path), bucket, key))

    result = {"validation_status": "completed", "training_iteration": 1}
    keys = write_and_upload_serverless_result(
        evidence, result, client=FakeS3()
    )
    assert keys[0] == (
        f"experiments/test/{RUN_NAME}/metadata/remote_artifacts.json"
    )
    assert keys[1] == (
        f"experiments/test/{RUN_NAME}/metadata/serverless_result.json"
    )
    assert keys[2] == (
        f"experiments/test/{RUN_NAME}/metadata/durability_manifest.json"
    )
    assert keys[0] in [item[2] for item in uploaded]
    assert keys[1] in [item[2] for item in uploaded]
    assert keys[2] in [item[2] for item in uploaded]
    stored = json.loads(
        (Path(evidence["results_dir"]) / "serverless_result.json").read_text()
    )
    assert stored == result


def test_success_path_mlflow_upload_errors_propagate(tmp_path, monkeypatch):
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir()
    (mlflow_dir / "metadata").write_text("data")
    monkeypatch.setattr(handler_module, "MLFLOW_DIR", mlflow_dir)

    class FailingS3:
        def upload_file(self, path, bucket, key):
            raise RuntimeError("upload failed")

    monkeypatch.setattr(handler_module, "_b2_client", lambda: FailingS3())
    with pytest.raises(RuntimeError, match="upload failed"):
        handler_module.upload_mlflow(RUN_NAME)


def test_handler_validates_input_and_strips_github_token_from_experiment_env(
    monkeypatch,
):
    job_input = {
        "run_argv": [
            "rl-harness",
            "experiments.test.experiment",
            "--upload-artifacts",
            "--run-id",
            RUN_NAME,
        ],
        "run_name": RUN_NAME,
        "experiment_repo_url": "https://github.com/example/experiments.git",
        "experiment_ref": SHA_A,
        "library_repo_url": "https://github.com/example/library.git",
        "library_ref": SHA_B,
        "image_digest": "sha256:" + "c" * 64,
        "ray_version": "2.56.0",
        "torch_version": "2.12.1",
        "gymnasium_version": "1.2.2",
        "push_results": False,
        "results_branch": "results",
    }
    assert validate_input({"id": "job", "input": job_input}) == job_input
    broken = dict(job_input, run_argv=["bash", "-lc", "unsafe"])
    with pytest.raises(ValueError, match="run_argv"):
        validate_input({"id": "job", "input": broken})

    monkeypatch.setenv("GH_TOKEN", "github-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret-2")
    monkeypatch.setenv("B2_APPLICATION_KEY", "b2-secret")
    env = experiment_env("mlflow-run")
    assert "GH_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["B2_APPLICATION_KEY"] == "b2-secret"
    assert env["MLFLOW_RUN_ID"] == "mlflow-run"


def test_worker_source_and_image_enforce_serverless_invariants():
    root = Path(__file__).resolve().parents[1]
    source = (root / "devops/serverless/handler.py").read_text()
    dockerfile = (root / "devops/serverless/Dockerfile").read_text()

    assert "runpod.serverless.progress_update" in source
    assert '"refresh_worker": True' in source
    assert "clean_workspace()" in source
    assert "torch.cuda.is_available()" in source
    assert "ray.__version__ != spec" in source
    assert "git.experiment_commit" in source
    assert "git.library_commit" in source
    assert "runpod.serverless.endpoint_id" in source
    assert "runpod.serverless.job_id" in source
    assert "runpod.gpu.actual" in source
    assert "terminate_self" not in source
    assert "podTerminate" not in source
    assert 'os.environ.get("RUNPOD_API_KEY"' not in source
    assert "GH_TOKEN" in source
    assert "--no-deps" in source
    assert 'spec["run_argv"]' in source
    assert '["bash", "-lc"' not in source
    assert 'if __name__ == "__main__":' in source
    assert "publish_compact_results" in source
    assert "git rebase" not in source
    assert "workload_success" in source
    assert "canonical_manifest_key" in source

    assert (
        "FROM ghcr.io/al-does/rl-harness-runpod@sha256:"
        "1257ac0a0f2b57b80022849a96fa1d9a5bfffff69fabbf233b3cf45dc665fb3c"
        in dockerfile
    )
    assert "runpod" in dockerfile
    assert "serverless_handler.py" in dockerfile
