import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from devops.vast.config import VastConfig
from devops.vast.provision import build_env
from devops.vast.scoring import rank_offers
from devops.vast.self_destruct import push_results
from harness.hardware import available_cpus, ensure_ray_initialized


def _offer(**overrides):
    offer = {
        "id": 1,
        "machine_id": 10,
        "dph_total": 0.4,
        "reliability2": 0.99,
        "verification": "verified",
        "duration": 3 * 86400,
        "disk_space": 100,
        "direct_port_count": 1,
        "cuda_max_good": 13.0,
        "cpu_cores_effective": 12.0,
        "rentable": True,
        "geolocation": "California, US",
    }
    offer.update(overrides)
    return offer


@pytest.mark.parametrize(
    ("field", "value"),
    [("cuda_max_good", 12.8), ("cpu_cores_effective", 11.9)],
)
def test_offer_hardware_gates_reject_incompatible_hosts(field, value):
    assert not rank_offers([_offer(**{field: value})], VastConfig(), disk=30, count=1)


def test_bootstrap_environment_carries_runtime_safeguards():
    cfg = VastConfig(MIN_CUDA=13.0, UV_SYNC_TIMEOUT_S=900)
    env = build_env(
        cfg,
        ref="abc123",
        run_cmd=None,
        self_destruct=False,
        instance_label="test",
        run_name="test",
        results_branch="results",
        github_token=None,
        api_key=None,
    )

    assert env["VAST_UV_SYNC_TIMEOUT_S"] == "900"
    assert env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_available_cpus_uses_smallest_host_affinity_and_cgroup_limit():
    values = {
        "/sys/fs/cgroup/cpu.max": "1150000 100000",
    }

    def fake_read_text(path: Path, *args, **kwargs):
        try:
            return values[str(path)]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    with (
        patch("harness.hardware.os.cpu_count", return_value=128),
        patch(
            "harness.hardware.os.sched_getaffinity",
            return_value=set(range(64)),
            create=True,
        ),
        patch.object(Path, "read_text", autospec=True, side_effect=fake_read_text),
    ):
        assert available_cpus() == pytest.approx(11.5)


def test_ray_cpu_pool_is_capped_to_container_quota():
    fake_ray = SimpleNamespace(is_initialized=Mock(return_value=False), init=Mock())

    with (
        patch.dict("sys.modules", {"ray": fake_ray}),
        patch("harness.hardware.available_cpus", return_value=11.5),
    ):
        ensure_ray_initialized()

    fake_ray.init.assert_called_once_with(
        num_cpus=11,
        runtime_env={"py_executable": shlex.quote(sys.executable)},
    )


def test_self_destruct_stages_only_compact_experiment_results(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run(args, cwd=None):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "devops.vast.self_destruct._run",
        fake_run,
    )

    assert push_results(
        branch="results",
        run_name="test",
        instance_id="1",
        repo=tmp_path,
    )
    assert calls[0] == [
        "git",
        "add",
        "-A",
        "--",
        "experiments/",
    ]
