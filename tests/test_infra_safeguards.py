import json
import shlex
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from devops.vast.config import VastConfig
from devops.vast.provision import build_env, build_parser, check_local_ssh, cmd_up
from devops.vast.quarantine import active_exclusions, load_quarantine, record_failure
from devops.vast.scoring import build_query, price_band_bounds, rank_offers
from devops.vast.self_destruct import destroy_self, push_results
from devops.vast.vast_client import VastClient, VastClientError
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


def test_offer_selection_supports_machine_query_and_machine_exclusions():
    cfg = VastConfig()
    query = build_query(cfg, disk=30, machine_id=10)
    offers = [
        _offer(id=123, machine_id=10),
        _offer(id=456, machine_id=20),
    ]

    ranked = rank_offers(
        offers,
        cfg,
        disk=30,
        count=2,
        excluded_machine_ids={10},
    )

    assert "machine_id=10" in query
    assert [offer.id for offer in ranked] == [456]


def test_explicit_regions_are_hard_filtered():
    cfg = VastConfig()
    offers = [
        _offer(id=1, machine_id=10, dph_total=0.20, geolocation="Shenzhen, CN"),
        _offer(id=2, machine_id=20, dph_total=0.30, geolocation="California, US"),
    ]

    soft = rank_offers(offers, cfg, disk=30, count=2, regions=["US", "CA"])
    hard = rank_offers(
        offers,
        cfg,
        disk=30,
        count=2,
        regions=["US", "CA"],
        require_preferred_region=True,
    )

    # Soft mode keeps non-preferred regions but ranks preferred ones earlier.
    assert [offer.id for offer in soft] == [2, 1]
    assert [offer.id for offer in hard] == [2]


def test_price_band_uses_upper_inner_quartile_when_pool_is_large():
    cfg = VastConfig(PRICE_BAND_MIN_HOSTS=8)
    offers = [
        _offer(
            id=100 + i,
            machine_id=1000 + i,
            dph_total=0.20 + i * 0.04,
            reliability2=0.990,
            cpu_cores_effective=12.0,
            inet_down=100.0,
        )
        for i in range(12)
    ]
    # Best reliability inside the upper-inner band (price 0.52).
    offers[8] = _offer(
        id=108,
        machine_id=1008,
        dph_total=0.52,
        reliability2=0.999,
        cpu_cores_effective=16.0,
        inet_down=100.0,
    )
    prices = [0.20 + i * 0.04 for i in range(12)]
    lo, hi, mode = price_band_bounds(prices, cfg)
    ranked = rank_offers(offers, cfg, disk=30, count=3)

    assert mode == "upper_inner_quartile"
    assert lo <= ranked[0].price <= hi
    assert all(lo <= offer.price <= hi for offer in ranked)
    assert ranked[0].id == 108
    assert all(offer.price >= lo for offer in ranked)


def test_price_band_falls_back_on_small_pools():
    cfg = VastConfig(PRICE_BAND_MIN_HOSTS=8, PRICE_BAND_FLOOR_MULT=1.35, PRICE_BAND_FLOOR_PAD=0.15)
    offers = [
        _offer(id=1, machine_id=10, dph_total=0.20),
        _offer(id=2, machine_id=20, dph_total=0.30),
        _offer(id=3, machine_id=30, dph_total=0.80),
    ]
    lo, hi, mode = price_band_bounds([0.20, 0.30, 0.80], cfg)
    ranked = rank_offers(offers, cfg, disk=30, count=3)

    assert mode == "floor_fallback"
    assert lo == pytest.approx(0.20)
    assert hi == pytest.approx(0.35)
    assert [offer.id for offer in ranked] == [1, 2]


def test_quarantine_persists_machine_and_ip_exclusions(tmp_path):
    cfg = VastConfig(QUARANTINE_PATH=tmp_path / "quarantine.json", QUARANTINE_TTL_S=3600)
    now = 1_700_000_000.0
    record_failure(
        cfg,
        machine_id=138964,
        public_ip="137.175.76.24",
        reason="uv sync stall",
        now=now,
    )
    machines, ips = active_exclusions(cfg, now=now + 10)
    assert machines == {138964}
    assert ips == {"137.175.76.24"}
    machines, ips = active_exclusions(cfg, now=now + 4000)
    assert machines == set()
    assert ips == set()
    data = load_quarantine(cfg)
    assert "138964" in data["machines"]


def test_vast_cli_parses_exact_offer_machine_and_exclusions():
    args = build_parser().parse_args(
        ["up", "--machine-id", "10", "--exclude-machine", "20", "30"]
    )

    assert args.machine_id == 10
    assert args.exclude_machine == [20, 30]


def test_vast_client_removes_api_keys_from_sdk_errors():
    sentinel = "TEST_KEY_DO_NOT_USE"

    class FakeSDK:
        def create_instance(self, **kwargs):
            raise RuntimeError(
                "410 Gone for "
                f"https://console.vast.ai/api/v0/asks/123/?api_key={sentinel}"
            )

    client = object.__new__(VastClient)
    client.api_key = sentinel
    client.v = FakeSDK()

    with pytest.raises(VastClientError) as caught:
        client.create_instance(
            123,
            image="test",
            disk=1,
            env={},
            label="test",
            onstart_cmd="true",
        )

    assert sentinel not in str(caught.value)
    assert "api_key=<REDACTED>" in str(caught.value)
    assert caught.value.__cause__ is None


def test_self_destruct_removes_api_keys_from_errors(monkeypatch):
    sentinel = "TEST_KEY_DO_NOT_USE"
    messages = []

    def fail_urlopen(*args, **kwargs):
        raise RuntimeError(
            f"request failed for https://console.vast.ai/?api_key={sentinel}"
        )

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    assert not destroy_self("123", sentinel, log=messages.append)
    assert sentinel not in "\n".join(messages)
    assert "api_key=<REDACTED>" in "\n".join(messages)


def test_bootstrap_uses_sparse_blobless_checkout():
    bootstrap = (
        Path(__file__).resolve().parents[1] / "devops" / "vast" / "bootstrap.sh"
    ).read_text()

    assert "git clone --depth 1 --filter=blob:none --sparse --no-checkout" in bootstrap
    assert "git sparse-checkout set --cone" in bootstrap
    assert " experiments " in bootstrap
    assert " results " not in bootstrap.split("git sparse-checkout set --cone", 1)[1].split(
        "|| fail", 1
    )[0]
    assert "git fetch --all" not in bootstrap


def test_bootstrap_environment_carries_runtime_safeguards():
    cfg = VastConfig(MIN_CUDA=13.0, UV_SYNC_TIMEOUT_S=900, UV_SYNC_STALL_S=480)
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
    assert env["VAST_UV_SYNC_STALL_S"] == "480"
    assert env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert env["VAST_EXPERIMENT_REPO_URL"] == cfg.EXPERIMENT_REPO_URL
    assert env["VAST_LIBRARY_REPO_URL"] == cfg.LIBRARY_REPO_URL
    assert env["VAST_LIBRARY_GIT_REF"] == cfg.LIBRARY_DEFAULT_REF
    assert env["VAST_EXPERIMENT_GIT_REF"] == "abc123"
    assert env["VAST_EXPERIMENT_DIR"] == "/root/work/alex-rl-experiments"


def test_bootstrap_environment_forwards_github_token_without_self_destruct():
    cfg = VastConfig()
    env = build_env(
        cfg,
        ref="abc123",
        run_cmd=None,
        self_destruct=False,
        instance_label="test",
        run_name="test",
        results_branch="results",
        github_token="ghp_test_token",
        api_key=None,
    )

    assert env["GITHUB_TOKEN"] == "ghp_test_token"
    assert "VAST_SELF_DESTRUCT" not in env


def test_bootstrap_environment_omits_b2_settings_by_default(monkeypatch):
    cfg = VastConfig()
    monkeypatch.setenv("B2_BUCKET", "bucket")
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("B2_PREFIX", "alex")

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

    assert "B2_BUCKET" not in env
    assert "B2_APPLICATION_KEY" not in env


def test_bootstrap_environment_forwards_b2_settings_when_requested(monkeypatch):
    cfg = VastConfig()
    monkeypatch.setenv("B2_BUCKET", "bucket")
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("B2_PREFIX", "alex")

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
        forward_b2=True,
    )

    assert env["B2_BUCKET"] == "bucket"
    assert env["B2_PREFIX"] == "alex"
    assert env["B2_APPLICATION_KEY"] == "secret"


def test_bootstrap_uses_token_authenticated_experiment_clone():
    bootstrap = (
        Path(__file__).resolve().parents[1] / "devops" / "vast" / "bootstrap.sh"
    ).read_text()

    assert "EXPERIMENT_CLONE_URL=" in bootstrap
    assert "x-access-token:${GITHUB_TOKEN}@github.com/${EXPERIMENT_SLUG}.git" in bootstrap
    assert '"$EXPERIMENT_CLONE_URL"' in bootstrap


def test_bootstrap_watches_uv_sync_for_stalls():
    bootstrap = (
        Path(__file__).resolve().parents[1] / "devops" / "vast" / "bootstrap.sh"
    ).read_text()
    assert "VAST_UV_SYNC_STALL_S" in bootstrap
    assert "uv sync stalled" in bootstrap


def _up_args(**overrides):
    args = dict(
        mode="ondemand",
        disk=None,
        image=None,
        regions=None,
        max_price=None,
        commit="abc123",
        branch=None,
        experiment_repo=None,
        library_branch=None,
        library_commit=None,
        count=1,
        offer_id=None,
        machine_id=None,
        exclude_machine=[],
        dry_run=False,
        yes=True,
        self_destruct=False,
        results_branch=None,
        run_name="test",
        github_token=None,
        max_age=0,
        run=None,
        teardown_on_error=False,
        no_open=True,
        bid=None,
        forward_b2=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_multi_box_readiness_is_concurrent(tmp_path, monkeypatch):
    lock = threading.Lock()
    active = 0
    max_active = 0
    next_instance = iter((101, 102))

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            self.api_key = api_key or "test-key"

        def search_offers(self, query, offer_type):
            return [
                _offer(id=1, machine_id=10),
                _offer(id=2, machine_id=20, dph_total=0.41),
            ]

        def ensure_ssh_key(self, path):
            return "ssh-rsa test"

        def create_instance(self, *args, **kwargs):
            return next(next_instance)

        def attach_ssh_key(self, *args, **kwargs):
            return None

        def destroy_instance(self, instance_id):
            raise AssertionError(f"ready box should not be destroyed: {instance_id}")

        def wait_until_running(self, instance_id, log):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return {"id": instance_id}

        def connection_info(self, inst, probe):
            return f"host-{inst['id']}", 22

    cfg = VastConfig(
        SSH_KEY_PATH=tmp_path / "id_rsa.pub",
        SSH_CONFIG_PATH=tmp_path / "ssh" / "vast.conf",
        STATE_PATH=tmp_path / "state.json",
        QUARANTINE_PATH=tmp_path / "quarantine.json",
    )

    monkeypatch.setattr("devops.vast.vast_client.VastClient", FakeClient)
    monkeypatch.setattr("devops.vast.provision.check_local_ssh", lambda cfg: None)
    monkeypatch.setattr("devops.vast.provision.wait_for_ready_ssh", lambda *args, **kwargs: True)
    monkeypatch.setattr("devops.vast.terminals.write_ssh_config", lambda *args: None)

    assert cmd_up(_up_args(count=2), cfg) == 0
    assert max_active == 2


def test_unready_host_is_destroyed_and_replaced(tmp_path, monkeypatch):
    created = []
    destroyed = []
    next_instance = iter((201, 202))

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            self.api_key = api_key or "test-key"

        def search_offers(self, query, offer_type):
            return [
                _offer(id=1, machine_id=10, public_ipaddr="10.0.0.1"),
                _offer(id=2, machine_id=20, dph_total=0.41, public_ipaddr="10.0.0.2"),
            ]

        def ensure_ssh_key(self, path):
            return "ssh-rsa test"

        def create_instance(self, offer_id, **kwargs):
            iid = next(next_instance)
            created.append((offer_id, iid))
            return iid

        def attach_ssh_key(self, *args, **kwargs):
            return None

        def destroy_instance(self, instance_id):
            destroyed.append(instance_id)
            return {}

        def wait_until_running(self, instance_id, log):
            return {"id": instance_id, "public_ipaddr": f"10.0.0.{instance_id - 200}"}

        def connection_info(self, inst, probe):
            return f"host-{inst['id']}", 22

    cfg = VastConfig(
        SSH_KEY_PATH=tmp_path / "id_rsa.pub",
        SSH_CONFIG_PATH=tmp_path / "ssh" / "vast.conf",
        STATE_PATH=tmp_path / "state.json",
        QUARANTINE_PATH=tmp_path / "quarantine.json",
    )

    def ready_only_second(host, port, identity, cfg, log, **kwargs):
        return host == "host-202"

    monkeypatch.setattr("devops.vast.vast_client.VastClient", FakeClient)
    monkeypatch.setattr("devops.vast.provision.check_local_ssh", lambda cfg: None)
    monkeypatch.setattr("devops.vast.provision.wait_for_ready_ssh", ready_only_second)
    monkeypatch.setattr("devops.vast.terminals.write_ssh_config", lambda *args: None)

    assert cmd_up(_up_args(count=1), cfg) == 0
    assert created == [(1, 201), (2, 202)]
    assert destroyed == [201]
    state = json.loads(cfg.STATE_PATH.read_text())
    assert [entry["id"] for entry in state["instances"]] == [202]
    machines, ips = active_exclusions(cfg)
    assert 10 in machines
    assert "10.0.0.1" in ips


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
        calls.append((args, cwd))
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
    assert calls[0][0] == [
        "git",
        "add",
        "-A",
        "--",
        "experiments/",
    ]


def test_self_destruct_qualifies_new_results_branch_ref(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, cwd=None):
        calls.append((args, cwd))
        if args[:3] == ["git", "diff", "--cached"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if args[:3] == ["git", "fetch", "origin"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("devops.vast.self_destruct._run", fake_run)

    assert push_results(
        branch="results",
        run_name="test",
        instance_id="1",
        repo=tmp_path,
    )
    assert (
        ["git", "push", "origin", "HEAD:refs/heads/results"],
        tmp_path,
    ) in calls


def test_self_destruct_defaults_to_experiment_repo_env(tmp_path, monkeypatch):
    repo = tmp_path / "experiment"
    (repo / "experiments").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    calls = []

    def fake_run(args, cwd=None):
        calls.append(cwd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("devops.vast.self_destruct._run", fake_run)
    monkeypatch.setenv("VAST_EXPERIMENT_DIR", str(repo))

    assert push_results(branch="results", run_name="test", instance_id="1")
    assert calls[0] == repo


def test_ssh_config_merges_instance_aliases_without_clobbering(tmp_path):
    from devops.vast.terminals import BoxConn, prune_ssh_aliases, write_ssh_config

    cfg = VastConfig(
        SSH_KEY_PATH=tmp_path / "id_rsa.pub",
        SSH_CONFIG_PATH=tmp_path / "ssh" / "vast.conf",
    )
    (tmp_path / "id_rsa.pub").write_text("ssh-rsa AAAA test\n")
    existing = BoxConn(
        alias="vast-111",
        host="1.1.1.1",
        port=1111,
        instance_id=111,
    )
    write_ssh_config([existing], cfg, log=lambda *_: None)
    write_ssh_config(
        [
            BoxConn(
                alias="vast-222",
                host="2.2.2.2",
                port=2222,
                instance_id=222,
            )
        ],
        cfg,
        log=lambda *_: None,
    )
    text = cfg.SSH_CONFIG_PATH.read_text()
    assert "Host vast-111" in text
    assert "HostName 1.1.1.1" in text
    assert "Host vast-222" in text
    assert "HostName 2.2.2.2" in text

    prune_ssh_aliases(["vast-222"], cfg, log=lambda *_: None)
    text = cfg.SSH_CONFIG_PATH.read_text()
    assert "Host vast-111" in text
    assert "Host vast-222" not in text


def test_cmd_up_uses_instance_id_ssh_aliases(tmp_path, monkeypatch):
    next_instance = iter((303,))
    written = []

    class FakeClient:
        def __init__(self, cfg, api_key=None):
            self.api_key = api_key or "test-key"

        def search_offers(self, query, offer_type):
            return [_offer(id=3, machine_id=30)]

        def ensure_ssh_key(self, path):
            return "ssh-rsa test"

        def create_instance(self, *args, **kwargs):
            return next(next_instance)

        def attach_ssh_key(self, *args, **kwargs):
            return None

        def wait_until_running(self, instance_id, log):
            return {"id": instance_id}

        def connection_info(self, inst, probe):
            return f"host-{inst['id']}", 22

    cfg = VastConfig(
        SSH_KEY_PATH=tmp_path / "id_rsa.pub",
        SSH_CONFIG_PATH=tmp_path / "ssh" / "vast.conf",
        STATE_PATH=tmp_path / "state.json",
        QUARANTINE_PATH=tmp_path / "quarantine.json",
    )

    monkeypatch.setattr("devops.vast.vast_client.VastClient", FakeClient)
    monkeypatch.setattr("devops.vast.provision.check_local_ssh", lambda cfg: None)
    monkeypatch.setattr(
        "devops.vast.provision.wait_for_ready_ssh", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "devops.vast.terminals.write_ssh_config",
        lambda boxes, *args, **kwargs: written.append([b.alias for b in boxes]),
    )

    assert cmd_up(_up_args(count=1), cfg) == 0
    assert written == [["vast-303"]]
    state = json.loads(cfg.STATE_PATH.read_text())
    assert state["instances"][0]["alias"] == "vast-303"


def test_check_local_ssh_requires_client_and_keypair(tmp_path, monkeypatch):
    cfg = VastConfig(SSH_KEY_PATH=tmp_path / "id_rsa.pub")
    monkeypatch.setattr("devops.vast.provision.shutil.which", lambda name: None)
    assert "ssh` client not found" in (check_local_ssh(cfg) or "")

    monkeypatch.setattr("devops.vast.provision.shutil.which", lambda name: "/usr/bin/ssh")
    assert "SSH keypair not found" in (check_local_ssh(cfg) or "")

    (tmp_path / "id_rsa").write_text("private\n")
    (tmp_path / "id_rsa.pub").write_text("ssh-rsa AAAA test\n")
    assert check_local_ssh(cfg) is None


def test_redact_instance_metadata_hides_control_plane_secrets():
    from devops.vast.redaction import redact_instance_metadata

    safe = redact_instance_metadata(
        {
            "id": 9,
            "actual_status": "running",
            "extra_env": {
                "VAST_GIT_REF": "abc",
                "GITHUB_TOKEN": "ghp_should_hide",
                "VAST_API_KEY": "vast_should_hide",
                "B2_APPLICATION_KEY": "b2_should_hide",
                "B2_BUCKET": "bucket",
            },
        }
    )
    env = safe["extra_env"]
    assert env["VAST_GIT_REF"] == "abc"
    assert env["B2_BUCKET"] == "bucket"
    assert env["GITHUB_TOKEN"] == "<REDACTED>"
    assert env["VAST_API_KEY"] == "<REDACTED>"
    assert env["B2_APPLICATION_KEY"] == "<REDACTED>"
    assert "ghp_should_hide" not in json.dumps(safe)
