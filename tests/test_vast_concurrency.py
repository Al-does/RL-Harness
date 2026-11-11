import multiprocessing
from types import SimpleNamespace

import pytest

from devops.vast.config import VastConfig
from devops.vast.provision import wait_for_ready_ssh
from devops.vast.terminals import BoxConn, _parse_host_blocks, write_ssh_config


def _upsert_alias(cfg, barrier, instance_id):
    barrier.wait(timeout=30)
    write_ssh_config(
        [BoxConn(f"vast-{instance_id}", "localhost", 22, instance_id)],
        cfg, log=lambda *_: None,
    )


def test_parallel_alias_writers(tmp_path):
    cfg = VastConfig(SSH_CONFIG_PATH=tmp_path / ".ssh/config.d/vast.conf")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(12)
    processes = [ctx.Process(target=_upsert_alias, args=(cfg, barrier, i))
                 for i in range(12)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    assert set(_parse_host_blocks(cfg.SSH_CONFIG_PATH.read_text())) == {
        f"vast-{i}" for i in range(12)
    }
    assert (tmp_path / ".ssh/config").read_text().count("Include") == 1
    assert not cfg.SSH_CONFIG_PATH.with_name("vast.conf.tmp").exists()


@pytest.mark.parametrize("stamp_results,expected_attempts,warn", [
    ([1, 0], 2, False),
    ([0], 1, False),
    ([1, 1, 1], 3, True),
])
@pytest.mark.parametrize("log_ready", [False, True])
def test_stamp_retries_until_success_or_readiness(
    tmp_path, monkeypatch, stamp_results, expected_attempts, warn, log_ready,
):
    attempts = []
    polls = []
    logs = []
    results = iter(stamp_results)

    def run(command, **kwargs):
        code = 1
        if "echo " in command[-1]:
            attempts.append(command[-1])
            code = next(results)
        elif command[-1] == "test -f /root/.vast_ready":
            polls.append(command)
            code = 0 if len(polls) == 3 and not log_ready else 1
        return SimpleNamespace(returncode=code, stdout="")

    client = SimpleNamespace(v=SimpleNamespace(
        logs=lambda *args, **kwargs: "bootstrap complete" if len(polls) == 3 else "",
    ))
    monkeypatch.setattr("devops.vast.provision.subprocess.run", run)
    monkeypatch.setattr("devops.vast.provision.time.sleep", lambda _: None)
    assert wait_for_ready_ssh(
        "localhost", 22, tmp_path / "id_rsa", VastConfig(READY_TIMEOUT_S=30),
        log=logs.append, instance_id=123, vast_client=client if log_ready else None,
    )
    assert len(attempts) == expected_attempts
    assert all("echo 123 > /root/vast_instance_id" in attempt for attempt in attempts)
    assert any("could not write" in line for line in logs) == warn


def test_readiness_without_instance_id_does_not_stamp(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command[-1])
        return SimpleNamespace(returncode=int(command[-1] != "test -f /root/.vast_ready"))

    monkeypatch.setattr("devops.vast.provision.subprocess.run", run)
    assert wait_for_ready_ssh("localhost", 22, tmp_path / "key", VastConfig())
    assert not any("vast_instance_id" in command for command in commands)
