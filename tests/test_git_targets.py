from pathlib import Path
import io
from types import SimpleNamespace

import pytest

from devops import git_targets as targets
from devops.vast import provision
from devops.vast.config import VastConfig


@pytest.mark.parametrize(
    "remote",
    [
        "git@github.com:owner/repo.git",
        "ssh://git@github.com/owner/repo",
        "https://github.com/owner/repo",
        "https://git-manager.devin.ai/proxy/github.com/owner/repo",
    ],
)
def test_normalize(remote):
    assert targets.normalize_github_url(remote) == "https://github.com/owner/repo.git"


def test_precedence_and_missing_origin(monkeypatch):
    monkeypatch.setattr(
        targets.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="git@github.com:fork/science.git\n"
        ),
    )
    assert (
        targets.resolve_target(
            "https://github.com/override/repo", Path("/checkout")
        ).source
        == "flag"
    )
    assert (
        targets.resolve_target(None, Path("/checkout")).url
        == "https://github.com/fork/science.git"
    )
    assert (
        targets.resolve_target(
            None, None, default="https://github.com/library/repo"
        ).source
        == "default"
    )
    monkeypatch.setattr(
        targets.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(
        ValueError, match="--experiment-repo PATH or --experiment-repo-url URL"
    ):
        targets.resolve_target(None, Path("/checkout"))
    with pytest.raises(ValueError, match="GitHub"):
        targets.normalize_github_url("https://gitlab.com/owner/repo")


def test_head_must_be_on_clone_remote(monkeypatch):
    def run(command, **kwargs):
        if command[1] == "ls-remote":
            return SimpleNamespace(
                returncode=0, stdout=f"{'b' * 40}\trefs/heads/main\n"
            )
        if command[1] == "branch":
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(targets.subprocess, "run", run)
    with pytest.raises(ValueError, match="not contained in clone remote"):
        targets.remote_ref_sha(
            "https://github.com/fork/repo.git", "a" * 40, None, require_contained=True
        )


def test_failure_precedes_offer_search(monkeypatch, capsys):
    monkeypatch.setattr(
        provision,
        "remote_ref_sha",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("git ls-remote failed")),
    )
    args = provision.build_parser().parse_args(
        [
            "up",
            "--experiment-repo-url",
            "https://github.com/fork/repo",
            "--branch",
            "main",
            "--dry-run",
        ]
    )
    assert provision.cmd_up(args, VastConfig()) == 2
    output = capsys.readouterr().out
    assert "ls-remote failed" in output
    assert "searching" not in output


@pytest.mark.parametrize(
    "disabled_flag,reason", [(False, "no --run"), (True, "--no-self-destruct")]
)
def test_dry_run_targets(monkeypatch, capsys, disabled_flag, reason):
    class Client:
        api_key = "test"

        def __init__(self, *a):
            pass

        def search_offers(self, *a, **kw):
            return []

    monkeypatch.setattr("devops.vast.vast_client.VastClient", Client)
    monkeypatch.setattr(provision, "remote_ref_sha", lambda *a, **kw: "a" * 40)
    args = provision.build_parser().parse_args(
        [
            "up",
            "--experiment-repo-url",
            "https://github.com/fork/repo",
            "--branch",
            "main",
            "--dry-run",
        ]
    )
    args.no_self_destruct = disabled_flag
    assert provision.cmd_up(args, VastConfig()) == 0
    output = capsys.readouterr().out
    for value in [
        "CLONE AND PUBLICATION TARGETS",
        "https://github.com/fork/repo.git",
        "ref main",
        "SHA " + "a" * 40,
        "source: flag",
        "Library:",
        f"Publication: disabled ({reason})",
        "B2 credentials forwarded: no",
    ]:
        assert value in output


@pytest.mark.parametrize("fetch_fails", [False, True])
def test_full_sha_fetchability(monkeypatch, fetch_fails):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert "secret-token" not in " ".join(command)
        assert kwargs["env"]["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")
        return SimpleNamespace(
            returncode=int(fetch_fails and command[1] == "fetch"), stdout=""
        )

    monkeypatch.setattr(targets.subprocess, "run", run)
    if fetch_fails:
        with pytest.raises(ValueError, match="not fetchable"):
            targets.remote_ref_sha(
                "https://github.com/fork/repo.git", "a" * 40, "secret-token"
            )
    else:
        assert (
            targets.remote_ref_sha(
                "https://github.com/fork/repo.git", "a" * 40, "secret-token"
            )
            == "a" * 40
        )
    assert [
        "git",
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "https://github.com/fork/repo.git",
        "a" * 40,
    ] in commands


@pytest.mark.parametrize("filter_rejected", [False, True])
def test_head_containment_filter_and_fallback(monkeypatch, filter_rejected):
    commands = []

    def run(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(
            returncode=int(filter_rejected and "--filter=blob:none" in command),
            stdout="clone/main\n" if command[1] == "branch" else "",
        )

    monkeypatch.setattr(targets.subprocess, "run", run)
    assert (
        targets.remote_ref_sha(
            "https://github.com/fork/repo.git", "a" * 40, None, require_contained=True
        )
        == "a" * 40
    )
    fetches = [
        command
        for command in commands
        if command[-1] == "+refs/heads/*:refs/remotes/clone/*"
    ]
    assert "--filter=blob:none" in fetches[0]
    assert len(fetches) == (2 if filter_rejected else 1)
    if filter_rejected:
        assert "--filter=blob:none" not in fetches[1]


@pytest.mark.parametrize("stdout,code", [("", 0), ("", 1)])
def test_remote_ref_failure(monkeypatch, stdout, code):
    monkeypatch.setattr(
        targets.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=code, stdout=stdout),
    )
    with pytest.raises(ValueError):
        targets.remote_ref_sha("https://github.com/fork/repo.git", "main", None)


def test_authenticated_branch_resolution(monkeypatch):
    def run(command, **kwargs):
        assert "secret-token" not in " ".join(command)
        assert kwargs["env"]["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")
        return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\trefs/heads/main\n")

    monkeypatch.setattr(targets.subprocess, "run", run)
    assert (
        targets.remote_ref_sha(
            "https://github.com/fork/repo.git", "main", "secret-token"
        )
        == "a" * 40
    )


@pytest.mark.parametrize(
    "payload,allowed",
    [
        (b'{"permissions":{"push":true}}', True),
        (b'{"permissions":{"push":false}}', False),
        (b"{}", False),
    ],
)
def test_publication_permissions(monkeypatch, payload, allowed):
    monkeypatch.setattr(
        targets.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload)
    )
    if allowed:
        targets.verify_push_permission("https://github.com/fork/repo.git", "token")
    else:
        with pytest.raises(ValueError, match="push permission"):
            targets.verify_push_permission("https://github.com/fork/repo.git", "token")


def test_publication_permission_probe_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(targets.urllib.request, "urlopen", fail)
    with pytest.raises(ValueError, match="Cannot verify"):
        targets.verify_push_permission("https://github.com/fork/repo.git", "token")
    with pytest.raises(ValueError, match="requires a GitHub token"):
        targets.verify_push_permission("https://github.com/fork/repo.git", None)
