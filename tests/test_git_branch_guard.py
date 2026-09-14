from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devops.git.branch_guard import (
    BranchMismatchError,
    assert_branch,
    current_branch,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    path.joinpath("README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_current_branch_returns_active_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    subprocess.run(
        ["git", "checkout", "-b", "experiment/mess3-paper-replication"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    assert current_branch(repo) == "experiment/mess3-paper-replication"


def test_assert_branch_passes_when_branch_matches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    subprocess.run(
        ["git", "checkout", "-b", "fix/vast-agent-issues"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    assert_branch(repo, "fix/vast-agent-issues", operation="cherry-pick")


def test_assert_branch_raises_on_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    subprocess.run(
        ["git", "checkout", "-b", "experiment/mess3-paper-replication"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    with pytest.raises(BranchMismatchError, match="cherry-pick"):
        assert_branch(
            repo,
            "fix/vast-agent-issues",
            operation="cherry-pick",
        )


def test_current_branch_rejects_detached_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    subprocess.run(
        ["git", "checkout", "--detach", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="detached HEAD"):
        current_branch(repo)
