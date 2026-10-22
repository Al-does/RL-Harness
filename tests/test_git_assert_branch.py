import subprocess
from pathlib import Path

import pytest

from devops.git.assert_branch import (
    BranchMismatchError,
    assert_branch,
    current_branch,
    main,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "README").write_text("seed\n")
    subprocess.run(["git", "add", "README"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_current_branch_returns_active_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    assert current_branch(repo) == "main"


def test_assert_branch_passes_when_branch_matches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    assert assert_branch(repo, "main") == "main"


def test_assert_branch_raises_on_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    with pytest.raises(BranchMismatchError) as excinfo:
        assert_branch(repo, "main", operation="git cherry-pick")
    assert excinfo.value.expected == "main"
    assert excinfo.value.actual == "feature"
    assert excinfo.value.operation == "git cherry-pick"


def test_cli_returns_nonzero_on_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert main(["main", "--repo", str(repo)]) == 1


def test_cli_succeeds_when_branch_matches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    assert main(["main", "--repo", str(repo)]) == 0
