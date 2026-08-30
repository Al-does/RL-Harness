import subprocess
from pathlib import Path

import pytest

from devops.git_guard import assert_branch, current_branch, main


def _init_repo(path: Path, branch: str = "main") -> None:
    subprocess.run(["git", "init", "-b", branch], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README").write_text("init\n")
    subprocess.run(["git", "add", "README"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_current_branch_returns_checked_out_branch(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, branch="main")
    subprocess.run(["git", "checkout", "-b", "feature/x"], cwd=repo, check=True, capture_output=True)

    assert current_branch(repo) == "feature/x"


def test_assert_branch_passes_on_match(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    assert_branch(repo, ("main",))


def test_assert_branch_fails_on_mismatch(tmp_path: Path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    with pytest.raises(SystemExit) as exc:
        assert_branch(repo, ("other-branch",))

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "expected {other-branch}" in captured.err
    assert "got 'main'" in captured.err


def test_main_runs_wrapped_command_on_expected_branch(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    with pytest.raises(SystemExit) as exc:
        main(["--repo", str(repo), "--expect", "main", "--", "git", "status", "--short"])

    assert exc.value.code == 0


def test_main_aborts_before_command_on_wrong_branch(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    marker = repo / "would-run"
    marker.write_text("no\n")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--repo",
                str(repo),
                "--expect",
                "wrong-branch",
                "--",
                "python",
                "-c",
                f"open({marker!r}, 'w').write('yes')",
            ]
        )

    assert exc.value.code == 1
    assert marker.read_text() == "no\n"
