"""Fail fast when concurrent agents leave a repository on the wrong branch."""

from __future__ import annotations

import subprocess
from pathlib import Path


class BranchMismatchError(RuntimeError):
    """Raised when the active branch does not match the expected branch."""

    def __init__(
        self,
        repo: Path,
        expected: str,
        actual: str,
        *,
        operation: str,
    ) -> None:
        self.repo = repo
        self.expected = expected
        self.actual = actual
        self.operation = operation
        super().__init__(
            f"Refusing {operation}: repository {repo} is on branch "
            f"'{actual}', expected '{expected}'. "
            "Concurrent agent sessions may have changed HEAD; "
            "re-checkout the intended branch or use an isolated worktree."
        )


def current_branch(repo: Path | str) -> str:
    """Return the symbolic branch name for ``repo``."""
    path = Path(repo)
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    if not branch:
        raise RuntimeError(
            f"Repository {path} has no symbolic branch (detached HEAD?)."
        )
    return branch


def assert_branch(
    repo: Path | str,
    expected: str,
    *,
    operation: str = "git mutation",
) -> None:
    """Fail fast if ``repo`` is not on ``expected`` before a mutating git command."""
    actual = current_branch(repo)
    if actual != expected:
        raise BranchMismatchError(
            Path(repo), expected, actual, operation=operation
        )
