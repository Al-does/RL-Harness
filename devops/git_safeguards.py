"""Branch assertions before local Git mutations.

Concurrent Cursor agents sharing one worktree can silently change HEAD.
Call ``assert_branch`` immediately before cherry-pick, commit, or push
operations so mutations fail fast on the wrong branch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class GitBranchMismatchError(RuntimeError):
    """Raised when the active branch does not match the expected branch."""

    def __init__(self, expected: str, actual: str, repo: Path) -> None:
        self.expected = expected
        self.actual = actual
        self.repo = repo
        super().__init__(
            f"expected git branch '{expected}' in {repo}, but HEAD is '{actual}'"
        )


def _repo_root(repo: Path | None) -> Path:
    if repo is None:
        return Path.cwd()
    return repo.resolve()


def current_branch(repo: Path | None = None) -> str:
    """Return the short name of the checked-out branch, or ``detached@<sha>``."""

    root = _repo_root(repo)
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    detached = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return f"detached@{detached.stdout.strip()}"


def assert_branch(expected: str, repo: Path | None = None) -> str:
    """Return the active branch after verifying it matches ``expected``."""

    actual = current_branch(repo)
    if actual != expected:
        raise GitBranchMismatchError(expected, actual, _repo_root(repo))
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Git branch safeguards for concurrent agent workflows."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    assert_cmd = sub.add_parser(
        "assert",
        help="Exit non-zero if HEAD is not on the expected branch.",
    )
    assert_cmd.add_argument("branch", help="Expected branch name")
    assert_cmd.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository root (defaults to the current directory)",
    )
    show_cmd = sub.add_parser("show", help="Print the current branch name.")
    show_cmd.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository root (defaults to the current directory)",
    )

    args = parser.parse_args(argv)
    if args.command == "assert":
        try:
            assert_branch(args.branch, args.repo)
        except GitBranchMismatchError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if args.command == "show":
        print(current_branch(args.repo))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
