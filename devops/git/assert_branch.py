"""Assert the active Git branch before mutating operations in shared worktrees.

Concurrent Cursor agent sessions can mutate one worktree's HEAD when they share
a checkout or when the agent root moves between sibling clones. Call
``assert_branch`` (or the CLI) immediately before ``git cherry-pick``, ``git
commit``, ``git push``, and similar write operations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class BranchMismatchError(Exception):
    """Raised when the active branch does not match the expected branch."""

    def __init__(
        self,
        repo: Path,
        expected: str,
        actual: str,
        *,
        operation: str | None = None,
    ) -> None:
        self.repo = repo
        self.expected = expected
        self.actual = actual
        self.operation = operation
        op = f" before {operation}" if operation else ""
        super().__init__(
            f"expected branch {expected!r} in {repo}{op}, but HEAD is {actual!r}"
        )


def current_branch(repo: Path) -> str:
    """Return the abbreviated name of the active branch in ``repo``."""
    completed = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git rev-parse failed in {repo}: {detail}")
    branch = completed.stdout.strip()
    if not branch or branch == "HEAD":
        raise RuntimeError(
            f"detached HEAD in {repo}; branch assertion requires a named branch"
        )
    return branch


def assert_branch(
    repo: Path,
    expected: str,
    *,
    operation: str | None = None,
) -> str:
    """Return the active branch when it matches ``expected``; otherwise raise."""
    actual = current_branch(repo)
    if actual != expected:
        raise BranchMismatchError(repo, expected, actual, operation=operation)
    return actual


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assert the active Git branch before a mutating git operation.",
    )
    parser.add_argument(
        "expected_branch",
        help="Branch name that must be checked out in the target repository.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository path (default: current working directory).",
    )
    parser.add_argument(
        "--operation",
        default=None,
        help="Optional label for the upcoming git command (included in errors).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo = args.repo.resolve()
    try:
        assert_branch(repo, args.expected_branch, operation=args.operation)
    except BranchMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
