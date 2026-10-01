"""Fail fast when a mutating git command runs on the wrong branch.

Parallel Cursor agents on one machine can silently change which branch is
checked out in a sibling clone when the agent root moves. Wrap check-then-write
git sequences with this helper so wrong-branch mutations abort before they
commit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def current_branch(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        _fail(f"git branch query failed in {repo}: {proc.stderr.strip()}")
    branch = proc.stdout.strip()
    if not branch:
        _fail(f"detached HEAD in {repo}; branch assertion requires a named branch")
    return branch


def assert_branch(repo: Path, expected: tuple[str, ...]) -> None:
    if not expected:
        raise ValueError("expected must contain at least one branch name")
    actual = current_branch(repo)
    if actual not in expected:
        names = ", ".join(expected)
        _fail(
            f"git branch assertion failed in {repo}: expected {{{names}}}, got {actual!r}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a command only when the git checkout is on an expected branch.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="repository to check (default: cwd)",
    )
    parser.add_argument(
        "--expect",
        action="append",
        dest="expected",
        required=True,
        metavar="BRANCH",
        help="allowed branch name (repeatable)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command after -- (e.g. -- git cherry-pick abc)",
    )
    args = parser.parse_args(argv)
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("command required after --")
    assert_branch(args.repo.resolve(), tuple(args.expected))
    proc = subprocess.run(cmd, cwd=args.repo)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
