"""Clean-worktree compact results publication for the ``results`` branch.

Never rebase experiment history onto the results branch. Overlay only the
compact result bundle, retry solely for genuine concurrent-update races, and
treat publication failure as a warning with a recoverable local/remote bundle.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PublicationResult:
    status: str  # succeeded | skipped | failed | warning
    detail: str
    branch: str
    commit: str | None = None
    attempts: int = 0
    concurrent_update: bool = False
    recoverable_bundle: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"succeeded", "skipped"}


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
    runner: Callable[..., subprocess.CompletedProcess],
) -> subprocess.CompletedProcess:
    return runner(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _git_auth_env(token: str) -> dict[str, str]:
    import base64

    encoded = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _is_non_fast_forward(stderr: str) -> bool:
    text = stderr.lower()
    return any(
        marker in text
        for marker in (
            "non-fast-forward",
            "fetch first",
            "rejected",
            "remote contains work",
            "! [rejected]",
        )
    )


def _is_content_conflict(stderr: str) -> bool:
    text = stderr.lower()
    return any(
        marker in text
        for marker in (
            "conflict",
            "merge conflict",
            "would be overwritten",
            "unmerged",
        )
    )


def collect_compact_bundle(
    experiment_repo: Path,
    *,
    relative_roots: tuple[str, ...] = ("experiments",),
) -> list[Path]:
    """Return tracked-or-new compact files under result roots."""
    files: list[Path] = []
    for root_name in relative_roots:
        root = experiment_repo / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(experiment_repo).as_posix()
            # Compact findings only; never publish ignored artifact trees.
            if "/artifacts/" in f"/{relative}/":
                continue
            if path.name.startswith("."):
                continue
            files.append(path)
    return sorted(files)


def publish_compact_results(
    *,
    experiment_repo: Path,
    remote_url: str,
    branch: str,
    commit_message: str,
    github_token: str,
    bot_name: str = "runpod-results-bot",
    bot_email: str = "runpod-results-bot@users.noreply.github.com",
    max_attempts: int = 6,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    bundle_files: list[Path] | None = None,
    work_root: Path | None = None,
) -> PublicationResult:
    """Publish from a clean worktree rooted at the current results branch tip.

    Algorithm:
    1. Collect the compact result bundle from the experiment checkout.
    2. Clone/fetch ``origin/<branch>`` into an isolated worktree.
    3. Overlay only bundle files (no experiment history rebase).
    4. Commit and push; on non-fast-forward, refresh tip and retry.
    5. Deterministic content conflicts fail immediately without looping.
    """
    run = runner or subprocess.run
    files = bundle_files if bundle_files is not None else collect_compact_bundle(
        experiment_repo
    )
    if not files:
        return PublicationResult(
            status="skipped",
            detail="no compact results to publish",
            branch=branch,
        )

    token_env = _git_auth_env(github_token)
    root = Path(work_root or tempfile.mkdtemp(prefix="rlh-results-"))
    worktree = root / "results-worktree"
    recoverable = root / "recoverable-bundle"
    recoverable.mkdir(parents=True, exist_ok=True)
    for path in files:
        relative = path.relative_to(experiment_repo)
        destination = recoverable / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    try:
        _run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", remote_url, str(worktree)],
            env=token_env,
            runner=run,
        )
        fetched = _run(
            ["git", "fetch", "--depth", "1", "origin", branch],
            cwd=worktree,
            env=token_env,
            check=False,
            capture_output=True,
            runner=run,
        )
        if fetched.returncode == 0:
            _run(
                ["git", "checkout", "--quiet", "-B", branch, "FETCH_HEAD"],
                cwd=worktree,
                runner=run,
            )
        else:
            # First publication may create the branch from an orphan empty tree.
            _run(["git", "checkout", "--orphan", branch], cwd=worktree, runner=run)
            _run(["git", "rm", "-rf", "--ignore-unmatch", "."], cwd=worktree, check=False, runner=run)

        _run(["git", "config", "user.name", bot_name], cwd=worktree, runner=run)
        _run(["git", "config", "user.email", bot_email], cwd=worktree, runner=run)

        for path in files:
            relative = path.relative_to(experiment_repo)
            destination = worktree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        _run(["git", "add", "-A", "--", "experiments"], cwd=worktree, runner=run)
        staged = _run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=worktree,
            check=False,
            runner=run,
        )
        if staged.returncode == 0:
            return PublicationResult(
                status="skipped",
                detail="compact results already present on results branch",
                branch=branch,
                recoverable_bundle=str(recoverable),
            )
        _run(
            ["git", "commit", "-m", commit_message],
            cwd=worktree,
            runner=run,
        )
        commit = _run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            capture_output=True,
            runner=run,
        ).stdout.strip()

        delay = 1.0
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            pushed = _run(
                ["git", "push", "origin", f"HEAD:refs/heads/{branch}"],
                cwd=worktree,
                env=token_env,
                check=False,
                capture_output=True,
                runner=run,
            )
            if pushed.returncode == 0:
                return PublicationResult(
                    status="succeeded",
                    detail=f"published compact results to {branch}",
                    branch=branch,
                    commit=commit,
                    attempts=attempt,
                    recoverable_bundle=str(recoverable),
                )
            stderr = (pushed.stderr or "") + (pushed.stdout or "")
            last_error = stderr.strip() or "git push failed"
            if _is_content_conflict(stderr) and not _is_non_fast_forward(stderr):
                return PublicationResult(
                    status="failed",
                    detail=(
                        "deterministic content conflict while overlaying "
                        f"compact results: {last_error[:300]}"
                    ),
                    branch=branch,
                    commit=commit,
                    attempts=attempt,
                    recoverable_bundle=str(recoverable),
                )
            if not _is_non_fast_forward(stderr):
                return PublicationResult(
                    status="failed",
                    detail=f"results publication failed: {last_error[:300]}",
                    branch=branch,
                    commit=commit,
                    attempts=attempt,
                    recoverable_bundle=str(recoverable),
                )
            # Genuine concurrent update: refresh tip, re-overlay, recommit.
            _run(
                ["git", "fetch", "origin", branch],
                cwd=worktree,
                env=token_env,
                runner=run,
            )
            _run(
                ["git", "reset", "--hard", "FETCH_HEAD"],
                cwd=worktree,
                runner=run,
            )
            for path in files:
                relative = path.relative_to(experiment_repo)
                destination = worktree / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            _run(["git", "add", "-A", "--", "experiments"], cwd=worktree, runner=run)
            staged = _run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=worktree,
                check=False,
                runner=run,
            )
            if staged.returncode != 0:
                _run(
                    ["git", "commit", "-m", commit_message],
                    cwd=worktree,
                    runner=run,
                )
                commit = _run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=worktree,
                    capture_output=True,
                    runner=run,
                ).stdout.strip()
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, 30.0)
        return PublicationResult(
            status="failed",
            detail=(
                "results publication exhausted concurrent-update retries: "
                f"{last_error[:300]}"
            ),
            branch=branch,
            commit=commit,
            attempts=max_attempts,
            concurrent_update=True,
            recoverable_bundle=str(recoverable),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return PublicationResult(
            status="failed",
            detail=f"results publication failed ({type(error).__name__})",
            branch=branch,
            recoverable_bundle=str(recoverable),
        )
