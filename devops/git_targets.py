"""GitHub clone target resolution shared by remote execution backends."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

from devops.runpod.execution.preflight import verify_remote_sha_fetchable


@dataclass(frozen=True)
class GitTarget:
    url: str
    source: str


def normalize_github_url(value: str) -> str:
    match = re.search(
        r"(?:^git@github\.com:|(?:^|/|@)github\.com/)([\w.-]+)/([\w.-]+)(?:/)?$", value
    )
    if not match:
        raise ValueError(
            "Expected a GitHub remote; supply --experiment-repo-url or --library-repo-url"
        )
    owner, repo = match.groups()
    return f"https://github.com/{owner}/{repo.removesuffix('.git')}.git"


def experiment_checkout(path: str | Path | None = None) -> Path | None:
    if path:
        return Path(path).expanduser().resolve()
    cwd = Path.cwd().resolve()
    return (
        cwd
        if (cwd / "experiments").is_dir() and (cwd / "pyproject.toml").is_file()
        else None
    )


def resolve_target(
    override: str | None, checkout: Path | None, *, default: str | None = None
) -> GitTarget:
    if override:
        return GitTarget(normalize_github_url(override), "flag")
    if checkout is not None:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=checkout,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return GitTarget(
                normalize_github_url(result.stdout.strip()), f"origin of {checkout}"
            )
    if default:
        return GitTarget(normalize_github_url(default), "default")
    raise ValueError(
        "No experiment origin available; use --experiment-repo PATH or --experiment-repo-url URL"
    )


def remote_ref_sha(
    url: str, ref: str, token: str | None, *, require_contained: bool = False
) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
            GIT_CONFIG_VALUE_0=f"Authorization: Basic {auth}",
        )
    result = subprocess.run(
        ["git", "ls-remote", url], env=env, capture_output=True, text=True, timeout=60
    )
    if result.returncode:
        raise ValueError(f"git ls-remote failed for {url}")
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        verify_remote_sha_fetchable(url, ref.lower(), label="clone", github_token=token)
        if require_contained:
            with tempfile.TemporaryDirectory(prefix="clone-preflight-") as directory:
                for command in (
                    ["git", "init", "--quiet"],
                    [
                        "git",
                        "fetch",
                        "--quiet",
                        url,
                        "+refs/heads/*:refs/remotes/clone/*",
                    ],
                ):
                    fetched = subprocess.run(
                        command,
                        cwd=directory,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if fetched.returncode:
                        raise ValueError(f"Cannot verify HEAD containment on {url}")
                contained = subprocess.run(
                    ["git", "branch", "-r", "--contains", ref],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                )
                if contained.returncode or not contained.stdout.strip():
                    raise ValueError(
                        f"HEAD {ref} is not contained in clone remote {url}; push it first or pass --branch"
                    )
        return ref.lower()
    refs = dict(
        (line.split()[1], line.split()[0])
        for line in result.stdout.splitlines()
        if len(line.split()) == 2
    )
    sha = refs.get(f"refs/heads/{ref}") or refs.get(ref)
    if not sha or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError(f"Ref {ref!r} does not exist on {url}")
    return sha


def verify_push_permission(url: str, token: str | None) -> None:
    if not token:
        raise ValueError("Publication requires a GitHub token with push permission")
    slug = url.removeprefix("https://github.com/").removesuffix(".git")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot verify push permission on {slug}") from error
    if not payload.get("permissions", {}).get("push"):
        raise ValueError(f"GitHub token lacks push permission on {slug}")
