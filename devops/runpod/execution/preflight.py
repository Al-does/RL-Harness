"""Launcher preflight: refs, image, resources, and cost plan before spend."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from harness.resources import (
    ResourceContract,
    parse_hardware_from_argv,
    resource_contract_from_profile,
)

_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_DIGEST_IMAGE = re.compile(r"^.+@sha256:([0-9a-fA-F]{64})$")


class PreflightError(ValueError):
    """Deterministic rejection before any billing resource is created."""


@dataclass(frozen=True)
class RefResolution:
    label: str
    requested: str
    resolved_sha: str
    remote_url: str
    fetchable: bool


@dataclass
class PreflightPlan:
    experiment: RefResolution
    library: RefResolution
    image: str
    image_digest: str
    image_available: bool
    resource_contract: ResourceContract
    available_gpus: float
    available_cpus: float | None
    estimate: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                "label": self.experiment.label,
                "requested": self.experiment.requested,
                "resolved_sha": self.experiment.resolved_sha,
                "remote_url": self.experiment.remote_url,
                "fetchable": self.experiment.fetchable,
            },
            "library": {
                "label": self.library.label,
                "requested": self.library.requested,
                "resolved_sha": self.library.resolved_sha,
                "remote_url": self.library.remote_url,
                "fetchable": self.library.fetchable,
            },
            "image": self.image,
            "image_digest": self.image_digest,
            "image_available": self.image_available,
            "resource_contract": self.resource_contract.to_dict(),
            "available_gpus": self.available_gpus,
            "available_cpus": self.available_cpus,
            "estimate": dict(self.estimate),
            "notes": list(self.notes),
        }


def require_sha(value: str, label: str) -> str:
    if not _SHA.fullmatch(value):
        raise PreflightError(
            f"{label} must be an explicit 40- or 64-hex commit SHA"
        )
    return value.lower()


def resolve_ref_with_rev_parse(
    value: str,
    *,
    label: str,
    repository: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> str:
    """Resolve a ref to a full SHA via ``git rev-parse`` when possible."""
    if _SHA.fullmatch(value):
        return value.lower()
    run = runner or subprocess.run
    args = ["git", "rev-parse", "--verify", f"{value}^{{commit}}"]
    try:
        completed = run(
            args,
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(
            f"{label} {value!r} could not be resolved with git rev-parse"
        ) from error
    sha = completed.stdout.strip().lower()
    if not _SHA.fullmatch(sha):
        raise PreflightError(f"{label} resolved to a non-SHA value")
    return sha


def _github_repo_from_url(remote_url: str) -> tuple[str, str]:
    parsed = urlparse(remote_url)
    if parsed.hostname != "github.com":
        raise PreflightError(f"unsupported git host for preflight: {remote_url}")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise PreflightError(f"cannot parse GitHub repository from {remote_url}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    return owner, repo


def verify_remote_sha_fetchable(
    remote_url: str,
    sha: str,
    *,
    label: str,
    github_token: str | None = None,
    opener: Callable[..., Any] | None = None,
) -> None:
    """Fail closed unless GitHub has the exact commit object."""
    owner, repo = _github_repo_from_url(remote_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "rl-harness-preflight",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    request = urllib.request.Request(api_url, headers=headers, method="GET")
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=20) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise PreflightError(
                    f"{label} SHA {sha} is not fetchable from {remote_url} "
                    f"(HTTP {status})"
                )
            payload = json.loads(response.read().decode("utf-8"))
    except PreflightError:
        raise
    except urllib.error.HTTPError as error:
        raise PreflightError(
            f"{label} SHA {sha} is not fetchable from {remote_url} "
            f"(HTTP {error.code})"
        ) from error
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise PreflightError(
            f"{label} SHA fetchability probe failed for {remote_url}: "
            f"{type(error).__name__}"
        ) from error
    actual = str(payload.get("sha") or "").lower()
    if actual != sha.lower():
        raise PreflightError(
            f"{label} SHA {sha} did not resolve to itself on GitHub"
        )


def _ghcr_anonymous_token(
    repository: str,
    *,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Fetch the anonymous pull token GHCR requires even for public packages."""
    token_url = (
        "https://ghcr.io/token?service=ghcr.io&scope="
        f"repository:{repository}:pull"
    )
    open_url = opener or urllib.request.urlopen
    request = urllib.request.Request(token_url, method="GET")
    with open_url(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = payload.get("token")
    if not isinstance(token, str) or not token:
        raise PreflightError(
            f"GHCR anonymous token missing for repository {repository}"
        )
    return token


def verify_image_digest_available(
    image: str,
    *,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Require a digest-pinned image and probe registry availability."""
    match = _DIGEST_IMAGE.fullmatch(image)
    if not match:
        raise PreflightError("image must be immutable and digest-pinned")
    digest = f"sha256:{match.group(1).lower()}"
    reference, _, _ = image.partition("@")
    if reference.startswith("ghcr.io/"):
        # Public GHCR packages still require the anonymous bearer handshake.
        # Private packages fail closed when the token cannot pull the manifest.
        name = reference.removeprefix("ghcr.io/")
        url = f"https://ghcr.io/v2/{name}/manifests/{digest}"
        open_url = opener or urllib.request.urlopen
        try:
            token = _ghcr_anonymous_token(name, opener=open_url)
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": (
                        "application/vnd.oci.image.manifest.v1+json, "
                        "application/vnd.docker.distribution.manifest.v2+json"
                    ),
                    "Authorization": f"Bearer {token}",
                },
                method="GET",
            )
            with open_url(request, timeout=20) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    raise PreflightError(
                        f"image {image} is not anonymously pullable "
                        f"(HTTP {status})"
                    )
        except PreflightError:
            raise
        except urllib.error.HTTPError as error:
            raise PreflightError(
                f"image {image} is not anonymously pullable "
                f"(HTTP {error.code})"
            ) from error
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise PreflightError(
                f"image availability probe failed for {image}: "
                f"{type(error).__name__}"
            ) from error
    elif "/" in reference:
        # Non-GHCR digests are accepted when syntactically pinned; operators
        # remain responsible for registry auth on private hosts.
        pass
    return digest


def build_resource_contract_for_run(
    run_argv: list[str],
    *,
    default_profile: str,
    available_gpus: float,
    available_cpus: float | None = None,
) -> ResourceContract:
    profile_name, smoke = parse_hardware_from_argv(
        run_argv,
        default_profile=default_profile,
    )
    contract = resource_contract_from_profile(profile_name, smoke=smoke)
    reason = contract.rejection_reason(
        available_gpus=available_gpus,
        available_cpus=available_cpus,
    )
    if reason:
        raise PreflightError(reason)
    return contract


def run_preflight(
    *,
    experiment_ref: str,
    library_ref: str,
    experiment_repo_url: str,
    library_repo_url: str,
    image: str,
    run_argv: list[str],
    available_gpus: float,
    estimate: dict[str, Any],
    default_profile: str = "cuda4090_gpuinfer",
    available_cpus: float | None = None,
    github_token: str | None = None,
    experiment_repository: str | None = None,
    library_repository: str | None = None,
    verify_image: bool = True,
    verify_remote_refs: bool = True,
    rev_parse_runner: Callable[..., subprocess.CompletedProcess] | None = None,
    image_opener: Callable[..., Any] | None = None,
) -> PreflightPlan:
    """Resolve and validate every deterministic input before provisioning."""
    notes: list[str] = []
    experiment_sha = resolve_ref_with_rev_parse(
        experiment_ref,
        label="experiment-ref",
        repository=experiment_repository,
        runner=rev_parse_runner,
    )
    library_sha = resolve_ref_with_rev_parse(
        library_ref,
        label="library-ref",
        repository=library_repository,
        runner=rev_parse_runner,
    )
    commit_opener = image_opener
    if verify_remote_refs:
        verify_remote_sha_fetchable(
            experiment_repo_url,
            experiment_sha,
            label="experiment",
            github_token=github_token,
            opener=commit_opener,
        )
        verify_remote_sha_fetchable(
            library_repo_url,
            library_sha,
            label="library",
            github_token=github_token,
            opener=commit_opener,
        )
    if verify_image:
        image_digest = verify_image_digest_available(image, opener=image_opener)
    else:
        match = _DIGEST_IMAGE.fullmatch(image)
        if not match:
            raise PreflightError("image must be immutable and digest-pinned")
        image_digest = f"sha256:{match.group(1).lower()}"
    contract = build_resource_contract_for_run(
        run_argv,
        default_profile=default_profile,
        available_gpus=available_gpus,
        available_cpus=available_cpus,
    )
    notes.extend(contract.notes)
    return PreflightPlan(
        experiment=RefResolution(
            label="experiment",
            requested=experiment_ref,
            resolved_sha=experiment_sha,
            remote_url=experiment_repo_url,
            fetchable=True,
        ),
        library=RefResolution(
            label="library",
            requested=library_ref,
            resolved_sha=library_sha,
            remote_url=library_repo_url,
            fetchable=True,
        ),
        image=image,
        image_digest=image_digest,
        image_available=True,
        resource_contract=contract,
        available_gpus=available_gpus,
        available_cpus=available_cpus,
        estimate=dict(estimate),
        notes=notes,
    )


def format_preflight_plan(plan: PreflightPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True)
