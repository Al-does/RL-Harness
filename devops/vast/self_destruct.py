"""On-box teardown: push compact experiment results, then destroy the box.

This module runs *on the vast box*, inside the training env — which is
`uv sync`ed WITHOUT the `devops` group, so ``vastai`` is NOT importable here.
The instance destroy therefore goes straight to the vast REST API over stdlib
``urllib`` (Authorization: Bearer <key>), keeping the training env clean while
still freeing the box.

Design guarantees:
  - push_results never fails when there are no new experiment results.
  - Publication overlays only ``experiments/**/results/**`` files (never
    checkpoints under ``artifacts/``). What belongs in ``results/`` vs
    ``artifacts/`` is chosen by each experiment recipe; the provisioner does
    not apply extension or size filters. Remote boxes do not rebase experiment
    history; they push to the launch branch (``VAST_EXPERIMENT_GIT_REF`` by
    default) and merge on genuine concurrent-update races.
  - push_results_and_destroy destroys only after Git publication and, when
    required, verified B2 durability; a failed transfer leaves the box running
    so its only copy can be recovered.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .redaction import redact_sensitive

VAST_API_BASE = os.environ.get("VAST_URL", "https://console.vast.ai")
LIBRARY_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID_FILE = Path("/root/vast_instance_id")
DURABILITY_REGISTRY = Path("/root/.vast_durability_manifests.json")


def experiment_repo_root() -> Path:
    """Return the on-box experiment checkout (science + results push target)."""
    env_dir = os.environ.get("VAST_EXPERIMENT_DIR")
    if env_dir:
        path = Path(env_dir)
        if path.is_dir():
            return path
    cwd = Path.cwd()
    if (cwd / "experiments").is_dir() and (cwd / "pyproject.toml").is_file():
        return cwd
    return LIBRARY_ROOT


def resolve_publish_branch(explicit: str | None = None) -> str:
    """Return the branch remote boxes should push compact results to."""
    for candidate in (
        explicit,
        os.environ.get("VAST_PUBLISH_BRANCH"),
        os.environ.get("VAST_RESULTS_BRANCH"),
        os.environ.get("VAST_EXPERIMENT_GIT_REF"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return "main"


def _is_gitignored(repo: Path, relative: str) -> bool:
    """True when ``relative`` is excluded by the experiment repo's gitignore."""

    check = _run(["git", "check-ignore", "-q", "--", relative], cwd=repo)
    return check.returncode == 0


def collect_compact_result_paths(repo: Path) -> list[Path]:
    """Return compact result files under ``experiments/**/results/**``."""
    experiments = repo / "experiments"
    if not experiments.is_dir():
        return []
    paths: list[Path] = []
    for path in experiments.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(repo).as_posix()
        if "/artifacts/" in f"/{relative}/":
            continue
        if "/.smoke/" in f"/{relative}/":
            continue
        if "/results/" not in f"/{relative}/":
            continue
        if _is_gitignored(repo, relative):
            continue
        paths.append(path)
    return sorted(paths)


def _load_durability_registry() -> list[str]:
    if not DURABILITY_REGISTRY.is_file():
        return []
    try:
        payload = json.loads(DURABILITY_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(entry) for entry in payload if isinstance(entry, str)]


def record_durability_manifests(
    repo: Path,
    manifest_paths: Iterable[Path],
    *,
    log=print,
) -> None:
    """Remember run manifests pushed before final self-destruct verification."""
    recorded = set(_load_durability_registry())
    for manifest in manifest_paths:
        path = manifest if manifest.is_absolute() else repo / manifest
        if path.is_file():
            recorded.add(str(path.resolve()))
    if not recorded:
        return
    DURABILITY_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    DURABILITY_REGISTRY.write_text(
        json.dumps(sorted(recorded), indent=2) + "\n"
    )
    _log(
        f"recorded {len(recorded)} durability manifest(s) for teardown",
        log,
    )


def manifests_from_durability_registry(repo: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in _load_durability_registry():
        path = Path(raw)
        if not path.is_absolute():
            path = repo / raw
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def pending_run_manifests(repo: Path) -> list[Path]:
    """Return terminal run manifests changed by the current remote run."""

    status = _run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            "experiments",
        ],
        cwd=repo,
    )
    if status.returncode != 0:
        return []
    manifests = []
    for entry in status.stdout.split("\0"):
        if not entry:
            continue
        relative = entry[3:]
        if relative.endswith("/results/run_manifest.json"):
            manifests.append(repo / relative)
            continue
        if relative.endswith("/run_manifest.json") and "/results/" in relative:
            manifests.append(repo / relative)
    return sorted(set(manifests))


def _verify_manifest_durability(manifests: list[Path], log=print) -> bool:
    for path in manifests:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            _log(f"cannot read durability manifest {path}: {error}", log)
            return False
        if manifest.get("status") not in {"completed", "failed"}:
            _log(f"run manifest is not terminal: {path}", log)
            return False
        remote = manifest.get("remote_artifacts")
        if not isinstance(remote, dict) or remote.get("status") != "completed":
            _log(f"B2 durability did not complete for {path}", log)
            return False
        remote_manifest = path.parent / "remote_artifacts.json"
        if not remote_manifest.is_file():
            _log(f"remote artifact manifest is missing: {remote_manifest}", log)
            return False
    return True


def required_durability_completed(repo: Path, log=print) -> bool:
    """Verify pending runs reached terminal state and completed B2 upload."""

    manifests = pending_run_manifests(repo)
    if not manifests:
        manifests = manifests_from_durability_registry(repo)
        if manifests:
            _log(
                f"checking {len(manifests)} durability registry manifest(s)",
                log,
            )
    if not manifests:
        _log("required durability found no pending run manifest", log)
        return False
    return _verify_manifest_durability(manifests, log)


def _run(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd or experiment_repo_root()),
        capture_output=True, text=True,
    )


def _log(msg: str, log=print, secrets: Iterable[str | None] = ()) -> None:
    known_secrets = (
        *secrets,
        os.environ.get("VAST_API_KEY"),
        os.environ.get("GH_TOKEN"),
        os.environ.get("B2_APPLICATION_KEY"),
        os.environ.get("B2_APPLICATION_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
    )
    log(f"[self_destruct] {redact_sensitive(msg, known_secrets)}")


def push_results(
    branch: str | None = None,
    run_name: str = "run",
    instance_id: Optional[str] = None,
    repo: Optional[Path] = None,
    attempts: int = 6,
    log=print,
) -> bool:
    """Commit and push compact experiment results to the launch branch.

    Returns True (success, no-op) when there is nothing new to push. Checkpoints,
    raw payloads, and other ignored ``artifacts/`` trees are never staged.
    """
    repo = repo or experiment_repo_root()
    branch = resolve_publish_branch(branch)
    result_paths = collect_compact_result_paths(repo)
    if not result_paths:
        _log("no new compact experiment results to push", log)
        return True

    for path in result_paths:
        add = _run(
            ["git", "add", "--", path.relative_to(repo).as_posix()],
            cwd=repo,
        )
        if add.returncode != 0:
            _log(f"git add failed: {add.stderr.strip()}", log)
            return False

    staged = _run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if staged.returncode == 0:
        _log("no new compact experiment results to push", log)
        return True

    label = (
        f"results: {run_name} (vast {instance_id})"
        if instance_id
        else f"results: {run_name}"
    )
    commit = _run(["git", "commit", "-m", label], cwd=repo)
    if commit.returncode != 0:
        _log(
            f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}",
            log,
        )
        return False

    manifest_paths = [
        path
        for path in result_paths
        if path.name == "run_manifest.json" and "/results/" in path.as_posix()
    ]
    if manifest_paths:
        record_durability_manifests(repo, manifest_paths, log=log)

    delay = 1.0
    for i in range(1, attempts + 1):
        fetched = _run(["git", "fetch", "origin", branch], cwd=repo)
        if fetched.returncode == 0:
            merged = _run(["git", "merge", "--no-edit", "FETCH_HEAD"], cwd=repo)
            if merged.returncode != 0:
                _run(["git", "merge", "--abort"], cwd=repo)
                detail = merged.stderr.strip() or merged.stdout.strip()
                if "CONFLICT" in detail.upper():
                    _log(
                        f"merge conflict while joining origin/{branch}; "
                        "resolve manually on a workstation: "
                        f"{detail[:300]}",
                        log,
                    )
                    return False
                _log(f"merge failed (attempt {i}): {detail}", log)
        # else: branch doesn't exist remotely yet; push creates it.

        pushed = _run(
            ["git", "push", "origin", f"HEAD:refs/heads/{branch}"],
            cwd=repo,
        )
        if pushed.returncode == 0:
            _log(f"pushed compact results to {branch}", log)
            return True

        _log(f"push rejected (attempt {i}/{attempts}): {pushed.stderr.strip()}", log)
        time.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 2, 30.0)

    _log(f"push failed after {attempts} attempts", log)
    return False


def _list_instances(api_key: str, log=print) -> list[dict]:
    url = f"{VAST_API_BASE}/api/v0/instances/"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        _log(
            f"could not list instances to resolve label: {e}",
            log,
            secrets=(api_key,),
        )
        return []
    instances = data.get("instances", []) or []
    return instances if isinstance(instances, list) else []


def _resolve_instance_id_by_label(label: str, api_key: str, log=print) -> Optional[str]:
    """Find this box's instance id by its unique label, via the vast REST API.

    The instance id (``new_contract``) is only known to the *local* provisioner
    after create, so it can't be injected into the pre-creation env. We inject a
    unique ``VAST_INSTANCE_LABEL`` instead and look the id up here.
    """
    for inst in _list_instances(api_key, log=log):
        if str(inst.get("label") or "") == label:
            return str(inst.get("id"))
    _log(f"no instance matched label {label!r}", log)
    return None


def _read_instance_id_file() -> Optional[str]:
    try:
        value = INSTANCE_ID_FILE.read_text().strip()
    except OSError:
        return None
    return value or None


def destroy_self(instance_id: str, api_key: str, log=print) -> bool:
    """DELETE the instance via the vast REST API (no vastai dependency)."""
    url = f"{VAST_API_BASE}/api/v0/instances/{instance_id}/"
    req = urllib.request.Request(
        url, data=b"{}", method="DELETE",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", "replace")
        _log(
            f"destroy request sent for instance {instance_id}: {body[:200]}",
            log,
            secrets=(api_key,),
        )
        return True
    except urllib.error.HTTPError as e:
        _log(
            f"destroy HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}",
            log,
            secrets=(api_key,),
        )
        return False
    except Exception as e:  # noqa: BLE001 — best-effort teardown
        _log(f"destroy error: {e}", log, secrets=(api_key,))
        return False


def _resolve_and_destroy(
    instance_id: Optional[str] = None,
    api_key: Optional[str] = None,
    label: Optional[str] = None,
    log=print,
) -> bool:
    """Resolve this box's id (explicit -> env -> label lookup) and destroy it.

    Shared by the run-finished teardown and the max-age watchdog so both go
    through the same REST destroy with the same graceful skips.
    """
    api_key = api_key or os.environ.get("VAST_API_KEY")
    instance_id = (
        instance_id
        or os.environ.get("VAST_INSTANCE_ID")
        or os.environ.get("CONTAINER_ID")
        or _read_instance_id_file()
    )
    label = label or os.environ.get("VAST_INSTANCE_LABEL")
    if not api_key:
        _log("missing VAST_API_KEY; skipping destroy", log)
        return False
    if not instance_id and label:
        instance_id = _resolve_instance_id_by_label(label, api_key, log=log)
    if not instance_id:
        _log("could not determine instance id; skipping destroy", log)
        return False
    return destroy_self(instance_id, api_key, log=log)


def push_results_and_destroy(
    *,
    branch: Optional[str] = None,
    run_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    api_key: Optional[str] = None,
    repo: Optional[Path] = None,
    log=print,
) -> None:
    """Push results, then destroy only when the push was successful.

    Preserving an on-box result is more important than immediate teardown: the
    max-age watchdog remains the cost backstop, while the completed run stays
    available for credential repair or manual recovery after a failed push.
    """
    run_name = run_name or os.environ.get("VAST_RUN_NAME", "run")
    instance_id = (
        instance_id
        or os.environ.get("VAST_INSTANCE_ID")
        or os.environ.get("CONTAINER_ID")
        or _read_instance_id_file()
    )
    api_key = api_key or os.environ.get("VAST_API_KEY")
    resolved_repo = repo or experiment_repo_root()
    durability_mode = os.environ.get("VAST_DURABILITY_MODE", "required")
    if durability_mode == "compact-only":
        durable = True
    elif durability_mode == "required":
        durable = required_durability_completed(resolved_repo, log=log)
    else:
        durable = False
        _log(
            f"invalid VAST_DURABILITY_MODE={durability_mode!r}; "
            "preserving box because durability must fail closed",
            log,
        )

    pushed = False
    try:
        pushed = push_results(
            branch=branch,
            run_name=run_name,
            instance_id=instance_id,
            repo=resolved_repo,
            log=log,
        )
    except Exception as e:  # noqa: BLE001 — preserve failed-push results for recovery
        _log(f"push_results raised; preserving box for recovery: {e}", log)

    if pushed and durable:
        _resolve_and_destroy(instance_id=instance_id, api_key=api_key, log=log)
    elif pushed:
        _log(
            "compact results pushed, but required B2 durability was not "
            "verified; preserving box for recovery until the max-age cap",
            log,
        )
    else:
        _log(
            "results push failed; preserving box for recovery until the max-age cap",
            log,
        )


def destroy_after_max_age(log=print) -> None:
    """Max-age watchdog teardown: the box lived past its wall-clock cap.

    This fires from an on-box timer, so it must be robust to a box that never
    ran (or crashed): only self-destruct boxes have a git identity + token
    origin, so we only try to salvage compact results when self-destruct is wired —
    otherwise we go straight to destroy. Either way the box is freed.
    """
    _log("max-age cap reached; tearing this box down", log)
    if enabled():
        try:
            push_results(
                branch=None,
                run_name=os.environ.get("VAST_RUN_NAME", "run"),
                instance_id=os.environ.get("VAST_INSTANCE_ID"),
                log=log,
            )
        except Exception as e:  # noqa: BLE001 — never let push block teardown
            _log(f"push_results raised (continuing to destroy): {e}", log)
    _resolve_and_destroy(log=log)


def enabled() -> bool:
    """True when the box was provisioned with self-destruct wired in."""
    return os.environ.get("VAST_SELF_DESTRUCT") == "1"


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="On-box teardown for vast.ai boxes.")
    p.add_argument(
        "--max-age", action="store_true",
        help="watchdog mode: box exceeded its wall-clock cap; destroy it "
             "(salvaging results/ first only if self-destruct is wired)",
    )
    args = p.parse_args()

    if args.max_age:
        destroy_after_max_age()
        return 0
    if not enabled():
        _log("VAST_SELF_DESTRUCT != 1; refusing to self-destruct")
        return 1
    push_results_and_destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
