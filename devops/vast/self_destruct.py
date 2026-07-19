"""On-box teardown: push compact experiment results, then destroy the box.

This module runs *on the vast box*, inside the training env — which is
`uv sync`ed WITHOUT the `devops` group, so ``vastai`` is NOT importable here.
The instance destroy therefore goes straight to the vast REST API over stdlib
``urllib`` (Authorization: Bearer <key>), keeping the training env clean while
still freeing the box.

Design guarantees:
  - push_results never fails when there are no new experiment results.
  - disjoint per-run folders make concurrent boxes' rebases auto-apply; the
    fetch+rebase+retry loop additionally survives non-fast-forward push races.
  - push_results_and_destroy destroys in a finally, so a push hiccup still frees
    the box (and stops billing).
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


def _run(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd or experiment_repo_root()),
        capture_output=True, text=True,
    )


def _log(msg: str, log=print, secrets: Iterable[str | None] = ()) -> None:
    known_secrets = (
        *secrets,
        os.environ.get("VAST_API_KEY"),
        os.environ.get("GITHUB_TOKEN"),
        os.environ.get("B2_APPLICATION_KEY"),
        os.environ.get("B2_APPLICATION_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
    )
    log(f"[self_destruct] {redact_sensitive(msg, known_secrets)}")


def push_results(
    branch: str,
    run_name: str,
    instance_id: Optional[str],
    repo: Optional[Path] = None,
    attempts: int = 6,
    log=print,
) -> bool:
    """Commit and push compact experiment results to ``branch``.

    Returns True (success, no-op) when there is nothing new to push. Complete
    checkpoints and raw data remain ignored beneath each ``artifacts/`` tree.
    """
    repo = repo or experiment_repo_root()
    add = _run(["git", "add", "-A", "--", "experiments/"], cwd=repo)
    if add.returncode != 0:
        _log(f"git add failed: {add.stderr.strip()}", log)
        return False

    staged = _run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if staged.returncode == 0:
        _log("no new compact experiment results to push", log)
        return True

    label = f"results: {run_name} (vast {instance_id})" if instance_id else f"results: {run_name}"
    commit = _run(["git", "commit", "-m", label], cwd=repo)
    if commit.returncode != 0:
        _log(f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}", log)
        return False

    delay = 1.0
    for i in range(1, attempts + 1):
        fetched = _run(["git", "fetch", "origin", branch], cwd=repo)
        if fetched.returncode == 0:
            # Rebase our disjoint per-run folder onto the current branch tip.
            rebased = _run(["git", "rebase", "--autostash", "FETCH_HEAD"], cwd=repo)
            if rebased.returncode != 0:
                _run(["git", "rebase", "--abort"], cwd=repo)
                _log(f"rebase failed (attempt {i}): {rebased.stderr.strip()}", log)
        # else: branch doesn't exist remotely yet; push creates it.

        pushed = _run(["git", "push", "origin", f"HEAD:{branch}"], cwd=repo)
        if pushed.returncode == 0:
            _log(f"pushed results to {branch}", log)
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
    """Push results (best-effort, logged) then destroy the box in a finally."""
    branch = branch or os.environ.get("VAST_RESULTS_BRANCH", "results")
    run_name = run_name or os.environ.get("VAST_RUN_NAME", "run")
    instance_id = (
        instance_id
        or os.environ.get("VAST_INSTANCE_ID")
        or os.environ.get("CONTAINER_ID")
        or _read_instance_id_file()
    )
    api_key = api_key or os.environ.get("VAST_API_KEY")
    resolved_repo = repo or experiment_repo_root()

    try:
        push_results(
            branch=branch,
            run_name=run_name,
            instance_id=instance_id,
            repo=resolved_repo,
            log=log,
        )
    except Exception as e:  # noqa: BLE001 — never let push block teardown
        _log(f"push_results raised (continuing to destroy): {e}", log)
    finally:
        _resolve_and_destroy(instance_id=instance_id, api_key=api_key, log=log)


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
                branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
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
