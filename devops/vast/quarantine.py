"""Local, gitignored quarantine of bad Vast machines / public IPs.

Persists across ``provision up`` invocations on this workstation so agents do
not re-rent hosts that already failed readiness. Never commit this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .config import VastConfig


def _empty() -> dict:
    return {"machines": {}, "public_ips": {}}


def load_quarantine(cfg: VastConfig) -> dict:
    path = Path(cfg.QUARANTINE_PATH)
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data.setdefault("machines", {})
    data.setdefault("public_ips", {})
    return data


def save_quarantine(cfg: VastConfig, data: dict) -> None:
    path = Path(cfg.QUARANTINE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _active_keys(section: dict, now: float) -> set[str]:
    active: set[str] = set()
    for key, entry in list(section.items()):
        if not isinstance(entry, dict):
            continue
        until = float(entry.get("until") or 0.0)
        if until > now:
            active.add(str(key))
    return active


def active_exclusions(cfg: VastConfig, now: Optional[float] = None) -> tuple[set[int], set[str]]:
    """Return (machine_ids, public_ips) still under quarantine."""
    now = time.time() if now is None else now
    data = load_quarantine(cfg)
    machines = set()
    for key in _active_keys(data.get("machines", {}), now):
        try:
            machines.add(int(key))
        except ValueError:
            continue
    ips = _active_keys(data.get("public_ips", {}), now)
    return machines, ips


def _bump(section: dict, key: str, reason: str, ttl_s: float, now: float) -> None:
    entry = section.get(key) if isinstance(section.get(key), dict) else {}
    fails = int(entry.get("fails") or 0) + 1
    section[key] = {
        "fails": fails,
        "reason": reason,
        "last_failed_at": now,
        "until": now + ttl_s,
    }


def record_failure(
    cfg: VastConfig,
    *,
    machine_id: object = None,
    public_ip: Optional[str] = None,
    reason: str = "readiness failure",
    now: Optional[float] = None,
) -> None:
    """Extend quarantine TTL for a machine and/or public IP after a failed rental."""
    now = time.time() if now is None else now
    data = load_quarantine(cfg)
    ttl = float(cfg.QUARANTINE_TTL_S)
    if machine_id is not None and str(machine_id).strip():
        _bump(data.setdefault("machines", {}), str(machine_id), reason, ttl, now)
    if public_ip:
        ip = str(public_ip).strip()
        if ip:
            _bump(data.setdefault("public_ips", {}), ip, reason, ttl, now)
    # Drop expired entries so the file stays small.
    for section_name in ("machines", "public_ips"):
        section = data.get(section_name, {})
        data[section_name] = {
            key: entry
            for key, entry in section.items()
            if isinstance(entry, dict) and float(entry.get("until") or 0.0) > now
        }
    save_quarantine(cfg, data)
