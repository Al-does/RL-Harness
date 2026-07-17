"""Thin wrapper over the ``vastai`` SDK.

Centralises auth, the ssh-direct ``runtype`` translation, readiness polling, and
connection-info extraction so the CLI stays declarative. Import lazily so the
rest of the toolkit (config/scoring, which are pure) can be used without the
``devops`` dependency group installed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from .config import CONFIG, VastConfig
from .redaction import redact_sensitive

# runtype string the vast CLI sends for `--ssh --direct` (see
# vastai/cli/commands/instances.py:get_runtype). Direct SSH needs
# direct_port_count>=1 on the offer, which our gates enforce.
SSH_DIRECT_RUNTYPE = "ssh_direc ssh_proxy"

_OFFER_TYPE_TO_SDK = {"ondemand": "on-demand", "interruptible": "bid"}
_DEAD_STATUSES = {"exited", "offline"}  # will never reach "running"


def resolve_api_key(cfg: VastConfig = CONFIG) -> Optional[str]:
    """Auth resolution order: VAST_API_KEY env -> ~/.vast_api_key -> SDK default."""
    env = os.environ.get("VAST_API_KEY")
    if env:
        return env.strip()
    if cfg.API_KEY_FILE.exists():
        key = cfg.API_KEY_FILE.read_text().strip()
        if key:
            return key
    return None  # let VastAI() fall back to its own stored-key resolution


class VastClientError(RuntimeError):
    pass


class VastClient:
    def __init__(self, cfg: VastConfig = CONFIG, api_key: Optional[str] = None):
        from vastai import VastAI  # lazy: only needed for live calls

        self.cfg = cfg
        self.api_key = api_key or resolve_api_key(cfg)
        self.v = VastAI(api_key=self.api_key)

    def _error(self, action: str, error: object) -> VastClientError:
        return VastClientError(
            f"{action}: {redact_sensitive(error, secrets=(getattr(self, 'api_key', None),))}"
        )

    # --- ssh keys -------------------------------------------------------
    def ensure_ssh_key(self, pubkey_path: Optional[Path] = None) -> str:
        """Register the local public key on the vast account if not already there."""
        pubkey_path = Path(pubkey_path or self.cfg.SSH_KEY_PATH).expanduser()
        if not pubkey_path.exists():
            raise VastClientError(f"SSH public key not found: {pubkey_path}")
        pubkey = pubkey_path.read_text().strip()
        existing = self._existing_ssh_keys()
        # Compare on the key body (type + base64), ignoring the trailing comment.
        body = " ".join(pubkey.split()[:2])
        if any(body in e for e in existing):
            return pubkey
        try:
            self.v.create_ssh_key(ssh_key=pubkey)
        except Exception as error:  # noqa: BLE001 — SDK exception types vary
            raise self._error("could not register SSH key", error) from None
        return pubkey

    def _existing_ssh_keys(self) -> list[str]:
        try:
            keys = self.v.show_ssh_keys()
        except Exception:
            return []
        rows = keys.get("ssh_keys", keys) if isinstance(keys, dict) else keys
        out = []
        for k in rows or []:
            val = k.get("public_key") or k.get("ssh_key") if isinstance(k, dict) else k
            if val:
                out.append(val.strip())
        return out

    # --- offers ---------------------------------------------------------
    def search_offers(self, query: str, offer_type: str = "ondemand", limit: int = 64) -> list[dict]:
        sdk_type = _OFFER_TYPE_TO_SDK.get(offer_type, offer_type)
        try:
            return self.v.search_offers(
                query=query, type=sdk_type, order="dph_total", limit=limit
            )
        except Exception as error:  # noqa: BLE001 — SDK exception types vary
            raise self._error("offer search failed", error) from None

    # --- instances ------------------------------------------------------
    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk: float,
        env: dict,
        label: str,
        onstart_cmd: str,
        bid: Optional[float] = None,
    ) -> int:
        """Create one instance (ssh+direct). Returns the new instance id.

        ``bid`` is only passed for interruptible rentals; when None the box is
        rented on-demand at dph_total (vast treats a null price as on-demand).
        """
        try:
            resp = self.v.create_instance(
                id=int(offer_id),
                image=image,
                disk=float(disk),
                env=env,
                label=label,
                onstart_cmd=onstart_cmd,
                runtype=SSH_DIRECT_RUNTYPE,
                price=bid,  # None => on-demand
                # Fail instead of silently creating a *stopped* (still billed) box
                # when the machine isn't available right now — lets us try the next.
                cancel_unavail=True,
            )
        except Exception as error:  # noqa: BLE001 — e.g. HTTP 410 Gone when the offer
            # was snapped up between search and create; treat as unavailable so
            # the caller falls through to the next-best offer.
            raise self._error(f"offer {offer_id} unavailable", error) from None
        new_id = resp.get("new_contract") if isinstance(resp, dict) else None
        if not new_id:
            detail = redact_sensitive(
                resp, secrets=(getattr(self, "api_key", None),)
            )
            raise VastClientError(f"offer {offer_id} unavailable: {detail}")
        # Even with cancel_unavail some hosts return a stopped contract; if so,
        # destroy it (so it stops billing) and signal the offer as unavailable.
        inst = self.show_instance(int(new_id))
        if inst and str(inst.get("intended_status")) == "stopped":
            self.destroy_instance(int(new_id))
            raise VastClientError(f"offer {offer_id} created a stopped box (unavailable); destroyed it")
        return int(new_id)

    def attach_ssh_key(self, instance_id: int, pubkey: str) -> None:
        """Attach the key to a specific instance (belt-and-suspenders over the
        account-level registration; tolerant of the 'already associated' case)."""
        try:
            self.v.attach_ssh(int(instance_id), pubkey)
        except Exception:  # noqa: BLE001 — already-associated / transient is fine
            pass

    def show_instance(self, instance_id: int) -> Optional[dict]:
        try:
            return self.v.show_instance(int(instance_id))
        except Exception as error:  # noqa: BLE001 — SDK exception types vary
            raise self._error(f"could not show instance {instance_id}", error) from None

    def destroy_instance(self, instance_id: int) -> dict:
        try:
            return self.v.destroy_instance(int(instance_id))
        except Exception as error:  # noqa: BLE001 — SDK exception types vary
            raise self._error(f"could not destroy instance {instance_id}", error) from None

    def label_instance(self, instance_id: int, label: str) -> dict:
        try:
            return self.v.label_instance(int(instance_id), label)
        except Exception as error:  # noqa: BLE001 — SDK exception types vary
            raise self._error(f"could not label instance {instance_id}", error) from None

    def wait_until_running(
        self,
        instance_id: int,
        timeout: Optional[float] = None,
        poll_s: Optional[float] = None,
        log=print,
    ) -> dict:
        """Poll until actual_status == 'running' with usable SSH connection info."""
        timeout = timeout if timeout is not None else self.cfg.RUNNING_TIMEOUT_S
        poll_s = poll_s if poll_s is not None else self.cfg.POLL_INTERVAL_S
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            inst = self.show_instance(instance_id)
            status = (inst or {}).get("actual_status")
            if status != last:
                log(f"  instance {instance_id}: status={status}")
                last = status
            if status == "running" and self.connection_info(inst)[0]:
                return inst
            if status in _DEAD_STATUSES:
                raise VastClientError(
                    f"instance {instance_id} reached terminal status {status!r}; "
                    "destroy it and retry a different offer"
                )
            time.sleep(poll_s)
        raise VastClientError(f"instance {instance_id} not running after {timeout:.0f}s")

    @staticmethod
    def _direct_endpoint(inst: dict) -> tuple[Optional[str], Optional[int]]:
        """Direct SSH: public_ipaddr + the host port mapped to 22/tcp."""
        ports = inst.get("ports") or {}
        mapped = ports.get("22/tcp")
        if mapped:
            try:
                return inst.get("public_ipaddr"), int(mapped[0]["HostPort"])
            except (KeyError, IndexError, ValueError, TypeError):
                pass
        return None, None

    @staticmethod
    def _proxy_endpoint(inst: dict) -> tuple[Optional[str], Optional[int]]:
        """Proxy SSH via the vast gateway: ssh_host + ssh_port."""
        host, port = inst.get("ssh_host"), inst.get("ssh_port")
        if host and port:
            return host, int(port)
        return None, None

    @staticmethod
    def _reachable(host: Optional[str], port: Optional[int], timeout: float = 4.0) -> bool:
        if not host or not port:
            return False
        import socket

        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    @classmethod
    def connection_info(
        cls, inst: Optional[dict], probe: bool = False
    ) -> tuple[Optional[str], Optional[int]]:
        """Return (host, port) for SSH.

        Mirrors vastai/cli/commands/misc.py:_ssh_url — the direct endpoint
        (public_ipaddr + mapped 22/tcp) is preferred, with the vast proxy as
        fallback. With ``probe=True`` the direct port is TCP-tested first and the
        proxy is used if it is unreachable (common: client networks block the
        high direct port), so the tool connects via whichever actually works.
        """
        if not inst:
            return None, None
        direct = cls._direct_endpoint(inst)
        proxy = cls._proxy_endpoint(inst)
        if probe:
            if cls._reachable(*direct):
                return direct
            if cls._reachable(*proxy):
                return proxy
        return direct if direct[0] else proxy
