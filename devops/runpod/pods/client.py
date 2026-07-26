"""Credential-safe RunPod Pods REST client."""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .config import CONFIG, RunPodConfig
from .redaction import redact_sensitive

_TERMINAL = {"EXITED", "ERROR", "TERMINATED"}


def resolve_api_key() -> str | None:
    value = os.environ.get("RUNPOD_API_KEY")
    return value.strip() if value and value.strip() else None


class RunPodClientError(RuntimeError):
    pass


class RunPodClient:
    def __init__(
        self,
        cfg: RunPodConfig = CONFIG,
        api_key: str | None = None,
    ):
        self.cfg = cfg
        self.api_key = api_key or resolve_api_key()
        if not self.api_key:
            raise RunPodClientError(
                "RUNPOD_API_KEY is required; refusing to access RunPod"
            )

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        url = f"{self.cfg.API_BASE.rstrip('/')}/{path.lstrip('/')}"
        if query:
            values = {
                key: value
                for key, value in query.items()
                if value is not None
            }
            url = f"{url}?{urllib.parse.urlencode(values, doseq=True)}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            if allow_not_found and error.code == 404:
                return None
            raw = error.read().decode("utf-8", "replace")[:500]
            detail = redact_sensitive(raw, (self.api_key,))
            raise RunPodClientError(
                f"RunPod {method} {path} failed with HTTP {error.code}: {detail}"
            ) from None
        except Exception as error:  # noqa: BLE001
            detail = redact_sensitive(error, (self.api_key,))
            raise RunPodClientError(
                f"RunPod {method} {path} failed: {detail}"
            ) from None

    def list_pods(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/pods")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            rows = payload.get("pods", [])
            return rows if isinstance(rows, list) else []
        return []

    def get_pod(
        self,
        pod_id: str,
        *,
        include_machine: bool = True,
    ) -> dict[str, Any] | None:
        payload = self._request(
            "GET",
            f"/pods/{pod_id}",
            query={"includeMachine": str(include_machine).lower()},
            allow_not_found=True,
        )
        return payload if isinstance(payload, dict) else None

    def pod_logs(
        self,
        pod_id: str,
        *,
        source: str | None = None,
        tail: int = 100,
        since: str | None = None,
        follow: bool = False,
        idle_timeout_s: float = 2.0,
        emit=None,
    ) -> list[dict[str, str]]:
        """Read the documented v2 Pod SSE log stream."""
        if source not in {None, "container", "system"}:
            raise ValueError("log source must be container or system")
        if not 0 <= tail <= 5000:
            raise ValueError("log tail must be between 0 and 5000")
        query = {"source": source, "tail": tail, "since": since}
        values = {
            key: value for key, value in query.items() if value is not None
        }
        url = (
            f"{self.cfg.V2_API_BASE.rstrip('/')}/pods/"
            f"{urllib.parse.quote(pod_id, safe='')}/logs?"
            f"{urllib.parse.urlencode(values)}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
                "User-Agent": "rl-harness-runpod/1.0 (RunPod Pods logs)",
            },
        )
        try:
            response = urllib.request.urlopen(
                request,
                timeout=60 if follow else max(1.0, idle_timeout_s),
            )
        except (TimeoutError, socket.timeout):
            if not follow:
                return []
            raise RunPodClientError("RunPod Pod log stream timed out") from None
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")[:500]
            detail = redact_sensitive(raw, (self.api_key,))
            raise RunPodClientError(
                f"RunPod Pod logs failed with HTTP {error.code}: {detail}"
            ) from None
        except Exception as error:  # noqa: BLE001
            detail = redact_sensitive(error, (self.api_key,))
            raise RunPodClientError(f"RunPod Pod logs failed: {detail}") from None

        def read_events():
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line.removeprefix("data:").strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and isinstance(
                    event.get("line"), str
                ):
                    yield {
                        "ts": str(event.get("ts") or ""),
                        "source": str(event.get("source") or ""),
                        "line": event["line"],
                    }

        collected: list[dict[str, str]] = []
        if follow:
            try:
                for event in read_events():
                    collected.append(event)
                    if emit:
                        emit(event)
            finally:
                response.close()
            return collected

        items: queue.Queue[object] = queue.Queue()
        done = object()

        def read_snapshot() -> None:
            try:
                for event in read_events():
                    items.put(event)
            except Exception as error:  # noqa: BLE001
                items.put(error)
            finally:
                items.put(done)

        threading.Thread(
            target=read_snapshot,
            name="runpod-log-snapshot",
            daemon=True,
        ).start()
        deadline = time.monotonic() + idle_timeout_s
        try:
            while time.monotonic() < deadline:
                try:
                    item = items.get(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except queue.Empty:
                    break
                if item is done:
                    break
                if isinstance(item, Exception):
                    if not collected:
                        raise RunPodClientError(
                            "RunPod Pod log stream ended unexpectedly"
                        ) from None
                    break
                event = item
                if isinstance(event, dict):
                    collected.append(event)
                    if emit:
                        emit(event)
        finally:
            response.close()
        return collected

    def _graphql(
        self,
        query: str,
        variables: dict[str, Any],
        *,
        operation: str,
        secrets: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            self.cfg.GRAPHQL_URL,
            data=json.dumps(
                {"query": query, "variables": variables}
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "rl-harness-runpod/1.0 (RunPod Pods client)",
            },
        )
        protected = (self.api_key, *secrets)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")[:500]
            detail = redact_sensitive(raw, protected)
            raise RunPodClientError(
                f"RunPod GraphQL {operation} failed with HTTP "
                f"{error.code}: {detail}"
            ) from None
        except Exception as error:  # noqa: BLE001
            detail = redact_sensitive(error, protected)
            raise RunPodClientError(
                f"RunPod GraphQL {operation} failed: {detail}"
            ) from None
        if not isinstance(payload, dict):
            raise RunPodClientError(
                f"RunPod GraphQL {operation} returned no object"
            )
        if payload.get("errors"):
            raise RunPodClientError(
                f"RunPod GraphQL {operation} returned errors"
            )
        return payload

    def get_pod_safety_fields(self, pod_id: str) -> dict[str, Any]:
        """Return authoritative rental, cloud, and GPU fields from GraphQL."""
        query = (
            "query PodRentalType($podId: String!) { "
            "pod(input: {podId: $podId}) { "
            "id podType machine { secureCloud gpuTypeId podHostId } } }"
        )
        payload = self._graphql(
            query,
            {"podId": pod_id},
            operation="Pod safety verification",
        )
        pod = (payload.get("data") or {}).get("pod")
        if not isinstance(pod, dict):
            raise RunPodClientError(
                "RunPod GraphQL Pod safety verification returned no Pod"
            )
        pod_type = str(pod.get("podType") or "").upper()
        if not pod_type:
            raise RunPodClientError(
                "RunPod GraphQL Pod rental verification omitted podType"
            )
        machine = pod.get("machine")
        if not isinstance(machine, dict):
            raise RunPodClientError(
                "RunPod GraphQL Pod placement verification omitted machine"
            )
        return {
            "podType": pod_type,
            "machine": {
                "secureCloud": machine.get("secureCloud"),
                "gpuTypeId": machine.get("gpuTypeId"),
                "podHostId": machine.get("podHostId"),
            },
        }

    def create_pod(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("interruptible") is not False:
            raise RunPodClientError(
                "GraphQL Pod create requires interruptible=false"
            )
        api_request = dict(request)
        api_request.pop("interruptible")
        env = api_request.get("env")
        secret_values: tuple[str, ...] = ()
        if isinstance(env, dict):
            secret_values = tuple(str(value) for value in env.values())
            api_request["env"] = [
                {"key": str(key), "value": str(value)}
                for key, value in env.items()
            ]
        query = """
mutation CreateOnDemandPod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    costPerHr
    podType
    machine { secureCloud gpuTypeId }
  }
}
"""
        payload = self._graphql(
            query,
            {"input": api_request},
            operation="on-demand Pod create",
            secrets=secret_values,
        )
        pod = (payload.get("data") or {}).get("podFindAndDeployOnDemand")
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RunPodClientError("RunPod create response omitted Pod id")
        return pod

    def terminate_pod(self, pod_id: str) -> None:
        self._request(
            "DELETE",
            f"/pods/{pod_id}",
            allow_not_found=True,
        )

    def wait_until_running(
        self,
        pod_id: str,
        *,
        timeout: float | None = None,
        poll_s: float | None = None,
        log=print,
    ) -> dict[str, Any]:
        timeout = self.cfg.RUNNING_TIMEOUT_S if timeout is None else timeout
        poll_s = self.cfg.POLL_INTERVAL_S if poll_s is None else poll_s
        deadline = time.monotonic() + timeout
        last_status: str | None = None
        while time.monotonic() < deadline:
            pod = self.get_pod(pod_id)
            if pod is None:
                raise RunPodClientError(
                    f"Pod {pod_id} disappeared before reaching RUNNING"
                )
            status = str(
                pod.get("desiredStatus")
                or pod.get("status")
                or "UNKNOWN"
            ).upper()
            if status != last_status:
                log(f"  Pod {pod_id}: status={status}")
                last_status = status
            if status == "RUNNING":
                return pod
            if status in _TERMINAL:
                raise RunPodClientError(
                    f"Pod {pod_id} reached terminal status {status}"
                )
            time.sleep(poll_s)
        raise RunPodClientError(
            f"Pod {pod_id} did not reach RUNNING within {timeout:.0f}s"
        )

    def pod_cost(
        self,
        pod_id: str,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        # RunPod's v1 billing endpoint currently returns an empty list when
        # queried with podId, even when the unfiltered response contains that
        # Pod. Fetch the grouped rows and filter locally.
        payload = self._request(
            "GET",
            "/billing/pods",
            query={
                "grouping": "podId",
                "bucketSize": "hour",
            },
        )
        records = (
            [
                row
                for row in payload
                if isinstance(row, dict) and str(row.get("podId")) == pod_id
            ]
            if isinstance(payload, list)
            else []
        )
        amount = sum(float(row.get("amount") or 0) for row in records)
        billed_ms = sum(
            int(row.get("timeBilledMs") or 0) for row in records
        )
        return {
            "pod_id": pod_id,
            "actual_cost_usd": amount,
            "time_billed_ms": billed_ms,
            "record_count": len(records),
            "queried_at": datetime.now(timezone.utc).isoformat(),
        }


def reject_explicitly_unsafe_pod(
    pod: dict[str, Any],
    cfg: RunPodConfig = CONFIG,
) -> None:
    """Reject unsafe returned fields while allowing pending fields to be absent."""
    interruptible = pod.get("interruptible")
    pod_type = str(pod.get("podType") or "").upper()
    if interruptible is True or pod_type in {"INTERRUPTABLE", "BID"}:
        raise RunPodClientError(
            "RunPod expected interruptible=false but returned "
            f"interruptible={interruptible!r}, podType={pod_type or None!r}"
        )

    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    cloud = str(pod.get("cloudType") or pod.get("cloud") or "").upper()
    secure_cloud = machine.get("secureCloud")
    if (cloud and cloud != cfg.CLOUD_TYPE) or secure_cloud is True:
        raise RunPodClientError(
            "RunPod did not return Community Cloud placement "
            f"(cloud={(cloud or None)!r}, secureCloud={secure_cloud!r})"
        )

    gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
    gpu_name = (
        gpu.get("displayName")
        or gpu.get("id")
        or pod.get("gpuTypeId")
        or machine.get("gpuDisplayName")
        or machine.get("gpuTypeId")
        or (
            machine.get("gpuType", {}).get("displayName")
            if isinstance(machine.get("gpuType"), dict)
            else None
        )
    )
    if gpu_name is not None and gpu_name not in cfg.GPU_TYPE_IDS:
        raise RunPodClientError(
            f"RunPod returned unexpected GPU {gpu_name!r}; expected one of "
            f"{', '.join(cfg.GPU_TYPE_IDS)}"
        )


def assert_safe_pod(
    pod: dict[str, Any],
    cfg: RunPodConfig = CONFIG,
) -> None:
    """Require a running Pod to prove every hard placement policy."""
    reject_explicitly_unsafe_pod(pod, cfg)
    interruptible = pod.get("interruptible")
    pod_type = str(pod.get("podType") or "").upper()
    if interruptible is not False and pod_type != "RESERVED":
        raise RunPodClientError(
            "RunPod did not prove on-demand rental; expected "
            "interruptible=false or podType=RESERVED, got "
            f"interruptible={interruptible!r}, podType={pod_type or None!r}"
        )

    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    cloud = str(pod.get("cloudType") or pod.get("cloud") or "").upper()
    secure_cloud = machine.get("secureCloud")
    community = cloud == cfg.CLOUD_TYPE if cloud else secure_cloud is False
    if not community:
        raise RunPodClientError(
            "RunPod did not prove Community Cloud placement "
            f"(cloud={(cloud or None)!r}, secureCloud={secure_cloud!r})"
        )

    gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
    gpu_name = (
        gpu.get("displayName")
        or gpu.get("id")
        or pod.get("gpuTypeId")
        or machine.get("gpuDisplayName")
        or machine.get("gpuTypeId")
        or (
            machine.get("gpuType", {}).get("displayName")
            if isinstance(machine.get("gpuType"), dict)
            else None
        )
    )
    if gpu_name not in cfg.GPU_TYPE_IDS:
        raise RunPodClientError(
            f"RunPod returned unexpected GPU {gpu_name!r}; expected one of "
            f"{', '.join(cfg.GPU_TYPE_IDS)}"
        )
