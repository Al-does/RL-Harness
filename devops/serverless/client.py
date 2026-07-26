"""Small credential-safe stdlib client for RunPod Serverless REST APIs."""

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

from .config import CONFIG, ServerlessConfig
from .redaction import redact_sensitive, sensitive_values


class ServerlessClientError(RuntimeError):
    pass


def resolve_api_key() -> str | None:
    value = os.environ.get("RUNPOD_API_KEY")
    return value.strip() if value and value.strip() else None


class ServerlessClient:
    def __init__(
        self,
        cfg: ServerlessConfig = CONFIG,
        api_key: str | None = None,
    ):
        self.cfg = cfg
        self.api_key = api_key or resolve_api_key()
        if not self.api_key:
            raise ServerlessClientError(
                "RUNPOD_API_KEY is required; refusing to access RunPod"
            )

    def _request_url(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        if query:
            values = {key: value for key, value in query.items() if value is not None}
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
                "User-Agent": self.cfg.USER_AGENT,
            },
        )
        safe_path = urllib.parse.urlsplit(url).path
        protected = (self.api_key, *sensitive_values(body))
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            if allow_not_found and error.code == 404:
                return None
            raw = error.read().decode("utf-8", "replace")[:1000]
            detail = redact_sensitive(raw, protected)
            raise ServerlessClientError(
                f"RunPod {method} {safe_path} failed with HTTP "
                f"{error.code}: {detail}"
            ) from None
        except Exception as error:  # noqa: BLE001
            detail = redact_sensitive(error, protected)
            raise ServerlessClientError(
                f"RunPod {method} {safe_path} failed: {detail}"
            ) from None

    def _management(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Any:
        return self._request_url(
            method,
            f"{self.cfg.API_BASE.rstrip('/')}/v2/{path.lstrip('/')}",
            **kwargs,
        )

    def _queue(
        self,
        method: str,
        endpoint_id: str,
        path: str,
        **kwargs,
    ) -> Any:
        endpoint = urllib.parse.quote(endpoint_id, safe="")
        return self._request_url(
            method,
            f"{self.cfg.QUEUE_BASE.rstrip('/')}/{endpoint}/{path.lstrip('/')}",
            **kwargs,
        )

    def list_endpoints(self) -> list[dict[str, Any]]:
        payload = self._management("GET", "serverless")
        if isinstance(payload, dict):
            rows = payload.get("endpoints", [])
            return rows if isinstance(rows, list) else []
        return payload if isinstance(payload, list) else []

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any] | None:
        payload = self._management(
            "GET",
            f"serverless/{urllib.parse.quote(endpoint_id, safe='')}",
            allow_not_found=True,
        )
        return payload if isinstance(payload, dict) else None

    def create_endpoint(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._management("POST", "serverless", body=body)
        if not isinstance(payload, dict) or not payload.get("id"):
            raise ServerlessClientError(
                "RunPod create endpoint response omitted endpoint id"
            )
        return payload

    def delete_endpoint(self, endpoint_id: str) -> None:
        self._management(
            "DELETE",
            f"serverless/{urllib.parse.quote(endpoint_id, safe='')}",
            allow_not_found=True,
        )

    def list_workers(self, endpoint_id: str) -> dict[str, Any]:
        payload = self._management(
            "GET",
            f"serverless/{urllib.parse.quote(endpoint_id, safe='')}/workers",
        )
        return payload if isinstance(payload, dict) else {"workers": []}

    def run_job(
        self, endpoint_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        payload = self._queue("POST", endpoint_id, "run", body=body)
        if not isinstance(payload, dict) or not payload.get("id"):
            raise ServerlessClientError("RunPod run response omitted job id")
        return payload

    def job_status(
        self, endpoint_id: str, job_id: str
    ) -> dict[str, Any] | None:
        payload = self._queue(
            "GET",
            endpoint_id,
            f"status/{urllib.parse.quote(job_id, safe='')}",
            allow_not_found=True,
        )
        return payload if isinstance(payload, dict) else None

    def cancel_job(
        self, endpoint_id: str, job_id: str
    ) -> dict[str, Any] | None:
        payload = self._queue(
            "POST",
            endpoint_id,
            f"cancel/{urllib.parse.quote(job_id, safe='')}",
            allow_not_found=True,
        )
        return payload if isinstance(payload, dict) else None

    def health(self, endpoint_id: str) -> dict[str, Any]:
        payload = self._queue("GET", endpoint_id, "health")
        return payload if isinstance(payload, dict) else {}

    def serverless_billing(
        self,
        endpoint_id: str,
        *,
        start_time: str,
        end_time: str,
    ) -> dict[str, Any]:
        payload = self._management(
            "GET",
            "billing/serverless",
            query={
                "serverlessId": endpoint_id,
                "startTime": start_time,
                "endTime": end_time,
                "bucketSize": "hour",
            },
        )
        records = (
            payload.get("records", [])
            if isinstance(payload, dict)
            else payload if isinstance(payload, list) else []
        )
        records = [
            row
            for row in records
            if isinstance(row, dict)
            and str(row.get("serverlessId", endpoint_id)) == endpoint_id
        ]
        fields = (
            "totalAmount",
            "gpuAmount",
            "cpuAmount",
            "diskAmount",
            "feeAmount",
        )
        totals = {
            field: sum(float(row.get(field) or 0) for row in records)
            for field in fields
        }
        return {
            "endpoint_id": endpoint_id,
            "actual_cost_usd": totals["totalAmount"],
            "components": totals,
            "record_count": len(records),
            "records": records,
            "queried_at": datetime.now(timezone.utc).isoformat(),
        }

    def worker_logs(
        self,
        endpoint_id: str,
        worker_id: str,
        *,
        source: str | None = None,
        tail: int = 100,
        since: str | None = None,
        follow: bool = False,
        idle_timeout_s: float = 2.0,
        emit=None,
    ) -> list[dict[str, str]]:
        if source not in {None, "container", "system"}:
            raise ValueError("log source must be container or system")
        if not 0 <= tail <= 5000:
            raise ValueError("log tail must be between 0 and 5000")
        endpoint = urllib.parse.quote(endpoint_id, safe="")
        worker = urllib.parse.quote(worker_id, safe="")
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in {
                    "source": source,
                    "tail": tail,
                    "since": since,
                }.items()
                if value is not None
            }
        )
        url = (
            f"{self.cfg.API_BASE.rstrip('/')}/v2/serverless/{endpoint}/"
            f"workers/{worker}/logs?{query}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
                "User-Agent": self.cfg.USER_AGENT,
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
            raise ServerlessClientError(
                "RunPod worker log stream timed out"
            ) from None
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")[:1000]
            raise ServerlessClientError(
                f"RunPod worker logs failed with HTTP {error.code}: "
                f"{redact_sensitive(raw, (self.api_key,))}"
            ) from None
        except Exception as error:  # noqa: BLE001
            raise ServerlessClientError(
                "RunPod worker logs failed: "
                f"{redact_sensitive(error, (self.api_key,))}"
            ) from None

        def read_events():
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line.removeprefix("data:").strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and isinstance(event.get("line"), str):
                    item = {
                        "ts": str(event.get("ts") or ""),
                        "source": str(event.get("source") or ""),
                        "line": event["line"],
                    }
                    yield item

        events: list[dict[str, str]] = []
        if follow:
            try:
                for item in read_events():
                    events.append(item)
                    if emit:
                        emit(item)
            finally:
                response.close()
            return events

        items: queue.Queue[object] = queue.Queue()
        done = object()

        def read_snapshot() -> None:
            try:
                for item in read_events():
                    items.put(item)
            except Exception as error:  # noqa: BLE001
                items.put(error)
            finally:
                items.put(done)

        threading.Thread(
            target=read_snapshot,
            name="serverless-log-snapshot",
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
                    if not events:
                        raise ServerlessClientError(
                            "RunPod worker log stream ended unexpectedly"
                        ) from None
                    break
                if isinstance(item, dict):
                    events.append(item)
                    if emit:
                        emit(item)
        finally:
            response.close()
        return events
