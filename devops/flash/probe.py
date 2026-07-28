"""Launch and verify no-Docker RunPod Flash capability probes."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from devops.serverless.client import ServerlessClient

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit bounded parallel probes to a deployed RunPod Flash endpoint."
    )
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least one")
    if args.jobs > args.max_workers:
        raise ValueError("--jobs cannot exceed --max-workers")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not 0 <= args.sleep_seconds <= 60:
        raise ValueError("--sleep-seconds must be between zero and 60")


def run_probe(args: argparse.Namespace, client: ServerlessClient) -> list[dict[str, Any]]:
    """Submit independent jobs, then poll all of them to a terminal state."""
    validate_args(args)
    jobs = [
        client.run_job(
            args.endpoint_id,
            {
                "input": {
                    "input_data": {
                        "probe": f"probe-{index + 1}",
                        "sleep_seconds": args.sleep_seconds,
                    }
                }
            },
        )
        for index in range(args.jobs)
    ]
    pending = {str(job["id"]): job for job in jobs}
    completed: list[dict[str, Any]] = []
    deadline = time.monotonic() + args.timeout
    while pending:
        if time.monotonic() >= deadline:
            for job_id in pending:
                client.cancel_job(args.endpoint_id, job_id)
            raise TimeoutError(
                f"{len(pending)} Flash probe job(s) exceeded --timeout"
            )
        for job_id in tuple(pending):
            observed = client.job_status(args.endpoint_id, job_id)
            if observed and str(observed.get("status", "")).upper() in _TERMINAL:
                completed.append(observed)
                del pending[job_id]
        if pending:
            time.sleep(1)
    return completed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run_probe(args, ServerlessClient())
    except Exception as error:  # noqa: BLE001
        print(f"Flash probe failed: {error}")
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if all(
        str(job.get("status", "")).upper() == "COMPLETED" for job in output
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
