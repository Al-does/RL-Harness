---
name: runpod-flash
description: Deploy and validate the no-Docker RunPod Flash capability path. Use for evaluating Flash artifact delivery, zero-minimum-worker sequential or parallel probes, and promotion to a future experiment backend. Do not use for image-backed Serverless or interactive Pods.
---

# RunPod Flash

Read `devops/flash/README.md` before operating this path. Flash is a separate,
no-Docker alternative; it packages source as an artifact in a provider-managed
runtime. It is not yet approved for RL experiment jobs.

## Safety rules

- Set worker minimum to zero. Never use active workers to hide cold starts.
- Pick one mode per deployment: one maximum worker for sequential work, or an
  explicit bounded maximum for parallel work.
- Submit no more jobs than the deployed maximum-worker bound.
- Run `flash build --no-deps` and inspect it before `flash deploy`.
- Use an isolated temporary directory containing only `worker.py`; do not
  package the repository or secrets.
- `RUNPOD_API_KEY` stays local. Do not add it to worker input or endpoint env.
- Delete test apps/environments after validation unless deliberately retained.
- A successful capability probe proves only CUDA/Flash artifact delivery. It
  does not approve the path for training, B2 upload, or result publication.

## Required environment

`RUNPOD_API_KEY` must be available as a Cursor Runtime Secret or process
environment variable. Install the local SDK with:

```bash
uv sync --group flash
```

## Sequential live probe

```bash
mkdir -p /tmp/rlh-flash-probe
cp devops/flash/worker.py /tmp/rlh-flash-probe/worker.py
cd /tmp/rlh-flash-probe
FLASH=/rl-harness/.venv/bin/flash
$FLASH app create rlh-flash-probe
$FLASH env create probe --app rlh-flash-probe
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe RL_HARNESS_FLASH_MAX_WORKERS=1 \
  $FLASH build --no-deps --python-version 3.12
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe RL_HARNESS_FLASH_MAX_WORKERS=1 \
  $FLASH deploy --app rlh-flash-probe --env probe --no-deps --python-version 3.12
uv run --directory /rl-harness --group flash python -m devops.flash.probe \
  --endpoint-id ENDPOINT_ID --jobs 1 --max-workers 1 --timeout 300
```

## Parallel live probe

Redeploy the same app with a worker cap, then submit no more than that cap:

```bash
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe RL_HARNESS_FLASH_MAX_WORKERS=2 \
  $FLASH deploy --app rlh-flash-probe --env probe --no-deps --python-version 3.12
uv run --directory /rl-harness --group flash python -m devops.flash.probe \
  --endpoint-id ENDPOINT_ID --jobs 2 --max-workers 2 --timeout 300
```

After the probe, delete it:

```bash
$FLASH app delete rlh-flash-probe
```
