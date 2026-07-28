# RunPod Flash (no-Docker capability path)

`devops.flash` is an alternative to the image-backed `devops.serverless`
backend. Flash uploads a small source artifact into a provider-managed GPU
runtime; this repository does not build, publish, or select a worker image.

It is presently a **capability path**, not a replacement for experiment
launches. It reports the CUDA runtime and available GPU without an application
image. Promote it only after an exact
Ray/Torch/Gymnasium experiment smoke also passes with verified B2 durability.

## Deployment modes

- **Sequential work:** deploy with `RL_HARNESS_FLASH_MAX_WORKERS=1` and submit
  one job at a time. The endpoint scales to zero after five idle seconds, so
  it has no active-worker charge.
- **Parallel work:** redeploy with `RL_HARNESS_FLASH_MAX_WORKERS=N`, then
  submit at most `N` probe/jobs concurrently. Workers still have a minimum of
  zero; `N` limits concurrent spend rather than reserving idle GPUs.

The deployment configuration is part of the Flash artifact. Changing the
endpoint name or worker bound requires a redeploy; do not use a reused endpoint
whose verified worker policy differs from the desired mode.

## Prerequisites

Set `RUNPOD_API_KEY`. The local launcher uses it only to submit/poll jobs; it
does not forward the key to the worker.

Install the optional local SDK:

```bash
uv sync --group flash
```

## Dry run and live capability probe

Use an isolated directory: Flash packages the current directory as its source
artifact. This avoids shipping the full checkout:

```bash
mkdir -p /tmp/rlh-flash-probe
cp devops/flash/worker.py /tmp/rlh-flash-probe/worker.py
cd /tmp/rlh-flash-probe
FLASH=/rl-harness/.venv/bin/flash
$FLASH app create rlh-flash-probe
$FLASH env create probe --app rlh-flash-probe
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe \
RL_HARNESS_FLASH_MAX_WORKERS=1 \
$FLASH build --no-deps --python-version 3.12
```

Inspect the build output before deploying. Then deploy it, record the endpoint
ID reported by Flash, and submit one bounded probe:

```bash
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe \
RL_HARNESS_FLASH_MAX_WORKERS=1 \
$FLASH deploy --app rlh-flash-probe --env probe --no-deps --python-version 3.12

uv run --directory /rl-harness --group flash python -m devops.flash.probe \
  --endpoint-id ENDPOINT_ID --jobs 1 --max-workers 1 --timeout 300
```

For a parallel probe, redeploy with `RL_HARNESS_FLASH_MAX_WORKERS=2`, then use
`--jobs 2 --max-workers 2`. Flash app deletion did not delete the underlying
Serverless endpoint in live testing, so explicitly delete that endpoint first:

```bash
uv run --directory /rl-harness --group flash python -m devops.flash.probe \
  --endpoint-id ENDPOINT_ID --delete-endpoint
$FLASH app delete rlh-flash-probe
```

## Current limitations

The existing image-backed worker remains the production experiment path because
it pins and verifies Ray, Torch, Gymnasium, CUDA, Git checkout, B2 durability,
and result publication. Do not send experiment jobs through Flash until those
same invariants are implemented and live-validated in the Flash worker.

The initial live probe returned CUDA 12.8 and Torch 2.9.1 despite requesting
CUDA 13.0. Treat the requested CUDA setting as unproven until the endpoint
metadata and worker output both prove the required version; the existing CUDA
13 experiment runtime is therefore not currently compatible with this path.
