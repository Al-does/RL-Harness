# RunPod Flash (no-Docker capability path)

`devops.flash` is an alternative to the image-backed `devops.serverless`
backend. Flash uploads a small source artifact into a provider-managed GPU
runtime; this repository does not build, publish, or select a worker image.

It is presently a **capability path**, not a replacement for experiment
launches. It validates that Flash can supply CUDA 13 and an available GPU
without an application image. Promote it only after an exact
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
uv run --directory /rl-harness --group flash flash app create rlh-flash-probe
uv run --directory /rl-harness --group flash flash env create probe --app rlh-flash-probe
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe \
RL_HARNESS_FLASH_MAX_WORKERS=1 \
uv run --directory /rl-harness --group flash flash build --no-deps --python-version 3.12
```

Inspect the build output before deploying. Then deploy it, record the endpoint
ID reported by Flash, and submit one bounded probe:

```bash
RL_HARNESS_FLASH_ENDPOINT=rlh-flash-probe \
RL_HARNESS_FLASH_MAX_WORKERS=1 \
uv run --directory /rl-harness --group flash flash deploy --app rlh-flash-probe --env probe --no-deps --python-version 3.12

uv run --directory /rl-harness --group flash python -m devops.flash.probe \
  --endpoint-id ENDPOINT_ID --jobs 1 --max-workers 1 --timeout 300
```

For a parallel probe, redeploy with `RL_HARNESS_FLASH_MAX_WORKERS=2`, then use
`--jobs 2 --max-workers 2`. Delete the app/environment after a probe unless it
is intentionally retained for more work:

```bash
uv run --directory /rl-harness --group flash flash app delete rlh-flash-probe
```

## Current limitations

The existing image-backed worker remains the production experiment path because
it pins and verifies Ray, Torch, Gymnasium, CUDA, Git checkout, B2 durability,
and result publication. Do not send experiment jobs through Flash until those
same invariants are implemented and live-validated in the Flash worker.
