# RunPod Flash (no user-managed Docker image)

`devops.flash` is an alternative to the image-backed `devops.serverless`
backend. Flash uploads source plus non-Torch Python dependencies into RunPod's
managed GPU runtime. This repository does not build or publish a worker image.

The worker clones exact experiment/harness SHAs, validates Ray 2.56,
Gymnasium 1.2.2, Torch `>=2.9,<3`, CUDA `>=12.8`, runs the experiment, and
requires verified B2 durability before reporting success. Exact runtime
versions are included in the result.

## Deployment modes

- **One-machine/sequential:** deploy with `--max-workers 1`. Put all work that
  must share one machine into one experiment invocation.
- **Parallel machines:** deploy with `--max-workers N` and submit up to `N`
  `up` commands concurrently.

Both modes enforce `workers.min=0` and a five-second idle timeout. The maximum
worker count is a concurrency/spend ceiling, not an idle reservation.

The deployment configuration is part of the Flash artifact. Changing the
endpoint name or worker bound requires a redeploy; do not use a reused endpoint
whose verified worker policy differs from the desired mode.

## Prerequisites

Set `RUNPOD_API_KEY`, `GH_TOKEN`, and the four required `B2_*` credentials.
Flash currently injects its RunPod key into its managed sentinel runtime; job
input never contains credentials. Git and B2 credentials are endpoint
environment values and are removed from the experiment subprocess where
appropriate.

Install the optional local SDK:

```bash
uv sync --group flash
```

## Deploy

```bash
uv run --group flash python -m devops.flash.provision deploy \
  --app rlh-flash-experiments --environment production \
  --max-workers 1 --dry-run
```

Repeat with `--yes` instead of `--dry-run`. Deployment stages only `worker.py`
and the shared experiment handler, builds the dependency artifact, and verifies
the resulting endpoint uses `runpod/flash`, FlashBoot, `workers.min=0`, and the
requested worker maximum.

## Run an experiment

```bash
uv run --group flash python -m devops.flash.provision up \
  --endpoint-id ENDPOINT_ID \
  --experiment-ref EXPERIMENT_SHA \
  --library-ref HARNESS_SHA \
  --run-name RUN_NAME \
  --run "rl-harness experiments.study.condition.experiment --smoke --upload-artifacts --run-id RUN_NAME" \
  --max-age 0.5 --queue-timeout 20 \
  --progress-interval 30 --no-progress-timeout 10 \
  --max-price 1.25 --max-estimated-cost 0.75 \
  --forward-b2 --dry-run
```

Repeat with `--yes`. `up` preflights exact remote SHAs and the resource
contract, submits one job, blocks through terminal state, and requires positive
training/checkpoint evidence plus `canonical_manifest_key`. It intentionally
keeps the endpoint: with minimum workers zero, reuse has no continuously idling
GPU charge.

## Monitor and recover

`up` automatically tails new redacted worker logs and prints periodic
heartbeats with provider status, execution phase, elapsed time, last-progress
age, and worker counts. The Flash worker emits a training heartbeat every 30
seconds even when the experiment itself is quiet. If an `IN_PROGRESS` job
produces no fresh evidence for `--no-progress-timeout`, the launcher cancels it
instead of waiting indefinitely.

Submission metadata and monitoring state are written to the gitignored
`devops/flash/state.json`. Recover an interrupted launcher with:

```bash
uv run --group flash python -m devops.flash.provision status
uv run --group flash python -m devops.flash.provision watch \
  ENDPOINT_ID PROVIDER_JOB_ID
uv run --group flash python -m devops.flash.provision logs \
  ENDPOINT_ID --follow
```

Keep the launcher attached until it proves durable workload success or reports
a diagnosed terminal failure. Provider `COMPLETED` alone is insufficient:
success still requires training/checkpoint evidence and a canonical B2
manifest.

Inspect or delete the endpoint explicitly:

```bash
uv run --group flash python -m devops.flash.provision inspect ENDPOINT_ID
uv run --group flash python -m devops.flash.provision destroy ENDPOINT_ID --yes
```

Deleting a Flash app alone did not delete its Serverless endpoint during live
testing; always run `destroy` first. Provider billing can post later. The
image-backed backend remains available when an exact Torch/CUDA image is more
important than cold-start speed.
