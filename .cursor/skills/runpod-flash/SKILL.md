---
name: runpod-flash
description: Deploy, run, verify, retrieve, and clean up no-user-image RunPod Flash RL jobs with zero minimum workers. Use for faster Serverless startup, sequential one-machine runs, or bounded parallel GPU jobs. Do not use for exact image runtimes or interactive Pods.
---

# RunPod Flash

Read `devops/flash/README.md` and `docs/runpod_execution.md` first. Flash
packages source and non-Torch dependencies into an artifact and uses RunPod's
managed Python 3.12/Torch GPU runtime.

## Safety rules

- Set worker minimum to zero. Never use active workers to hide cold starts.
- Pick one mode per deployment: one maximum worker for sequential work, or an
  explicit bounded maximum for parallel work.
- Submit no more jobs than the deployed maximum-worker bound.
- Run the identical `deploy` or `up` command with `--dry-run` before `--yes`.
- Never put credentials in job input. The launcher verifies endpoint
  credentials by name and redacts values.
- Require full immutable experiment and harness SHAs.
- Require `--forward-b2`, a positive training iteration, uploaded checkpoint,
  and canonical durability manifest.
- Keep `up` attached in a persistent terminal until durable success or a
  diagnosed terminal failure. An agent must not stop merely because a job was
  submitted or remains `IN_QUEUE`/`IN_PROGRESS`.
- Treat periodic heartbeats as required evidence. If monitoring is interrupted,
  run `status`, then resume with `watch` before ending the task.
- Keep the endpoint after jobs for reuse; `workers.min=0` means no idle GPU.
- Explicitly destroy endpoints when no longer needed. Deleting a Flash app is
  not sufficient.

## Required environment

`RUNPOD_API_KEY` must be available as a Cursor Runtime Secret or process
environment variable. Install the local SDK with:

```bash
uv sync --group flash
```

## Deploy for sequential work

```bash
uv run --group flash python -m devops.flash.provision deploy \
  --app rlh-flash-experiments --environment production \
  --max-workers 1 --dry-run
```

Repeat with `--yes`. Put all work that must share one machine into one
experiment invocation.

## Deploy for parallel work

Deploy with an explicit cap and submit at most that many `up` commands
concurrently:

```bash
uv run --group flash python -m devops.flash.provision deploy \
  --app rlh-flash-parallel --environment production \
  --max-workers 4 --dry-run
```

## Run and clean up

```bash
uv run --group flash python -m devops.flash.provision up \
  --endpoint-id ENDPOINT_ID \
  --experiment-ref EXPERIMENT_SHA --library-ref HARNESS_SHA \
  --run-name RUN_NAME \
  --run "rl-harness experiments.study.condition.experiment --smoke --upload-artifacts --run-id RUN_NAME" \
  --max-age 0.5 --queue-timeout 20 \
  --progress-interval 30 --no-progress-timeout 10 \
  --max-price 1.25 --max-estimated-cost 0.75 \
  --forward-b2 --dry-run

uv run --group flash python -m devops.flash.provision destroy ENDPOINT_ID --yes
```

Repeat the `up` command with `--yes` only after reviewing preflight. Record
endpoint policy, exact Python/Ray/Torch/CUDA versions, training iteration,
checkpoint keys, canonical manifest key, and durable retrieval evidence.

## Mandatory supervision and recovery

`up` prints new redacted worker logs plus a heartbeat containing provider
status, current phase, elapsed time, last-progress age, and worker counts. The
worker itself emits a training heartbeat every 30 seconds. The launcher cancels
an `IN_PROGRESS` job after `--no-progress-timeout` without fresh worker
evidence; do not weaken this silently.

If the attached launcher is interrupted, it has already recorded the endpoint
and provider job IDs in `devops/flash/state.json`:

```bash
uv run --group flash python -m devops.flash.provision status
uv run --group flash python -m devops.flash.provision watch \
  ENDPOINT_ID PROVIDER_JOB_ID
uv run --group flash python -m devops.flash.provision logs \
  ENDPOINT_ID --follow
```

Before an agent stops working, it must reach one of these outcomes:

1. `workload_success=true` with a canonical manifest key, then retrieve and
   hash-verify artifacts when validation was requested.
2. A terminal failure with the relevant redacted worker exception reported.
3. A cancelled timeout/stall with endpoint/job IDs and recovery action reported.

Never hand off a silently running job without explicitly stating who or what is
continuing to monitor it.
