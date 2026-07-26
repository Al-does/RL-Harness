# RunPod Serverless backend

This backend runs exactly one asynchronous experiment job on a disposable
queue-based **RunPod Serverless** endpoint. It is independent from RunPod Pods
(`devops/runpod/pods`) and Vast (`devops/vast`).

Authoritative references:

- [REST API v2 Serverless endpoints](https://docs.runpod.io/api-reference-v2/serverless/create-a-serverless-endpoint)
- [queue operations and execution policies](https://docs.runpod.io/serverless/endpoints/send-requests)
- [handler progress and worker refresh](https://docs.runpod.io/serverless/workers/handler-functions)
- [Serverless billing v2](https://docs.runpod.io/api-reference-v2/billing/get-serverless-billing-history)
- [Serverless pricing](https://docs.runpod.io/serverless/pricing)

## Safety model

Every launch creates a new endpoint named `rlh-serverless-*`, submits exactly
one `/run` job, and then blocks while polling it through a terminal state.
`up` owns the lifecycle: its `finally` path cancels when appropriate and always
deletes the endpoint on success, failure, timeout, interruption, or API error.
It returns success only for provider `COMPLETED` plus verified durable output.
`status` is crash recovery and delayed-billing settlement, not the normal job
monitor.

Policy is fixed to one GPU from the configured pool, one GPU per worker,
`min=0`, `max=1`, a five-second idle timeout, and positive provider execution
and TTL limits no longer than seven days. The create response must echo and
prove the requested image digest, GPU policy, worker bounds, idle timeout,
execution timeout, and FlashBoot before the job is submitted. The worker image
and both repositories are immutable digests/commit SHAs.

The TTL reserves the requested queue window, a startup allowance, and the full
execution limit. `up` cancels an `IN_QUEUE` job at `--queue-timeout` and
enforces the endpoint wall deadline; provider TTL is an independent backstop.

Endpoint deletion terminates workers and cancels queued/in-progress jobs.
`reap` uses each tracked endpoint's recorded lifecycle deadline. For untracked
managed endpoints it uses the default execution, queue, and startup allowance,
and discovers them by name prefix even when local state is missing.

`--max-price` gates the current conservative GPU rate constant.
`--max-estimated-cost` gates a clearly labeled **estimated spend ceiling** that
assumes the sole worker can be billed continuously for the entire provider TTL
plus idle time. This covers sequential cold starts/retries while preserving the
one-worker-at-any-instant policy, and includes container disk plus a 20% fee
reserve. It is not a provider-enforced hard dollar cap. Actual endpoint billing
can post after deletion and is collected from `GET /v2/billing/serverless`.

Cancellation may kill a worker without running Python `finally` blocks.
The harness performs a final upload, but does not automatically upload periodic
checkpoints. Long-running experiment code must explicitly checkpoint and upload
to B2 during execution; final smoke evidence is still validated after normal
completion. Serverless container disk is ephemeral.

## Publish the worker image

The Serverless Dockerfile derives from the digest-pinned Pods image, preserving
Python 3.13, Ray 2.56.0, Torch 2.12.1, Gymnasium 1.2.2, and CUDA 13. It adds the
latest compatible RunPod worker SDK only inside the image.

```bash
docker build -f devops/serverless/Dockerfile \
  -t ghcr.io/OWNER/rl-harness-runpod-serverless:DATE .
docker push ghcr.io/OWNER/rl-harness-runpod-serverless:DATE
# Resolve the registry's manifest digest, then use:
IMAGE=ghcr.io/OWNER/rl-harness-runpod-serverless@sha256:...
```

There is intentionally no mutable or invented default image. `--image` is
required until a published digest is chosen.

## Configure secrets

Set `RUNPOD_API_KEY`, `GH_TOKEN`, `B2_BUCKET`, `B2_ENDPOINT`,
`B2_APPLICATION_KEY_ID`, and `B2_APPLICATION_KEY` in the process environment or
Cursor Runtime Secrets; `B2_PREFIX` is optional. `up` refuses to proceed without
`--forward-b2` and complete B2 configuration.

The account `RUNPOD_API_KEY` is used only by the local stdlib client and is
never sent to the endpoint. `GH_TOKEN` and mandatory B2 values are endpoint
environment variables. Job input contains no secrets. The worker uses
`GH_TOKEN` only for exact-SHA checkouts and strips it from the experiment
subprocess; B2 credentials remain because the harness needs them.

The command is parsed with `shlex`, sent as an argument list, restricted to
`rl-harness experiments.*.experiment`, and executed without a shell. It must
contain one `--upload-artifacts` and one `--run-id` equal to `--run-name`.
Pinned experiment and library code remains trusted code and is the security
boundary: it can execute Python and access the B2 credentials intentionally
available to the experiment.

## Dry run, then launch

Use full 40-character commit SHAs:

```bash
uv run python -m devops.serverless.provision up \
  --experiment-ref EXPERIMENT_SHA \
  --library-ref HARNESS_SHA \
  --image "$IMAGE" \
  --run-name serverless-smoke \
  --run "rl-harness experiments.study.condition.experiment --smoke --upload-artifacts --run-id serverless-smoke" \
  --max-age 1 --queue-timeout 30 --ttl 1.75 \
  --max-price 1.25 --max-estimated-cost 2.75 \
  --forward-b2 --self-destruct --dry-run
```

Review the full-TTL estimate. Repeat the identical command without `--dry-run`
and add `--yes`. The command does not detach; wait for verified completion and
endpoint cleanup:

```bash
uv run python -m devops.serverless.provision up ... --yes
```

`--self-destruct` only requests a compact results push to the results branch.
The worker writes `serverless_result.json` before that push. It validates
`run_manifest.json`, `remote_artifacts.json`, and `progress.jsonl`: completion,
nonempty hashed uploads, a checkpoint-like artifact, and positive
`training_iteration` are mandatory. It uploads both the remote manifest and
the compact result under `<artifact-prefix>/metadata/`, and uploads finalized
MLflow data. Any successful-path MLflow upload failure fails the job. The
worker never invokes Pod termination; endpoint cleanup belongs to `up`.

## Observe and clean up

```bash
# Recover an interrupted launcher and collect/settle posted actual billing
uv run python -m devops.serverless.provision status

# Redacted endpoint, worker, health, and tracked job metadata
uv run python -m devops.serverless.provision inspect ENDPOINT_ID

# Logs exist only while a worker is active
uv run python -m devops.serverless.provision logs ENDPOINT_ID --tail 100

# Cancel and delete now
uv run python -m devops.serverless.provision destroy \
  --id ENDPOINT_ID --yes

# Delete managed endpoints beyond their recorded/default lifecycle deadline
uv run python -m devops.serverless.provision reap --yes
```

Billing aggregation is delayed. `up` reports any immediately visible amount as
provisional. Run `status` later: it records revision history and retains state
until the configured post-deletion settlement delay has elapsed and two
successive polls report an identical total. Revisions are stored in gitignored
`cost_history.json` with GPU/CPU/disk/fee components.

## Retrieve durable artifacts

The harness `--upload-artifacts` flow writes `remote_artifacts.json` with B2
keys, sizes, and SHA-256 hashes. The worker also uploads that manifest to the
deterministic key returned in the terminal output. Retrieve and verify every
file using a local copy:

```bash
uv run python -m devops.serverless.provision retrieve \
  --manifest experiments/.../results/RUN_ID/remote_artifacts.json \
  --destination artifacts/recovered/RUN_ID
```

Or use `--manifest-key <artifact-prefix>/metadata/remote_artifacts.json` and
optionally `--bucket BUCKET`. Credentials are read from the environment and
never printed.
