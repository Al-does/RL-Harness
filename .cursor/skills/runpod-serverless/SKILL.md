---
name: runpod-serverless
description: Launch, monitor, persist, retrieve, cost-account, and reap disposable RunPod Serverless RL experiment jobs through devops/serverless. Use when the user explicitly asks for RunPod Serverless, queue-based GPU jobs, Serverless endpoint cleanup, or Serverless billing. Do not use for interactive SSH or ordinary RunPod Pods.
---

# RunPod Serverless

Use `devops/serverless/` for one asynchronous experiment job on one disposable,
queue-based RunPod Serverless endpoint. Read `devops/serverless/README.md` and
`docs/runpod_execution.md` before changing or operating it.

Serverless workers are provider-managed and ephemeral. **They do not support
SSH.** Use active worker logs and job progress/output for diagnosis. Use
`devops/runpod/pods/` interactive mode when a shell, profiler, or persistent
debugging session is required.

## Safety rules

- Run the identical launch with `--dry-run` before `--yes`.
- Dry-run must print the full preflight plan: resolved SHAs, image digest,
  resource contract, capacity, and conservative cost ceiling. If preflight
  rejects (bad SHA, over-capacity GPUs, undigested image), do **not** provision.
- Use full experiment and harness commit SHAs and a publicly pullable,
  digest-pinned worker image.
- Keep `--max-age`, `--queue-timeout`, TTL, `--max-price`, and
  `--max-estimated-cost` positive and explicit.
- Treat `--max-estimated-cost` as a conservative estimate, not a
  provider-enforced dollar cap. The estimate assumes the sole worker can bill
  for the full TTL.
- Keep the endpoint policy at one GPU, `workers.min=0`, `workers.max=1`, and a
  short idle timeout.
- Require CUDA 13 placement. The launcher verifies the v2 endpoint policy, then
  uses RunPod's narrowly scoped v1 compatibility update because v2 currently
  omits CUDA placement controls.
- Always use `--forward-b2`; the command must include `--upload-artifacts` and a
  `--run-id` equal to `--run-name`.
- Choose a hardware profile that fits one GPU (`--smoke` or
  `--hardware cuda4090`). Do not launch non-smoke `cuda4090_gpuinfer` on a
  1-GPU endpoint (1.8 GPUs requested).
- Never forward `RUNPOD_API_KEY` to the worker or put secrets in job input.
- Let `up` block through terminal status and endpoint deletion. Do not detach
  it. Use `status` and `reap` only as crash-recovery backstops.
- Cancellation can prevent finalizers from running. Long experiments must
  upload periodic checkpoints from experiment code; the harness final upload
  only protects normal completion.
- Treat Git `results` publication as best-effort. Workload success is training
  plus verified B2 durability (`canonical_manifest_key`). Publication failure
  is a warning with a recoverable bundle.
- For retryable cold-start/queue failures, prefer `--reuse-endpoint` or
  `--fallback pods` instead of creating a new endpoint that re-pulls the image.

## Prerequisites

Required Cursor Runtime Secrets or process environment:

- `RUNPOD_API_KEY`
- `GH_TOKEN`
- `B2_BUCKET`
- `B2_ENDPOINT`
- `B2_APPLICATION_KEY_ID`
- `B2_APPLICATION_KEY`

`B2_PREFIX` is optional.

The GHCR worker package must be public because registry credentials are not
stored in RunPod. The current published image is:

```text
ghcr.io/al-does/rl-harness-runpod-serverless@sha256:3e0ad745f08603793df6a9ec61dfbafceb3035e9d1eaecf56794b2c25a069da5
```

If anonymous pull fails, make the package public in GitHub's package settings
before launching. Do not work around private visibility by persisting a registry
token in RunPod.

## Launch workflow

Run from this harness checkout:

```bash
uv run python -m devops.serverless.provision up \
  --experiment-ref EXPERIMENT_SHA \
  --library-ref HARNESS_SHA \
  --image ghcr.io/al-does/rl-harness-runpod-serverless@sha256:DIGEST \
  --run-name RUN_NAME \
  --run "rl-harness experiments.study.condition.experiment --smoke --upload-artifacts --run-id RUN_NAME" \
  --max-age 0.5 \
  --queue-timeout 15 \
  --ttl 1 \
  --max-price 1.12 \
  --max-estimated-cost 1.40 \
  --forward-b2 \
  --fallback pods \
  --dry-run
```

Review the preflight plan (refs, digest, resource contract, CUDA policy, worker
bounds, TTL, estimate, fallback). Repeat the identical command with `--yes`
instead of `--dry-run`. Keep the process attached until it reports
`terminal_reason`, `workload_success`, `publication_status`, and endpoint
cleanup.

## Observe, recover, and retrieve

Progress lines distinguish capacity queueing from image initialization even
when the provider status remains `IN_QUEUE`.

```bash
# Active workers only; logs disappear after endpoint deletion.
uv run python -m devops.serverless.provision logs ENDPOINT_ID --tail 100

# Recover an interrupted launcher and poll delayed billing.
uv run python -m devops.serverless.provision status

# Redacted endpoint, worker, health, and tracked job metadata.
uv run python -m devops.serverless.provision inspect ENDPOINT_ID

# Cancel and delete explicitly.
uv run python -m devops.serverless.provision destroy \
  --id ENDPOINT_ID --yes

# Delete managed endpoints beyond their lifecycle deadline.
uv run python -m devops.serverless.provision reap --yes

# Download every B2 artifact/result and verify size and SHA-256.
uv run --extra storage python -m devops.serverless.provision retrieve \
  --manifest-key ARTIFACT_PREFIX/metadata/durability_manifest.json \
  --destination artifacts/recovered/RUN_NAME
```

Billing aggregation can be delayed. Report posted actual cost separately from
the conservative estimate and run `status` again later. Before finishing,
confirm no managed endpoint remains unless it was explicitly retained for retry.

## Success evidence

For a validation run, record:

1. Exact experiment SHA, harness SHA, and image digest.
2. Preflight resource contract and endpoint capacity.
3. Endpoint policy and CUDA 13 placement.
4. Actual GPU and exact Ray, Torch, and Gymnasium versions.
5. Positive training iteration and created checkpoints.
6. `canonical_manifest_key` plus hash-verified retrieval of artifacts **and**
   compact JSON/plots.
7. `workload_success=true` with independently reported `publication_status`.
8. Endpoint deletion (or explicit retain-for-retry) with billing metadata.
9. Posted actual billing, or explicitly labeled pending billing plus estimate.
