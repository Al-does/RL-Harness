# RunPod image size and cold-start reduction

## Current layering

Pods image (`devops/runpod/pods/Dockerfile`):

1. `ubuntu:24.04` + ca-certificates, curl, git, openssh-server, tini
2. uv-managed Python 3.13 venv
3. Pinned `ray[rllib]`, `torch` (CUDA 13 wheel), `gymnasium`, matplotlib,
   scipy, boto3, mlflow-skinny
4. Tiny final `COPY` of `container_entrypoint.py` only

Serverless image (`devops/serverless/Dockerfile`):

1. `FROM` the digest-pinned Pods image
2. `uv pip install runpod`
3. `COPY` handler only

`.dockerignore` for Serverless keeps the build context to the handler file so
source trees never invalidate the heavy dependency layer.

## What dominates cold start

- Torch CUDA 13 runtime libraries are the large layer. Starting from Ubuntu
  (instead of a PyTorch CUDA base) already avoids duplicating CUDA stacks.
- Provider image pull/extract happens while job status remains `IN_QUEUE`, which
  previously looked like capacity queueing.
- Recreating a disposable endpoint on every retry forced a fresh pull.

## Structural reductions (implemented)

1. **Preflight before pull** — nonexistent SHAs, over-capacity resource
   contracts, and undigested/private images fail without creating endpoints.
2. **Endpoint reuse** — `--reuse-endpoint` and optional retain-on-retryable-
   failure avoid repeating 12–25 minute pulls for safe retries.
3. **Flashboot kept on** — already required in endpoint policy.
4. **Handler/entrypoint stay last COPY** — code edits do not rebuild Torch.
5. **No baked experiment/harness source** — clones exact SHAs at bootstrap.
6. **Automatic Serverless→Pods fallback** — when cold-start/queue fails and
   `--fallback pods` is set, spend moves to the working Pods path instead of
   looping Serverless pulls.

## Further options (not enabled by default)

- Split analysis-only deps (matplotlib/scipy) into an optional overlay image for
  training-only workers.
- Maintain a warm `workers.min=1` pool only for short validation windows; keep
  default `workers.min=0` for spend safety.
- Publish a CPU-smoke image for preflight integration tests (never for CUDA
  training).
