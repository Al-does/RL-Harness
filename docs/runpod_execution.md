# RunPod phased execution model

RunPod Pods (`devops/runpod/pods/`) and Serverless (`devops/serverless/`) share
one execution model under `devops/runpod/execution/`. The model exists so agents
cannot spend long periods provisioning machines only to discover deterministic
input, resource, or publication errors.

## Phases and independent statuses

Every job reports these phases:

| Phase | Meaning |
|-------|---------|
| `PREFLIGHT` | Resolve/verify refs, image, resources, and cost plan |
| `PROVISIONING` | Create or reuse endpoint/Pod; capacity vs image-init |
| `BOOTSTRAP` | Clone exact SHAs, install sources, validate runtime |
| `TRAINING` | Run the experiment command |
| `ANALYSIS` | Optional offline/portable analysis hooks |
| `DURABLE_UPLOAD` | Verified B2 upload of artifacts + compact results |
| `RESULTS_PUBLICATION` | Best-effort Git overlay onto `results` |
| `CLEANUP` | Delete endpoint/Pod; retain billing metadata |

**Workload success** means experiment execution succeeded **and** durable upload
was verified. Results-branch publication is reported separately
(`publication_status`) and never flips workload success.

## Preflight (before any spend)

1. Resolve refs with `git rev-parse` when they are not already full SHAs.
2. Verify both exact SHAs via the GitHub Commits API (fail closed on 404).
3. Require digest-pinned images and probe anonymous GHCR pullability.
4. Build a declarative `ResourceContract` from the hardware profile + `--smoke`
   without constructing a Ray cluster.
5. Reject jobs whose total GPU/CPU demand exceeds endpoint capacity.
6. Print the complete resource and cost plan (dry-run stops here).

Serverless default remote profile is `cuda4090_gpuinfer`. A non-smoke trial on
that profile requests `1.0 + 8×0.1 = 1.8` GPUs and is rejected on a 1-GPU
endpoint. Use `--smoke`, `--hardware cuda4090`, or a larger GPU count.

## Durability

`harness.storage.b2.upload_run_artifacts` uploads:

- the full `artifacts/` tree (checkpoints, Tune trees, logs)
- the compact `results/` tree (JSON, plots, manifests, provenance)

A canonical `durability_manifest.json` (key returned as
`canonical_manifest_key`) lists every object with SHA-256 and size. Legacy
`remote_artifacts.json` remains for older tools.

## Results publication

Publication uses a clean worktree rooted at the current `results` tip:

1. Collect only the compact result bundle from the experiment checkout.
2. Overlay those files onto a fresh results-branch worktree.
3. Commit and push; never rebase experiment history onto `results`.
4. Retry only genuine non-fast-forward concurrent updates.
5. Deterministic content conflicts fail immediately.
6. Failures are warnings with a recoverable local/remote bundle.

## Portable checkpoints

`analysis.portable_checkpoint` stores:

- module class + config
- RLModule state (`module_state.pt`)
- environment specification
- checkpoint step
- experiment/harness SHAs
- analysis protocol

`load_portable_module` restores weights without starting Ray or environment
runners. Use `export_portable_from_algorithm_checkpoint` once if you still have
a full RLlib checkpoint.

## Provisioning UX and fallback

- Launcher progress distinguishes **capacity queueing** from **image
  initialization** even when the provider still reports `IN_QUEUE`.
- `--reuse-endpoint` submits to a healthy endpoint without recreating it.
- `--keep-endpoint-on-retryable-failure` retains an endpoint after queue timeout.
- `--fallback pods` automatically hands off to Pods after Serverless
  queue/image-init/provisioning failures.
- Deterministic preflight failures never create endpoints.
- All failure paths delete endpoints unless explicitly retained, and keep enough
  state for billing reconciliation.

## Image cold-start (structural)

See `devops/runpod/execution/image_coldstart.md`. Short version: Torch CUDA
wheels dominate image size; Serverless already layers only the SDK + handler on
the Pods digest; cold-start cost is reduced by endpoint reuse, Flashboot, and
failing preflight before pull—not by raising queue timeouts.
