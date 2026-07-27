# Agent prompt: add RunPod Serverless

Add a **RunPod Serverless** backend under `devops/serverless/`. Treat
Serverless as RunPod's separate, higher-level managed product—not as another
Pod cloud or pricing tier. Fetch the current RunPod Serverless REST API, Python
SDK, worker, endpoint, image, job, timeout, and billing documentation before
designing anything; do not rely on remembered API shapes.

Background already available:

- `devops/runpod/pods/` is a working Pods backend with config, authenticated
  clients, redaction, digest resolution, Docker image, lifecycle CLI,
  self-termination, B2 persistence, MLflow provenance, cost collection, tests,
  and a runbook.
- Its public image pins Ray/RLlib 2.56.0, Torch 2.12.1, Gymnasium 1.2.2, CUDA
  13, and Python 3.13. Reuse dependency/build-layer choices where compatible,
  but Serverless will require its own handler/worker contract and image.
- Experiment code is cloned at runtime at explicit experiment and harness SHAs.
  Durable checkpoints use the existing Backblaze B2 integration.
- Secrets are environment-only: `RUNPOD_API_KEY`, `GH_TOKEN`, and optional
  `B2_*`. Never print, persist, or bake them.
- Known integration details: use an explicit User-Agent for RunPod GraphQL;
  MLflow 3.14 file storage requires `MLFLOW_ALLOW_FILE_STORE=true`; runtime
  must assert CUDA availability and exact framework versions.

First compare Serverless semantics with the existing Vast/Pods experiment-job
contract. If long-running RLlib jobs, checkpoint persistence, cancellation, or
a clearly labeled conservative estimated spend ceiling cannot be represented
safely, stop and explain the mismatch before implementation. The estimate is a
safety gate, not a provider-enforced hard dollar cap. Do not change or break
`devops/runpod/pods/` or `devops/vast/`.

Implement a documented dry-run-first workflow, immutable image digest and git
SHA provenance, explicit GPU/worker policy, success/failure cleanup, provider
timeout, cancellation/reaping, redacted inspection, estimated and actual cost,
B2 checkpoint retrieval, `.env.example`, tests, and a short runbook. Validate
with one minimal real GPU smoke job only after dry run, using the conservative
estimated spend ceiling and confirming training steps, durable checkpoint
retrieval, worker cleanup, and billing.
