---
status: resolved
severity: medium
area: experiment checkpoint restore
discovered: 2026-07-19
reproduction: confirmed
---

# CUDA map_location makes saved generator state unrestorable

## Context

- **Git revision / worktree:** experiment commit `5715121` reproduced the
  failure; fixed by `aede982` on
  `Al-does/alex-rl-experiments:experiment/mess3-paper-replication`
- **Command:** `pytest -q tests/test_paper_supervised_replication.py -k cuda_compile`
- **Environment:** Python 3.14.5; PyTorch 2.12.1+cu130; RTX 4090
- **Training context:** experiment
  `experiments.mess3_belief_geometry_2026_07.paper_supervised_replication`;
  seed 42; CUDA smoke/checkpoint verification
- **Related records:**
  `experiments/mess3_belief_geometry_2026_07/paper_supervised_replication/results/paper-sgd-compiled-retry-seed42-20260719/operations_summary.json`

## Expected behavior

A checkpoint saved from an in-place compiled CUDA model should restore model,
optimizer, and sampler RNG state into a fresh process.

## Observed behavior

Loading with `torch.load(..., map_location=cuda_device)` also moved the
serialized generator-state byte tensor to CUDA. Restore then failed:

```text
TypeError: RNG state must be a torch.ByteTensor
```

The existing CPU resume test passed and did not expose the device-specific
failure.

## Minimal reproduction

Save `torch.Generator(device="cuda").get_state()` in a checkpoint, load the
checkpoint with `map_location=torch.device("cuda")`, and pass the resulting
tensor directly to a CUDA generator's `set_state()`.

## Suspected cause and scope

Generator state is serialized as a byte tensor but `set_state()` expects that
state tensor on CPU even for a CUDA generator. Any checkpoint loader applying a
global CUDA `map_location` must move RNG state back to CPU before restoration.
This is independent of `torch.compile`.

## Resolution history

- 2026-07-19 — Reproduced on an RTX 4090 with a compiled/eager equivalence and
  checkpoint-resume test.
- 2026-07-19 — Fixed in experiment commit `aede982` by calling
  `generator.set_state(checkpoint["generator_state"].cpu())`; the CUDA test and
  compiled smoke run passed.
