---
status: open
severity: medium
area: devops/vast/self_destruct.py
discovered: 2026-11-05
reproduction: confirmed
---

# Vast self_destruct skips publication and durability for study-local `runs/<run-id>/` folders

## Context

- **Git revision / worktree:** harness `9e1bff5` (plus the inlined
  `bootstrap.sh` fix from PR #85); experiment repo
  `Al-does/alex-rl-experiments@ad6b9b6a` on branch
  `devin/1793887268-mrrxor-grokking-ppo`; remote worktree dirty only under
  `experiments/mrrxor_grokking_jax_2026_11/runs/20261105-grid-seed42/`.
- **Command:** `uv run --group devops python -m devops.vast.provision up --run
  "uv sync --group jax-cuda && python -c 'assert __import__(\"jax\").devices()[0].platform == \"gpu\"' && python -m experiments.mrrxor_grokking_jax_2026_11.sweep --seed 42 --hardware cuda4090 --run-id 20261105-grid-seed42"
  --forward-b2 --on-demand --max-age-hours 24 --branch devin/1793887268-mrrxor-grokking-ppo`
- **Environment:** Python 3.13 on the box (CPython 3.13 via `.python-version`);
  pure JAX/Optax study, no Ray/RLlib runtime.
- **Training context:** experiment
  `experiments.mrrxor_grokking_jax_2026_11.ppo_sweep.experiment` via the study
  module launcher; seed 42; full mode; hardware profile `cuda4090`.
- **Related records:**
  `experiments/mrrxor_grokking_jax_2026_11/runs/20261105-grid-seed42/run_manifest.json`
  and `.../runs/20261105-grid-seed42/<arm>/remote_artifacts.json` in the
  experiment repo (commit `d1997ec4` on the launch branch, pushed manually).

## Expected behavior

After the run exited with `EXIT=0`, `python -m devops.vast.self_destruct`
should have verified the B2 uploads recorded by the run, committed the
compact results to the launch branch, and destroyed the box.

## Observed behavior

```text
EXIT=0
[self_destruct] required durability found no pending run manifest
[self_destruct] no new compact experiment results to push
[self_destruct] compact results pushed, but required B2 durability was not verified; preserving box for recovery until the max-age cap
```

Nothing was pushed, and the box stayed up billing until destroyed by hand.
Checkpoints had in fact been uploaded: every arm's `remote_artifacts.json`
reported `file_count=83`, and a manual `list_objects_v2` on each
`s3://slop-bucket/experiments/mrrxor_grokking_jax_2026_11/ppo_sweep/20261105-grid-seed42-<arm>/`
prefix found 85 objects (~300 MB) plus the canonical manifest key.

## Minimal reproduction

In any experiment checkout, run a leaf whose `--results-dir` is not under a
`results/` directory (this study resolves
`experiments/<study>/runs/<run-id>/` and `.../runs/<run-id>/artifacts` before
the harness writes its first manifest), then run
`python -m devops.vast.self_destruct` with `VAST_SELF_DESTRUCT=1` and
`VAST_REQUIRE_DURABILITY` enabled. Both `collect_compact_result_paths` and
`pending_run_manifests` return empty lists.

## Suspected cause and scope

`collect_compact_result_paths` requires `/results/` in the relative path and
`pending_run_manifests` only accepts `run_manifest.json` paths containing
`/results/`. The experiment-repo rule in `experiments/AGENTS.md` permits
study-local run folders, and `harness.cli` accepts arbitrary
`--results-dir`/`--artifacts-dir`, so any study that consolidates outputs into
one run folder silently loses publication and durability verification on
Vast (and the equivalent RunPod publication paths, if they share this
discovery logic). A fix should discover result roots from the run's own
`run_manifest.json` (`results_dir`/`artifacts_dir` recorded by the harness)
instead of a hard-coded `results/` path component, or the harness should
refuse `--results-dir` values that publication cannot discover.

## Resolution history

- 2026-11-05 — Recorded. Box 59282874 preserved until manual destroy;
  compact results pushed by hand; B2 durability verified manually.
