# Multi-repo workflow

## Entry point

Colleagues start at
[`rl-experiments`](https://github.com/Al-does/rl-experiments):

1. **Fork** it (keeps the name `rl-experiments` — no rename).
2. Clone their fork.
3. Run `./scripts/bootstrap_local.sh` — clones
   [`rl-harness`](https://github.com/Al-does/RL-Harness) beside it if needed
   and editable-installs the library.
4. Replace the example study with their science.

The shared library is this repository (`rl-harness`). Personal science stays on
each researcher's fork of `rl-experiments`.

## Repos

| Repo | Role |
|---|---|
| [`rl-experiments`](https://github.com/Al-does/rl-experiments) | **Headline entry point** — fork this |
| [`RL-Harness`](https://github.com/Al-does/RL-Harness) | Shared library — PRs for reusable code |
| Personal forks / repos (e.g. `alex-rl-experiments`) | Ongoing science histories |

## Default local layout

```text
parent/
  rl-harness/        # shared library (cloned by bootstrap if missing)
  rl-experiments/    # your fork
```

Experiment repos declare an editable path dependency:

```toml
[tool.uv.sources]
rl-harness = { path = "../rl-harness", editable = true }
```

Until the API stabilizes, everyone tracks library `main` (pull often). Run
manifests record the library commit SHA (and package version later).

## Who commits where

| Change | Repository | Action |
|---|---|---|
| `experiment.py`, findings, compact results | your fork of `rl-experiments` | push to your fork (not a science PR upstream) |
| `harness/`, `learners/`, `losses/`, `envs/`, `analysis/`, `devops/` | `rl-harness` | branch + PR |
| Improvements to the starter itself | upstream `rl-experiments` | optional small PR |

A change that touches library + science requires **two commits** (one per repo).

## Why fork (not rename / not “Use this template”)

- Fork keeps the folder and repo name `rl-experiments` for everyone.
- No “clone then rename my-jane-experiments” step.
- Science stays on the researcher’s GitHub account; library contributions still
  go to `rl-harness` via normal PRs.

## vast.ai

Boxes clone the experiment fork (results push target) and this library as
siblings, then `uv sync` in the experiment repo. Point provisioning at *your*
fork’s URL. See `devops/vast/README.md`.
