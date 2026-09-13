---
status: open
severity: medium
area: tests/test_infra_safeguards.py
discovered: 2026-09-12
reproduction: confirmed
---

# Two Vast publication tests treat every result file as ignored

## Context

- Git revision: `55a5f9e27a90c88e81b15ad6211fe63c748e256b`, clean detached
  worktree on current `origin/main`.
- Environment: Python 3.14; existing locked harness environment.
- Discovered during simplex analysis/export promotion. That work changes
  `analysis/simplex.py`, viewer assets, packaging and their tests; it does not
  change Vast publishing or the failing test file.
- No remote infrastructure or training was started.

## Expected behavior

Both tests should exercise staging compact experiment results and pushing to
the launch branch.

## Observed behavior

The full fast suite reported 880 passed, 2 failed, 5 deselected. Both failing
tests reproduce independently on the clean base revision:

```text
test_self_destruct_stages_only_compact_experiment_results
test_self_destruct_pushes_to_launch_branch_with_merge_not_rebase

[self_destruct] no new compact experiment results to push
AssertionError: expected git add / git push absent from captured calls
```

The only captured subprocess call is `git check-ignore -q -- <result-file>`.

## Minimal reproduction

```bash
uv run pytest -q \
  tests/test_infra_safeguards.py::test_self_destruct_stages_only_compact_experiment_results \
  tests/test_infra_safeguards.py::test_self_destruct_pushes_to_launch_branch_with_merge_not_rebase
```

## Suspected cause and scope

Both tests monkeypatch `devops.vast.self_destruct._run` with a default successful
return code of zero. That also handles `git check-ignore`, whose zero status
means the result is ignored. `experiment_result_files` correctly skips that
file, so the intended staging and publication branches are never exercised.
The mocks should model ignored/unignored status explicitly. Evidence here is
about the test setup; it does not establish a production publishing defect.

## Resolution history

- 2026-09-12 — Recorded after reproduction on both the promotion branch and a
  clean base worktree. Left for a focused test-maintenance change.
