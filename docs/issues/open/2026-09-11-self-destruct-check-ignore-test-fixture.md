---
status: open
severity: medium
area: devops/vast/self_destruct.py + tests/test_infra_safeguards.py
discovered: 2026-09-11
reproduction: confirmed
---

# Self-destruct result-push tests treat compact results as ignored

## Context

- **Git revision / worktree:** `9a5fed83b71bdf3d63a1784556b9c9743b3bca7e`; clean
- **Command:** `uv run pytest -q tests/test_infra_safeguards.py::test_self_destruct_stages_only_compact_experiment_results tests/test_infra_safeguards.py::test_self_destruct_pushes_to_launch_branch_with_merge_not_rebase`
- **Environment:** Python 3.14 via `uv`
- **Related records:** none

## Expected behavior

The two result-publication safeguards pass on `main` and reach their assertions
for `git add` and `git push`.

## Observed behavior

Both tests fail before staging. Their `fake_run()` returns success for the
production `git check-ignore -q -- <result>` call, so `push_results()` correctly
interprets the compact result as ignored and reports:

```text
[self_destruct] no new compact experiment results to push
```

The failures are reproducible on clean `main` and are unrelated to the
complete-episode EnvRunner change that exposed them during the fast suite.

## Minimal reproduction

```bash
uv run pytest -q \
  tests/test_infra_safeguards.py::test_self_destruct_stages_only_compact_experiment_results \
  tests/test_infra_safeguards.py::test_self_destruct_pushes_to_launch_branch_with_merge_not_rebase
```

Observed result: `2 failed`.

## Suspected cause and scope

The test fixtures predate the `git check-ignore` safeguard. Their catch-all
success return now means "the result is ignored." The fixtures should model an
unignored compact result by returning a nonzero status for `git check-ignore`,
then retain the existing staging and launch-branch assertions.

The scope appears limited to these two mocked result-push tests.

## Resolution history

- 2026-09-11 — Recorded after reproducing both failures on clean `main`.
