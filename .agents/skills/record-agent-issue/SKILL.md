---
name: record-agent-issue
description: Records reproducible bugs, failed tests or runs, regressions, and post-refactor integration failures in this RLlib harness. Use when debugging a coding, training, Ray/RLlib/Tune, GPU, remote-execution, analysis, or infrastructure problem, or when asked to log an issue for later triage.
---

# Record agent-discovered issue

Use this skill to preserve actionable engineering problems that cannot safely
be completed within the assigned task. The queue is
`docs/issues/`; read its `README.md` before recording.

## 1. Decide whether to record

Record an issue when it is reproducible or has credible evidence of a defect,
regression, missing integration, or documented workaround. Include failures
found while coding, testing, running smoke checks, training, analysis, or
remote execution.

Do not record:

- a transient infrastructure failure with no evidence of a repository defect;
- an expected validation error, incorrect command, or missing local dependency
  that the agent can correct immediately;
- raw output, checkpoints, credentials, API keys, tokens, or personal data;
- scientific interpretation; put that in the experiment's `findings.md`.

When a fix is obvious, safe, and within the assigned scope, implement and
verify it first. Record it only if the issue is likely to recur or the history
will materially help future agents. Otherwise, record the unresolved issue.

## 2. Check for duplicates

Search `docs/issues/open/` by symptom, exception text, affected component, and
experiment module. Update a matching record rather than creating a duplicate.

For a distinct issue, copy `docs/issues/TEMPLATE.md` to:

```text
docs/issues/open/YYYY-MM-DD-short-kebab-slug.md
```

Use the current date and a concise, stable symptom name. Do not allocate
numeric IDs or maintain a central index.

## 3. Capture useful evidence

Fill in every frontmatter field and make the record self-contained:

- exact minimal reproduction command and expected versus observed behavior;
- Git revision and relevant dirty-worktree paths;
- owning package or experiment path;
- compact error excerpt, not a full traceback or log;
- evidence-backed suspected cause and scope;
- paths to `run_manifest.json`, result summaries, or ignored artifacts when
  they help a future investigator locate details.

For RL training failures, also record experiment module, seed, smoke/full mode,
hardware profile, and Ray/RLlib versions when they are available. Reference
artifacts by path only; artifacts are not durable on self-destructing remote
machines.

Set `reproduction` to `confirmed`, `partial`, or `not-yet` honestly. A
well-described partial failure is useful; do not invent a root cause.

## 4. Hand off clearly

Tell the user the issue record path and a one-sentence summary. Keep the
assigned task's final report separate from the issue record.

Future triage moves conclusively fixed, non-reproducible, or superseded records
to `docs/issues/resolved/` and updates their resolution history. Do not delete
records.

## Common gotchas

- `run_manifest.json` is per-run provenance, not the cross-cutting issue queue.
- `findings.md` captures scientific interpretation, not engineering defects.
- `artifacts/` is ignored and can disappear with a remote machine; preserve
  the reproduction and compact evidence in the issue record.
- Do not add issue state to `harness/`, `RunContext`, the CLI, or an
  experiment's scientific configuration.
