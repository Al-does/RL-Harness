# Agent-discovered issues

This is the repository's durable, Git-tracked queue for reproducible
engineering failures found by coding and training agents. It supplements, but
does not replace:

- per-run `run_manifest.json` records for execution provenance;
- experiment `findings.md` files for scientific interpretation;
- ignored `artifacts/` trees for large logs, checkpoints, and raw data.

Do not add issue state to the harness runtime, CLI, `RunContext`, or experiment
configuration.

## Lifecycle

1. Before creating a record, search `open/` for the same symptom or component.
   Update an existing record rather than duplicating it.
2. Create a self-contained record in `open/` from `TEMPLATE.md`.
3. When the issue is fixed, confirmed non-actionable, or superseded, move it to
   `resolved/` and update its frontmatter and resolution history. Do not delete
   the record.

Use filenames of the form `YYYY-MM-DD-short-kebab-slug.md`. The filename date
is the discovery date; keep a later update history inside the record.

## Record fields

All records use YAML frontmatter. Required fields are:

- `status`: `open`, `resolved`, `not-reproducible`, or `superseded`;
- `severity`: `blocker`, `high`, `medium`, or `low`;
- `area`: the owning package or experiment path;
- `discovered`: ISO date;
- `reproduction`: `confirmed`, `partial`, or `not-yet`.

Put a minimal command and observed behavior in every issue. Training failures
also record the experiment module, seed, smoke/full mode, hardware profile, and
relevant compact result or manifest paths when available. Reference artifacts
by path only; never commit raw logs, checkpoints, credentials, tokens, or
other secrets.

## Weekly triage

A scheduled agent runs weekly using the project skill
`.cursor/skills/weekly-issue-triage/SKILL.md`. It inspects `open/*.md`, groups
duplicates, reproduces the highest-severity issues, and either implements a safe
fix or improves the record with current evidence. Each fix branches from latest
`main` and opens a PR targeting `main` for human review; the agent never merges
or pushes directly to `main`. Each run publishes a dated report under
`docs/issues/triage-reports/` and opens a triage report PR for human review.
Conclusively closed entries move to `resolved/` with history preserved.
