---
name: weekly-issue-triage
description: Weekly triage of agent-discovered bugs in docs/issues/open/. Reproduce, fix with a simplification bias, open review PRs to bugfix/triage (never main), move resolved records, and publish a structured triage report for human review.
---

# Weekly issue triage

Run this skill on a schedule to work through the engineering issue queue in
`docs/issues/`. Read `docs/issues/README.md` first. Issue records are written
by agents using the `record-agent-issue` skill.

## Goals

1. Reduce open issues with safe, reviewable fixes.
2. Prefer **simplifying** changes (delete dead code, narrow scope, fix root
   cause) over adding layers, flags, or workarounds.
3. When more code is unavoidable, say why in the triage report.
4. Never merge to `main`. Open PRs for human review only.

## Branch policy

- **Integration branch (PR base):** `bugfix/triage`
  - If it does not exist, create it from `main` and push it before opening fix PRs.
- **Fix branches:** `automated/bugfix/YYYY-MM-DD-<issue-slug>`
  - One branch and one PR per issue by default.
  - Use a single branch/PR for multiple issues only when one shared fix
    genuinely resolves them (same root cause, same change set). Name the branch
    after the primary issue slug and list every resolved issue in the PR body.
- **Report branch:** `automated/triage-report/YYYY-MM-DD`
  - Always create this branch for the weekly summary, even when no code fixes land.

All PRs target `bugfix/triage`, not `main`.

## Triage procedure

### 1. Inventory

- List every file in `docs/issues/open/*.md`.
- Skip empty queues; still publish a short triage report PR noting no open issues.
- Sort candidates:
  1. `severity`: blocker → high → medium → low
  2. `reproduction`: confirmed → partial → not-yet
  3. `discovered` date (oldest first within the same tier)

### 2. Per-issue workflow

For each issue, up to **10 fix attempts per weekly run** (stop after that;
defer the rest to the next run):

1. Read the full record and note `area`, reproduction command, and suspected cause.
2. Search the codebase and `docs/issues/` for duplicates; merge or cross-link in
   the record instead of fixing twice.
3. Reproduce with the recorded minimal command when possible.
4. Classify the fix before coding:
   - **Straightforward** — root cause is clear; change is localized and low risk.
   - **Deliberated** — multiple plausible causes or approaches; document what
     you considered and why you chose the fix.
5. Implement the smallest correct fix. Bias:
   - remove incorrect or redundant code;
   - fix the underlying bug instead of adding guards;
   - add code only when the bug truly requires new behavior or coverage.
6. Verify with the minimal reproduction command and any existing tests that
   touch the affected area. Do not run full training unless the issue is
   training-specific and a smoke run is recorded in the issue.
7. Update the issue record:
   - append to **Resolution history** with date and evidence;
   - if fixed, set `status: resolved`, set `reproduction: confirmed`, and move the
     file to `docs/issues/resolved/`;
   - if not fixed, improve the record (new evidence, narrowed scope, updated
     suspected cause) and leave it in `open/`.
8. If you changed code, open **one PR per issue** into `bugfix/triage`:
   - branch: `automated/bugfix/YYYY-MM-DD-<issue-slug>`
   - one issue record per PR unless step 2 identified issues that share a single
     fix; then one PR may close multiple records, with every affected issue path
     listed in the PR body and triage report.
   - do not batch unrelated fixes into one PR.

Do **not** fix issues that are not reproducible and have no credible defect
evidence; update the record or move to `resolved/` with status
`not-reproducible` only when evidence supports that conclusion.

### 3. Weekly triage report (required)

On `automated/triage-report/YYYY-MM-DD`, add or update:

`docs/issues/triage-reports/YYYY-MM-DD.md`

Use this structure:

```markdown
# Issue triage report — YYYY-MM-DD

## Summary

<One paragraph: how many open issues, how many fixed, how many updated, how many deferred.>

## Fixes (PRs for review)

| Issue | Severity | Fix type | PR | Notes |
|-------|----------|----------|-----|-------|
| [slug](path/to/record) | high | Straightforward | #123 | … |

## Deliberated fixes

For each fix that required tradeoffs:

### <issue slug>

- **Bug:** <plain-language description>
- **Approach chosen:** …
- **Alternatives considered:** …
- **Tradeoffs:** …
- **Why not simpler:** … (omit if the fix was a deletion or simplification)

## Updated but not fixed

| Issue | Reason deferred | Next step |
|-------|-----------------|-----------|

## No change

| Issue | Reason |
|-------|--------|

## Reviewer action

Inspect PRs targeting `bugfix/triage`. Merge that branch to `main` only after
review. Do not auto-merge.
```

Open a PR from the report branch into `bugfix/triage`. The report PR body must
repeat the **Summary**, list every fix PR with links, and call out **Deliberated
fixes** and **Tradeoffs** prominently so the reviewer knows what needs attention.

### 4. PR template for code fixes

Title: `fix(<area>): <concise symptom>`

Body:

```markdown
## Issue

- Record: `docs/issues/...` (list every record when one PR fixes multiple issues)
- **Bug:** <what was broken, in plain language>

## Fix type

Straightforward | Deliberated

## What changed

<Bullet list of concrete changes.>

## Tradeoffs

<None — simplification/deletion> | <describe tradeoffs>

## Verification

- [ ] Minimal reproduction command: `...`
- [ ] Tests: `...`

## Reviewer notes

<Anything non-obvious the reviewer should check.>
```

Mark fix PRs as **ready for review** (not draft) unless reproduction was only partial.

## Constraints

- Follow `AGENTS.md` ownership and dependency rules.
- Do not add issue state to harness runtime, CLI, or experiment configuration.
- Do not commit secrets, raw logs, checkpoints, or credentials.
- Do not merge PRs or push to `main`.
- Do not delete issue records; move to `resolved/` and preserve history.
- If zero open issues, still open the report PR noting a clean queue.

## Handoff to the human reviewer

When the run finishes, state clearly:

1. Link to the **triage report PR**
2. Links to each **fix PR**, or say none were opened
3. Which issues need **human investigation** (deliberated fixes, partial
   reproduction, or deferred items)

The triage report PR is the primary notification surface; the reviewer should
watch the repo or enable GitHub notifications for PRs on `bugfix/triage`.
