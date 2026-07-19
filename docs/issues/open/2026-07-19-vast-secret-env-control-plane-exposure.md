---
status: open
severity: high
area: devops/vast/provision.py
discovered: 2026-07-19
reproduction: confirmed
---

# Vast instance metadata returns injected secrets in plaintext

## Context

- **Git revision / worktree:** `019ebf5`; clean at discovery
- **Command:** `uv run --group devops vastai show instance <instance-id> --raw`
- **Environment:** Vast.ai on-demand RTX 4090 instance; Python 3.14.5
- **Related records:** `devops/vast/README.md` lines describing on-box token
  visibility; no secret values are preserved in this record

## Expected behavior

Provisioning diagnostics should not return credential values in ordinary tool
output, and boxes should receive only credentials required by their requested
operations.

## Observed behavior

The raw instance response included an `extra_env` object containing unredacted
Vast, GitHub, and B2 credential values. The response was emitted into an agent
tool transcript. In addition, `build_env()` currently calls
`b2_env_for_remote()` for every box, including commands that do not request
artifact upload.

## Minimal reproduction

1. Provision a box with the default max-age cap and locally configured GitHub
   and B2 credentials.
2. Run `vastai show instance <id> --raw`.
3. Inspect `extra_env`; credential names and plaintext values are present.

Never paste the response into an issue or test fixture.

## Suspected cause and scope

Vast persists create-time environment variables in control-plane metadata, and
the raw SDK/CLI response does not redact them. Some credentials are needed for
clone, upload, or self-destruction, but B2 credentials are currently injected
unconditionally. Mitigation likely requires least-privilege conditional
injection, short-lived scoped credentials where supported, safe diagnostic
wrappers, and explicit warnings against raw instance dumps.

## Resolution history

- 2026-07-19 — Recorded after plaintext values appeared in a diagnostic tool
  result. The exposed credentials require rotation outside this repository.
