---
status: resolved
severity: high
area: devops/vast/vast_client.py
discovered: 2026-07-17
reproduction: confirmed
---

# Vast HTTP errors leak the API key

## Context

- **Git revision / worktree:** `d91bdffc7aa6839a8bd45161e2b03920c270a5f5`
  on `MESS3-supervised`; dirty only in unrelated pre-existing paths
  (`AGENTS.md`, `.cursor/Dockerfile`, `.cursor/environment.json`)
- **Command:** `uv run --group devops python - <<'PY'` using the synthetic
  reproduction below
- **Environment:** Python with the repository's `devops` dependency group;
  live Vast provider access is not required for the synthetic reproduction
- **Related records:** `docs/issues/open/2026-07-17-vast-cannot-exclude-bad-machines.md`

## Expected behavior

Errors returned by the Vast SDK should identify the unavailable offer without
including credentials. Any URL included in a wrapped or logged exception must
have sensitive query parameters such as `api_key` removed or redacted.

## Observed behavior

`VastClient.create_instance()` catches the SDK/requests exception and embeds
`str(e)` unchanged in a new `VastClientError`. A requests `HTTPError` normally
includes its request URL; when the Vast SDK put the API key in that URL's query
string, the complete key was therefore included in the exception and printed
by `cmd_up()` during a failed exact-offer attempt.

Only a redacted compact shape is retained here:

```text
offer <id> unavailable: 4xx Client Error: ... for url:
https://.../api/...?...&api_key=<REDACTED>
```

No real API key or terminal excerpt containing one is copied into this record.

## Minimal reproduction

This synthetic reproduction uses a non-secret sentinel and verifies that the
current wrapper preserves it:

```bash
uv run --group devops python - <<'PY'
import requests

from devops.vast.vast_client import VastClient, VastClientError

sentinel = "TEST_KEY_DO_NOT_USE"

class FakeSDK:
    def create_instance(self, **kwargs):
        response = requests.Response()
        response.status_code = 410
        response.url = (
            "https://console.vast.ai/api/v0/asks/123/"
            f"?api_key={sentinel}"
        )
        response.raise_for_status()

client = object.__new__(VastClient)
client.v = FakeSDK()

try:
    client.create_instance(
        123,
        image="test",
        disk=1,
        env={},
        label="test",
        onstart_cmd="true",
    )
except VastClientError as error:
    assert sentinel in str(error), "sentinel was sanitized"
    print(str(error).replace(sentinel, "<REDACTED>"))
else:
    raise AssertionError("expected VastClientError")
PY
```

Observed compact output has this form:

```text
offer 123 unavailable: 410 Client Error: None for url:
https://console.vast.ai/api/v0/asks/123/?api_key=<REDACTED>
```

## Suspected cause and scope

The `except Exception as e` branch in `VastClient.create_instance()` raises
`VastClientError(f"offer {offer_id} unavailable: {e}")`. This trusts the
third-party exception's string representation even though HTTP exceptions may
contain request URLs and query credentials. `cmd_up()` then prints the wrapped
error.

Sanitize exception messages and URLs at the `VastClient` boundary before they
can reach logs. At minimum, redact credential-like query parameters
case-insensitively while retaining status and endpoint context. Add a
regression test with a fake `HTTPError` URL containing a sentinel API key and
assert that neither `VastClientError` nor CLI log output contains the sentinel.
Audit other broad exception logging in `devops/vast` for the same boundary
problem.

## Resolution history

- 2026-07-17 — Recorded after a failed offer request printed an HTTP error URL
  containing the Vast API key; the credential itself was deliberately omitted.
- 2026-07-17 — Resolved in `1c52d39`. SDK errors and on-box teardown logs now
  redact credential query fields and known secret values before formatting,
  and wrapped SDK failures suppress the original secret-bearing exception
  chain. Synthetic SDK and teardown regressions confirm the sentinel key is
  absent from both exceptions and logs.
