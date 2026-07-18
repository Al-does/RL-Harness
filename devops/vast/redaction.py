"""Credential-safe formatting for errors from Vast and GitHub HTTP clients."""

from __future__ import annotations

import re
from collections.abc import Iterable

_REDACTED = "<REDACTED>"
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|github[_-]?token|token)=)[^&#\s]*"
)
_FIELD_SECRET = re.compile(
    r"""(?ix)
    (\b(?:api[_-]?key|access[_-]?token|github[_-]?token)\b
    \s*[:=]\s*["']?)
    ([^&,\s"' }\]]+)
    """
)
_BEARER_SECRET = re.compile(r"(?i)(\bauthorization\s*:\s*bearer\s+)\S+")


def redact_sensitive(value: object, secrets: Iterable[str | None] = ()) -> str:
    """Return a useful error string with credentials removed.

    Third-party HTTP exceptions commonly include their request URL. Vast API
    URLs may carry ``api_key`` in the query string, so exception text is
    untrusted at every logging boundary.
    """

    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    text = _QUERY_SECRET.sub(rf"\1{_REDACTED}", text)
    text = _FIELD_SECRET.sub(rf"\1{_REDACTED}", text)
    return _BEARER_SECRET.sub(rf"\1{_REDACTED}", text)
