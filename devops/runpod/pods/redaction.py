"""Credential-safe formatting for RunPod errors and Pod metadata."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping

_REDACTED = "<REDACTED>"
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|github[_-]?token|token)=)[^&#\s]*"
)
_BEARER_SECRET = re.compile(r"(?i)(\bauthorization\s*:\s*bearer\s+)\S+")
_SECRET_ENV_KEY = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key|access[_-]?key|"
    r"application[_-]?key|private[_-]?key)"
)


def redact_sensitive(
    value: object,
    secrets: Iterable[str | None] = (),
) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    text = _QUERY_SECRET.sub(rf"\1{_REDACTED}", text)
    return _BEARER_SECRET.sub(rf"\1{_REDACTED}", text)


def redact_pod_metadata(pod: object) -> object:
    """Return Pod metadata with secret-bearing environment values removed."""
    if not isinstance(pod, Mapping):
        return pod
    safe = copy.deepcopy(dict(pod))
    env = safe.get("env")
    if isinstance(env, Mapping):
        safe["env"] = {
            str(key): (
                _REDACTED
                if _SECRET_ENV_KEY.search(str(key)) and value not in (None, "")
                else (
                    f"<OMITTED:{len(value)} chars>"
                    if isinstance(value, str) and len(value) > 256
                    else value
                )
            )
            for key, value in env.items()
        }
    return safe
