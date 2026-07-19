"""Credential-safe formatting for errors and Vast control-plane metadata."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping

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
_SECRET_ENV_KEY = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key|access[_-]?key|"
    r"application[_-]?key|private[_-]?key)$"
)
_ENV_METADATA_KEYS = ("extra_env", "env", "environment")


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


def _is_secret_env_key(key: object) -> bool:
    name = str(key)
    if _SECRET_ENV_KEY.search(name):
        return True
    upper = name.upper()
    return upper in {
        "GITHUB_TOKEN",
        "VAST_API_KEY",
        "B2_APPLICATION_KEY",
        "B2_APPLICATION_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
    }


def _redact_env_mapping(env: Mapping[object, object]) -> dict:
    return {
        str(key): (_REDACTED if _is_secret_env_key(key) and value not in (None, "") else value)
        for key, value in env.items()
    }


def redact_instance_metadata(instance: object) -> object:
    """Return instance metadata safe for logs/tool transcripts.

    Vast persists create-time environment variables in control-plane fields such
    as ``extra_env``. Never dump raw ``show instance --raw`` output into agent
    transcripts; use this helper (or ``provision inspect``) instead.
    """

    if not isinstance(instance, Mapping):
        return instance
    safe = copy.deepcopy(dict(instance))
    for key in _ENV_METADATA_KEYS:
        value = safe.get(key)
        if isinstance(value, Mapping):
            safe[key] = _redact_env_mapping(value)
    return safe
