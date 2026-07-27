"""Credential-safe formatting for Serverless API data and errors."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping

REDACTED = "<REDACTED>"
_SECRET_KEY = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key|access[_-]?key|"
    r"application[_-]?key|private[_-]?key)"
)
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token)=)[^&#\s]*"
)
_BEARER = re.compile(r"(?i)(\bauthorization\s*:\s*bearer\s+)\S+")


def redact_sensitive(
    value: object, secrets: Iterable[str | None] = ()
) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    text = _QUERY_SECRET.sub(rf"\1{REDACTED}", text)
    return _BEARER.sub(rf"\1{REDACTED}", text)


def sensitive_values(value: object) -> tuple[str, ...]:
    """Collect values below secret-named keys for HTTP error redaction."""
    found: list[str] = []

    def visit(child: object, secret_context: bool = False) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                visit(nested, secret_context or bool(_SECRET_KEY.search(str(key))))
        elif isinstance(child, (list, tuple)):
            for nested in child:
                visit(nested, secret_context)
        elif secret_context and child not in (None, ""):
            found.append(str(child))

    visit(value)
    return tuple(found)


def redact_metadata(value: object) -> object:
    """Deep-copy API metadata while removing secret-named values."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if _SECRET_KEY.search(str(key)) and child not in (None, "")
                else redact_metadata(child)
            )
            for key, child in copy.deepcopy(dict(value)).items()
        }
    if isinstance(value, list):
        return [redact_metadata(child) for child in value]
    if isinstance(value, str) and len(value) > 512:
        return f"<OMITTED:{len(value)} chars>"
    return value
