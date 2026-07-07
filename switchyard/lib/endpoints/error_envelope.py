# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP error envelopes for Switchyard LLM-serving endpoints."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse

from switchyard.lib.proxy_context import ERROR_SOURCE_PROVIDER, ERROR_SOURCE_SWITCHYARD

_DEFAULT_UPSTREAM_MESSAGE = "upstream returned HTTP {status}"

#: Response header naming the layer that originated the error: ``switchyard``
#: (this proxy rejected or failed the request itself) or ``provider`` (an
#: upstream LLM failure passed through). Layers above Switchyard (e.g. a
#: LiteLLM front proxy) are expected to tag their own failures the same way
#: and propagate this header from below — Switchyard cannot see them. The
#: values live in :mod:`switchyard.lib.proxy_context` so FastAPI-free backend
#: code can stamp them.
ERROR_SOURCE_HEADER = "x-switchyard-error-source"

#: Response header carrying the upstream model actually attempted when the
#: surfaced failure happened, when a routing selection took place.
UPSTREAM_MODEL_HEADER = "x-switchyard-upstream-model"


def error_payload(
    message: str,
    *,
    error_type: str,
    code: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Return the normalized JSON body used by all LLM HTTP endpoints."""
    error: dict[str, object] = {
        "message": message,
        "type": error_type,
        "code": code,
    }
    if extra:
        error.update(extra)
    return {"error": error}


def error_response(
    status_code: int,
    message: str,
    *,
    error_type: str,
    code: str,
    extra: Mapping[str, object] | None = None,
    error_source: str | None = ERROR_SOURCE_SWITCHYARD,
    upstream_model: str | None = None,
) -> JSONResponse:
    """Build a JSONResponse with Switchyard's normalized error envelope.

    Stamps the failure-source headers: every direct caller synthesizes a
    Switchyard-originated envelope, so ``error_source`` defaults to
    ``switchyard``; the upstream passthrough path overrides it with
    ``provider``. Headers rather than body fields keep the passthrough
    contract intact — provider error bodies flow through unmodified.
    """
    headers: dict[str, str] = {}
    if error_source:
        headers[ERROR_SOURCE_HEADER] = error_source
    if upstream_model:
        headers[UPSTREAM_MODEL_HEADER] = upstream_model
    return JSONResponse(
        status_code=status_code,
        content=error_payload(message, error_type=error_type, code=code, extra=extra),
        headers=headers or None,
    )


def upstream_error_response(
    status_code: int,
    body: object,
    *,
    error_source: str = ERROR_SOURCE_PROVIDER,
    upstream_model: str | None = None,
) -> JSONResponse:
    """Normalize an upstream provider error body into Switchyard's envelope.

    ``error_source`` defaults to ``provider`` — this path renders upstream
    failures — but a backend that deliberately routes its own rejection
    through the upstream-status stash (e.g. the ``caller_required`` 401)
    overrides it back to ``switchyard`` via ``ctx``.
    """
    parsed = _upstream_error_fields(status_code, body)
    return error_response(
        status_code,
        parsed.message,
        error_type=parsed.error_type,
        code=parsed.code,
        extra=parsed.extra,
        error_source=error_source,
        upstream_model=upstream_model,
    )


@dataclass(frozen=True)
class _UpstreamErrorFields:
    """Internal value object for provider error fields after normalization."""

    message: str
    error_type: str = "upstream_error"
    code: str = "upstream_error"
    extra: Mapping[str, object] | None = None


def _upstream_error_fields(status_code: int, body: object) -> _UpstreamErrorFields:
    """Extract stable error fields from common provider error shapes."""
    default_message = _DEFAULT_UPSTREAM_MESSAGE.format(status=status_code)
    if isinstance(body, str):
        return _UpstreamErrorFields(message=body or default_message)
    if isinstance(body, Mapping):
        return _fields_from_mapping(status_code, body)
    if isinstance(body, list):
        return _UpstreamErrorFields(message=_compact_json(body))
    return _UpstreamErrorFields(message=default_message)


def _fields_from_mapping(status_code: int, body: Mapping[str, object]) -> _UpstreamErrorFields:
    """Handle OpenAI-style ``{"error": {...}}`` and flat error dictionaries."""
    error = body.get("error")
    source = error if isinstance(error, Mapping) else body
    default_message = _DEFAULT_UPSTREAM_MESSAGE.format(status=status_code)

    message = _string_field(source, "message") or _compact_json(body) or default_message
    error_type = _string_field(source, "type") or "upstream_error"
    code = _string_or_number_field(source, "code") or (
        error_type if error_type != "upstream_error" else "upstream_error"
    )
    extra = {
        key: value
        for key, value in {
            "param": _string_or_number_field(source, "param"),
        }.items()
        if value is not None
    }
    return _UpstreamErrorFields(
        message=message,
        error_type=error_type,
        code=code,
        extra=extra,
    )


def _string_field(source: Mapping[str, object], key: str) -> str | None:
    value = source.get(key)
    return value if isinstance(value, str) and value else None


def _string_or_number_field(source: Mapping[str, object], key: str) -> str | None:
    value = source.get(key)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int | float):
        return str(value)
    return None


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return str(value)
