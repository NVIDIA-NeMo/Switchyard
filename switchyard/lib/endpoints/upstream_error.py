# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-side helper: convert chain exceptions into upstream-status responses.

Python LLM backends (e.g. :class:`LatencyServiceLLMBackend`) stash the
upstream HTTP status / body into ``ctx.metadata`` before raising the
upstream provider's exception. Rust backends attach typed
``status_code`` and ``body`` attributes to ``SwitchyardUpstreamError`` so
endpoints can preserve provider failures without parsing exception text.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal

from fastapi.responses import JSONResponse

from switchyard.lib import metrics
from switchyard.lib.endpoints.error_envelope import error_response, upstream_error_response
from switchyard.lib.proxy_context import (
    CTX_UPSTREAM_ATTEMPTS_RECORDED,
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
)
from switchyard_rust.core import SwitchyardUpstreamError

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext

Inbound = Literal["anthropic", "openai", "openai-responses"]

_log = logging.getLogger(__name__)


def record_upstream_attempt_success(ctx: ProxyContext) -> None:
    """Record one successful (HTTP 200) ``switchyard_upstream_attempts_total``.

    The endpoint-layer fallback that wires the upstream-attempt counter for
    every chain whose backend does not record per-attempt itself: the Rust
    native / passthrough / multi backends issue exactly one upstream attempt
    per call and have no Python retry loop, so one served client request maps
    to one successful attempt observable here.

    No-op when a backend already counted its own attempts
    (:data:`CTX_UPSTREAM_ATTEMPTS_RECORDED` set) — e.g.
    :class:`LatencyServiceLLMBackend`, whose retry fan-out must not be
    double-counted here.
    """
    if ctx.metadata.get(CTX_UPSTREAM_ATTEMPTS_RECORDED):
        return
    _record_attempt(200)


def record_upstream_attempt_failure(ctx: ProxyContext, exc: BaseException) -> None:
    """Record one failed ``switchyard_upstream_attempts_total`` when attributable.

    Counts an attempt only for failures attributable to the upstream call — a
    Python backend that stashed :data:`CTX_UPSTREAM_HTTP_STATUS`, or a Rust
    backend's :class:`SwitchyardUpstreamError` (HTTP status carried verbatim;
    a status-less upstream error is a network / pre-status failure recorded as
    ``None`` → ``retryable_error``). Internal chain failures (translation,
    processor, validation) are not upstream attempts and are skipped.

    No-op when a backend already counted its own attempts
    (:data:`CTX_UPSTREAM_ATTEMPTS_RECORDED` set).
    """
    if ctx.metadata.get(CTX_UPSTREAM_ATTEMPTS_RECORDED):
        return
    status = ctx.metadata.get(CTX_UPSTREAM_HTTP_STATUS)
    if isinstance(status, int):
        _record_attempt(status)
        return
    if isinstance(exc, SwitchyardUpstreamError):
        rust_status = getattr(exc, "status_code", None)
        _record_attempt(rust_status if isinstance(rust_status, int) else None)


def _record_attempt(status_code: int | None) -> None:
    """Record one upstream attempt, classifying status into outcome/code.

    A non-HTTP failure (``status_code=None``) is bucketed as ``retryable_error``
    — exactly the fault a health-aware router should absorb by retrying.
    """
    outcome = "retryable_error" if status_code is None else metrics.classify(status_code)
    metrics.record_upstream_attempt(outcome=outcome, code=metrics.code_label(status_code))


def upstream_response_from_ctx(
    ctx: ProxyContext,
    *,
    inbound: Inbound = "openai",
    exc: BaseException | None = None,
) -> JSONResponse | None:
    """Return a structured upstream-status response when one can be recovered.

    Python backends store status/body in ``ctx.metadata``. Rust backends attach
    typed upstream status/body attributes to ``SwitchyardUpstreamError``.
    ``inbound`` is accepted for endpoint compatibility; HTTP errors use one
    Switchyard envelope across all LLM routes so clients see stable fields.
    Returns ``None`` when neither source carries an upstream status and the
    endpoint should re-raise.
    """
    status = ctx.metadata.get(CTX_UPSTREAM_HTTP_STATUS)
    if isinstance(status, int):
        return upstream_error_response(status, ctx.metadata.get(CTX_UPSTREAM_HTTP_BODY))
    return upstream_response_from_error(exc, inbound=inbound)


def upstream_response_from_error(
    exc: BaseException | None,
    *,
    inbound: Inbound = "openai",
) -> JSONResponse | None:
    """Return a normalized response for typed Rust upstream HTTP failures."""
    if not isinstance(exc, SwitchyardUpstreamError):
        return None
    status = getattr(exc, "status_code", None)
    raw_body = getattr(exc, "body", None)
    if not isinstance(status, int) or not isinstance(raw_body, str):
        return None
    body = _parse_body(raw_body)
    return upstream_error_response(status, body)


def _parse_body(raw: str) -> object:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def internal_chain_error_response(exc: BaseException, inbound: Inbound) -> JSONResponse:
    """Translate an unexpected chain failure into the client error envelope.

    Used when an exception escapes dispatch or response-processing that is not
    a known upstream HTTP error (no status stashed in ctx) and not a
    ``SwitchyardUpstreamError``. LLM clients expect a JSON error object rather
    than FastAPI's plain-text 500; callers must log the traceback before calling
    this helper so the full context is preserved server-side. ``inbound`` is
    retained in the signature because endpoint callers already pass it, but the
    HTTP envelope is intentionally shared across inbound formats.
    """
    message = repr(exc)[:200]
    return error_response(
        500,
        message,
        error_type="internal_error",
        code="internal_chain_error",
    )


def handle_chain_exception(
    exc: BaseException,
    ctx: ProxyContext,
    *,
    inbound: Inbound,
    log_msg: str,
) -> JSONResponse:
    """Handle an unexpected chain exception: check for upstream status, log, and return envelope."""
    record_upstream_attempt_failure(ctx, exc)
    upstream = upstream_response_from_ctx(ctx, inbound=inbound, exc=exc)
    if upstream is not None:
        return upstream
    _log.error(log_msg, exc_info=exc)
    return internal_chain_error_response(exc, inbound=inbound)


def context_exhausted_response(exc: BaseException, inbound: Inbound) -> JSONResponse:
    """Translate :class:`SwitchyardContextPoolExhaustedError` into a 400.

    Raised by the chain executor when every routing target has been evicted
    after consecutive context-window overflows; FastAPI endpoints catch it
    and call this helper to produce the shared Switchyard HTTP error envelope.
    """
    error: dict[str, object] = {
        "type": "invalid_request_error",
        "message": str(exc),
        "code": "context_length_exceeded",
    }
    return JSONResponse(status_code=400, content={"error": error})
