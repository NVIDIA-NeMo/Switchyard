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

from switchyard.lib.endpoints import outcome_metrics
from switchyard.lib.endpoints.error_envelope import (
    error_response,
    upstream_error_response,
)
from switchyard.lib.proxy_context import (
    CTX_ERROR_SOURCE,
    CTX_UPSTREAM_ATTEMPTS_RECORDED,
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
    CTX_UPSTREAM_MODEL,
    ERROR_SOURCE_PROVIDER,
    ERROR_SOURCE_SWITCHYARD,
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
    outcome_metrics.record_upstream_attempt(200)


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
        outcome_metrics.record_upstream_attempt(status)
        return
    if isinstance(exc, SwitchyardUpstreamError):
        rust_status = getattr(exc, "status_code", None)
        outcome_metrics.record_upstream_attempt(
            rust_status if isinstance(rust_status, int) else None
        )


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
        return upstream_error_response(
            status,
            ctx.metadata.get(CTX_UPSTREAM_HTTP_BODY),
            # A stashed status without an explicit source is an upstream
            # passthrough; backends that reuse this channel for their own
            # rejections (caller_required 401, translation 400) mark the
            # stash ``switchyard`` so the header stays truthful.
            error_source=_ctx_error_source(ctx, default=ERROR_SOURCE_PROVIDER),
            upstream_model=_ctx_upstream_model(ctx),
        )
    return upstream_response_from_error(exc, inbound=inbound)


def _ctx_error_source(ctx: ProxyContext, *, default: str) -> str:
    """Failure origin stamped by the backend, or ``default`` for the path."""
    source = ctx.metadata.get(CTX_ERROR_SOURCE)
    return source if isinstance(source, str) and source else default


def _ctx_upstream_model(ctx: ProxyContext) -> str | None:
    """Upstream model recorded at failure time, when a selection happened."""
    model = ctx.metadata.get(CTX_UPSTREAM_MODEL)
    return model if isinstance(model, str) and model else None


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


def internal_chain_error_response(
    exc: BaseException,
    inbound: Inbound,
    *,
    error_source: str = ERROR_SOURCE_SWITCHYARD,
    upstream_model: str | None = None,
) -> JSONResponse:
    """Translate an unexpected chain failure into the client error envelope.

    Used when an exception escapes dispatch or response-processing that is not
    a known upstream HTTP error (no status stashed in ctx) and not a
    ``SwitchyardUpstreamError``. LLM clients expect a JSON error object rather
    than FastAPI's plain-text 500; callers must log the traceback before calling
    this helper so the full context is preserved server-side. ``inbound`` is
    retained in the signature because endpoint callers already pass it, but the
    HTTP envelope is intentionally shared across inbound formats.

    ``error_source`` defaults to ``switchyard`` (an unexpected internal
    failure) but a backend that failed on a status-less upstream fault (e.g.
    a network error after retries) marks ``ctx`` so this 500 is labeled
    ``provider`` instead.
    """
    message = repr(exc)[:200]
    return error_response(
        500,
        message,
        error_type="internal_error",
        code="internal_chain_error",
        error_source=error_source,
        upstream_model=upstream_model,
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
    return internal_chain_error_response(
        exc,
        inbound=inbound,
        error_source=_ctx_error_source(ctx, default=ERROR_SOURCE_SWITCHYARD),
        upstream_model=_ctx_upstream_model(ctx),
    )


def context_exhausted_response(exc: BaseException, inbound: Inbound) -> JSONResponse:
    """Translate :class:`SwitchyardContextPoolExhaustedError` into a 400.

    Raised by the chain executor when every routing target has been evicted
    after consecutive context-window overflows; FastAPI endpoints catch it
    and call this helper to produce the shared Switchyard HTTP error envelope.
    """
    # Chain-executor rejection, not an upstream failure — ``error_response``
    # stamps the ``switchyard`` source header by default.
    return error_response(
        400,
        str(exc),
        error_type="invalid_request_error",
        code="context_length_exceeded",
    )
