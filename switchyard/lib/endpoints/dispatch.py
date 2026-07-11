# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared endpoint dispatch for Python chains and model registries."""

from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from fastapi.responses import JSONResponse, Response, StreamingResponse

from switchyard.lib.endpoints.error_envelope import error_response
from switchyard.lib.endpoints.route_selection import route_selection_headers
from switchyard.lib.endpoints.upstream_error import record_upstream_attempt_success
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import TranslatedResponse
from switchyard.lib.route_table import RouteTable
from switchyard_rust.core import ChatRequest

_MISSING_MODEL_LABEL = "<missing>"


def model_not_found_response(model: str) -> JSONResponse:
    """Build the OpenAI-compatible error payload for unknown model IDs."""
    return error_response(
        404,
        f"No route registered for model {model}",
        error_type="model_not_found",
        code="model_not_found",
    )


def _model_label(model: object | None) -> str:
    """Return a stable human label for model-not-found errors."""
    return str(model) if model else _MISSING_MODEL_LABEL


def invalid_request_response(message: str, *, code: str = "invalid_request_error") -> JSONResponse:
    """Build the OpenAI-compatible error payload for invalid requests."""
    return error_response(
        400,
        message,
        error_type="invalid_request_error",
        code=code,
    )


async def dispatch_chat_request(
    app_state: object,
    chat_request: ChatRequest,
    ctx: ProxyContext,
) -> TranslatedResponse | Response:
    """Dispatch one request through the configured app state.

    Single-chain apps return already-translated Python payloads because
    ``Switchyard`` still owns its terminal translator.
    """
    if isinstance(app_state, RouteTable):
        model = _model_label(chat_request.model)
        try:
            table_chain = app_state.lookup_switchyard(model)
        except KeyError:
            # No upstream call happened — a 404 is not an upstream attempt.
            return model_not_found_response(model)
        result = await table_chain.call(chat_request, ctx=ctx)
        record_upstream_attempt_success(ctx)
        return result

    chain: Any = app_state
    result = cast(TranslatedResponse, await chain.call(chat_request, ctx=ctx))
    record_upstream_attempt_success(ctx)
    return result


def model_entries(app_state: object) -> list[dict[str, Any]]:
    """Return OpenAI-compatible model entries for table app state."""
    if isinstance(app_state, RouteTable):
        return app_state.registered_model_entries()
    return []


def model_listing_warnings(app_state: object) -> list[str]:
    """Return non-fatal model listing warnings for table-backed apps."""
    if isinstance(app_state, RouteTable):
        return app_state.model_listing_warnings()
    return []


def model_listing_default(app_state: object) -> str | None:
    """Return the default model id advertised by ``GET /v1/models``."""
    if isinstance(app_state, RouteTable):
        return app_state.default_model()
    return None


def serialize_chain_result(
    result: Any,
    *,
    stream: bool,
    sse_iter: Callable[[Any], AsyncIterator[str]],
    ctx: ProxyContext,
) -> Response:
    """Serialize a chain result to the appropriate HTTP response.

    Returns the result as-is if it is already a ``Response``, wraps it in a
    ``StreamingResponse`` when streaming is requested, or JSON-serializes it.
    Any route selection recorded on *ctx* is stamped as ``x-switchyard-*``
    response headers (streaming included — the backend call completed before
    the response object is built, so the selection is final). ``ctx`` is
    required so a new endpoint cannot silently opt out of spend attribution.
    """
    if isinstance(result, Response):
        return result
    headers = route_selection_headers(ctx)
    if stream and hasattr(result, "__aiter__"):
        return StreamingResponse(
            sse_iter(result), media_type="text/event-stream", headers=headers
        )
    if hasattr(result, "model_dump"):
        return JSONResponse(content=result.model_dump(), headers=headers)
    return JSONResponse(content=result, headers=headers)
