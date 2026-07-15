# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint serving a ``Switchyard`` as ``POST /v1/messages`` (Anthropic Messages API).

Paper-thin by design: wrap the raw JSON body in a Rust-backed Anthropic request,
run the chain, serialize the result.  All Anthropic ↔ OpenAI format
conversion lives inside the chain (``TranslationEngine`` and
``TranslationEngine``), so the endpoint itself contains zero
translation logic.

Streaming contract:

- When the request body carries ``"stream": true``, the chain's
  translation engine surfaces an async iterator of Anthropic event dicts; the
  endpoint frames them into Anthropic-style named-event SSE
  (``event: message_start\\ndata: {...}\\n\\n``, …) via
  :func:`iter_anthropic_sse`.
- Non-streaming requests return the Anthropic ``Message`` body as JSON.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import Response

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.endpoints.dispatch import dispatch_chat_request, serialize_chain_result
from switchyard.lib.endpoints.sse_helpers import iter_anthropic_sse
from switchyard.lib.endpoints.upstream_error import (
    context_exhausted_response,
    handle_chain_exception,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.request_metadata import (
    RequestMetadata,
    attach_caller_api_key,
    attach_request_metadata,
)
from switchyard_rust.core import (
    ChatRequest,
    SwitchyardContextPoolExhaustedError,
    SwitchyardContextWindowExceededError,
)

log = logging.getLogger(__name__)


def _strip_unsupported_output_config(body: dict[str, Any]) -> None:
    """Drop ``output_config.format`` from an inbound Anthropic body in place.

    Claude Code 2.1.1x sends ``output_config.format`` (a structured-output
    schema) that upstream Anthropic model groups reject with HTTP 400.
    ``output_config.effort`` is accepted, so only the ``format`` key is
    removed; if that leaves ``output_config`` empty it is dropped entirely.
    """
    oc = body.get("output_config")
    if isinstance(oc, dict) and "format" in oc:
        oc.pop("format", None)
        if not oc:
            body.pop("output_config", None)


class AnthropicMessagesEndpoint(NemoSwitchyardEndpoint):
    """Composable endpoint that exposes ``POST /v1/messages``."""

    def register(self, app: FastAPI) -> None:
        """Attach ``POST /v1/messages`` onto *app*."""
        router = APIRouter()

        @router.post("/v1/messages", response_model=None)
        async def anthropic_messages(
            request: Request,
            body: Annotated[dict[str, Any], Body(...)],
        ) -> Response:
            """Anthropic-compatible Messages endpoint."""
            obj = request.app.state.switchyard
            _strip_unsupported_output_config(body)
            model = str(body.get("model", "<none>"))
            stream = bool(body.get("stream"))
            log.debug(
                "POST /v1/messages model=%s stream=%s keys=%s",
                model,
                stream,
                list(body.keys()),
            )
            ctx = ProxyContext()
            attach_request_metadata(
                ctx,
                RequestMetadata.from_headers(request.headers),
                request.headers,
            )
            attach_caller_api_key(ctx, request.headers)

            chat_request = ChatRequest.anthropic(body)
            # Reject semantically invalid input (e.g. empty messages) at the
            # inbound boundary; raises SwitchyardInvalidRequestError -> 400.
            chat_request.validate()

            try:
                result: Any = await dispatch_chat_request(obj, chat_request, ctx)
                if not isinstance(result, Response):
                    log.debug(
                        "POST /v1/messages chain returned model=%s stream=%s result=%s",
                        model,
                        stream,
                        type(result).__name__,
                    )
                return serialize_chain_result(
                    result, stream=stream, sse_iter=iter_anthropic_sse, ctx=ctx
                )
            except (SwitchyardContextPoolExhaustedError, SwitchyardContextWindowExceededError) as exc:
                return context_exhausted_response(exc, inbound="anthropic")
            except Exception as exc:
                return handle_chain_exception(
                    exc,
                    ctx,
                    inbound="anthropic",
                    log_msg=f"POST /v1/messages chain raised model={model}",
                )

        app.include_router(router, tags=["Anthropic Compatible"])
