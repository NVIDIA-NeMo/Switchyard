# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint serving a ``Switchyard`` as ``POST /v1/responses`` (OpenAI Responses API).

Paper-thin by design: wrap the raw JSON body in a Rust-backed Responses request,
run the chain, serialize the result. All Responses ↔ Chat Completions
format conversion lives inside the chain's ``TranslationEngine``, so the
endpoint itself contains zero translation logic.

Streaming contract:

- When the request body carries ``"stream": true``, the chain's
  translation engine surfaces an async iterator of pre-formatted Responses API
  SSE frames; :func:`iter_preframed_sse` forwards them verbatim through
  a ``StreamingResponse`` with mid-stream error quarantine.
- Non-streaming requests return the Responses ``Response`` body as JSON.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import Response

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.endpoints.dispatch import dispatch_chat_request, serialize_chain_result
from switchyard.lib.endpoints.sse_helpers import iter_preframed_sse
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


class ResponsesEndpoint(NemoSwitchyardEndpoint):
    """Composable endpoint that exposes ``POST /v1/responses``."""

    def register(self, app: FastAPI) -> None:
        """Attach ``POST /v1/responses`` onto *app*."""
        router = APIRouter()

        @router.post("/v1/responses", response_model=None)
        async def responses(
            request: Request,
            body: Annotated[dict[str, Any], Body(...)],
        ) -> Response:
            """OpenAI-compatible Responses endpoint."""
            obj = request.app.state.switchyard
            model = str(body.get("model", "<none>"))
            stream = bool(body.get("stream"))
            log.debug(
                "POST /v1/responses model=%s stream=%s keys=%s",
                model,
                stream,
                list(body.keys()),
            )

            chat_request = ChatRequest.openai_responses(body)
            ctx = ProxyContext()
            attach_request_metadata(
                ctx,
                RequestMetadata.from_headers(request.headers),
                request.headers,
            )
            attach_caller_api_key(ctx, request.headers)
            try:
                result: Any = await dispatch_chat_request(obj, chat_request, ctx)
                if not isinstance(result, Response):
                    log.debug(
                        "POST /v1/responses chain returned model=%s stream=%s result=%s",
                        model,
                        stream,
                        type(result).__name__,
                    )
                return serialize_chain_result(
                    result, stream=stream, sse_iter=iter_preframed_sse, ctx=ctx
                )
            except (SwitchyardContextPoolExhaustedError, SwitchyardContextWindowExceededError) as exc:
                return context_exhausted_response(exc, inbound="openai-responses")
            except Exception as exc:
                return handle_chain_exception(
                    exc,
                    ctx,
                    inbound="openai-responses",
                    log_msg=f"POST /v1/responses chain raised model={model}",
                )

        app.include_router(router, tags=["OpenAI Responses"])
