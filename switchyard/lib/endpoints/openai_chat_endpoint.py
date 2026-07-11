# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint serving a ``Switchyard`` as ``POST /v1/chat/completions``.

The class is stateless — at request time it reads the switchyard from
``request.app.state.switchyard``.  Wire-up is performed by the
``build_switchyard_app()`` convenience factory.

Streaming contract:

- When the request body carries ``"stream": true``, the chain's
  translation engine surfaces an async iterator of ``ChatCompletionChunk``; the
  endpoint wraps it in a ``StreamingResponse`` emitting OpenAI-style
  SSE frames (``data: {...}\\n\\n`` + ``data: [DONE]\\n\\n``).
- Upstream failures (auth, rate-limit, connection) surface before the
  ``StreamingResponse`` is constructed — they propagate as exceptions
  to the global handler and map to proper HTTP error responses.  Only
  mid-stream iteration errors land in the SSE error branch of
  :func:`iter_chat_completion_sse`.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import Response

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.endpoints.dispatch import dispatch_chat_request, serialize_chain_result
from switchyard.lib.endpoints.sse_helpers import iter_chat_completion_sse
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


class OpenAIChatEndpoint(NemoSwitchyardEndpoint):
    """Composable endpoint that exposes ``POST /v1/chat/completions``.

    Reads the raw JSON body, wraps it in a Rust-backed OpenAI chat request (no
    validation or field-stripping, so provider-specific fields pass
    through transparently), runs the chain, and either JSON-serializes
    the result (non-streaming) or wraps the async chunk iterator in an
    SSE ``StreamingResponse`` (streaming).

    Streaming support is limited to same-format passthrough today —
    i.e. OpenAI Chat Completions inbound against an OpenAI-native
    backend.  Cross-format streaming (Anthropic / Responses inbound)
    raises ``NotImplementedError`` from ``TranslationEngine``
    until streaming translation lands for those formats.
    """

    def register(self, app: FastAPI) -> None:
        """Attach ``POST /v1/chat/completions`` onto *app*."""
        router = APIRouter()

        @router.post("/v1/chat/completions", response_model=None)
        async def chat_completions(
            request: Request,
            body: Annotated[dict[str, Any], Body(...)],
        ) -> Response:
            obj = request.app.state.switchyard
            chat_request = ChatRequest.openai_chat(body)
            # Reject semantically invalid input (e.g. empty messages) at the
            # inbound boundary; raises SwitchyardInvalidRequestError -> 400.
            chat_request.validate()
            ctx = ProxyContext()
            attach_request_metadata(
                ctx,
                RequestMetadata.from_headers(request.headers),
                request.headers,
            )
            attach_caller_api_key(ctx, request.headers)
            stream = bool(body.get("stream"))
            try:
                result: Any = await dispatch_chat_request(obj, chat_request, ctx)
                return serialize_chain_result(
                    result, stream=stream, sse_iter=iter_chat_completion_sse, ctx=ctx
                )
            except (SwitchyardContextPoolExhaustedError, SwitchyardContextWindowExceededError) as exc:
                return context_exhausted_response(exc, inbound="openai")
            except Exception as exc:
                return handle_chain_exception(
                    exc, ctx, inbound="openai", log_msg="POST /v1/chat/completions chain raised"
                )

        app.include_router(router, tags=["OpenAI Compatible"])
