# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint serving a ``Switchyard`` as Gemini ``generateContent``.

Paper-thin by design: wrap the raw JSON body in a Rust-backed Gemini request,
run the chain, serialize the result. All Gemini ↔ OpenAI/Anthropic format
conversion lives inside the chain (``TranslationEngine``), so the endpoint
itself contains zero translation logic.

Unlike the other inbound formats, Gemini encodes the model and the streaming
choice in the URL rather than the body:

- ``POST /v1beta/models/{model}:generateContent`` — buffered JSON response.
- ``POST /v1beta/models/{model}:streamGenerateContent`` — SSE stream of
  ``GenerateContentResponse`` chunks (clients append ``?alt=sse``).

Both values are injected into the wrapped body as synthetic ``model`` and
``stream`` fields; the Gemini codec and backend understand that convention
and move them back into the upstream URL.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import Response

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.endpoints.dispatch import dispatch_chat_request, serialize_chain_result
from switchyard.lib.endpoints.sse_helpers import iter_gemini_sse
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


class GeminiEndpoint(NemoSwitchyardEndpoint):
    """Composable endpoint that exposes Gemini ``generateContent`` routes."""

    def register(self, app: FastAPI) -> None:
        """Attach the Gemini ``generateContent`` routes onto *app*."""
        router = APIRouter()

        @router.post("/v1beta/models/{model}:generateContent", response_model=None)
        async def gemini_generate_content(
            model: str,
            request: Request,
            body: Annotated[dict[str, Any], Body(...)],
        ) -> Response:
            return await _handle(request, model, body, stream=False)

        @router.post("/v1beta/models/{model}:streamGenerateContent", response_model=None)
        async def gemini_stream_generate_content(
            model: str,
            request: Request,
            body: Annotated[dict[str, Any], Body(...)],
        ) -> Response:
            return await _handle(request, model, body, stream=True)

        async def _handle(
            request: Request,
            model: str,
            body: dict[str, Any],
            *,
            stream: bool,
        ) -> Response:
            obj = request.app.state.switchyard
            log.debug(
                "POST /v1beta/models/%s:%s keys=%s",
                model,
                "streamGenerateContent" if stream else "generateContent",
                list(body.keys()),
            )
            # The URL carries what other formats put in the body; the codec
            # and backend expect both as synthetic body fields.
            body["model"] = model
            body["stream"] = stream
            ctx = ProxyContext()
            attach_request_metadata(
                ctx,
                RequestMetadata.from_headers(request.headers),
                request.headers,
            )
            attach_caller_api_key(ctx, request.headers)

            chat_request = ChatRequest.gemini(body)
            # Reject semantically invalid input (e.g. empty contents) at the
            # inbound boundary; raises SwitchyardInvalidRequestError -> 400.
            chat_request.validate()

            try:
                result: Any = await dispatch_chat_request(obj, chat_request, ctx)
                if not isinstance(result, Response):
                    log.debug(
                        "Gemini generateContent chain returned model=%s stream=%s result=%s",
                        model,
                        stream,
                        type(result).__name__,
                    )
                return serialize_chain_result(result, stream=stream, sse_iter=iter_gemini_sse)
            except (SwitchyardContextPoolExhaustedError, SwitchyardContextWindowExceededError) as exc:
                return context_exhausted_response(exc, inbound="gemini")
            except Exception as exc:
                return handle_chain_exception(
                    exc,
                    ctx,
                    inbound="gemini",
                    log_msg=f"Gemini generateContent chain raised model={model}",
                )

        app.include_router(router, tags=["Gemini Compatible"])
