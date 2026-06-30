# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convenience factory for serving a ``Switchyard`` or model table.

The default setup registers all three inbound endpoints so the app
can serve OpenAI Chat Completions, Anthropic Messages, and OpenAI
Responses API clients simultaneously — the chain handles format
translation internally.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response

from switchyard.lib import metrics, observability
from switchyard.lib.endpoints.anthropic_messages_endpoint import (
    AnthropicMessagesEndpoint,
)
from switchyard.lib.endpoints.base import Endpoint
from switchyard.lib.endpoints.dispatch import invalid_request_response
from switchyard.lib.endpoints.metrics_endpoint import register_metrics_endpoint
from switchyard.lib.endpoints.models_endpoint import ModelsEndpoint
from switchyard.lib.endpoints.openai_chat_endpoint import (
    OpenAIChatEndpoint,
)
from switchyard.lib.endpoints.responses_endpoint import (
    ResponsesEndpoint,
)
from switchyard_rust.core import SwitchyardInvalidRequestError

#: Inbound LLM-serving paths whose response status codes feed the
#: client-side outcome counter. Other routes (/v1/models, /metrics, /health)
#: are excluded — they don't represent router-served LLM traffic and would
#: distort the error-rate ratio.
_LLM_ROUTES: frozenset[str] = frozenset({
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/responses",
})

if TYPE_CHECKING:
    from switchyard.lib.route_table import SwitchyardApp


async def _run_lifecycle_method(component: object, method_name: str) -> None:
    method = getattr(component, method_name, None)
    if not callable(method):
        return
    result = method()
    if inspect.isawaitable(result):
        await result


async def _shutdown_components(components: Iterable[object]) -> None:
    for component in components:
        await _run_lifecycle_method(component, "shutdown")


def build_switchyard_app(switchyard: SwitchyardApp) -> FastAPI:
    """Create a FastAPI app serving *switchyard* over all inbound formats.

    Registers three LLM endpoints plus a liveness probe:

    - ``POST /v1/chat/completions`` (OpenAI Chat Completions)
    - ``POST /v1/messages``         (Anthropic Messages)
    - ``POST /v1/responses``        (OpenAI Responses API)
    - ``GET  /v1/models``           (local model discovery)
    - ``GET  /health``              (liveness — always 200 when the process is up)

    All three LLM routes go through the same chain — translation
    between wire formats is handled by ``TranslationEngine``
    inside the backend and ``TranslationEngine`` inside the
    translator.

    Example::

        from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
        from switchyard.lib.backends import OpenAiNativeBackend
        from switchyard_rust.translation import TranslationEngine
        from switchyard.lib.switchyard import Switchyard
        from switchyard import build_switchyard_app
        import uvicorn

        switchyard = Switchyard(
            backend=OpenAiNativeBackend(LlmTarget(model="gpt-4o", format=BackendFormat.OPENAI)),
            translator=TranslationEngine(),
        )
        uvicorn.run(build_switchyard_app(switchyard), port=4000)
    """
    components = _switchyard_components(switchyard)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        started: list[object] = []
        startup_complete = False
        try:
            for component in components:
                await _run_lifecycle_method(component, "startup")
                started.append(component)
            startup_complete = True
            yield
        finally:
            shutdown_targets = components if startup_complete else started
            await _shutdown_components(reversed(shutdown_targets))

    app = FastAPI(title="Switchyard", lifespan=_lifespan)

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> Response:
        """Map body validation failures on LLM routes to the Switchyard 400 envelope.

        FastAPI raises RequestValidationError for both malformed JSON and wrong
        body types (e.g. array instead of object) when a route declares a typed
        body parameter. Non-LLM routes and non-body errors fall through to
        FastAPI's default 422 handler.
        """
        if request.url.path in _LLM_ROUTES:
            body_errors = [e for e in exc.errors() if e.get("loc", (None,))[0] == "body"]
            if body_errors:
                is_json_parse = any(e["type"] == "json_invalid" for e in body_errors)
                message = (
                    "Request body is not valid JSON"
                    if is_json_parse
                    else "Request body must be a JSON object"
                )
                return invalid_request_response(message, code="invalid_body")
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(SwitchyardInvalidRequestError)
    async def _invalid_request_handler(
        _request: Request, exc: SwitchyardInvalidRequestError
    ) -> Response:
        """Map Rust-side request validation failures to the 400 envelope.

        ``ChatRequest.validate()`` (called by the inbound endpoints) raises
        this from the Rust core when a body is structurally valid but
        semantically invalid. The only such check today is a present-but-empty
        ``messages`` array, so the envelope uses ``code="empty_messages"``;
        revisit if more validations start sharing this error.
        """
        return invalid_request_response(str(exc), code="empty_messages")

    app.state.switchyard = switchyard

    # Bring up the OTel SDK (no-op when the otel extra is absent / disabled) so the
    # /metrics scrape endpoint and OTLP export are live for the server's lifetime.
    observability.init_observability()
    register_metrics_endpoint(app)

    @app.middleware("http")
    async def _record_client_outcome(request, call_next):  # type: ignore[no-untyped-def]
        """Tally every LLM-route response into the outcome counter.

        Runs after the endpoint produces its response, so it sees the
        final status code regardless of how it was generated (success,
        upstream-error passthrough, internal exception, model-not-found).
        """
        response = await call_next(request)
        if request.url.path in _LLM_ROUTES:
            metrics.record_client_response(
                outcome=metrics.classify(response.status_code)
            )
        return response

    # Route tables can contain hundreds of per-model components that contribute
    # the same fixed-path endpoint. Registering each copy creates unreachable
    # duplicate routes and recursively nests FastAPI lifespan contexts.
    registered_once_endpoint_types: set[type[Endpoint]] = set()
    for endpoint in [
        OpenAIChatEndpoint(),
        AnthropicMessagesEndpoint(),
        ResponsesEndpoint(),
        ModelsEndpoint(),
    ]:
        endpoint.register(app)
        if endpoint.register_once:
            registered_once_endpoint_types.add(type(endpoint))

    for component in components:
        get_endpoint = getattr(component, "get_endpoint", None)
        if not callable(get_endpoint):
            continue
        contributed = cast(Endpoint | None, get_endpoint())
        if contributed is None:
            continue
        endpoint_type = type(contributed)
        if contributed.register_once and endpoint_type in registered_once_endpoint_types:
            continue
        contributed.register(app)
        if contributed.register_once:
            registered_once_endpoint_types.add(endpoint_type)

    @app.get("/health", include_in_schema=False)
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _switchyard_components(
    switchyard: SwitchyardApp,
) -> list[object]:
    iter_components = getattr(switchyard, "iter_components", None)
    if not callable(iter_components):
        if callable(getattr(switchyard, "startup", None)) or callable(
            getattr(switchyard, "shutdown", None)
        ):
            return [switchyard]
        return []
    component_iter = cast(Callable[[], Iterable[object]], iter_components)
    return list(component_iter())
