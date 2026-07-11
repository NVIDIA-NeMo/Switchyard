# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for route-selection spend-attribution headers.

For tokenomics reporting, a front proxy (e.g. LiteLLM) must be able to tie
its provider spend-log rows back to the Switchyard logical route that
selected them:

* outbound — every upstream attempt carries an ``x-litellm-spend-logs-metadata``
  request header whose JSON payload records the route selection plus a
  per-request correlation id;
* inbound — the Switchyard response returns ``x-switchyard-*`` headers with
  the successful attempt's selection and the same correlation id, so the
  parent spend-log row can be enriched to match the provider row.

The app-level tests run the real ``OpenAILLMClient`` against a loopback HTTP
stub, so the outbound header is asserted on the wire — not on a mock.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import fixture

from switchyard.lib.backends.health_poller import HealthPoller
from switchyard.lib.backends.latency_service_llm_backend import (
    SPEND_LOGS_METADATA_HEADER,
)
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.endpoints.dispatch import serialize_chain_result
from switchyard.lib.endpoints.route_selection import (
    ROUTER_CORRELATION_ID_HEADER,
    ROUTER_MODEL_HEADER,
    SELECTED_MODEL_HEADER,
    SELECTED_PROVIDER_HEADER,
    route_selection_headers,
)
from switchyard.lib.endpoints.upstream_error import handle_chain_exception
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.lib.proxy_context import CTX_ROUTE_SELECTION, ProxyContext
from switchyard.server.switchyard_app import build_switchyard_app
from tests._chain_test_helpers import _backend_payload, _OpenAICompatStub, _sse_body, _stream_chunk

ROUTE_MODEL = "nvidia/switchyard/test-route"
ENDPOINT_ID = "openai/test-model"
UPSTREAM_MODEL = "openai/openai/test-model"


def _selection(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "router_model": ROUTE_MODEL,
        "router_strategy": "latency",
        "router_selected_endpoint": ENDPOINT_ID,
        "router_selected_model": UPSTREAM_MODEL,
        "router_selected_provider": "openai",
        "router_correlation_id": "11111111-2222-3333-4444-555555555555",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Unit: ctx → response-header mapping
# ---------------------------------------------------------------------------


class TestRouteSelectionHeaders:
    def test_empty_without_selection(self) -> None:
        assert route_selection_headers(ProxyContext()) == {}

    def test_maps_selection_to_response_headers(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection()

        assert route_selection_headers(ctx) == {
            ROUTER_MODEL_HEADER: ROUTE_MODEL,
            SELECTED_MODEL_HEADER: UPSTREAM_MODEL,
            SELECTED_PROVIDER_HEADER: "openai",
            ROUTER_CORRELATION_ID_HEADER: "11111111-2222-3333-4444-555555555555",
        }

    def test_skips_absent_fields_instead_of_stamping_placeholders(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection(router_model=None)

        headers = route_selection_headers(ctx)

        assert ROUTER_MODEL_HEADER not in headers
        assert headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL

    def test_ignores_non_mapping_value(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = "bogus"

        assert route_selection_headers(ctx) == {}

    def test_skips_values_unsafe_as_header_material(self) -> None:
        """The client-controlled router_model must be re-validated as a header.

        A CRLF-bearing value would be a response-splitting vector, and a
        non-latin-1 value would crash Starlette response construction after
        the upstream call already succeeded and was billed.
        """
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection(router_model="gpt\r\nx-evil: 1")
        headers = route_selection_headers(ctx)
        assert ROUTER_MODEL_HEADER not in headers
        assert headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL

        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection(router_model="gpt-4中文")
        headers = route_selection_headers(ctx)
        assert ROUTER_MODEL_HEADER not in headers
        assert headers[ROUTER_CORRELATION_ID_HEADER] == (
            "11111111-2222-3333-4444-555555555555"
        )


class TestSerializeChainResultHeaders:
    @staticmethod
    async def _sse_iter(_result: Any) -> Any:
        yield "data: {}\n\n"

    def test_json_response_carries_selection_headers(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection()

        response = serialize_chain_result(
            {"ok": True}, stream=False, sse_iter=self._sse_iter, ctx=ctx
        )

        assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL
        assert (
            response.headers[ROUTER_CORRELATION_ID_HEADER]
            == "11111111-2222-3333-4444-555555555555"
        )

    def test_streaming_response_carries_selection_headers(self) -> None:
        # The backend call completes before the StreamingResponse is built, so
        # the selection is final by the time SSE headers are committed.
        class _EmptyStream:
            def __aiter__(self) -> _EmptyStream:
                return self

            async def __anext__(self) -> str:
                raise StopAsyncIteration

        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection()

        response = serialize_chain_result(
            _EmptyStream(), stream=True, sse_iter=self._sse_iter, ctx=ctx
        )

        assert response.media_type == "text/event-stream"
        assert response.headers[ROUTER_MODEL_HEADER] == ROUTE_MODEL

    def test_selection_free_ctx_no_selection_headers(self) -> None:
        response = serialize_chain_result(
            {"ok": True}, stream=False, sse_iter=self._sse_iter, ctx=ProxyContext()
        )

        assert ROUTER_CORRELATION_ID_HEADER not in response.headers


class TestErrorPathSelectionHeaders:
    def test_failure_after_billed_success_keeps_selection_headers(self) -> None:
        """A post-backend failure must still expose the billed selection.

        The upstream call succeeded (provider spend-log row written with the
        stamped correlation id) before e.g. response translation raised; the
        error response must carry the selection headers or that provider row
        becomes unjoinable.
        """
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection()

        response = handle_chain_exception(
            RuntimeError("response translation failed"),
            ctx,
            inbound="openai",
            log_msg="test",
        )

        assert response.status_code == 500
        assert response.headers[ROUTER_CORRELATION_ID_HEADER] == (
            "11111111-2222-3333-4444-555555555555"
        )
        assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL

    def test_failure_without_selection_has_no_selection_headers(self) -> None:
        response = handle_chain_exception(
            RuntimeError("backend never succeeded"),
            ProxyContext(),
            inbound="openai",
            log_msg="test",
        )

        assert response.status_code == 500
        assert ROUTER_CORRELATION_ID_HEADER not in response.headers


# ---------------------------------------------------------------------------
# End-to-end: HTTP in → wire header out → response headers back
# ---------------------------------------------------------------------------


@fixture
def latency_app() -> Iterator[tuple[FastAPI, _OpenAICompatStub]]:
    """Latency-service app whose single endpoint targets a loopback stub.

    The backend uses the real ``OpenAILLMClient``/OpenAI SDK, so the stub
    records the actual HTTP headers the upstream would receive. Only the
    health poller is stubbed out (no Latency Service in tests).
    """
    with _OpenAICompatStub() as stub:
        config = LatencyServiceBackendConfig(
            latency_service_url="http://latency-service.test:8080",
            route_model=ROUTE_MODEL,
            endpoints=[
                LatencyServiceEndpoint(
                    model=ENDPOINT_ID,
                    upstream_model=UPSTREAM_MODEL,
                    base_url=stub.base_url,
                    api_key="test-key",
                ),
            ],
        )
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            switchyard = ProfileSwitchyard(
                LatencyServiceProfileConfig.from_config(config)
                .build()
                .with_runtime_components(enable_stats=config.enable_stats)
            )
            yield build_switchyard_app(switchyard), stub


def _wire_spend_logs_payload(stub: _OpenAICompatStub) -> dict[str, object]:
    """Parse the spend-logs metadata header the stub received on the wire."""
    headers = stub.requests[-1]["headers"]
    assert isinstance(headers, dict)
    payload = json.loads(str(headers[SPEND_LOGS_METADATA_HEADER]))
    assert isinstance(payload, dict)
    return payload


def test_chat_completion_roundtrips_selection_and_correlation_id(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    app, stub = latency_app
    stub.respond_json(_backend_payload(content="hello", model=UPSTREAM_MODEL))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    payload = _wire_spend_logs_payload(stub)
    assert payload["router_model"] == ROUTE_MODEL
    assert payload["router_strategy"] == "latency"
    assert payload["router_selected_endpoint"] == ENDPOINT_ID
    assert payload["router_selected_model"] == UPSTREAM_MODEL
    assert payload["router_selected_provider"] == "openai"
    assert response.headers[ROUTER_MODEL_HEADER] == ROUTE_MODEL
    assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL
    assert response.headers[SELECTED_PROVIDER_HEADER] == "openai"
    # The join key: the response header must equal the id the provider row got.
    assert response.headers[ROUTER_CORRELATION_ID_HEADER] == payload["router_correlation_id"]


def test_streaming_chat_completion_carries_selection_headers(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    app, stub = latency_app
    stub.respond_sse(_sse_body([_stream_chunk(content="hel"), _stream_chunk(finish="stop")]))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text
    payload = _wire_spend_logs_payload(stub)
    assert response.headers[ROUTER_CORRELATION_ID_HEADER] == payload["router_correlation_id"]
    assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL


def test_anthropic_messages_endpoint_carries_selection_headers(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    """Anthropic inbound rides the same dispatch path and gets the same headers."""
    app, stub = latency_app
    stub.respond_json(_backend_payload(content="hello", model=UPSTREAM_MODEL))

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": ROUTE_MODEL,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    payload = _wire_spend_logs_payload(stub)
    assert response.headers[ROUTER_CORRELATION_ID_HEADER] == payload["router_correlation_id"]
    assert response.headers[SELECTED_PROVIDER_HEADER] == "openai"


def test_responses_endpoint_carries_selection_headers(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    """OpenAI Responses inbound rides the same dispatch path and gets the same headers."""
    app, stub = latency_app
    stub.respond_json(_backend_payload(content="hello", model=UPSTREAM_MODEL))

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": ROUTE_MODEL, "input": "hi"},
        )

    assert response.status_code == 200
    payload = _wire_spend_logs_payload(stub)
    assert response.headers[ROUTER_CORRELATION_ID_HEADER] == payload["router_correlation_id"]
    assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL


def test_upstream_error_response_has_no_selection_headers(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    """No upstream call succeeded — the error response must not claim a selection.

    The failed attempt itself is still stamped on the wire (the provider row
    for the failure keeps its attribution); the client-facing headers are
    reserved for a recorded selection, i.e. a billed upstream success — which
    never happened here. (A failure *after* a billed success does carry them;
    see TestErrorPathSelectionHeaders.)
    """
    app, stub = latency_app
    stub.respond_json({"error": {"message": "bad key"}}, status=401)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 401
    assert ROUTER_CORRELATION_ID_HEADER not in response.headers
    assert SPEND_LOGS_METADATA_HEADER in stub.requests[-1]["headers"]  # type: ignore[operator]


def test_hostile_model_string_never_breaks_the_response(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    """A model string that is not legal header material must not 500 or inject.

    Single-chain profile serving forwards any client model string, so
    router_model is client-controlled; the billed 200 must survive, with the
    unsafe header skipped and the config-derived headers intact.
    """
    app, stub = latency_app

    with TestClient(app) as client:
        for hostile_model in ("gpt-4中文", "gpt\r\nx-evil: 1"):
            stub.respond_json(_backend_payload(content="hello", model=UPSTREAM_MODEL))
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": hostile_model,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            assert response.status_code == 200
            assert ROUTER_MODEL_HEADER not in response.headers
            assert "x-evil" not in response.headers
            assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL
            # The outbound header is JSON (ensure_ascii escapes control and
            # non-ASCII chars), so the wire payload still records the model.
            assert _wire_spend_logs_payload(stub)["router_model"] == hostile_model


def test_client_body_extra_headers_forwarded_not_collided(
    latency_app: tuple[FastAPI, _OpenAICompatStub],
) -> None:
    """A passthrough body carrying an SDK-style extra_headers field keeps working.

    Pre-spend-logs it rode ``**body`` into the SDK's header kwarg; it must
    still reach the wire — and must not be able to spoof the spend-logs
    header, which wins the name conflict.
    """
    app, stub = latency_app
    stub.respond_json(_backend_payload(content="hello", model=UPSTREAM_MODEL))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {
                    "x-client-tag": "42",
                    SPEND_LOGS_METADATA_HEADER: "spoofed",
                },
            },
        )

    assert response.status_code == 200
    headers = stub.requests[-1]["headers"]
    assert isinstance(headers, dict)
    assert headers["x-client-tag"] == "42"
    payload = _wire_spend_logs_payload(stub)
    assert payload["router_selected_endpoint"] == ENDPOINT_ID  # not "spoofed"
    assert response.headers[ROUTER_CORRELATION_ID_HEADER] == payload["router_correlation_id"]
