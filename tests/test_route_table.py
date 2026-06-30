# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RouteTable model-based HTTP dispatch."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.endpoints.anthropic_messages_endpoint import (
    AnthropicMessagesEndpoint,
)
from switchyard.lib.endpoints.openai_chat_endpoint import OpenAIChatEndpoint
from switchyard.lib.endpoints.responses_endpoint import ResponsesEndpoint
from switchyard.lib.proxy_context import CTX_CALLER_API_KEY
from switchyard.lib.route_table import RouteTable
from switchyard.lib.route_table_builders import build_passthrough_table
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app


def _make_chain(name: str = "chain") -> MagicMock:
    chain = MagicMock(spec=Switchyard)
    chain.call = AsyncMock(return_value={"chain": name})
    chain.iter_components.return_value = [MagicMock(name=f"{name}-component")]
    return chain


def _make_app(table: RouteTable) -> TestClient:
    app = FastAPI()
    app.state.switchyard = table
    OpenAIChatEndpoint().register(app)
    AnthropicMessagesEndpoint().register(app)
    ResponsesEndpoint().register(app)
    return TestClient(app, raise_server_exceptions=False)


class TestRouteTable:
    def test_lookup_returns_registered_chain(self) -> None:
        chain = _make_chain("specific")
        table = RouteTable()
        table.register("gpt-4o", chain)

        assert table.lookup_switchyard("gpt-4o") is chain

    def test_lookup_raises_key_error_for_unknown_model(self) -> None:
        table = RouteTable()
        table.register("gpt-4o", _make_chain())

        with pytest.raises(KeyError):
            table.lookup_switchyard("not-registered")

    def test_register_overwrites_existing_key(self) -> None:
        first = _make_chain("first")
        second = _make_chain("second")
        table = RouteTable()
        table.register("m", first)
        table.register("m", second)

        assert table.lookup_switchyard("m") is second

    def test_registered_models_preserves_registration_order(self) -> None:
        table = RouteTable()
        table.register("first", _make_chain("first"))
        table.register("second", _make_chain("second"))

        assert table.registered_models() == ["first", "second"]

    def test_explicit_default_can_differ_from_registration_order(self) -> None:
        table = RouteTable()
        table.register("first", _make_chain("first"))
        table.register("second", _make_chain("second"), default=True)

        assert table.default_model() == "second"

        table.set_default_model("first")
        assert table.default_model() == "first"

        with pytest.raises(KeyError):
            table.set_default_model("missing")

    def test_state_key_matches_switchyard(self) -> None:
        assert RouteTable.state_key == Switchyard.state_key == "switchyard"

    def test_iter_components_deduplicates_shared_instances(self) -> None:
        shared = MagicMock(name="shared")
        unique_a = MagicMock(name="unique-a")
        unique_b = MagicMock(name="unique-b")
        first = MagicMock(spec=Switchyard)
        first.iter_components.return_value = [shared, unique_a]
        second = MagicMock(spec=Switchyard)
        second.iter_components.return_value = [shared, unique_b]
        table = RouteTable()
        table.register("first", first)
        table.register("second", second)

        components = table.iter_components()

        assert components.count(shared) == 1
        assert unique_a in components
        assert unique_b in components
        assert len(components) == 3


def test_build_switchyard_app_accepts_table() -> None:
    table = RouteTable()
    app = build_switchyard_app(table)

    assert app.state.switchyard is table


def test_models_endpoint_lists_registered_models() -> None:
    table = RouteTable()
    table.register(
        "switchyard-default-random-12345678",
        _make_chain("random"),
        metadata={
            "display_name": "Switchyard random routing",
            "description": "Random routes strong and weak.",
            "switchyard": {
                "profile": "random_routing",
                "strong_model": "strong/model",
                "weak_model": "weak/model",
                "strong_probability": 0.5,
            },
        },
    )
    table.register("strong/model", _make_chain("strong"))

    with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
        response = client.get("/v1/models?limit=1000")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["has_more"] is False
    assert body["default_model"] == "switchyard-default-random-12345678"
    assert body["model_pool"] == [
        "switchyard-default-random-12345678",
        "strong/model",
    ]
    assert [item["id"] for item in body["data"]] == [
        "switchyard-default-random-12345678",
        "strong/model",
    ]
    assert body["data"][0]["display_name"] == "Switchyard random routing"
    assert body["data"][0]["description"] == "Random routes strong and weak."
    assert body["data"][0]["switchyard"]["profile"] == "random_routing"
    assert body["data"][0]["capabilities"]["streaming"] is True
    assert body["data"][0]["capabilities"]["supported_inbound_formats"] == [
        "openai-chat-completions",
        "openai-responses",
        "anthropic-messages",
    ]


def test_models_endpoint_uses_table_default_model() -> None:
    table = RouteTable()
    table.register("strong/model", _make_chain("strong"))
    table.register("weak/model", _make_chain("weak"))
    table.register("switchyard-route", _make_chain("random"), default=True)

    with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
        response = client.get("/v1/models?limit=1000")

    assert response.status_code == 200
    body = response.json()
    assert body["first_id"] == "strong/model"
    assert body["default_model"] == "switchyard-route"
    assert body["model_pool"] == [
        "strong/model",
        "weak/model",
        "switchyard-route",
    ]
    assert [item["id"] for item in body["data"]] == [
        "strong/model",
        "weak/model",
        "switchyard-route",
    ]


def test_models_endpoint_lists_discovered_catalog_models() -> None:
    def discover(base_url: str, api_key: str) -> list[str]:
        assert base_url == "https://primary.example/v1"
        assert api_key == "k-primary"
        return ["catalog/a", "catalog/b"]

    table = build_passthrough_table(
        (
            LlmTarget(
                model="configured/model",
                format=BackendFormat.AUTO,
                api_key="k-primary",
                base_url="https://primary.example/v1",
            ),
        ),
        discovery_fn=discover,
    )

    with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["model_pool"] == [
        "configured/model",
        "catalog/a",
        "catalog/b",
    ]
    assert [item["switchyard"]["source"] for item in body["data"]] == [
        "configured",
        "discovered",
        "discovered",
    ]


def test_large_discovered_catalog_registers_metrics_endpoint_once() -> None:
    discovered_models = [f"catalog/model-{index}" for index in range(340)]
    table = build_passthrough_table(
        (
            LlmTarget(
                model="configured/model",
                format=BackendFormat.AUTO,
                api_key="k-primary",
                base_url="https://primary.example/v1",
            ),
        ),
        discovery_fn=lambda _base_url, _api_key: discovered_models,
    )

    app = build_switchyard_app(table)

    # The OTel /metrics scrape endpoint mounts exactly once regardless of how
    # many models the catalog discovers. (/v1/stats was removed in the OTel
    # migration.)
    assert sum(route.path == "/metrics" for route in app.routes) == 1
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 200


def test_models_endpoint_returns_static_models_and_warning_when_discovery_fails() -> None:
    def discover(_base_url: str, _api_key: str) -> list[str]:
        raise RuntimeError("catalog timed out")

    table = build_passthrough_table(
        (
            LlmTarget(
                model="configured/model",
                format=BackendFormat.AUTO,
                api_key="k-primary",
                base_url="https://primary.example/v1",
            ),
        ),
        discovery_fn=discover,
    )

    with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["model_pool"] == ["configured/model"]
    assert body["warnings"] == [
        "Model discovery failed for https://primary.example/v1: catalog timed out"
    ]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "registered", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": "registered", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "registered", "input": "hi"}),
    ],
)
def test_http_dispatch_uses_registered_chain(path: str, body: dict[str, object]) -> None:
    registered = _make_chain("registered")
    table = RouteTable()
    table.register("registered", registered)

    with _make_app(table) as client:
        response = client.post(path, json=body)

    assert response.status_code == 200
    assert response.json() == {"chain": "registered"}
    registered.call.assert_awaited_once()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "registered", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": "registered", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "registered", "input": "hi"}),
    ],
)
def test_http_dispatch_attaches_caller_api_key_to_context(
    path: str,
    body: dict[str, object],
) -> None:
    registered = _make_chain("registered")
    table = RouteTable()
    table.register("registered", registered)

    with _make_app(table) as client:
        response = client.post(
            path,
            json=body,
            headers={"Authorization": "Bearer caller-key"},
        )

    assert response.status_code == 200
    ctx = registered.call.await_args.kwargs["ctx"]
    assert ctx.metadata[CTX_CALLER_API_KEY] == "caller-key"  # pragma: allowlist secret


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "unknown", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": "unknown", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "unknown", "input": "hi"}),
    ],
)
def test_http_dispatch_returns_404_for_unknown_model(
    path: str, body: dict[str, object],
) -> None:
    registered = _make_chain("registered")
    table = RouteTable()
    table.register("registered", registered)

    with _make_app(table) as client:
        response = client.post(path, json=body)

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "No route registered for model unknown",
            "type": "model_not_found",
            "code": "model_not_found",
        }
    }
    registered.call.assert_not_awaited()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "missing", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": "missing", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "missing", "input": "hi"}),
    ],
)
def test_http_dispatch_returns_404_without_default(path: str, body: dict[str, object]) -> None:
    table = RouteTable()

    with _make_app(table) as client:
        response = client.post(path, json=body)

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "No route registered for model missing",
            "type": "model_not_found",
            "code": "model_not_found",
        }
    }
