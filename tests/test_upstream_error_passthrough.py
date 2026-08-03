# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate that upstream HTTP errors keep stable provider details.

Before this fix, a 401 from the upstream LLM (typically a bad API key
or expired credential) became a generic 500 at the client because the
compatibility executor wrapped the Python ``openai.APIStatusError`` in a
``SwitchyardError::Backend(error.to_string())``, which surfaced as a
plain Python ``RuntimeError`` and FastAPI defaulted it to 500.

Python backends stash upstream status/body on ``ProxyContext.metadata``.
Rust backends surface a typed upstream exception with ``status_code`` and
``body`` attributes. The endpoints recover either signal, preserve the
HTTP status plus stable provider fields, and return the normalized
Switchyard error envelope instead of FastAPI's default plain-text 500.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from switchyard.cli.route_bundle import build_route_bundle_table
from switchyard.lib.endpoints.error_envelope import ERROR_SOURCE_HEADER
from switchyard.lib.endpoints.upstream_error import (
    internal_chain_error_response,
    upstream_response_from_ctx,
)
from switchyard.lib.proxy_context import (
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
    ProxyContext,
)
from switchyard.server.switchyard_app import build_switchyard_app
from tests._chain_test_helpers import _OpenAICompatStub

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestUpstreamResponseFromCtx:
    def test_returns_none_when_no_status_recorded(self) -> None:
        ctx = ProxyContext()
        assert upstream_response_from_ctx(ctx) is None

    def test_wraps_string_body_in_error_envelope(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 401
        ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = "Unauthorized"

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 401

    def test_normalizes_dict_body_into_switchyard_envelope(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 429
        ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = {
            "error": {"message": "rate limited", "type": "rate_limit"},
        }

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 429
        body = json.loads(response.body)
        assert body == {
            "error": {
                "message": "rate limited",
                "type": "rate_limit",
                "code": "rate_limit",
            }
        }

    def test_preserves_provider_error_param_when_present(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 400
        ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = {
            "error": {
                "message": "bad input",
                "type": "invalid_request_error",
                "code": "invalid_value",
                "param": "messages.0.role",
            },
        }

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 400
        assert json.loads(response.body) == {
            "error": {
                "message": "bad input",
                "type": "invalid_request_error",
                "code": "invalid_value",
                "param": "messages.0.role",
            }
        }

    def test_synthesizes_envelope_when_body_missing(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 503

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 503
        body = json.loads(response.body)
        assert body == {
            "error": {
                "message": "upstream returned HTTP 503",
                "type": "upstream_error",
                "code": "upstream_error",
            }
        }

    def test_ignores_non_int_status(self) -> None:
        """Defensive: a stray non-int value must not crash the error path."""
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = "401"  # wrong type — ignore

        assert upstream_response_from_ctx(ctx) is None

    def test_internal_chain_error_uses_openai_error_envelope(self) -> None:
        response = internal_chain_error_response(RuntimeError("connection refused"), "openai")

        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        body = json.loads(response.body)
        assert "connection refused" in body["error"]["message"]
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["code"] == "internal_chain_error"

    def test_internal_chain_error_uses_same_envelope_for_anthropic_inbound(self) -> None:
        response = internal_chain_error_response(RuntimeError("connection refused"), "anthropic")

        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        body = json.loads(response.body)
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["code"] == "internal_chain_error"
        assert "connection refused" in body["error"]["message"]

    def test_internal_chain_error_truncates_long_repr(self) -> None:
        long_msg = "x" * 500
        response = internal_chain_error_response(RuntimeError(long_msg), "openai")
        body = json.loads(response.body)
        assert len(body["error"]["message"]) <= 200


# ---------------------------------------------------------------------------
# Anthropic and Responses endpoints share the same helper
# ---------------------------------------------------------------------------


def test_rust_openai_route_upstream_401_returns_structured_openai_error() -> None:
    """Rust OpenAI-native backend errors must not fall through to FastAPI's 500."""
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
            status=401,
        )
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "bad-key",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "bad-key": {
                    "type": "passthrough",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "bad-key",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        },
    }


def test_rust_openai_route_upstream_401_returns_same_error_shape_for_anthropic_inbound() -> None:
    """Anthropic inbound clients should receive the same HTTP error envelope."""
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
            status=401,
        )
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "bad-key",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "bad-key": {
                    "type": "passthrough",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "bad-key",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        },
    }


def test_rust_openai_route_upstream_401_returns_same_error_shape_for_responses_inbound() -> None:
    """Responses inbound clients should receive the same HTTP error envelope."""
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
            status=401,
        )
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "bad-key",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "bad-key": {
                    "type": "passthrough",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "bad-key",
                    "input": "ping",
                },
            )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        },
    }


# ---------------------------------------------------------------------------
# Failure-source headers on the wire
# ---------------------------------------------------------------------------
# The header tests above the endpoint layer call helpers directly; these prove
# the annotation actually survives the real FastAPI handlers + middleware to
# the HTTP response a client receives.


def test_model_not_found_404_header_labels_switchyard_on_the_wire() -> None:
    """RouteTable dispatch 404s carry the switchyard source header."""
    with _OpenAICompatStub() as upstream:
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "k",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "registered": {
                    "type": "passthrough",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "no-such-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert response.headers[ERROR_SOURCE_HEADER] == "switchyard"
