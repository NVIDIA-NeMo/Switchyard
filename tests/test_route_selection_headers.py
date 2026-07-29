# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route-selection spend-attribution response headers.

For tokenomics reporting, a front proxy (e.g. LiteLLM) must be able to tie
its provider spend-log rows back to the Switchyard logical route that
selected them: the Switchyard response returns ``x-switchyard-*`` headers with
the successful attempt's selection plus a per-request correlation id, so the
parent spend-log row can be enriched to match the provider row.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse

from switchyard.lib.endpoints.dispatch import serialize_chain_result
from switchyard.lib.endpoints.route_selection import (
    ROUTER_CORRELATION_ID_HEADER,
    ROUTER_MODEL_HEADER,
    SELECTED_MODEL_HEADER,
    SELECTED_PROVIDER_HEADER,
    route_selection_headers,
)
from switchyard.lib.endpoints.upstream_error import handle_chain_exception
from switchyard.lib.proxy_context import CTX_ROUTE_SELECTION, ProxyContext

ROUTE_MODEL = "nvidia/switchyard/test-route"
ENDPOINT_ID = "openai/test-model"
UPSTREAM_MODEL = "openai/openai/test-model"


def _selection(**overrides: object) -> dict[str, object]:
    """Recorded route selection, with *overrides* applied."""
    payload: dict[str, object] = {
        "router_model": ROUTE_MODEL,
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
    """Mapping of the ctx selection record to ``x-switchyard-*`` headers."""

    def test_empty_without_selection(self) -> None:
        """A ctx without a recorded selection yields no headers."""
        assert route_selection_headers(ProxyContext()) == {}

    def test_maps_selection_to_response_headers(self) -> None:
        """Every exposed selection field maps to its response header."""
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection()

        assert route_selection_headers(ctx) == {
            ROUTER_MODEL_HEADER: ROUTE_MODEL,
            SELECTED_MODEL_HEADER: UPSTREAM_MODEL,
            SELECTED_PROVIDER_HEADER: "openai",
            ROUTER_CORRELATION_ID_HEADER: "11111111-2222-3333-4444-555555555555",
        }

    def test_skips_absent_fields_instead_of_stamping_placeholders(self) -> None:
        """An absent field is skipped — never stamped as a placeholder value."""
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection(router_model=None)

        headers = route_selection_headers(ctx)

        assert ROUTER_MODEL_HEADER not in headers
        assert headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL

    def test_ignores_non_mapping_value(self) -> None:
        """A malformed (non-mapping) ctx record yields no headers."""
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
    """``serialize_chain_result`` stamps the selection on every branch."""

    @staticmethod
    async def _sse_iter(_result: Any) -> Any:
        """Minimal SSE iterator satisfying the serializer signature."""
        yield "data: {}\n\n"

    def test_json_response_carries_selection_headers(self) -> None:
        """A JSON-serialized result carries the selection headers."""
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
        """The SSE branch stamps the headers too.

        The backend call completes before the StreamingResponse is built, so
        the selection is final by the time SSE headers are committed.
        """
        class _EmptyStream:
            """Async iterator yielding nothing, standing in for a backend stream."""

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
        """A selection-free ctx adds no ``x-switchyard-*`` headers."""
        response = serialize_chain_result(
            {"ok": True}, stream=False, sse_iter=self._sse_iter, ctx=ProxyContext()
        )

        assert ROUTER_CORRELATION_ID_HEADER not in response.headers

    def test_prebuilt_response_passes_through_with_selection_merged(self) -> None:
        """A result that is already a ``Response`` gains the selection headers.

        No current chain path yields a pre-built Response after a billed
        upstream success, but the serializer contract — a recorded selection
        is never dropped — must hold on that branch too, with the response's
        own status, body, and headers preserved.
        """
        ctx = ProxyContext()
        ctx.metadata[CTX_ROUTE_SELECTION] = _selection()
        prebuilt = JSONResponse(
            content={"ok": True}, status_code=404, headers={"x-existing": "1"}
        )

        response = serialize_chain_result(
            prebuilt, stream=False, sse_iter=self._sse_iter, ctx=ctx
        )

        assert response is prebuilt
        assert response.status_code == 404
        assert response.headers["x-existing"] == "1"
        assert json.loads(response.body) == {"ok": True}
        assert response.headers[SELECTED_MODEL_HEADER] == UPSTREAM_MODEL
        assert response.headers[ROUTER_CORRELATION_ID_HEADER] == (
            "11111111-2222-3333-4444-555555555555"
        )


class TestErrorPathSelectionHeaders:
    """``handle_chain_exception`` keeps a billed selection on error responses."""

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
        """No billed success → the error response claims no selection."""
        response = handle_chain_exception(
            RuntimeError("backend never succeeded"),
            ProxyContext(),
            inbound="openai",
            log_msg="test",
        )

        assert response.status_code == 500
        assert ROUTER_CORRELATION_ID_HEADER not in response.headers
