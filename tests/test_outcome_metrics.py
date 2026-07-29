# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit + integration tests for the outcome counters."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchyard.lib.endpoints import outcome_metrics
from switchyard.lib.endpoints.upstream_error import (
    record_upstream_attempt_failure,
    record_upstream_attempt_success,
)
from switchyard.lib.proxy_context import (
    CTX_UPSTREAM_ATTEMPTS_RECORDED,
    CTX_UPSTREAM_HTTP_STATUS,
    ProxyContext,
)
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    SwitchyardUpstreamError,
)
from switchyard_rust.translation import TranslationEngine


@pytest.fixture(autouse=True)
def _reset_counters():
    outcome_metrics._reset_for_tests()
    yield
    outcome_metrics._reset_for_tests()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassify:
    @pytest.mark.parametrize("code", [200, 201, 204, 299])
    def test_2xx_is_success(self, code: int) -> None:
        assert outcome_metrics.classify(code) == "success"

    @pytest.mark.parametrize("code", [429, 500, 504])
    def test_spec_codes_are_retryable_error(self, code: int) -> None:
        """Exactly the codes the success criterion lists count as retryable."""
        assert outcome_metrics.classify(code) == "retryable_error"

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 502, 503])
    def test_other_codes_are_other_error(self, code: int) -> None:
        """Bad-payload / bad-key / non-spec 5xx fall outside the criterion."""
        assert outcome_metrics.classify(code) == "other_error"


# ---------------------------------------------------------------------------
# Code label (the per-status dimension for the distribution dashboard)
# ---------------------------------------------------------------------------


class TestCodeLabel:
    def test_none_is_the_no_status_sentinel(self) -> None:
        """Non-HTTP failures have no status line → the ``none`` sentinel."""
        assert outcome_metrics.code_label(None) == outcome_metrics.NO_STATUS_CODE
        assert outcome_metrics.code_label(None) == "none"

    @pytest.mark.parametrize("code", sorted(outcome_metrics.KNOWN_STATUS_CODES))
    def test_known_codes_emitted_verbatim(self, code: int) -> None:
        assert outcome_metrics.code_label(code) == str(code)

    @pytest.mark.parametrize(
        ("code", "expected"),
        [(418, "4xx"), (451, "4xx"), (599, "5xx"), (100, "1xx"), (302, "3xx")],
    )
    def test_unknown_codes_clamp_to_class(self, code: int, expected: str) -> None:
        """An oddball upstream code collapses to its class, bounding cardinality."""
        assert outcome_metrics.code_label(code) == expected

    @pytest.mark.parametrize("code", [0, 99, 600, 700])
    def test_out_of_range_codes_clamp_to_other(self, code: int) -> None:
        assert outcome_metrics.code_label(code) == "other"


# ---------------------------------------------------------------------------
# Render shape
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_initial_state_is_all_zero(self) -> None:
        out = "\n".join(outcome_metrics.render_lines())
        assert 'switchyard_client_responses_total{outcome="success"} 0' in out
        assert 'switchyard_client_responses_total{outcome="retryable_error"} 0' in out
        assert 'switchyard_client_responses_total{outcome="other_error"} 0' in out
        # Upstream attempts carry a code label; the canonical codes are
        # seeded at 0 so their series exist before the first matching attempt.
        assert 'switchyard_upstream_attempts_total{outcome="success",code="200"} 0' in out
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="429"} 0'
            in out
        )
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="none"} 0'
            in out
        )
        assert "switchyard_router_retry_recovered_total 0" in out

    def test_render_includes_help_and_type_lines(self) -> None:
        """Prometheus exposition needs HELP+TYPE before each metric family."""
        out = "\n".join(outcome_metrics.render_lines())
        for metric in (
            "switchyard_client_responses_total",
            "switchyard_upstream_attempts_total",
            "switchyard_router_retry_recovered_total",
        ):
            assert f"# HELP {metric}" in out
            assert f"# TYPE {metric}" in out

    def test_render_reflects_recorded_state(self) -> None:
        outcome_metrics.record_client_response(200)
        outcome_metrics.record_client_response(429)
        outcome_metrics.record_client_response(401)
        outcome_metrics.record_upstream_attempt(500)
        outcome_metrics.record_upstream_attempt(None)
        outcome_metrics.record_retry_recovered()

        out = "\n".join(outcome_metrics.render_lines())
        assert 'switchyard_client_responses_total{outcome="success"} 1' in out
        assert 'switchyard_client_responses_total{outcome="retryable_error"} 1' in out
        assert 'switchyard_client_responses_total{outcome="other_error"} 1' in out
        # The two retryable attempts split across their codes — the whole
        # point of the new label — rather than collapsing into one bucket.
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="500"} 1'
            in out
        )
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="none"} 1'
            in out
        )
        assert "switchyard_router_retry_recovered_total 1" in out

    def test_distinct_codes_get_distinct_series(self) -> None:
        """429 / 500 / 504 must be separately countable, not merged."""
        for _ in range(3):
            outcome_metrics.record_upstream_attempt(429)
        outcome_metrics.record_upstream_attempt(500)
        outcome_metrics.record_upstream_attempt(504)
        # An unknown 4xx clamps to its class rather than spawning a new series.
        outcome_metrics.record_upstream_attempt(418)

        out = "\n".join(outcome_metrics.render_lines())
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="429"} 3'
            in out
        )
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="500"} 1'
            in out
        )
        assert (
            'switchyard_upstream_attempts_total{outcome="retryable_error",code="504"} 1'
            in out
        )
        assert (
            'switchyard_upstream_attempts_total{outcome="other_error",code="4xx"} 1'
            in out
        )


# ---------------------------------------------------------------------------
# Endpoint-layer fallback — wires the upstream-attempt counter for backends
# (Rust native / passthrough / multi) that issue one attempt and can't reach
# the Python-only outcome_metrics themselves.
# ---------------------------------------------------------------------------


def _upstream_count(out: str, outcome: str, code: str) -> str:
    return f'switchyard_upstream_attempts_total{{outcome="{outcome}",code="{code}"}}'


class TestEndpointUpstreamAttemptFallback:
    def test_success_records_one_200(self) -> None:
        record_upstream_attempt_success(ProxyContext())
        out = "\n".join(outcome_metrics.render_lines())
        assert f"{_upstream_count(out, 'success', '200')} 1" in out

    def test_rust_upstream_http_error_records_its_status(self) -> None:
        """A Rust backend's typed ``SwitchyardUpstreamError.status_code`` is used."""
        exc = SwitchyardUpstreamError("boom")
        exc.status_code = 500
        record_upstream_attempt_failure(ProxyContext(), exc)
        out = "\n".join(outcome_metrics.render_lines())
        assert f"{_upstream_count(out, 'retryable_error', '500')} 1" in out

    def test_python_backend_ctx_status_takes_priority(self) -> None:
        """A Python backend's stashed ctx status is recorded even without a typed exc."""
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 401
        record_upstream_attempt_failure(ctx, RuntimeError("opaque"))
        out = "\n".join(outcome_metrics.render_lines())
        assert f"{_upstream_count(out, 'other_error', '401')} 1" in out

    def test_status_less_upstream_error_is_retryable_none(self) -> None:
        """An upstream failure with no HTTP status (network) maps to code=none."""
        record_upstream_attempt_failure(ProxyContext(), SwitchyardUpstreamError("conn reset"))
        out = "\n".join(outcome_metrics.render_lines())
        assert f"{_upstream_count(out, 'retryable_error', 'none')} 1" in out

    def test_internal_error_is_not_an_upstream_attempt(self) -> None:
        """A non-upstream chain failure (e.g. translation/processor) records nothing."""
        record_upstream_attempt_failure(ProxyContext(), ValueError("internal bug"))
        out = "\n".join(outcome_metrics.render_lines())
        # Every seeded series stays at 0 — no attempt was attributed.
        assert f"{_upstream_count(out, 'success', '200')} 0" in out
        assert f"{_upstream_count(out, 'retryable_error', 'none')} 0" in out

    def test_dedup_flag_suppresses_fallback(self) -> None:
        """A backend that records its own attempts opts the endpoint out."""
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_ATTEMPTS_RECORDED] = True
        record_upstream_attempt_success(ctx)
        record_upstream_attempt_failure(ctx, SwitchyardUpstreamError("boom"))
        out = "\n".join(outcome_metrics.render_lines())
        assert f"{_upstream_count(out, 'success', '200')} 0" in out
        assert f"{_upstream_count(out, 'retryable_error', 'none')} 0" in out


# ---------------------------------------------------------------------------
# Client-response middleware — only the /v1/* LLM routes feed the client
# outcome counter; the operational routes (/metrics, /health, /v1/models) do
# not, so a scraper polling them cannot inflate the success count.
# ---------------------------------------------------------------------------


class _CannedBackend(LLMBackend):
    """Passthrough-style backend: returns one canned OpenAI completion per call."""

    def supported_request_types(self) -> list[ChatRequestType]:
        return [
            ChatRequestType.OPENAI_CHAT,
            ChatRequestType.OPENAI_RESPONSES,
            ChatRequestType.ANTHROPIC,
        ]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        return ChatResponse.openai_completion(
            ChatCompletion(
                id="chatcmpl-test",
                object="chat.completion",
                created=1700000000,
                model="mock-model",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="hi"),
                        finish_reason="stop",
                    )
                ],
                usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )
        )


def _canned_app() -> object:
    return build_switchyard_app(
        Switchyard(backend=_CannedBackend(), translator=TranslationEngine())
    )


class TestClientResponseMiddleware:
    def test_successful_chat_completion_counts_as_success(self) -> None:
        """A served /v1/chat/completions success increments the client outcome counter."""
        with TestClient(_canned_app(), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200

        metrics = "\n".join(outcome_metrics.render_lines())
        assert 'switchyard_client_responses_total{outcome="success"} 1' in metrics
        assert 'switchyard_client_responses_total{outcome="retryable_error"} 0' in metrics

    def test_operational_routes_are_not_counted_as_client_responses(self) -> None:
        """Polling /metrics, /health, /v1/models must not feed the client counter —
        otherwise a scraper on a fixed cadence would inflate the success count."""
        with TestClient(_canned_app(), raise_server_exceptions=False) as client:
            for _ in range(5):
                client.get("/metrics")
                client.get("/health")
                client.get("/v1/models")

        metrics = "\n".join(outcome_metrics.render_lines())
        assert 'switchyard_client_responses_total{outcome="success"} 0' in metrics
