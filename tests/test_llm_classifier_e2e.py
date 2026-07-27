# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production offline e2e test for the LLM classifier routing chain.

Exercises the full classifier-driven deterministic routing path with
the real production code paths wired together:

    inbound HTTP request
       -> Switchyard chain
           LLMClassifierRequestProcessor    (real OpenAILLMClient -> openai SDK -> httpx)
           SignalTierSelectorRequestProcessor
       -> DeterministicRoutingLLMBackend
           -> per-tier Rust OpenAI backend           (real reqwest -> loopback HTTP)
       -> TranslationEngine
       -> outbound HTTP response

Classifier HTTPS calls are intercepted by ``respx``.  Backend tier
calls go through local loopback HTTP servers because Rust reqwest
bypasses ``respx``.  The test stays offline while still exercising the
real classifier client, tier mapping, and per-tier backend dispatch.

"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.llm_classifier import (
    LLMClassifierConfig,
    LLMClassifierRequestProcessor,
    SignalTierSelectorConfig,
    SignalTierSelectorRequestProcessor,
)
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.translation import TranslationEngine
from tests._chain_test_helpers import (
    _CLASSIFIER_BASE,
    _CLASSIFIER_MODEL,
    _CLASSIFIER_URL,
    _COMPLEX_MODEL,
    _SIMPLE_MODEL,
    _backend_payload,
    _classifier_payload,
    _ClassifierHarness,
    _last_body,
    _OpenAICompatStub,
    _signals_json,
    _sse_body,
    _stream_chunk,
)

# ---------------------------------------------------------------------------
# Fixture — assembles the classifier-driven chain by hand.
# ---------------------------------------------------------------------------


def _build_switchyard(*, simple_base_url: str, complex_base_url: str) -> Switchyard:
    """Hand-assemble the classifier routing chain.

    There's no recipe for this yet, so we wire it the same way a
    downstream integrator would: classifier -> tier selector ->
    deterministic backend with one Rust OpenAI backend per tier.
    """
    classifier = LLMClassifierRequestProcessor(
        LLMClassifierConfig(
            model=_CLASSIFIER_MODEL,
            api_key="classifier-key-not-validated",
            base_url=_CLASSIFIER_BASE,
        ),
    )
    tier_selector = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(default_tier="simple"),
    )
    backend = DeterministicRoutingLLMBackend.from_tiers(
        tiers={
            "simple": LlmTarget(
                model=_SIMPLE_MODEL,
                format=BackendFormat.OPENAI,
                api_key="simple-key",
                base_url=simple_base_url,
            ),
            "complex": LlmTarget(
                model=_COMPLEX_MODEL,
                format=BackendFormat.OPENAI,
                api_key="complex-key",
                base_url=complex_base_url,
            ),
        },
        default_tier="simple",
    )
    return Switchyard(
        request_processors=[classifier, tier_selector],
        backend=backend,
        translator=TranslationEngine(),
    )


@pytest.fixture
async def classifier_harness() -> AsyncIterator[_ClassifierHarness]:
    """ASGI-transport client wired through the full classifier chain."""
    with _OpenAICompatStub() as simple, _OpenAICompatStub() as complex:
        app = build_switchyard_app(
            _build_switchyard(
                simple_base_url=simple.base_url,
                complex_base_url=complex.base_url,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            yield _ClassifierHarness(client=client, simple=simple, complex=complex)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMClassifierE2E:
    """Production offline e2e — classifier picks a tier, backend serves it."""

    @respx.mock
    async def test_classifier_picks_complex_tier(
        self, classifier_harness: _ClassifierHarness,
    ) -> None:
        """High-confidence ``complex`` signals route to the complex backend."""
        classifier_route = respx.post(_CLASSIFIER_URL).mock(
            return_value=httpx.Response(
                200, json=_classifier_payload(_signals_json()),
            )
        )
        classifier_harness.complex.respond_json(
            _backend_payload(content="complex-reply", model=_COMPLEX_MODEL)
        )
        classifier_harness.simple.respond_json(
            _backend_payload(content="simple-reply", model=_SIMPLE_MODEL)
        )

        resp = await classifier_harness.client.post(
            "/v1/chat/completions",
            json={
                "model": "client-requested-model",
                "messages": [{"role": "user", "content": "debug this traceback please"}],
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "complex-reply"

        # The classifier ran, and the routing reached the complex tier
        # (not the simple one).
        assert classifier_route.called
        assert classifier_harness.complex.called
        assert not classifier_harness.simple.called

        # The classifier saw the inbound request body in its user turn.
        classifier_req = json.loads(classifier_route.calls.last.request.content)
        messages = classifier_req["messages"]
        assert messages[0]["role"] == "system"
        assert "routing classifier" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "debug this traceback" in messages[1]["content"]
        # Classifier defaults to strict Structured Outputs: response_format
        # carries the pydantic schema with strict: true.
        rf = classifier_req.get("response_format")
        assert rf is not None and rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "RouteSignals"
        assert rf["json_schema"]["strict"] is True
        sent_schema = rf["json_schema"]["schema"]
        assert sent_schema["type"] == "object"
        assert sent_schema["additionalProperties"] is False
        # Strict mode requires every property in `required`, even ones with defaults.
        assert set(sent_schema["required"]) == set(sent_schema["properties"].keys())

        # The downstream backend got the tier-specific model name, not
        # the client's requested model.
        backend_req = _last_body(classifier_harness.complex)
        assert backend_req["model"] == _COMPLEX_MODEL

    @respx.mock
    async def test_classifier_picks_simple_tier(
        self, classifier_harness: _ClassifierHarness,
    ) -> None:
        """``recommended_tier: simple`` routes to the simple backend."""
        respx.post(_CLASSIFIER_URL).mock(
            return_value=httpx.Response(
                200,
                json=_classifier_payload(
                    _signals_json(
                        task_type="chat",
                        complexity="simple",
                        reasoning_depth="none",
                        precision_requirement="low",
                        recommended_tier="simple",
                        confidence=0.95,
                        reason_code="simple_qa",
                    ),
                ),
            )
        )
        classifier_harness.complex.respond_json(
            _backend_payload(content="complex-reply", model=_COMPLEX_MODEL)
        )
        classifier_harness.simple.respond_json(
            _backend_payload(content="simple-reply", model=_SIMPLE_MODEL)
        )

        resp = await classifier_harness.client.post(
            "/v1/chat/completions",
            json={
                "model": "client-requested-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "simple-reply"
        assert classifier_harness.simple.called
        assert not classifier_harness.complex.called

    @respx.mock
    async def test_classifier_abstain_falls_back_to_default_tier(
        self, classifier_harness: _ClassifierHarness,
    ) -> None:
        """``abstain: true`` bypasses ``recommended_tier`` and uses the default."""
        respx.post(_CLASSIFIER_URL).mock(
            return_value=httpx.Response(
                200,
                json=_classifier_payload(
                    _signals_json(
                        # Even though recommended_tier is complex, abstain=True
                        # should force the tier selector to its default ("simple").
                        recommended_tier="complex",
                        abstain=True,
                        confidence=0.1,
                        reason_code="ambiguous",
                    ),
                ),
            )
        )
        classifier_harness.complex.respond_json(
            _backend_payload(content="complex-reply", model=_COMPLEX_MODEL)
        )
        classifier_harness.simple.respond_json(
            _backend_payload(content="simple-reply", model=_SIMPLE_MODEL)
        )

        resp = await classifier_harness.client.post(
            "/v1/chat/completions",
            json={
                "model": "client-requested-model",
                "messages": [{"role": "user", "content": "??"}],
            },
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "simple-reply"
        assert classifier_harness.simple.called
        assert not classifier_harness.complex.called

    @respx.mock
    async def test_classifier_failure_fails_open_to_default_tier(
        self, classifier_harness: _ClassifierHarness,
    ) -> None:
        """A classifier 500 stamps abstain signals and routes to the default tier.

        ``fail_open=True`` is the configured default for
        :class:`LLMClassifierConfig`, so a broken classifier upstream
        must not break inference — it falls through to the default tier
        with abstain semantics.
        """
        classifier_route = respx.post(_CLASSIFIER_URL).mock(
            return_value=httpx.Response(500, json={"error": "boom"}),
        )
        classifier_harness.complex.respond_json(
            _backend_payload(content="complex-reply", model=_COMPLEX_MODEL)
        )
        classifier_harness.simple.respond_json(
            _backend_payload(content="simple-reply", model=_SIMPLE_MODEL)
        )

        resp = await classifier_harness.client.post(
            "/v1/chat/completions",
            json={
                "model": "client-requested-model",
                "messages": [{"role": "user", "content": "anything"}],
            },
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["message"]["content"] == "simple-reply"
        # Classifier was attempted (and failed) at least once.
        assert classifier_route.called
        assert classifier_harness.simple.called
        assert not classifier_harness.complex.called

    @respx.mock
    async def test_classifier_picks_complex_tier_streaming(
        self, classifier_harness: _ClassifierHarness,
    ) -> None:
        """Streaming path: classifier -> complex tier -> SSE chunks back."""
        respx.post(_CLASSIFIER_URL).mock(
            return_value=httpx.Response(
                200, json=_classifier_payload(_signals_json()),
            )
        )
        chunks = [
            _stream_chunk(content="foo"),
            _stream_chunk(content="bar"),
            _stream_chunk(finish="stop"),
        ]
        classifier_harness.complex.respond_sse(_sse_body(chunks))
        classifier_harness.simple.respond_json(
            _backend_payload(content="should-not-fire", model=_SIMPLE_MODEL)
        )

        resp = await classifier_harness.client.post(
            "/v1/chat/completions",
            json={
                "model": "client-requested-model",
                "messages": [{"role": "user", "content": "debug stream"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200, resp.text
        assert "text/event-stream" in resp.headers["content-type"]
        assert classifier_harness.complex.called
        assert not classifier_harness.simple.called

        data_frames = [
            line[6:]
            for line in resp.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert data_frames, "expected at least one outbound SSE data frame"
        decoded = [json.loads(line) for line in data_frames]
        joined = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in decoded
            if c.get("choices")
        )
        assert joined == "foobar"
