# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-backend e2e tests for the LLM classifier routing chain.

Mirrors :mod:`tests.test_llm_classifier_e2e` (which mocks all HTTPS via
``respx``), but with no mocking — every call hits the selected live backend::

    inbound ChatRequest
       -> LLMClassifierRequestProcessor    -> real classifier LLM call
       -> SignalTierSelectorRequestProcessor
       -> DeterministicRoutingLLMBackend
           -> per-tier LlmClient                    -> real tier LLM call
       -> ChatResponse

Sibling pattern to :mod:`tests.e2e.test_random_routing_llm_backend`:
in-process — drive the processors + backend manually rather than via
``Switchyard.call()`` so we get back the typed
:class:`ChatResponse` (the translator would unwrap it to the
SDK shape).  No subprocess and no HTTP transport — overkill for
verifying the experimental chain against a live LLM.

Because the classifier's output is non-deterministic, tests do not
assert *which* tier was picked.  They assert:

* the classifier ran (RouteSignals stamped on context)
* a known tier was picked and a real backend produced a response
* the response is well-formed (non-empty content or reasoning)
* tier-specific model rewriting happened on the request

Prerequisites:
    - ``OPENROUTER_API_KEY`` or ``NVIDIA_API_KEY`` env var (skips otherwise).

Run with::

    OPENROUTER_API_KEY=sk-or-... uv run pytest tests/e2e/test_llm_classifier_e2e.py -v
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.chat_response.base import ChatResponse
from switchyard.lib.processors.llm_classifier import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    CTX_DETERMINISTIC_TIER_DECISION,
    LLMClassifierConfig,
    LLMClassifierRequestProcessor,
    RouteSignals,
    SignalTierSelectorConfig,
    SignalTierSelectorRequestProcessor,
    TierSelectionDecision,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

from .conftest import get_nvidia_config

# Hard timeout — if the live backend hangs we want a clean fail.
# Two real LLM calls per test (classifier + tier backend); 120s budget.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]

logger = logging.getLogger("e2e.llm_classifier")

HTTP_TIMEOUT_S = 60.0

_nvidia = get_nvidia_config()
_skip_reason = "OPENROUTER_API_KEY or NVIDIA_API_KEY not set"

BACKEND_BASE_URL = _nvidia["base_url"]

# Same model for the classifier and every tier — this test validates
# the end-to-end wiring, not routing intelligence.  Using one model
# keeps the test fast and lets us pin a model known to support
# ``response_format={"type": "json_object"}`` on the selected backend.
#
# All four ``RouteTier`` labels are registered as tiers so that
# whichever ``recommended_tier`` the live classifier picks is always
# a valid backend label — without this, e.g. a ``reasoning`` pick
# would silently fall back to the default tier and we'd lose the
# real-world coverage of the dispatch path.
CLASSIFIER_MODEL = _nvidia["model"]
TIER_MODEL = _nvidia["model"]
TIER_LABELS = ("simple", "medium", "complex", "reasoning")


# ---------------------------------------------------------------------------
# Helpers — build the chain by hand.  There's no recipe yet for the
# experimental classifier routing chain, so tests assemble the four
# roles directly (mirroring how a downstream integrator would).
# ---------------------------------------------------------------------------


def _build_chain(
    *,
    classifier_base_url: str = BACKEND_BASE_URL,
    classifier_api_key: str | None = None,
    classifier_timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[list[Any], DeterministicRoutingLLMBackend]:
    """Build (processors, backend) for the classifier routing chain."""
    classifier = LLMClassifierRequestProcessor(
        LLMClassifierConfig(
            model=CLASSIFIER_MODEL,
            api_key=classifier_api_key or _nvidia["api_key"],
            base_url=classifier_base_url,
            timeout_s=classifier_timeout_s,
        ),
    )
    tier_selector = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(default_tier="simple"),
    )
    backend = DeterministicRoutingLLMBackend.from_tiers(
        tiers={
            label: LlmTarget(
                model=TIER_MODEL,
                format=BackendFormat.OPENAI,
                api_key=_nvidia["api_key"],
                base_url=BACKEND_BASE_URL,
                timeout=HTTP_TIMEOUT_S,
            )
            for label in TIER_LABELS
        },
        default_tier="simple",
    )
    return [classifier, tier_selector], backend


async def _call_through_chain(
    processors: list[Any],
    backend: DeterministicRoutingLLMBackend,
    request: ChatRequest,
) -> tuple[ProxyContext, ChatResponse]:
    """Run a request through the processors and the backend in-process.

    Skips terminal translation on purpose: returning the typed
    :class:`ChatResponse` makes assertions cleaner than
    re-parsing the SDK-shape ``openai.types.chat.ChatCompletion``.
    """
    ctx = ProxyContext()
    for proc in processors:
        request = await proc.process(ctx, request)
    response = await backend.call(ctx, request)
    return ctx, response


def _chat_request(*, user_message: str, **body_overrides: Any) -> ChatRequest:
    body: dict = {
        "model": "placeholder-rewritten-by-router",
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 2048,
    }
    body.update(body_overrides)
    return ChatRequest.openai_chat(body)


def _assert_chain_metadata(ctx: ProxyContext) -> None:
    """Common post-call assertions on what the chain stamped onto ctx."""
    signals = ctx.metadata.get(CTX_DETERMINISTIC_ROUTE_SIGNALS)
    assert isinstance(signals, RouteSignals), (
        f"classifier did not stamp RouteSignals; metadata keys: "
        f"{list(ctx.metadata.keys())}"
    )
    # Pydantic enforces ``0 <= confidence <= 1`` at parse time, so a
    # parsed RouteSignals already satisfies the range — but we
    # re-assert as a guard against future schema drift.
    assert 0.0 <= signals.confidence <= 1.0

    decision = ctx.metadata.get(CTX_DETERMINISTIC_TIER_DECISION)
    assert isinstance(decision, TierSelectionDecision), (
        f"tier selector did not stamp TierSelectionDecision; "
        f"metadata keys: {list(ctx.metadata.keys())}"
    )

    tier = ctx.metadata.get(CTX_DETERMINISTIC_ROUTING_TIER)
    assert tier in set(TIER_LABELS), f"unexpected tier picked: {tier!r}"
    assert decision.tier == tier

    actual_model = ctx.selected_model
    assert actual_model == TIER_MODEL, f"unexpected actual model: {actual_model!r}"


def _assistant_text(response: ChatResponse) -> tuple[str, str]:
    choice = response.body["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    return content, reasoning


def _finish_reason(response: ChatResponse) -> object:
    return response.body["choices"][0].get("finish_reason")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestLLMClassifierLiveE2E:
    """Real classifier + real tier backend via the selected live backend."""

    async def test_simple_ask_routes_and_returns_content(self) -> None:
        """A trivial ask round-trips: classifier runs, a tier serves it.

        Does not assert *which* tier is picked — the classifier is a
        real LLM whose output varies.  Asserts the wiring: signals
        stamped, tier decision recorded, response well-formed.
        """
        processors, backend = _build_chain()
        request = _chat_request(user_message="hi")

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)
        _assert_chain_metadata(ctx)

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        signals = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
        tier = ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]
        actual_model = ctx.selected_model
        logger.info(
            "  [classifier-e2e][simple] tier=%s model=%s "
            "task=%s recommended=%s confidence=%.2f abstain=%s "
            "content[:80]=%r",
            tier, actual_model,
            signals.task_type.value, signals.recommended_tier.value,
            signals.confidence, signals.abstain,
            content[:80],
        )

    async def test_complex_ask_routes_and_returns_content(self) -> None:
        """A multi-step debugging ask round-trips through the chain.

        Different prompt from the simple test, exercises the classifier
        on a harder request.  Same well-formedness assertions: we don't
        require any specific tier choice, only that the chain completes
        cleanly and the picked tier produces text.
        """
        processors, backend = _build_chain()
        request = _chat_request(
            user_message=(
                "I'm seeing a Python traceback ending in "
                "`KeyError: 'session_id'` from a Flask endpoint. "
                "Walk me through how you'd narrow down whether the "
                "key is missing because of a race condition with "
                "session writes or a misconfigured middleware order. "
                "Then propose one focused diagnostic to confirm it."
            ),
        )

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)
        _assert_chain_metadata(ctx)

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        signals = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
        tier = ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]
        logger.info(
            "  [classifier-e2e][complex] tier=%s "
            "task=%s recommended=%s confidence=%.2f reason=%s "
            "content[:80]=%r",
            tier,
            signals.task_type.value, signals.recommended_tier.value,
            signals.confidence, signals.reason_code.value,
            content[:80],
        )

    async def test_classifier_failure_fails_open_to_default_tier(self) -> None:
        """A broken classifier upstream must not break inference.

        Points the classifier at an unresolvable host while keeping
        both tier backends live.  ``fail_open=True`` is the configured
        default, so the chain should stamp abstain signals, fall back
        to the default tier (``simple`` here), and still return real
        content from the live tier backend.
        """
        # Short classifier timeout so we don't burn the whole
        # pytest-timeout budget waiting for the bogus URL to fail.
        processors, backend = _build_chain(
            classifier_base_url="https://classifier-does-not-exist.invalid/v1",
            classifier_api_key="not-used",
            classifier_timeout_s=5.0,
        )
        request = _chat_request(user_message="say hi briefly")

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)

        signals = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
        assert isinstance(signals, RouteSignals)
        assert signals.abstain is True, (
            f"expected abstain after classifier failure; got "
            f"abstain={signals.abstain}, confidence={signals.confidence}"
        )

        decision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
        assert decision.source == "abstain", decision
        tier = ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]
        assert tier == "simple", (
            f"expected fallback to default tier 'simple', got {tier!r}"
        )

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live default-tier "
            f"backend after classifier failure; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        logger.info(
            "  [classifier-e2e][fail-open] tier=%s decision_source=%s "
            "content[:80]=%r",
            tier, decision.source, content[:80],
        )
