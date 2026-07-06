# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the stage_router pickers."""

from dataclasses import dataclass
from typing import Any

import pytest

from switchyard.lib.processors.stage_router import (
    CAPABLE,
    EFFICIENT,
    StageRouterDecisionLog,
    TierClassifier,
)
from switchyard.lib.processors.stage_router.decision_log import CONTEXT_KEY
from switchyard.lib.processors.stage_router.picker import (
    pick_capable_first,
    pick_efficient_first,
)
from switchyard_rust.components import DimensionCollector
from switchyard_rust.core import ChatRequest, ProxyContext


async def _ctx(messages: list[dict]) -> ProxyContext:
    collector = DimensionCollector()
    ctx = ProxyContext()
    await collector.process(ctx, ChatRequest.openai_chat({"messages": messages}))
    return ctx


def _msg_tool_call(name: str) -> dict:
    return {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": "{}"}}]}


def _msg_tool_result(content: str) -> dict:
    return {"role": "tool", "tool_call_id": "x", "content": content}


@dataclass
class _StubClassifierResponse:
    tier: str | None

    @property
    def choices(self) -> list[Any]:
        if self.tier is None:
            return [type("M", (), {"message": type("C", (), {"content": "garbage"})()})()]
        content = f'{{"tier": "{self.tier}"}}'
        return [type("M", (), {"message": type("C", (), {"content": content})()})()]


class _StubLLMClient:
    """In-memory classifier client; one canned response per call."""

    def __init__(self, tier: str | None) -> None:
        self._tier = tier
        self.calls = 0

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: int,
        response_format: dict[str, str],
        max_tokens: int,
        extra_body: dict[str, Any] | None,
    ) -> _StubClassifierResponse:
        self.calls += 1
        return _StubClassifierResponse(tier=self._tier)


def _stub_classifier(tier: str | None) -> TierClassifier:
    return TierClassifier(
        model="stub",
        api_key="stub",
        client=_StubLLMClient(tier),
    )


@pytest.mark.asyncio
async def test_critical_severity_overrides_to_capable():
    ctx = await _ctx([
        _msg_tool_result("Out of memory: cannot allocate memory"),
        {"role": "user", "content": "retry"},
    ])
    assert await pick_capable_first(ctx, confidence_threshold=0.7) == CAPABLE
    assert await pick_efficient_first(ctx, confidence_threshold=0.7) == CAPABLE


@pytest.mark.asyncio
async def test_tests_passed_with_edits_drops_to_efficient():
    messages = [_msg_tool_call("Edit")] * 3 + [_msg_tool_result("all tests passed")] * 3
    messages += [{"role": "user", "content": "ok continue"}] * 12
    ctx = await _ctx(messages)
    assert await pick_capable_first(ctx, confidence_threshold=0.7) == EFFICIENT


@pytest.mark.asyncio
async def test_no_signal_returns_default_tier():
    ctx = ProxyContext()  # signal not stamped
    assert await pick_capable_first(ctx, confidence_threshold=0.7) == CAPABLE
    assert await pick_efficient_first(ctx, confidence_threshold=0.7) == EFFICIENT


@pytest.mark.asyncio
async def test_low_confidence_consults_classifier_when_configured():
    """Empty trajectory yields confidence 0.0 — picker must call the classifier."""
    messages = [{"role": "user", "content": "hi"}]
    ctx = await _ctx(messages)
    classifier = _stub_classifier(tier="efficient")
    assert await pick_capable_first(
        ctx, confidence_threshold=0.7, classifier=classifier,
    ) == EFFICIENT
    assert classifier._client.calls == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_low_confidence_falls_back_to_default_without_classifier():
    messages = [{"role": "user", "content": "hi"}]
    ctx = await _ctx(messages)
    assert await pick_capable_first(ctx, confidence_threshold=0.7) == CAPABLE
    assert await pick_efficient_first(ctx, confidence_threshold=0.7) == EFFICIENT


@pytest.mark.asyncio
async def test_classifier_fall_open_on_unknown_tier():
    """A malformed classifier verdict must not override the picker default."""
    messages = [{"role": "user", "content": "hi"}]
    ctx = await _ctx(messages)
    classifier = _stub_classifier(tier=None)
    assert await pick_capable_first(
        ctx, confidence_threshold=0.7, classifier=classifier,
    ) == CAPABLE


@pytest.mark.asyncio
async def test_threshold_zero_skips_classifier_entirely():
    """``confidence_threshold=0`` means accept every scorer verdict, however efficient."""
    messages = [{"role": "user", "content": "hi"}]
    ctx = await _ctx(messages)
    classifier = _stub_classifier(tier="capable")
    # Empty trajectory → scorer near zero, default sign decides; classifier not consulted.
    await pick_capable_first(ctx, confidence_threshold=0.0, classifier=classifier)
    assert classifier._client.calls == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_dimensions_branch_routes_by_scorer_sign():
    """Confident scorer verdict (≥ threshold) bypasses the classifier.

    A single TodoWrite in the recent window pushes ``planning_active`` to
    1.0 (weight -0.70) — that alone clears the 0.7 confidence threshold,
    so ``_pick`` takes the dimensions branch and returns EFFICIENT by sign
    without consulting the classifier.
    """
    log = StageRouterDecisionLog()
    classifier = _stub_classifier(tier="capable")  # would push CAPABLE if reached
    ctx = await _ctx([
        _msg_tool_call("TodoWrite"),
        _msg_tool_result("ok"),
        {"role": "user", "content": "next"},
    ])
    tier = await pick_capable_first(
        ctx, confidence_threshold=0.7, classifier=classifier, decision_log=log,
    )
    assert ctx.metadata[CONTEXT_KEY] == "dimensions"
    assert tier == EFFICIENT  # negative score → EFFICIENT regardless of capable-first fallback
    assert classifier._client.calls == 0  # type: ignore[attr-defined]
    assert log.snapshot()["dimensions"] == 1


@pytest.mark.asyncio
async def test_decision_log_counts_sources():
    """Each decision path increments exactly one bucket in the shared log."""
    log = StageRouterDecisionLog()
    # override path: critical severity → CAPABLE, source=override
    ctx_critical = await _ctx([
        _msg_tool_result("Out of memory: cannot allocate memory"),
        {"role": "user", "content": "retry"},
    ])
    await pick_capable_first(ctx_critical, confidence_threshold=0.7, decision_log=log)
    assert ctx_critical.metadata[CONTEXT_KEY] == "override"

    # fall_open path: low confidence, no classifier → default tier
    ctx_neutral = await _ctx([{"role": "user", "content": "hi"}])
    await pick_capable_first(ctx_neutral, confidence_threshold=0.99, decision_log=log)
    assert ctx_neutral.metadata[CONTEXT_KEY] == "fall_open"

    # classifier path: low confidence, classifier configured → classifier verdict
    classifier = _stub_classifier(tier="efficient")
    ctx_class = await _ctx([{"role": "user", "content": "hi"}])
    await pick_capable_first(
        ctx_class, confidence_threshold=0.99, classifier=classifier, decision_log=log,
    )
    assert ctx_class.metadata[CONTEXT_KEY] == "llm-classifier"

    snapshot = log.snapshot()
    assert snapshot["override"] == 1
    assert snapshot["fall_open"] == 1
    assert snapshot["llm-classifier"] == 1
    assert log.total() == 3
