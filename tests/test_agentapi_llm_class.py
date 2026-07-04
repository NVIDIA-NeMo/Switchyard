# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLM-classifier router (ports llm_class.rs)."""

from __future__ import annotations

import pytest

from switchyard.lib.agentapi.chat import ChatRequest, ChatResponse, EnrichmentData
from switchyard.lib.agentapi.llm_class import ClassifierTier, LlmClassifier
from switchyard.lib.agentapi.optimizer import (
    ModelInference,
    RequestInput,
    ResponseInput,
    Return,
)


def _algorithm(threshold: float) -> LlmClassifier:
    return LlmClassifier(
        classifier_model="router/classifier",
        strong_model="frontier/model",
        weak_model="cheap/model",
        threshold=threshold,
    )


async def _run_flow(threshold: float, user_prompt: str, score_text: str):
    opt = _algorithm(threshold).optimizer()

    # 1. Feed the user request; first inference is the classifier call.
    await opt.feed(RequestInput(ChatRequest(user_prompt, "client/model")), EnrichmentData())
    classify = await opt.optimize()
    assert isinstance(classify, ModelInference)
    call = classify.response.requests[0]
    assert user_prompt in call.prompt
    assert call.model == "router/classifier"

    # 2. Feed the mocked classifier score; next inference is the routed call.
    await opt.feed(ResponseInput(ChatResponse(score_text)), EnrichmentData())
    routed = await opt.optimize()
    assert isinstance(routed, ModelInference)
    decision = routed.response.decision_info
    routed_model = routed.response.requests[0].model

    # 3. Feed the routed model response; next optimize returns to the agent.
    await opt.feed(ResponseInput(ChatResponse("mocked completion")), EnrichmentData())
    assert isinstance(await opt.optimize(), Return)
    return routed_model, decision


async def test_score_at_or_above_threshold_routes_strong():
    model, decision = await _run_flow(0.5, "solve this proof", "0.9")
    assert model == "frontier/model"
    assert decision.tier is ClassifierTier.STRONG
    assert decision.score == 0.9


async def test_score_below_threshold_routes_weak():
    model, decision = await _run_flow(0.5, "say hello", "0.2")
    assert model == "cheap/model"
    assert decision.tier is ClassifierTier.WEAK
    assert decision.score == 0.2


async def test_score_exactly_at_threshold_routes_strong():
    model, decision = await _run_flow(0.5, "borderline", "0.5")
    assert model == "frontier/model"
    assert decision.tier is ClassifierTier.STRONG


async def test_unparseable_score_defaults_to_strong():
    model, decision = await _run_flow(0.5, "hi", "not-a-number")
    assert model == "frontier/model"
    assert decision.tier is ClassifierTier.STRONG
    assert decision.score is None


async def test_optimize_before_feed_errors():
    opt = _algorithm(0.5).optimizer()
    with pytest.raises(ValueError):
        await opt.optimize()


async def test_optimize_before_classifier_response_errors():
    opt = _algorithm(0.5).optimizer()
    await opt.feed(RequestInput(ChatRequest("hi", "client/model")), EnrichmentData())
    assert isinstance(await opt.optimize(), ModelInference)  # classifier call emitted
    with pytest.raises(ValueError):
        await opt.optimize()  # no score fed yet
