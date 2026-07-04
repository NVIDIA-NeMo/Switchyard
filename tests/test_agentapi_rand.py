# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the weighted random router (ports rand.rs)."""

from __future__ import annotations

import pytest

from switchyard.lib.agentapi.chat import ChatRequest, EnrichmentData
from switchyard.lib.agentapi.optimizer import (
    AgentApiOptimizer,
    ModelInference,
    RequestInput,
    ResponseInput,
    Return,
)
from switchyard.lib.agentapi.rand import RandomRouter, WeightedModel


def _algorithm(models: list[WeightedModel], seed: int) -> RandomRouter:
    return RandomRouter(models=models, rng_seed=seed)


async def _route_once(opt: AgentApiOptimizer) -> str:
    await opt.feed(RequestInput(ChatRequest("hi", "client/model")), EnrichmentData())
    decision = await opt.optimize()
    assert isinstance(decision, ModelInference)
    return decision.response.requests[0].model


async def test_all_weight_on_one_model_always_selects_it():
    algo = _algorithm(
        [WeightedModel("a/model", 0.0), WeightedModel("b/model", 1.0), WeightedModel("c/model", 0.0)],
        seed=7,
    )
    opt = algo.optimizer()
    for _ in range(25):
        assert await _route_once(opt) == "b/model"


async def test_selection_frequencies_track_weights():
    algo = _algorithm(
        [WeightedModel("a/model", 1.0), WeightedModel("b/model", 3.0), WeightedModel("c/model", 6.0)],
        seed=42,
    )
    opt = algo.optimizer()
    counts = {"a/model": 0, "b/model": 0, "c/model": 0}
    draws = 20_000
    for _ in range(draws):
        counts[await _route_once(opt)] += 1
    assert abs(counts["a/model"] / draws - 0.1) < 0.02
    assert abs(counts["b/model"] / draws - 0.3) < 0.02
    assert abs(counts["c/model"] / draws - 0.6) < 0.02


async def test_decision_reports_draw_within_total_weight():
    opt = _algorithm([WeightedModel("a/model", 2.0), WeightedModel("b/model", 3.0)], seed=7).optimizer()
    await opt.feed(RequestInput(ChatRequest("hi", "client/model")), EnrichmentData())
    decision = await opt.optimize()
    assert isinstance(decision, ModelInference)
    info = decision.response.decision_info
    assert info.total_weight == 5.0
    assert 0.0 <= info.draw < 5.0


async def test_returns_to_agent_after_response_is_fed():
    opt = _algorithm([WeightedModel("frontier/model", 1.0)], seed=7).optimizer()
    await opt.feed(RequestInput(ChatRequest("hi", "client/model")), EnrichmentData())
    first = await opt.optimize()
    assert isinstance(first, ModelInference)
    assert first.response.requests[0].model == "frontier/model"
    await opt.feed(ResponseInput_from("mocked completion"), EnrichmentData())
    assert isinstance(await opt.optimize(), Return)


async def test_optimize_before_feed_errors():
    opt = _algorithm([WeightedModel("a/model", 1.0)], seed=7).optimizer()
    with pytest.raises(ValueError):
        await opt.optimize()


async def test_no_positive_weight_errors():
    opt = _algorithm([WeightedModel("a/model", 0.0), WeightedModel("b/model", 0.0)], seed=7).optimizer()
    await opt.feed(RequestInput(ChatRequest("hi", "client/model")), EnrichmentData())
    with pytest.raises(ValueError):
        await opt.optimize()


async def test_factory_mints_independent_deterministic_optimizers():
    algo = _algorithm([WeightedModel("a/model", 1.0), WeightedModel("b/model", 1.0)], seed=42)
    first, second = algo.optimizer(), algo.optimizer()

    async def first_draw(opt: AgentApiOptimizer) -> float:
        await opt.feed(RequestInput(ChatRequest("hi", "client/model")), EnrichmentData())
        decision = await opt.optimize()
        assert isinstance(decision, ModelInference)
        return decision.response.decision_info.draw

    assert await first_draw(first) == await first_draw(second)


def ResponseInput_from(text: str) -> ResponseInput:
    from switchyard.lib.agentapi.chat import ChatResponse

    return ResponseInput(ChatResponse(text))
