# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agentapi optimizer role interfaces."""

from __future__ import annotations

import pytest

from switchyard.lib.agentapi.chat import ChatRequest, ChatResponse, EnrichmentData
from switchyard.lib.agentapi.optimizer import (
    AgentApiOptimizer,
    Decision,
    MetadataInput,
    ModelInference,
    OptimizerResponse,
    RequestInput,
    ResponseInput,
    Return,
)


def test_input_variants_carry_payloads():
    assert RequestInput(ChatRequest("hi", "m")).request.prompt == "hi"
    assert ResponseInput(ChatResponse("done")).response.completion == "done"
    assert MetadataInput({"k": "v"}).metadata == {"k": "v"}


def test_decision_variants():
    resp = OptimizerResponse(requests=[ChatRequest("hi", "m")])
    inf = ModelInference(resp)
    assert isinstance(inf, Decision)
    assert inf.response.requests[0].model == "m"
    assert isinstance(Return(), Decision)


def test_optimizer_is_abstract():
    with pytest.raises(TypeError):
        AgentApiOptimizer()  # optimize() is abstract


async def test_feed_has_noop_default():
    class Opt(AgentApiOptimizer):
        async def optimize(self) -> Decision:
            return Return()

    opt = Opt()
    # Default feed does nothing and does not raise.
    await opt.feed(RequestInput(ChatRequest("hi", "m")), EnrichmentData())
    assert isinstance(await opt.optimize(), Return)
