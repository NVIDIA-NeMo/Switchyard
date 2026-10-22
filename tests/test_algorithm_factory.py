# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment factories reuse the host-owned libsy step API."""

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from switchyard.libsy import AlgorithmFactory, ConfiguredAlgorithm, LlmResponse, Step

DEPLOYMENT = """
schema_version = 1
[llm_clients.host]
format = "openai_chat"
base_url = "https://host.invalid/v1"
api_key_env = "SWITCHYARD_FACTORY_UNUSED_LLM_KEY"
[decision_clients.host]
format = "system_one"
endpoint = "https://decision.invalid"
api_key_env = "SWITCHYARD_FACTORY_UNUSED_DECISION_KEY"
timeout_ms = 1000
[targets]
fast = { id = "gateway/fast", llm_client = "host" }
capable = { id = "gateway/capable", llm_client = "host" }
judge = { id = "gateway/judge", llm_client = "host" }
[decision_targets]
decision_judge = { id = "gateway/decision", decision_client = "host" }
[routes.parent]
id = "switchyard/parent"
type = "passthrough"
target = "fast"
[routes.parent.subagents]
type = "passthrough"
target = "capable"
[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "judge"
weak_target = "fast"
strong_target = "capable"
base_threshold = 0.5
[routes.decision]
id = "switchyard/decision"
type = "llm_classifier"
classifier_target = "decision_judge"
weak_target = "fast"
strong_target = "capable"
[routes.decision.decision]
cutoff = 0.4
candidates = { a = "capable", b = "fast" }
evidence = { description = "candidate comparison" }
"""


def request_body() -> dict[str, Any]:
    return {
        "model": "switchyard/parent",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }


async def test_factory_keeps_public_route_ids_and_subagent_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction needs no credentials, and each run keeps its own model scope."""
    monkeypatch.delenv("SWITCHYARD_FACTORY_UNUSED_LLM_KEY", raising=False)
    monkeypatch.delenv("SWITCHYARD_FACTORY_UNUSED_DECISION_KEY", raising=False)
    factory = AlgorithmFactory.from_toml(DEPLOYMENT)
    routes = factory.build()
    assert set(routes) == {"switchyard/parent", "switchyard/classifier", "switchyard/decision"}
    route = routes["switchyard/parent"]
    assert isinstance(route, ConfiguredAlgorithm)

    async def select(headers: Mapping[str, str]) -> list[str]:
        steps = [
            step
            async for step in route.run_stream(MappingProxyType(request_body()), headers=headers)
        ]
        assert len(steps) == 1
        assert isinstance(steps[0], Step.Done)
        return steps[0].outcome.selected_model_ids

    assert await asyncio.gather(
        select(MappingProxyType({})),
        select(MappingProxyType({"x-openai-subagent": "review"})),
    ) == [["gateway/fast"], ["gateway/capable"]]


@pytest.mark.parametrize("kind", ["classifier", "decision"])
async def test_configured_algorithms_yield_host_owned_judges(kind: str) -> None:
    """Model and decision judges use existing reply types and retain fallback order."""
    route = AlgorithmFactory.from_toml(DEPLOYMENT).build()[f"switchyard/{kind}"]
    stream = route.run_stream(request_body())
    step = await anext(stream)
    if kind == "classifier":
        assert isinstance(step, Step.CallModel)
        assert step.call.models == ["gateway/judge"]

        async def model_response() -> LlmResponse.Agg:
            return LlmResponse.Agg(
                {
                    "outputs": [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        '{"crux":"bounded task","primary_rule":"SUP-1",'
                                        '"capability_boundary":"supported","p_solve":0.9}'
                                    ),
                                }
                            ],
                        }
                    ]
                }
            )

        await step.call.respond(model_response())
    else:
        assert isinstance(step, Step.CallDecision)
        assert step.call.model == "gateway/decision"
        assert step.call.request["model"] == "gateway/decision"

        async def decision_response() -> dict[str, Any]:
            return {
                "model": "gateway/decision",
                "answers": {
                    "route": {
                        "value": {
                            "type": "choice",
                            "data": {
                                "selected": "no_advantage",
                                "probabilities": {"advantage": 0.1, "no_advantage": 0.9},
                            },
                        }
                    }
                },
            }

        await step.call.respond(decision_response())

    step = await anext(stream)
    assert isinstance(step, Step.Done)
    assert step.outcome.selected_model_ids == ["gateway/fast", "gateway/capable"]
    assert step.outcome.request["model"] == "gateway/fast"
    assert step.outcome.response is None
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


def test_factory_rejects_invalid_deployments() -> None:
    """Native parse and target validation failures remain Python ValueErrors."""
    with pytest.raises(ValueError):
        AlgorithmFactory.from_toml("schema_version =")
    with pytest.raises(ValueError, match="missing"):
        AlgorithmFactory.from_toml(
            DEPLOYMENT.replace('target = "fast"', 'target = "missing"')
        ).build()
