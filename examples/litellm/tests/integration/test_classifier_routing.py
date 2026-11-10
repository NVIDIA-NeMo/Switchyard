# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from typing import Any

from litellm import Router
from switchyard_litellm import ClassifierRoutingPlugin, LiteLLMJudge

from switchyard.libsy import LlmResponse

MODEL_GROUP = "deployer-route"
JUDGE_GROUP = "deployer-judge"
CAPABLE_MODEL = "openrouter/example/deployer-capable"
EFFICIENT_MODEL = "openrouter/example/deployer-efficient"
JUDGE_MODEL = "openrouter/example/deployer-judge"

SUPPORTED_VERDICT = json.dumps(
    {"crux": "x", "primary_rule": "SUP-1", "capability_boundary": "supported", "p_solve": 0.9}
)


def model_list() -> list[dict[str, Any]]:
    """Provide real LiteLLM deployments whose provider calls are replaced by responses."""
    return [
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {"model": CAPABLE_MODEL, "mock_response": "served by capable"},
        },
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {"model": EFFICIENT_MODEL, "mock_response": "served by efficient"},
        },
        {
            "model_name": JUDGE_GROUP,
            "litellm_params": {"model": JUDGE_MODEL, "mock_response": SUPPORTED_VERDICT},
        },
    ]


class CountingJudge:
    """Serve judge calls through the Router while counting them."""

    def __init__(self) -> None:
        self.inner = LiteLLMJudge(JUDGE_GROUP)
        self.calls = 0

    async def __call__(self, request: Mapping[str, object], model: str) -> LlmResponse.Agg:
        self.calls += 1
        return await self.inner(request, model)


def classifier_router() -> tuple[Router, CountingJudge]:
    """Build a Router whose classifier judge calls go back through the same Router."""
    judge = CountingJudge()
    plugin = ClassifierRoutingPlugin(judge=judge, judge_model=JUDGE_GROUP, base_threshold=0.5)
    router = Router(model_list=model_list(), plugins=[plugin])
    judge.inner.router = router
    return router, judge


async def test_judge_call_through_the_same_router_is_not_routed_again() -> None:
    router, judge = classifier_router()

    response = await router.acompletion(
        model=MODEL_GROUP, messages=[{"role": "user", "content": "Say hello."}]
    )

    assert response.choices[0].message.content == "served by efficient"
    assert judge.calls == 1


async def test_client_supplied_judge_metadata_is_still_classified() -> None:
    router, judge = classifier_router()

    response = await router.acompletion(
        model=MODEL_GROUP,
        messages=[{"role": "user", "content": "Say hello."}],
        metadata={"switchyard_judge": True},
    )

    assert response.choices[0].message.content == "served by efficient"
    assert judge.calls == 1
