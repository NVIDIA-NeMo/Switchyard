# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import litellm
import pytest
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from switchyard_litellm import RandomRoutingPlugin, StageRoutingPlugin

MODEL_GROUP = "deployer-route"
CAPABLE_MODEL = "openrouter/example/deployer-capable"
EFFICIENT_MODEL = "openrouter/example/deployer-efficient"


def stage_plugin() -> StageRoutingPlugin:
    """Build the deployer-configurable Stage policy used by the integration test."""
    return StageRoutingPlugin(
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )


def model_list() -> list[dict[str, Any]]:
    """Provide real LiteLLM deployments whose provider calls are replaced by responses."""
    return [
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {
                "model": CAPABLE_MODEL,
                "mock_response": "served by capable",
            },
        },
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {
                "model": EFFICIENT_MODEL,
                "mock_response": "served by efficient",
            },
        },
    ]


def critical_tool_history() -> list[dict[str, Any]]:
    """Build OpenAI chat history containing a decisive Stage escalation signal."""
    return [
        {"role": "user", "content": "Fix the failing tests."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": '{"command":"pytest"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "fatal runtime error: out of memory",
        },
    ]


async def test_stage_plugin_controls_real_litellm_router_pipeline() -> None:
    router = Router(model_list=model_list(), plugins=[stage_plugin()])

    efficient = await router.acompletion(
        model=MODEL_GROUP,
        messages=[{"role": "user", "content": "Say hello."}],
    )
    capable = await router.acompletion(
        model=MODEL_GROUP,
        messages=critical_tool_history(),
    )

    assert efficient.choices[0].message.content == "served by efficient"
    assert capable.choices[0].message.content == "served by capable"


async def test_seeded_random_plugin_reaches_both_litellm_candidates() -> None:
    router = Router(model_list=model_list(), plugins=[RandomRoutingPlugin(seed=6)])

    served = {
        (
            await router.acompletion(
                model=MODEL_GROUP,
                messages=[{"role": "user", "content": f"Request {index}"}],
            )
        )
        .choices[0]
        .message.content
        for index in range(2)
    }

    assert served == {"served by capable", "served by efficient"}


class DeploymentRequestRecorder(CustomLogger):
    """Record the request after all deployment callbacks have transformed it."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict[str, Any]] = []

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, Any],
        call_type: object,
    ) -> None:
        self.requests.append(kwargs)


async def test_stage_rewrite_reaches_the_selected_litellm_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = StageRoutingPlugin(
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
        escalation_note="The efficient tier failed.",
        capable_system_prompt="Use the capable tier.",
    )
    recorder = DeploymentRequestRecorder()
    monkeypatch.setattr(litellm, "callbacks", [plugin, recorder])
    router = Router(model_list=model_list(), plugins=[plugin])

    response = await router.acompletion(
        model=MODEL_GROUP,
        messages=critical_tool_history(),
        temperature=0.2,
        max_tokens=64,
    )

    assert response.choices[0].message.content == "served by capable"
    assert recorder.requests
    deployment_request = recorder.requests[-1]
    assert deployment_request["messages"][0] == {
        "role": "system",
        "content": "Use the capable tier.",
    }
    assert deployment_request["messages"][-1] == {
        "role": "user",
        "content": "The efficient tier failed.",
    }
    assert deployment_request["temperature"] == 0.2
    assert deployment_request["max_tokens"] == 64
