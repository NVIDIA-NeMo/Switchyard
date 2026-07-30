# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import respx
from switchyard_litellm import LiteLLMSyClient

from switchyard.libsy import LlmTarget, algorithms

BASE_URL = "http://gateway.test/v1"


def request_body() -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Route this request."}],
            }
        ]
    }


def gateway_response(model: str) -> dict[str, object]:
    return {
        "id": f"chatcmpl-{model}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": model},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
        },
    }


@respx.mock
async def test_random_router_can_drive_both_litellm_targets() -> None:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        return httpx.Response(200, json=gateway_response(model))

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=respond)
    strong_client = LiteLLMSyClient("strong", base_url=BASE_URL)
    fast_client = LiteLLMSyClient("fast", base_url=BASE_URL)
    targets = [
        LlmTarget("strong", strong_client),
        LlmTarget("fast", fast_client),
    ]
    try:
        strong_router = algorithms.random(targets, weights=[1, 0], seed=42)
        fast_router = algorithms.random(targets, weights=[0, 1], seed=42)
        strong_decisions, strong_response = await strong_router.run(request_body())
        fast_decisions, fast_response = await fast_router.run(request_body())
    finally:
        await strong_client.aclose()
        await fast_client.aclose()

    assert [item["selected_model"] for item in strong_decisions] == ["strong"]
    assert [item["selected_model"] for item in fast_decisions] == ["fast"]
    assert strong_response["outputs"][0]["content"][0]["text"] == "strong"
    assert fast_response["outputs"][0]["content"][0]["text"] == "fast"
    assert seen == ["strong", "fast"]
