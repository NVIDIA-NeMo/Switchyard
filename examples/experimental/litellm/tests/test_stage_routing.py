# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import respx
from switchyard_litellm import LiteLLMSyClient

from switchyard.libsy import Algorithm, Decision, Step, algorithms

BASE_URL = "http://gateway.test/v1"


def request_body(*, critical_error: bool = False) -> dict[str, object]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Fix the failing tests."}],
        }
    ]
    if critical_error:
        messages.extend([
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "Bash",
                        "arguments": {"command": "pytest"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "fatal runtime error: out of memory",
                            }
                        ],
                        "is_error": True,
                    }
                ],
            },
        ])
    return {"model": "auto", "messages": messages}


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


async def _run_router(
    router: Algorithm,
    request: dict[str, object],
    client: LiteLLMSyClient,
) -> tuple[list[Decision], dict[str, object]]:
    decisions: list[Decision] = []
    async for step in router.run_stream(request):
        match step:
            case Step.Decision(decision):
                decisions.append(decision)
            case Step.CallModel(call):
                call.respond(await client.call(call.request))
            case Step.Done(response):
                return decisions, response
    raise AssertionError("algorithm stream ended without a response")


@respx.mock
async def test_stage_router_drives_both_litellm_models() -> None:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        return httpx.Response(200, json=gateway_response(model))

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=respond)
    client = LiteLLMSyClient(base_url=BASE_URL)
    router = algorithms.stage_router(
        "strong",
        "fast",
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    try:
        fast_decisions, fast_response = await _run_router(
            router, request_body(), client
        )
        strong_decisions, strong_response = await _run_router(
            router, request_body(critical_error=True), client
        )
    finally:
        await client.aclose()

    assert [item.selected_model_id for item in fast_decisions] == ["fast"]
    assert [item.selected_model_id for item in strong_decisions] == ["strong"]
    assert fast_response["outputs"][0]["content"][0]["text"] == "fast"
    assert strong_response["outputs"][0]["content"][0]["text"] == "strong"
    assert seen == ["fast", "strong"]
