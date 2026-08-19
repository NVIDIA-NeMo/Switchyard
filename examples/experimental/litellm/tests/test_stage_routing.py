# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import respx
from switchyard_litellm import LiteLLMSyClient

from switchyard.libsy import Algorithm, LlmResponse, Step, algorithms

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
) -> tuple[str, dict[str, object]]:
    async for step in router.run_stream(request):
        match step:
            case Step.CallModel(call):
                call.respond(LlmResponse.Agg(await client.call(call.request)))
            case Step.Done(outcome):
                match outcome.response:
                    case LlmResponse.Agg(response):
                        return outcome.selected_model_id, response
                    case LlmResponse.Stream(_):
                        raise AssertionError("LiteLLM example expects buffered responses")
                    case None:
                        return outcome.selected_model_id, await client.call(outcome.request)
    raise AssertionError("algorithm stream ended without an outcome")


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
        fast_target, fast_response = await _run_router(
            router, request_body(), client
        )
        strong_target, strong_response = await _run_router(
            router, request_body(critical_error=True), client
        )
    finally:
        await client.aclose()

    assert fast_target == "fast"
    assert strong_target == "strong"
    assert fast_response["outputs"][0]["content"][0]["text"] == "fast"
    assert strong_response["outputs"][0]["content"][0]["text"] == "strong"
    assert seen == ["fast", "strong"]
