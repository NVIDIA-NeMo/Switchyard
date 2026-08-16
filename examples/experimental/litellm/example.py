#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route one coding-agent turn through LiteLLM with libsy's Stage router."""

import asyncio

from switchyard_litellm import LiteLLMSyClient

from switchyard.libsy import Step, algorithms


def sy_request() -> dict[str, object]:
    """Build a normalized turn whose critical tool failure needs the capable tier."""
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Fix the failing tests.",
                    }
                ],
            },
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
        ],
        "reasoning": {"effort": "low"},
        "output": {"max_output_tokens": 96},
    }


async def main() -> None:
    """Run the Stage router and print its normalized result."""
    client = LiteLLMSyClient()
    router = algorithms.stage_router(
        "strong",
        "fast",
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    try:
        async for step in router.run_stream(sy_request()):
            match step:
                case Step.Decision(decision):
                    print("Decision:", decision.selected_model_id, decision.reasoning)
                case Step.CallModel(call):
                    try:
                        response = await client.call(call.request)
                    except Exception as error:
                        call.fail(error)
                    else:
                        call.respond(response)
                case Step.Done(response):
                    print("Response:", response)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
