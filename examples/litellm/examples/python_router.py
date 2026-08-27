#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send one request through LiteLLM's Router and the Switchyard Stage plugin."""

import asyncio

import litellm
from litellm.router import Router
from litellm.types.llms.openai import AllMessageValues
from switchyard_litellm import StageRoutingPlugin

MODEL_GROUP = "switchyard"
MODEL_LIST: list[dict[str, object]] = [
    {
        "model_name": MODEL_GROUP,
        "litellm_params": {
            "model": "openrouter/openai/gpt-5.6-sol",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    },
    {
        "model_name": MODEL_GROUP,
        "litellm_params": {
            "model": "openrouter/openai/gpt-5.6-terra",
            "api_key": "os.environ/OPENROUTER_API_KEY",
        },
    },
]
STAGE_ROUTING_PLUGIN = StageRoutingPlugin(
    picker="efficient_first",
    confidence_threshold=0.5,
    recent_window=3,
)


def critical_tool_history() -> list[AllMessageValues]:
    """Build a turn whose critical tool result should select GPT-5.6 Sol."""
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


async def main() -> None:
    """Route the example through the plugin-controlled LiteLLM pipeline."""
    router = Router(model_list=MODEL_LIST, plugins=[STAGE_ROUTING_PLUGIN])
    litellm.callbacks.append(STAGE_ROUTING_PLUGIN)
    try:
        response = await router.acompletion(
            model=MODEL_GROUP,
            messages=critical_tool_history(),
            max_tokens=64,
        )
        print("Selected model:", response.model)
        print("Response:", response.choices[0].message.content)
    finally:
        litellm.callbacks.remove(STAGE_ROUTING_PLUGIN)


if __name__ == "__main__":
    asyncio.run(main())
