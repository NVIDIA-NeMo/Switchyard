#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route one text request through LiteLLM with libsy's random algorithm."""

import asyncio

from switchyard_litellm import LiteLLMSyClient

from switchyard.libsy import LlmTarget, algorithms


def sy_request() -> dict[str, object]:
    """Build a normalized libsy text request."""
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain weighted random LLM routing in one sentence.",
                    }
                ],
            }
        ],
        "reasoning": {"effort": "low"},
        "output": {"max_output_tokens": 96},
    }


async def main() -> None:
    """Run the weighted router and print its normalized result."""
    strong_client = LiteLLMSyClient("strong")
    fast_client = LiteLLMSyClient("fast")
    router = algorithms.random(
        [
            LlmTarget("strong", strong_client),
            LlmTarget("fast", fast_client),
        ],
        weights=[1, 3],
        seed=42,
    )
    try:
        decisions, response = await router.run(sy_request())
        print("Random:", decisions, response)
    finally:
        await strong_client.aclose()
        await fast_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
