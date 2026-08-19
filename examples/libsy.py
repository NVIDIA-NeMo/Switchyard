#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a libsy algorithm stream from Python."""

import asyncio

from switchyard.libsy import Step, algorithms


class EchoClient:
    """Return a fixed completion for the model on the request."""

    async def call(self, request):
        return {
            "model": request["model"],
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                    "stop_reason": "end_turn",
                }
            ],
        }


async def main() -> None:
    """Run random routing and serve its selected target."""
    request = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    }
    client = EchoClient()
    algorithm = algorithms.random(
        ["fast", "quality"],
        weights=[1, 3],
        seed=42,
    )

    async for step in algorithm.run_stream(request):
        match step:
            case Step.Done(outcome):
                response = outcome.response
                if response is None:
                    response = await client.call(outcome.request)
                print("Selected model:", outcome.selected_model_id)
                print("Response:", response)


if __name__ == "__main__":
    asyncio.run(main())
