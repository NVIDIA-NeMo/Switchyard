#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a libsy algorithm stream from Python."""

import asyncio
from collections.abc import AsyncIterator, Mapping

from switchyard.libsy import LlmResponse, Step, algorithms


class EchoClient:
    """Return a fixed completion for any selected target."""

    async def call(
        self,
        request: Mapping[str, object],
        model: str,
    ) -> LlmResponse.Agg | LlmResponse.Stream:
        if request.get("stream") is True:

            async def events() -> AsyncIterator[Mapping[str, object]]:
                yield {
                    "preservation": None,
                    "normalized": [{"MessageStart": {"id": "echo", "model": model}}],
                }
                yield {
                    "preservation": None,
                    "normalized": [{"TextDelta": {"index": 0, "text": "Hello"}}],
                }
                yield {
                    "preservation": None,
                    "normalized": [{"MessageStop": {"reason": "end_turn"}}],
                }

            return LlmResponse.Stream(events())

        return LlmResponse.Agg(
            {
                "model": model,
                "outputs": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            }
        )


async def main() -> None:
    """Run random routing and serve its selected target."""
    request = {
        "model": "auto",
        "stream": True,
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
            case Step.CallModel(call):
                call.respond(await client.call(call.request, call.models[0]))
            case Step.Done(outcome):
                print("Decision:", outcome.selected_model_ids[0])
                response = outcome.response or await client.call(
                    outcome.request,
                    outcome.selected_model_ids[0],
                )
                match response:
                    case LlmResponse.Agg(aggregate_response):
                        print("Response:", aggregate_response)
                    case LlmResponse.Stream(response_stream):
                        async for event in response_stream:
                            print("Response event:", event)


if __name__ == "__main__":
    asyncio.run(main())
