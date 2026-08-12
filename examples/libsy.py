#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a libsy algorithm stream from Python."""

import asyncio
from collections.abc import Mapping

from switchyard.libsy import Step, algorithms


class EchoClient:
    """Return a fixed completion for any selected target."""

    async def call(
        self,
        request: Mapping[str, object],
        model: str,
    ) -> Mapping[str, object]:
        return {
            "model": model,
            "outputs": [
                {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
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
            case Step.Decision(decision):
                print("Decision:", decision.selected_model_id, decision.reasoning)
            case Step.CallModel(call):
                model = call.decision.selected_model_id
                call.respond(await client.call(call.request, model))
            case Step.Done(response):
                print("Response:", response)


if __name__ == "__main__":
    asyncio.run(main())
