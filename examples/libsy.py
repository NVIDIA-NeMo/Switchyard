#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run libsy's no-op and random-routing algorithms from Python."""

import asyncio
from collections.abc import Mapping

from switchyard.libsy import LlmTarget, algorithms


class EchoClient:
    """Return its configured model as the completion."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "model": self.model,
            "outputs": [
                {"role": "assistant", "content": [{"type": "text", "text": self.model}]}
            ],
        }


async def main() -> None:
    """Run both algorithms and print their aggregate results."""
    request = {
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    }

    noop_decisions, noop_response = await algorithms.noop().run(request)
    print("No-op:", noop_decisions, noop_response)

    random = algorithms.random(
        [
            LlmTarget("fast", EchoClient("fast")),
            LlmTarget("quality", EchoClient("quality")),
        ]
    )
    random_decisions, random_response = await random.run(request)
    print("Random:", random_decisions, random_response)


if __name__ == "__main__":
    asyncio.run(main())
