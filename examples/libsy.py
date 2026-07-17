#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run libsy algorithms to completion and as a host-driven step stream."""

import asyncio

from switchyard import libsy


class ExampleRoutedLlmClient:
    """Return a local response for the model selected by an algorithm."""

    async def call(
        self,
        context: libsy.protocol.Context,
        request: libsy.protocol.Request,
        decision: libsy.protocol.Decision,
    ) -> libsy.protocol.Response:
        return libsy.protocol.Response(
            libsy.protocol.AggLlmResponse(
                model=decision.selected_model,
                outputs=[
                    libsy.protocol.ResponseOutput(
                        libsy.protocol.Role.ASSISTANT,
                        [
                            libsy.protocol.ContentBlock.Text(
                                f"Response from {decision.selected_model}"
                            )
                        ],
                    )
                ],
            )
        )


def response_text(response: libsy.protocol.Response) -> str:
    """Read the first text block from an aggregate response."""
    block = response.aggregate.outputs[0].content[0]
    match block:
        case libsy.protocol.ContentBlock.Text(text=text):
            return text
        case _:
            raise TypeError("expected a text response")


def user_message(text: str) -> libsy.protocol.Message:
    """Build a typed user message."""
    content: libsy.protocol.ContentBlockValue = libsy.protocol.ContentBlock.Text(text)
    return libsy.protocol.Message(libsy.protocol.Role.USER, [content])


async def main() -> None:
    """Exercise no-op and random routing through the libsy execution APIs."""
    algorithm = libsy.algorithms.noop()
    request = libsy.protocol.Request(
        libsy.protocol.LlmRequest(
            model="example-model",
            messages=[user_message("Hello from Python")],
        )
    )

    decisions, response = await algorithm.run(request)
    print("run")
    for decision in decisions:
        print(f"  decision: {decision.selected_model}")
    print(f"  response: {response_text(response)}")

    print("run_stream")
    async for step in algorithm.run_stream(request):
        match step:
            case libsy.Step.Decision(decision=decision):
                print(f"  decision: {decision.selected_model}")
            case libsy.Step.CallLlm(call=call):
                raise RuntimeError(f"unexpected LLM call for {call.decision.selected_model}")
            case libsy.Step.ReturnToAgent(response=response):
                print(f"  response: {response_text(response)}")

    client = ExampleRoutedLlmClient()
    random_algorithm = libsy.algorithms.random(
        libsy.LlmTargetSet(
            [
                libsy.LlmTarget("fast-model", llm_client=client),
                libsy.LlmTarget("smart-model", llm_client=client),
            ]
        )
    )
    decisions, response = await random_algorithm.run(request)

    print("random routing")
    for decision in decisions:
        print(f"  decision: {decision.selected_model}")
    print(f"  response: {response_text(response)}")


if __name__ == "__main__":
    asyncio.run(main())
