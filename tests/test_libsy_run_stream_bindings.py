# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from switchyard import libsy


async def test_noop_algorithm_is_driven_with_structural_step_matching() -> None:
    run = libsy.algorithms.noop().run_stream(
        libsy.protocol.Request(libsy.protocol.LlmRequest(model="requested-model"))
    )
    observed: list[tuple[str, str | None]] = []

    async for step in run:
        match step:
            case libsy.Step.Decision(decision=decision):
                observed.append(("decision", decision.selected_model))
            case libsy.Step.ReturnToAgent(response=response):
                observed.append(("return", response.selected_model))
                match response.aggregate.outputs[0].content[0]:
                    case libsy.protocol.ContentBlock.Text(text="OK"):
                        pass
                    case block:
                        pytest.fail(f"unexpected content block: {block!r}")
            case libsy.Step.CallLlm():
                pytest.fail("the no-op algorithm must not request an LLM call")

    assert observed == [
        ("decision", "requested-model"),
        ("return", "requested-model"),
    ]
    with pytest.raises(RuntimeError, match="RunStream has already been consumed"):
        async for _ in run:
            pass


def test_run_stream_starts_lazily_without_an_event_loop() -> None:
    run = libsy.algorithms.noop().run_stream(
        libsy.protocol.Request(libsy.protocol.LlmRequest(model="requested-model"))
    )

    async def consume() -> list[str | None]:
        selected_models: list[str | None] = []
        async for step in run:
            match step:
                case libsy.Step.Decision(decision=decision):
                    selected_models.append(decision.selected_model)
                case _:
                    pass
        return selected_models

    assert asyncio.run(consume()) == ["requested-model"]


async def test_random_algorithm_is_driven_through_the_same_handle() -> None:
    algorithm = libsy.algorithms.random(libsy.LlmTargetSet([libsy.LlmTarget("only/model")]))
    request = libsy.protocol.Request(libsy.protocol.LlmRequest(model="auto"))
    context = libsy.protocol.Context(values={"request_id": "request-1"})
    observed: list[tuple[str, str | None]] = []

    async for step in algorithm.run_stream(request, context=context):
        match step:
            case libsy.Step.Decision(decision=decision):
                observed.append(("decision", decision.selected_model))
            case libsy.Step.CallLlm(call=call):
                observed.append(("call", call.decision.selected_model))
                assert call.context.values == {"request_id": "request-1"}
                call.respond(
                    libsy.protocol.Response(
                        libsy.protocol.AggLlmResponse(model=call.decision.selected_model)
                    )
                )
            case libsy.Step.ReturnToAgent(response=response):
                observed.append(("return", response.selected_model))

    assert observed == [
        ("decision", "only/model"),
        ("call", "only/model"),
        ("return", "only/model"),
    ]
