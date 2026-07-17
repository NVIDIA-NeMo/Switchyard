# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from switchyard import libsy


def test_algorithm_handle_cannot_be_constructed_or_subclassed_in_python() -> None:
    with pytest.raises(TypeError):
        libsy.Algorithm()
    with pytest.raises(TypeError):

        class PythonAlgorithm(libsy.Algorithm):
            pass


class _EchoRoutedLlmClient:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                libsy.protocol.Context,
                libsy.protocol.Request,
                libsy.protocol.Decision,
            ]
        ] = []

    async def call(
        self,
        context: libsy.protocol.Context,
        request: libsy.protocol.Request,
        decision: libsy.protocol.Decision,
    ) -> libsy.protocol.Response:
        self.calls.append((context, request, decision))
        return libsy.protocol.Response(libsy.protocol.AggLlmResponse(model=decision.selected_model))


class _FailingRoutedLlmClient:
    async def call(
        self,
        context: libsy.protocol.Context,
        request: libsy.protocol.Request,
        decision: libsy.protocol.Decision,
    ) -> libsy.protocol.Response:
        raise RuntimeError(f"failed to call {decision.selected_model}")


async def test_noop_algorithm_runs_to_completion() -> None:
    decisions, response = await libsy.algorithms.noop().run(
        libsy.protocol.Request(libsy.protocol.LlmRequest(model="requested-model"))
    )

    assert [decision.selected_model for decision in decisions] == ["requested-model"]
    assert response.selected_model == "requested-model"
    match response.aggregate.outputs[0].content[0]:
        case libsy.protocol.ContentBlock.Text(text="OK"):
            pass
        case block:
            pytest.fail(f"unexpected content block: {block!r}")


async def test_random_algorithm_run_uses_the_target_client() -> None:
    client = _EchoRoutedLlmClient()
    target = libsy.LlmTarget("only/model", llm_client=client)
    algorithm = libsy.algorithms.random(libsy.LlmTargetSet([target]))
    request = libsy.protocol.Request(libsy.protocol.LlmRequest(model="auto"))
    context = libsy.protocol.Context(values={"request_id": "request-1"})

    assert type(algorithm) is libsy.Algorithm
    decisions, response = await algorithm.run(request, context=context)

    assert [decision.selected_model for decision in decisions] == ["only/model"]
    assert response.selected_model == "only/model"
    assert len(client.calls) == 1
    called_context, called_request, called_decision = client.calls[0]
    assert called_context.values == {"request_id": "request-1"}
    assert called_request.requested_model == "auto"
    assert called_decision.selected_model == "only/model"


async def test_python_target_client_errors_reach_the_algorithm_caller() -> None:
    target = libsy.LlmTarget("broken/model", llm_client=_FailingRoutedLlmClient())
    algorithm = libsy.algorithms.random(libsy.LlmTargetSet([target]))

    with pytest.raises(libsy.LibsyError, match="failed to call broken/model"):
        await algorithm.run(libsy.protocol.Request(libsy.protocol.LlmRequest(model="auto")))
