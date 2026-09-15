# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Python judge deadlines and replies after a routing run is cancelled."""

import asyncio
from collections.abc import AsyncIterator, Awaitable

import pytest

from switchyard.libsy import LibsyError, LlmResponse, Step, TaskClassifierConfig, algorithms


def judge_run() -> AsyncIterator[Step.CallModel | Step.Done]:
    algorithm = algorithms.llm_task_classifier(config=TaskClassifierConfig(0.5))
    return algorithm.run_stream(
        {
            "model": "route",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        },
        {
            "judge": ["judge"],
            "capable": ["strong"],
            "efficient": ["weak"],
            "any": ["strong", "weak"],
        },
    )


@pytest.mark.parametrize("failure", [False, True], ids=["respond", "fail"])
async def test_reply_after_cancelling_run_is_ignored(failure: bool) -> None:
    stream = judge_run()
    step = await anext(stream)
    assert isinstance(step, Step.CallModel)
    call = step.call
    del stream
    # `del stream` releases the iterator and aborts its Rust task; wait for cancellation.
    await asyncio.sleep(0.05)
    response = LlmResponse.Agg(
        {
            "model": "judge",
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "{}"}],
                    "stop_reason": "end_turn",
                }
            ],
        }
    )
    if failure:
        call.fail(TimeoutError("judge exceeded the host deadline"))
    else:
        call.respond(response)
    with pytest.raises(LibsyError, match="already been completed"):
        call.respond(response)
    with pytest.raises(LibsyError, match="already been completed"):
        call.fail(RuntimeError("second completion"))


@pytest.mark.parametrize("source", ["buffered", "streamed", "expired-stream"])
async def test_python_host_deadline_returns_the_fallback(source: str) -> None:
    cancelled = asyncio.Event()

    class ExpiredStream:
        def __aiter__(self) -> AsyncIterator[dict[str, object]]:
            return self

        def __anext__(self) -> Awaitable[dict[str, object]]:
            cancelled.set()
            raise TimeoutError()

    async def slow_response() -> dict[str, object]:
        try:
            await asyncio.sleep(10)
            raise AssertionError("judge should have been cancelled")
        finally:
            cancelled.set()

    async def events() -> AsyncIterator[dict[str, object]]:
        yield {
            "preservation": None,
            "normalized": [{"MessageStart": {"id": "judge", "model": "judge"}}],
        }
        try:
            await asyncio.wait_for(slow_response(), timeout=0.05)
        except asyncio.TimeoutError as error:
            raise TimeoutError("judge stream exceeded the host deadline") from error

    async def route() -> list[str]:
        async for step in judge_run():
            match step:
                case Step.CallModel(call):
                    assert call.category == "judge"
                    if source == "expired-stream":
                        call.respond(LlmResponse.Stream(ExpiredStream()))
                    elif source == "streamed":
                        call.respond(LlmResponse.Stream(events()))
                    else:
                        try:
                            await asyncio.wait_for(slow_response(), timeout=0.05)
                        except asyncio.TimeoutError as error:
                            call.fail(TimeoutError(str(error)))
                case Step.Done(outcome):
                    assert outcome.metadata is not None
                    assert outcome.metadata.evidence == {
                        "source": "fail_open",
                        "reason_code": "timeout",
                    }
                    return outcome.selected_model_ids
        raise AssertionError("routing ended without a fallback")

    assert (await asyncio.wait_for(route(), timeout=1))[0] == "strong"
    assert cancelled.is_set()
