# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation crosses the Python/Rust boundary and awaits Python cleanup."""

import asyncio
import gc
from typing import Any

import pytest

from switchyard.libsy import (
    ContextWindowExceededError,
    LibsyError,
    LlmResponse,
    Step,
    TaskClassifierConfig,
    algorithms,
)


@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("cancel_handler", [False, True])
async def test_response_cancellation_cleans_up_python_work(
    decision: bool, cancel_handler: bool
) -> None:
    config = TaskClassifierConfig.decision(cutoff=0.5) if decision else TaskClassifierConfig(0.5)
    algorithm = algorithms.llm_task_classifier(config=config)
    stream = algorithm.run_stream(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
        {
            "judge": ["judge"],
            "capable": ["strong"],
            "efficient": ["weak"],
            "any": ["strong", "weak"],
        },
    )
    step = await anext(stream)
    assert isinstance(step, Step.CallDecision if decision else Step.CallModel)
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def work() -> LlmResponse.Agg:
        try:
            started.set()
            await asyncio.Future()
            raise AssertionError("pending work completed")
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    handler = asyncio.create_task(step.call.respond(work()))
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel_handler:
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(handler, timeout=2)
        outcome = await anext(stream)
        assert isinstance(outcome, Step.Done)
        assert outcome.outcome.selected_model_ids[0] == "strong"
    else:
        del stream
        gc.collect()
        await asyncio.wait_for(handler, timeout=2)
    assert cleaned_up.is_set()


CUSTOM_CHOICES = {
    "choices": [
        {"id": "complex", "description": "Needs the capable model"},
        {"id": "simple", "description": "The efficient model is enough"},
    ],
    "apply_cutoff_to": "complex",
}


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize(
    ("options", "score", "expected"),
    [
        ({}, 0.6, "weak"),
        ({}, 0.61, "strong"),
        ({"cutoff": None}, 0.0, "weak"),
        ({"cutoff": 0.5}, 0.5, "weak"),
        ({"cutoff": 0.5}, 0.9, "strong"),
        ({"cutoff": 0.5, **CUSTOM_CHOICES}, 0.9, "strong"),
    ],
)
async def test_decision_response_and_failure_keep_their_routing_behavior(
    failure: bool, options: dict[str, Any], score: float, expected: str
) -> None:
    algorithm = algorithms.llm_task_classifier(config=TaskClassifierConfig.decision(**options))
    capable, efficient = [
        choice["id"] for choice in options.get("choices", [{"id": "capable"}, {"id": "efficient"}])
    ]
    stream = algorithm.run_stream(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
        {
            "judge": ["judge"],
            "capable": ["strong"],
            "efficient": ["weak"],
            "any": ["strong", "weak"],
        },
    )
    step = await anext(stream)
    assert isinstance(step, Step.CallDecision)

    async def answer() -> dict[str, object]:
        return {
            "id": None,
            "model": "judge",
            "answers": {
                "route": {
                    "value": {
                        "type": "choice",
                        "data": {
                            "selected": capable,
                            "probabilities": {capable: score, efficient: 1 - score},
                        },
                    },
                    "provider_confidence": None,
                }
            },
        }

    if failure:
        await step.call.fail(ContextWindowExceededError("judge context overflow"))
    else:
        await step.call.respond(answer())
    outcome = await anext(stream)
    assert isinstance(outcome, Step.Done)
    assert outcome.outcome.selected_model_ids[0] == ("strong" if failure else expected)
    with pytest.raises(LibsyError, match="already been completed"):
        await step.call.respond(answer())
