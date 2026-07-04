# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the @route decorator."""

from __future__ import annotations

import pytest

from switchyard.lib.agentapi.decorator import route
from switchyard.lib.agentapi.llm_class import LlmClassifier
from switchyard.lib.agentapi.rand import RandomRouter, WeightedModel


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _FakeResponse:
    """Mimics a litellm/OpenAI response: resp.choices[0].message.content."""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeMessage(content)]


async def test_rand_calls_wrapped_fn_once_and_rewrites_model():
    calls: list[dict] = []

    @route(RandomRouter([WeightedModel("frontier/model", 1.0)], rng_seed=1))
    async def chat(model: str, messages: list[dict]) -> _FakeResponse:
        calls.append({"model": model, "content": messages[-1]["content"]})
        return _FakeResponse(f"answer from {model}")

    resp = await chat(model="auto", messages=[{"role": "user", "content": "hello"}])

    assert len(calls) == 1
    assert calls[0]["model"] == "frontier/model"
    assert calls[0]["content"] == "hello"
    assert resp.choices[0].message.content == "answer from frontier/model"


async def test_classifier_calls_wrapped_fn_twice_and_returns_routed_response():
    calls: list[dict] = []

    @route(
        LlmClassifier(
            classifier_model="router/clf",
            strong_model="big/model",
            weak_model="small/model",
            threshold=0.5,
        )
    )
    async def chat(model: str, messages: list[dict]) -> _FakeResponse:
        content = messages[-1]["content"]
        calls.append({"model": model, "content": content})
        # The classifier round must return a parseable score; the routed round
        # returns the real answer.
        reply = "0.9" if model == "router/clf" else f"answer from {model}"
        return _FakeResponse(reply)

    resp = await chat(model="auto", messages=[{"role": "user", "content": "prove it"}])

    assert len(calls) == 2
    # Round 1: classifier model, classifier preamble prompt containing user text.
    assert calls[0]["model"] == "router/clf"
    assert "prove it" in calls[0]["content"]
    assert "frontier model" in calls[0]["content"]
    # Round 2: routed (strong) model, original user prompt.
    assert calls[1]["model"] == "big/model"
    assert calls[1]["content"] == "prove it"
    # Returned response is the routed call's, not the classifier's.
    assert resp.choices[0].message.content == "answer from big/model"


async def test_custom_adapters_drive_non_litellm_shape():
    @route(
        RandomRouter([WeightedModel("target/model", 1.0)], rng_seed=1),
        get_prompt=lambda kw: kw["text"],
        apply_request=lambda kw, req: kw.update(engine=req.model, text=req.prompt),
        get_completion=lambda resp: resp,
    )
    async def chat(engine: str, text: str) -> str:
        return f"{engine}:{text}"

    result = await chat(engine="auto", text="hi")
    assert result == "target/model:hi"


def test_decorating_sync_function_raises():
    with pytest.raises(TypeError):

        @route(RandomRouter([WeightedModel("m", 1.0)], rng_seed=1))
        def chat(model: str, prompt: str) -> str:  # not async
            return prompt
