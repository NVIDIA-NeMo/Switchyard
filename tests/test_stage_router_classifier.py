# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`TierClassifier`.

The classifier is tested in isolation with a stubbed LLM client. Failure-mode
coverage matters more than success — any real-world flakiness must fall open
to ``None`` so the picker keeps moving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from switchyard.lib.processors.stage_router.classifier import (
    CAPABLE_TIER,
    EFFICIENT_TIER,
    RECENT_MESSAGES_KEY,
    TierClassifier,
)
from switchyard_rust.components import DimensionCollector, get_tool_result_signal
from switchyard_rust.core import ChatRequest, ProxyContext


@dataclass
class _Resp:
    content: str | None = None
    raise_exc: BaseException | None = None
    no_choices: bool = False

    @property
    def choices(self) -> list[Any]:
        if self.no_choices:
            return []
        return [type("M", (), {"message": type("C", (), {"content": self.content})()})()]


class _StubClient:
    def __init__(self, response: _Resp | None) -> None:
        self._response = response
        self.calls = 0

    async def acompletion(self, **_kwargs: object) -> _Resp:
        self.calls += 1
        if self._response is None:
            raise RuntimeError("upstream unavailable")
        if self._response.raise_exc is not None:
            raise self._response.raise_exc
        return self._response


async def _build_signal() -> Any:
    collector = DimensionCollector()
    ctx = ProxyContext()
    request = ChatRequest.openai_chat({
        "messages": [{"role": "user", "content": "hello"}]
    })
    await collector.process(ctx, request)
    signal = get_tool_result_signal(ctx)
    assert signal is not None
    return ctx, signal


@pytest.mark.asyncio
async def test_returns_capable_on_valid_capable_response():
    ctx, signal = await _build_signal()
    classifier = TierClassifier(
        model="m", api_key="k",
        client=_StubClient(_Resp(content=json.dumps({"tier": "capable"}))),
    )
    assert await classifier.classify(ctx, signal) == CAPABLE_TIER


@pytest.mark.asyncio
async def test_returns_efficient_on_valid_efficient_response():
    ctx, signal = await _build_signal()
    classifier = TierClassifier(
        model="m", api_key="k",
        client=_StubClient(_Resp(content=json.dumps({"tier": "efficient"}))),
    )
    assert await classifier.classify(ctx, signal) == EFFICIENT_TIER


@pytest.mark.asyncio
async def test_falls_open_on_malformed_json():
    ctx, signal = await _build_signal()
    classifier = TierClassifier(
        model="m", api_key="k",
        client=_StubClient(_Resp(content="not json at all")),
    )
    assert await classifier.classify(ctx, signal) is None


@pytest.mark.asyncio
async def test_falls_open_on_unexpected_tier_value():
    ctx, signal = await _build_signal()
    classifier = TierClassifier(
        model="m", api_key="k",
        client=_StubClient(_Resp(content=json.dumps({"tier": "elite"}))),
    )
    assert await classifier.classify(ctx, signal) is None


@pytest.mark.asyncio
async def test_falls_open_on_network_error():
    ctx, signal = await _build_signal()
    classifier = TierClassifier(
        model="m", api_key="k",
        client=_StubClient(_Resp(raise_exc=TimeoutError("upstream slow"))),
    )
    assert await classifier.classify(ctx, signal) is None


@pytest.mark.asyncio
async def test_falls_open_on_empty_choices():
    ctx, signal = await _build_signal()
    classifier = TierClassifier(
        model="m", api_key="k",
        client=_StubClient(_Resp(no_choices=True)),
    )
    assert await classifier.classify(ctx, signal) is None


@pytest.mark.asyncio
async def test_default_window_omits_recent_messages():
    """recent_turn_window=0 keeps the prompt to the aggregate state only."""
    ctx, signal = await _build_signal()
    ctx.metadata[RECENT_MESSAGES_KEY] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self) -> None:
            self.calls = 0

        async def acompletion(self, **kwargs: object) -> _Resp:
            self.calls += 1
            captured["messages"] = kwargs["messages"]
            return _Resp(content=json.dumps({"tier": "capable"}))

    classifier = TierClassifier(
        model="m", api_key="k", recent_turn_window=0, client=_Recorder(),
    )
    await classifier.classify(ctx, signal)
    user_prompt = captured["messages"][1]["content"]
    assert "Recent turns" not in user_prompt


@pytest.mark.asyncio
async def test_disable_reasoning_passes_enable_thinking_false_extra_body():
    """Reasoning models on the NVIDIA Inference Hub misroute JSON output into
    `reasoning_content` unless `enable_thinking=False` is hinted via vLLM's
    `chat_template_kwargs`. The classifier must send this by default."""
    ctx, signal = await _build_signal()
    captured: dict[str, object] = {}

    class _Recorder:
        async def acompletion(self, **kwargs: object) -> _Resp:
            captured.update(kwargs)
            return _Resp(content=json.dumps({"tier": "efficient"}))

    classifier = TierClassifier(model="m", api_key="k", client=_Recorder())
    await classifier.classify(ctx, signal)
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # max_tokens must be wide enough for reasoning_content + JSON when the hint
    # is ignored upstream; canonical LLM-classifier uses 4096.
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0


@pytest.mark.asyncio
async def test_disable_reasoning_false_omits_extra_body():
    """Opt-out path: `disable_reasoning=False` sends no extra_body."""
    ctx, signal = await _build_signal()
    captured: dict[str, object] = {}

    class _Recorder:
        async def acompletion(self, **kwargs: object) -> _Resp:
            captured.update(kwargs)
            return _Resp(content=json.dumps({"tier": "capable"}))

    classifier = TierClassifier(
        model="m", api_key="k", disable_reasoning=False, client=_Recorder(),
    )
    await classifier.classify(ctx, signal)
    assert captured["extra_body"] is None


@pytest.mark.asyncio
async def test_window_4_appends_last_four_messages():
    """recent_turn_window=4 includes the last four messages, in order."""
    ctx, signal = await _build_signal()
    ctx.metadata[RECENT_MESSAGES_KEY] = [
        {"role": "user", "content": "msg-0"},
        {"role": "assistant", "content": "msg-1"},
        {"role": "user", "content": "msg-2"},
        {"role": "assistant", "content": "msg-3"},
        {"role": "user", "content": "msg-4"},
        {"role": "assistant", "content": "msg-5"},
    ]
    captured: dict[str, object] = {}

    class _Recorder:
        async def acompletion(self, **kwargs: object) -> _Resp:
            captured["messages"] = kwargs["messages"]
            return _Resp(content=json.dumps({"tier": "efficient"}))

    classifier = TierClassifier(
        model="m", api_key="k", recent_turn_window=4, client=_Recorder(),
    )
    await classifier.classify(ctx, signal)
    user_prompt = captured["messages"][1]["content"]
    assert "Recent turns" in user_prompt
    # last 4 only, in order — earlier messages dropped
    assert "msg-2" in user_prompt
    assert "msg-3" in user_prompt
    assert "msg-4" in user_prompt
    assert "msg-5" in user_prompt
    assert "msg-0" not in user_prompt
    assert "msg-1" not in user_prompt
