# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ReasoningEffortNormalizer``."""

from __future__ import annotations

import pytest

from switchyard.lib.processors.reasoning_effort_normalizer import (
    ReasoningEffortNormalizer,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest


def _make_request(reasoning_effort: str | None) -> ChatRequest:
    body: dict = {
        "model": "azure/anthropic/claude-opus-4-6",
        "messages": [{"role": "user", "content": "hi"}],
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return ChatRequest.openai_chat(body)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
async def test_valid_effort_passes_through(effort: str) -> None:
    req = _make_request(effort)
    out = await ReasoningEffortNormalizer().process(ProxyContext(), req)
    assert out.body["reasoning_effort"] == effort


async def test_xhigh_alias_maps_to_high() -> None:
    req = _make_request("xhigh")
    out = await ReasoningEffortNormalizer().process(ProxyContext(), req)
    assert out.body["reasoning_effort"] == "high"


async def test_unknown_effort_falls_back_to_high() -> None:
    req = _make_request("super-mega")
    out = await ReasoningEffortNormalizer().process(ProxyContext(), req)
    assert out.body["reasoning_effort"] == "high"


async def test_absent_effort_is_noop() -> None:
    req = _make_request(None)
    out = await ReasoningEffortNormalizer().process(ProxyContext(), req)
    assert "reasoning_effort" not in out.body
