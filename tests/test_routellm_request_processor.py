# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``RouteLLMRequestProcessor`` — scoring + tier stamping."""

from __future__ import annotations

from typing import Any

from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.processors.routellm_request_processor import (
    CTX_ROUTELLM_TIER,
    RouteLLMRequestProcessor,
    _extract_user_prompt,
)
from switchyard.lib.profiles.routellm import RouteLLMConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest


class FakeClassifier:
    """Stand-in for a routellm router. Score is whatever you set it to."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.calls: list[str] = []

    def calculate_strong_win_rate(self, prompt: str) -> float:
        self.calls.append(prompt)
        return self._score


def _config(threshold: float = 0.5) -> RouteLLMConfig:
    return RouteLLMConfig(
        strong=LlmTarget(model="strong-model"),
        weak=LlmTarget(model="weak-model"),
        threshold=threshold,
    fallback_target_on_evict="strong")


def _openai_request(messages: list[dict[str, Any]]) -> ChatRequest:
    return ChatRequest.openai_chat({"model": "x", "messages": messages})


class TestPromptExtraction:
    def test_string_content(self):
        req = _openai_request([{"role": "user", "content": "hello"}])
        assert _extract_user_prompt(req) == "hello"

    def test_list_content_text_blocks(self):
        req = _openai_request([
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]},
        ])
        assert _extract_user_prompt(req) == "first\nsecond"

    def test_picks_last_user_turn(self):
        req = _openai_request([
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "newest"},
        ])
        assert _extract_user_prompt(req) == "newest"

    def test_no_user_message(self):
        req = _openai_request([{"role": "system", "content": "you are helpful"}])
        assert _extract_user_prompt(req) is None

    def test_empty_string_content(self):
        req = _openai_request([{"role": "user", "content": ""}])
        assert _extract_user_prompt(req) is None

    def test_responses_string_input(self):
        req = ChatRequest.openai_responses({"model": "x", "input": "hello via responses"})
        assert _extract_user_prompt(req) == "hello via responses"

    def test_responses_list_input(self):
        req = ChatRequest.openai_responses({
            "model": "x",
            "input": [{"role": "user", "content": "responses-msg"}],
        })
        assert _extract_user_prompt(req) == "responses-msg"


class TestRouteLLMRequestProcessor:
    async def test_score_above_threshold_picks_strong(self):
        proc = RouteLLMRequestProcessor(_config(threshold=0.5), classifier=FakeClassifier(0.8))
        await proc.startup()
        ctx = ProxyContext()
        req = _openai_request([{"role": "user", "content": "what is 2+2?"}])
        await proc.process(ctx, req)
        assert ctx.metadata[CTX_ROUTELLM_TIER] == "strong"

    async def test_score_below_threshold_picks_weak(self):
        proc = RouteLLMRequestProcessor(_config(threshold=0.5), classifier=FakeClassifier(0.2))
        await proc.startup()
        ctx = ProxyContext()
        req = _openai_request([{"role": "user", "content": "trivial question"}])
        await proc.process(ctx, req)
        assert ctx.metadata[CTX_ROUTELLM_TIER] == "weak"

    async def test_score_equal_threshold_picks_strong(self):
        # `>=` semantics: score >= threshold → strong tier.
        proc = RouteLLMRequestProcessor(_config(threshold=0.5), classifier=FakeClassifier(0.5))
        await proc.startup()
        ctx = ProxyContext()
        req = _openai_request([{"role": "user", "content": "edge"}])
        await proc.process(ctx, req)
        assert ctx.metadata[CTX_ROUTELLM_TIER] == "strong"

    async def test_no_user_prompt_defaults_to_strong(self):
        # Conservative — when classification can't run, route to the
        # higher-quality tier.
        clf = FakeClassifier(0.0)  # would pick weak if it ran
        proc = RouteLLMRequestProcessor(_config(threshold=0.5), classifier=clf)
        await proc.startup()
        ctx = ProxyContext()
        req = _openai_request([{"role": "system", "content": "no user turn"}])
        await proc.process(ctx, req)
        assert ctx.metadata[CTX_ROUTELLM_TIER] == "strong"
        assert clf.calls == []  # never scored

    async def test_classifier_receives_extracted_prompt(self):
        clf = FakeClassifier(0.7)
        proc = RouteLLMRequestProcessor(_config(threshold=0.5), classifier=clf)
        await proc.startup()
        ctx = ProxyContext()
        req = _openai_request([
            {"role": "user", "content": "what can you do?"},
            {"role": "assistant", "content": "lots"},
            {"role": "user", "content": "tell me about quantum chromodynamics"},
        ])
        await proc.process(ctx, req)
        assert clf.calls == ["tell me about quantum chromodynamics"]

    async def test_process_before_startup_warns_and_defaults_strong(self):
        # No startup() call → classifier is None → safe default + warn.
        proc = RouteLLMRequestProcessor(_config(threshold=0.5))  # no injection
        ctx = ProxyContext()
        req = _openai_request([{"role": "user", "content": "anything"}])
        await proc.process(ctx, req)
        assert ctx.metadata[CTX_ROUTELLM_TIER] == "strong"

    async def test_injected_classifier_skips_resource_cache(self):
        # Sanity: shutdown() is a no-op when the classifier was injected.
        proc = RouteLLMRequestProcessor(_config(), classifier=FakeClassifier(0.7))
        await proc.startup()
        await proc.shutdown()
        # No exception raised: shutdown does not unload classifier instances
        # injected by the caller.
