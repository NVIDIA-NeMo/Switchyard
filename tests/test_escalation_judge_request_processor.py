# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the escalation-router judge processor and its latch policy."""

from __future__ import annotations

import json
from typing import Any, cast

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)
from switchyard.lib.processors.escalation_judge_request_processor import (
    CTX_ESCALATION_VERDICT,
    EscalationJudgeConfig,
    EscalationJudgeRequestProcessor,
)
from switchyard.lib.processors.llm_classifier import ClassifierCompletion
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_affinity import CTX_SESSION_KEY, SessionAffinity
from switchyard_rust.core import ChatRequest


class _FakeJudgeClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    async def classify(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> ClassifierCompletion:
        self.calls.append({
            "model": model,
            "system_prompt": system_prompt,
            "request_summary": request_summary,
        })
        if isinstance(self.response, Exception):
            raise self.response
        return ClassifierCompletion(content=self.response)


def _verdict_json(escalate: bool, reason: str = "same error repeating") -> str:
    return json.dumps({"escalate": escalate, "reason": reason})


def _request(n_assistant_turns: int = 3, **body_overrides: Any) -> ChatRequest:
    """Chat request with enough prior assistant turns to pass the judge gate."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "fix the failing tests"},
    ]
    for i in range(n_assistant_turns):
        messages.append({"role": "assistant", "content": f"attempt {i + 1}"})
        messages.append({"role": "user", "content": f"tool result {i + 1}: error"})
    body: dict[str, Any] = {"model": "client-model", "messages": messages}
    body.update(body_overrides)
    return ChatRequest.openai_chat(cast(Any, body))


def _processor(
    response: str | Exception,
    *,
    affinity: SessionAffinity | None = None,
    session_key_depth: int = 0,
    **config_overrides: Any,
) -> tuple[EscalationJudgeRequestProcessor, _FakeJudgeClient, SessionAffinity]:
    affinity = affinity or SessionAffinity(enabled=True)
    fake = _FakeJudgeClient(response)
    processor = EscalationJudgeRequestProcessor(
        EscalationJudgeConfig(model="judge-model", **config_overrides),
        affinity=affinity,
        client=fake,
        session_key_depth=session_key_depth,
    )
    return processor, fake, affinity


async def test_no_escalate_routes_weak_without_pin() -> None:
    processor, fake, affinity = _processor(_verdict_json(False, ""))
    ctx = ProxyContext()
    req = _request()

    await processor.process(ctx, req)

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert len(fake.calls) == 1
    assert ctx.metadata[CTX_ESCALATION_VERDICT]["escalate"] is False
    assert await affinity.pinned(ProxyContext(), _request()) is None


async def test_escalate_routes_strong_and_latches() -> None:
    processor, fake, _ = _processor(_verdict_json(True))
    ctx = ProxyContext()

    await processor.process(ctx, _request())
    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    assert ctx.metadata[CTX_ESCALATION_VERDICT]["reason"] == "same error repeating"

    # Later turn of the same conversation: pinned, judge not called again.
    later_ctx = ProxyContext()
    await processor.process(later_ctx, _request(n_assistant_turns=5))
    assert later_ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    assert later_ctx.metadata[CTX_ESCALATION_VERDICT]["source"] == "pinned"
    assert len(fake.calls) == 1


async def test_judge_skipped_before_min_turn() -> None:
    processor, fake, _ = _processor(_verdict_json(True), min_judge_turn=3)
    ctx = ProxyContext()

    await processor.process(ctx, _request(n_assistant_turns=1))

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert fake.calls == []


async def test_judge_failure_fails_open_to_weak_without_pin() -> None:
    processor, fake, affinity = _processor(RuntimeError("judge down"))
    ctx = ProxyContext()

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert ctx.metadata[CTX_ESCALATION_VERDICT]["source"] == "fail_open"
    assert await affinity.pinned(ProxyContext(), _request()) is None
    # A garbage verdict is the same failure surface.
    processor, fake, affinity = _processor("not json at all")
    ctx = ProxyContext()
    await processor.process(ctx, _request())
    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert await affinity.pinned(ProxyContext(), _request()) is None


async def test_markdown_fenced_verdict_is_parsed() -> None:
    processor, _, _ = _processor("```json\n" + _verdict_json(True) + "\n```")
    ctx = ProxyContext()

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"


async def test_summary_has_header_anchors_and_truncated_window() -> None:
    processor, fake, _ = _processor(
        _verdict_json(False, ""),
        window_message_chars=80,
        first_user_chars=200,
    )
    req = _request(n_assistant_turns=3, messages=[
        {"role": "system", "content": "harness boilerplate " * 200},
        {"role": "user", "content": "fix the failing tests in tests/test_api.py"},
        {"role": "assistant", "content": "running pytest " + "x" * 500},
        {"role": "user", "content": "tool result: ImportError " + "y" * 500},
        {"role": "assistant", "content": "attempt 2"},
        {"role": "assistant", "content": "attempt 3"},
    ])

    await processor.process(ProxyContext(), req)

    summary = fake.calls[0]["request_summary"]
    assert summary.startswith("Conversation turn ")
    # Anchors survive: capped system + full task statement.
    assert "harness boilerplate" in summary
    assert "fix the failing tests in tests/test_api.py" in summary
    # Oversized system anchor and window messages are head/tail truncated.
    assert "...[trimmed]" in summary
    # Window content is present with role labels.
    assert "[assistant] attempt 3" in summary
    for line in summary.splitlines()[1:]:
        assert len(line) <= 2_000


async def test_global_cap_drops_oldest_window_messages_first() -> None:
    processor, fake, _ = _processor(
        _verdict_json(False, ""),
        max_request_chars=1_000,
        window_message_chars=200,
    )
    messages: list[dict[str, str]] = [
        {"role": "user", "content": "the task"},
    ]
    for i in range(20):
        messages.append({"role": "assistant", "content": f"attempt {i} " + "z" * 150})

    await processor.process(ProxyContext(), _request(messages=messages))

    summary = fake.calls[0]["request_summary"]
    assert len(summary) <= 1_000
    # The newest evidence survives; the oldest window lines are dropped.
    assert "attempt 19" in summary
    assert "attempt 5" not in summary


async def test_deep_session_key_diverges_on_early_responses() -> None:
    processor, _, _ = _processor(_verdict_json(False, ""), session_key_depth=2)

    def _trial(first_response: str) -> ChatRequest:
        return _request(messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "identical task text"},
            {"role": "assistant", "content": first_response},
            {"role": "user", "content": "tool result"},
            {"role": "assistant", "content": "next"},
        ])

    ctx_a, ctx_b = ProxyContext(), ProxyContext()
    await processor.process(ctx_a, _trial("I'll start by reading the tests."))
    await processor.process(ctx_b, _trial("Let me look at the repo layout."))

    assert ctx_a.metadata[CTX_SESSION_KEY] != ctx_b.metadata[CTX_SESSION_KEY]

    # The key is a prefix hash: the same conversation grown later keeps it.
    grown = _trial("I'll start by reading the tests.")
    grown.body["messages"].extend([
        {"role": "assistant", "content": "much later turn"},
        {"role": "user", "content": "more results"},
    ])
    ctx_grown = ProxyContext()
    await processor.process(ctx_grown, grown)
    assert ctx_grown.metadata[CTX_SESSION_KEY] == ctx_a.metadata[CTX_SESSION_KEY]


async def test_deep_key_incomplete_prefix_skips_affinity() -> None:
    processor, fake, affinity = _processor(
        _verdict_json(True),
        session_key_depth=4,
        min_judge_turn=1,
    )
    ctx = ProxyContext()
    # Only one post-first-user message: key prefix incomplete.
    req = _request(messages=[
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": "first response"},
    ])

    await processor.process(ctx, req)

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert CTX_SESSION_KEY not in ctx.metadata
    assert fake.calls == []


async def test_escalation_latch_isolated_by_deep_key() -> None:
    """Trial 2 of an identical task starts weak when trajectories diverge."""
    affinity = SessionAffinity(enabled=True)
    processor, fake, _ = _processor(
        _verdict_json(True), affinity=affinity, session_key_depth=1,
    )

    def _trial(first_response: str, extra_turns: int = 2) -> ChatRequest:
        messages = [
            {"role": "user", "content": "identical task text"},
            {"role": "assistant", "content": first_response},
        ]
        for i in range(extra_turns):
            messages.append({"role": "user", "content": f"tool result {i}"})
            messages.append({"role": "assistant", "content": f"attempt {i}"})
        return _request(messages=messages)

    ctx1 = ProxyContext()
    await processor.process(ctx1, _trial("trial one path"))
    assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"

    # Same task text, different first response: fresh latch, judge runs.
    fake.response = _verdict_json(False, "")
    ctx2 = ProxyContext()
    await processor.process(ctx2, _trial("trial two path"))
    assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert len(fake.calls) == 2
