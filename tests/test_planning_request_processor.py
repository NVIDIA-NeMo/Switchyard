# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the planning request processor (slim two-field schema)."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from switchyard.lib.processors.plan_execute import (
    CTX_PLANNER_DECISION,
    PlannerCompletion,
    PlannerDecision,
    PlanningConfig,
    PlanningError,
    PlanningRequestProcessor,
    PlanningTriggerMode,
    parse_planner_decision,
)
from switchyard.lib.processors.plan_execute.request_processor import (
    _completion_content,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

_PLAN_TEXT = (
    "I'll approach this in steps:\n"
    "1. Read /app/api_bug.py and the existing tests to map the bugs.\n"
    "2. Implement /app/api.py satisfying every test behavior.\n"
    "3. Run the test suite and iterate until it passes.\n\n"
    "Starting now."
)
_REVISION_TEXT = (
    "Pausing to reassess — the test harness uses pytest-asyncio rather than "
    "the asyncio fixture I assumed. Updated approach: switch to async fixtures "
    "and re-run. Resuming work."
)
_WRAPPED_PLAN = f"<plan>\n{_PLAN_TEXT}\n</plan>"
_WRAPPED_REVISION = f"<plan>\n{_REVISION_TEXT}\n</plan>"


class _FakePlannerClient:
    """Test double — captures the system_prompt + request_summary the
    processor selected, so tests can assert on planner-prompt routing."""

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    async def plan(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> PlannerCompletion:
        self.calls.append({
            "model": model,
            "system_prompt": system_prompt,
            "request_summary": request_summary,
        })
        if isinstance(self.response, Exception):
            raise self.response
        # ``usage=None`` mirrors the production code's tolerance — the
        # audit-line builder treats missing usage as zero counts.
        return PlannerCompletion(content=self.response, usage=None)


def _decision_with_plan_json(plan_text: str = _PLAN_TEXT) -> str:
    """Planner output that emits a plan (the slim shape)."""
    return json.dumps({"plan_needed": True, "plan_text": plan_text})


def _decision_no_plan_json() -> str:
    """Planner output declining to plan (fast path)."""
    return json.dumps({"plan_needed": False, "plan_text": ""})


def _openai_request_first_turn() -> ChatRequest:
    """Outbound conversation with no prior plan in history (initial-plan trigger)."""
    return ChatRequest.openai_chat(cast(Any, {
        "model": "client-model",
        "messages": [{"role": "user", "content": "refactor auth middleware"}],
    }))


def _openai_request_with_prior_plan() -> ChatRequest:
    """Outbound conversation where a prior assistant turn contains a
    ``<plan>...</plan>`` block — this is the revision-mode trigger."""
    prior_assistant = (
        "<plan>\nPrior approach: 1. inspect /app, 2. patch, 3. test.\n</plan>\n\n"
        "I checked /app/middleware.py — found the auth bug."
    )
    return ChatRequest.openai_chat(cast(Any, {
        "model": "client-model",
        "messages": [
            {"role": "user", "content": "refactor auth middleware"},
            {"role": "assistant", "content": prior_assistant},
            {"role": "user", "content": "the test is failing now"},
        ],
    }))


def _openai_request_with_assistant_no_plan() -> ChatRequest:
    """Outbound conversation with prior assistant turns but no
    ``<plan>...</plan>`` block — should still take the initial-plan
    path (the assistant turns alone don't trigger revision mode)."""
    return ChatRequest.openai_chat(cast(Any, {
        "model": "client-model",
        "messages": [
            {"role": "user", "content": "refactor auth middleware"},
            {"role": "assistant", "content": "Inspecting /app/middleware..."},
            {"role": "user", "content": "looking for the bug"},
        ],
    }))


async def test_planning_processor_prefills_wrapped_plan_on_initial_turn() -> None:
    """When the conversation has no prior plan, planner uses the initial
    system prompt and the resulting plan is appended as a trailing
    assistant turn wrapped in ``<plan>...</plan>``."""
    fake = _FakePlannerClient(_decision_with_plan_json())
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model", cadence_n=1),
        client=fake,
    )
    req = _openai_request_first_turn()
    ctx = ProxyContext()

    returned = await processor.process(ctx, req)

    assert returned is req

    decision = ctx.metadata[CTX_PLANNER_DECISION]
    assert isinstance(decision, PlannerDecision)
    assert decision.plan_needed is True
    assert decision.plan_text == _PLAN_TEXT  # the raw, unwrapped form

    # Prefill: last message is now an assistant turn carrying the
    # wrapped plan.
    messages = req.body["messages"]
    assert messages[-1] == {"role": "assistant", "content": _WRAPPED_PLAN}

    # Initial-prompt path: the planner saw the initial system prompt.
    assert "drafting the agent's opening thought" in fake.calls[0]["system_prompt"]
    assert fake.calls[0]["model"] == "planner-model"


async def test_planning_processor_uses_revision_prompt_when_prior_plan_in_history() -> None:
    """When an assistant turn already contains a ``<plan>...</plan>``
    block, the planner uses the revision system prompt and the prior
    plan is shown inline at the top of the request_summary."""
    fake = _FakePlannerClient(_decision_with_plan_json(_REVISION_TEXT))
    processor = PlanningRequestProcessor(
        # cadence_n=1: this test exercises a turn with a prior
        # assistant message, which would otherwise be cadence-skipped
        # at the default ``cadence_n=2``.
        PlanningConfig(model="planner-model", cadence_n=1),
        client=fake,
    )
    req = _openai_request_with_prior_plan()
    ctx = ProxyContext()

    await processor.process(ctx, req)

    # Revision-prompt path.
    assert "software architect reassessing" in fake.calls[0]["system_prompt"]
    # Prior plan was shown inline at the top of the summary so the
    # planner can decide whether to revise.
    request_summary = fake.calls[0]["request_summary"]
    assert request_summary.startswith("Prior plan:\n")
    assert "Prior approach: 1. inspect /app" in request_summary

    # The revision plan was prefilled in wrapped form at the tail.
    messages = req.body["messages"]
    assert messages[-1] == {"role": "assistant", "content": _WRAPPED_REVISION}


async def test_planning_processor_uses_initial_prompt_when_no_plan_in_history() -> None:
    """Prior assistant turns *without* a ``<plan>`` block do NOT trigger
    revision mode — the planner still uses the initial prompt because
    no plan has been emitted yet for this task."""
    fake = _FakePlannerClient(_decision_with_plan_json())
    processor = PlanningRequestProcessor(
        # cadence_n=1: this test has a prior assistant message which
        # would be cadence-skipped at the default ``cadence_n=2``.
        PlanningConfig(model="planner-model", cadence_n=1),
        client=fake,
    )
    req = _openai_request_with_assistant_no_plan()
    ctx = ProxyContext()

    await processor.process(ctx, req)

    # Initial-prompt path even though the conversation already has
    # assistant turns — the discriminator is "is there a <plan> block
    # in any prior assistant turn," not "any prior assistant turn."
    assert "drafting the agent's opening thought" in fake.calls[0]["system_prompt"]
    # No "Prior plan:" preamble in the request_summary.
    assert "Prior plan:" not in fake.calls[0]["request_summary"]


async def test_planning_processor_skips_injection_when_planner_declines() -> None:
    """``plan_needed=False`` leaves the request body untouched."""
    fake = _FakePlannerClient(_decision_no_plan_json())
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model", cadence_n=1),
        client=fake,
    )
    req = _openai_request_first_turn()
    ctx = ProxyContext()

    await processor.process(ctx, req)

    decision = ctx.metadata[CTX_PLANNER_DECISION]
    assert isinstance(decision, PlannerDecision)
    assert decision.plan_needed is False
    assert decision.plan_text == ""

    # Request body was not mutated.
    messages = req.body["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


async def test_planning_processor_can_be_disabled() -> None:
    fake = _FakePlannerClient(_decision_with_plan_json())
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model", trigger_mode=PlanningTriggerMode.DISABLED),
        client=fake,
    )
    req = _openai_request_first_turn()
    ctx = ProxyContext()

    await processor.process(ctx, req)

    assert fake.calls == []
    assert CTX_PLANNER_DECISION not in ctx.metadata
    assert req.body["messages"][0]["role"] == "user"


def test_parse_planner_decision_accepts_json_fence() -> None:
    """``parse_planner_decision`` strips a leading markdown fence."""
    decision = parse_planner_decision(f"```json\n{_decision_with_plan_json()}\n```")
    assert decision.plan_needed is True
    assert decision.plan_text == _PLAN_TEXT


def test_parse_planner_decision_accepts_no_plan_shape() -> None:
    decision = parse_planner_decision(_decision_no_plan_json())
    assert decision.plan_needed is False
    assert decision.plan_text == ""


def test_parse_planner_decision_rejects_inconsistent_shapes() -> None:
    """``plan_needed=True`` requires non-empty ``plan_text``."""
    with pytest.raises(PlanningError):
        parse_planner_decision(json.dumps({"plan_needed": True, "plan_text": ""}))


def test_planner_decision_normalizes_false_with_spurious_text() -> None:
    """``plan_needed=False`` with non-empty ``plan_text`` is normalized
    to empty — we trust the gate."""
    decision = PlannerDecision(plan_needed=False, plan_text="leaked")
    assert decision.plan_text == ""


def _completion(content: Any, tool_arguments: str | None = None) -> Any:
    """Build a minimal OpenAI-SDK-shaped completion for ``_completion_content``."""
    message = SimpleNamespace(content=content, tool_calls=None)
    if tool_arguments is not None:
        message.tool_calls = [
            SimpleNamespace(function=SimpleNamespace(arguments=tool_arguments)),
        ]
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_completion_content_prefers_content() -> None:
    """When ``content`` is present it wins over any tool call."""
    result = _completion(content="real content", tool_arguments='{"ignored": true}')
    assert _completion_content(result) == "real content"


def test_completion_content_falls_back_to_tool_call_arguments() -> None:
    """Anthropic-via-LiteLLM: empty ``content``, JSON in ``tool_calls`` — the
    forced-tool path the planner must read."""
    plan_json = _decision_with_plan_json()
    result = _completion(content="", tool_arguments=plan_json)
    raw = _completion_content(result)
    # Round-trips through the same parser the production path uses.
    decision = parse_planner_decision(raw)
    assert decision.plan_needed is True
    assert decision.plan_text == _PLAN_TEXT


def test_completion_content_handles_none_content_with_tool_call() -> None:
    """``content=None`` (not just empty string) still falls back to the tool call."""
    result = _completion(content=None, tool_arguments=_decision_no_plan_json())
    assert parse_planner_decision(_completion_content(result)).plan_needed is False


def test_completion_content_raises_when_no_content_and_no_tool_call() -> None:
    """Empty content with no tool call is a genuine empty completion."""
    with pytest.raises(PlanningError):
        _completion_content(_completion(content="", tool_arguments=None))


async def test_planning_processor_fail_open_continues_without_plan() -> None:
    fake = _FakePlannerClient("not json")
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model", cadence_n=1),
        client=fake,
    )
    ctx = ProxyContext()

    await processor.process(ctx, _openai_request_first_turn())

    assert CTX_PLANNER_DECISION not in ctx.metadata


async def test_planning_processor_fail_closed_raises() -> None:
    fake = _FakePlannerClient(RuntimeError("boom"))
    processor = PlanningRequestProcessor(
        PlanningConfig(
            model="planner-model",
            fail_open=False,
        ),
        client=fake,
    )

    with pytest.raises(PlanningError):
        await processor.process(ProxyContext(), _openai_request_first_turn())


async def test_planning_processor_prefills_anthropic_messages() -> None:
    """Anthropic wire format uses the same prefill mechanic — append the
    plan as an assistant turn at the tail of ``messages``.  The
    top-level ``system`` field stays untouched (it's terminus-2's
    behavior contract, not a plan slot)."""
    fake = _FakePlannerClient(_decision_with_plan_json())
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model"),
        client=fake,
    )
    req = ChatRequest.anthropic(cast(Any, {
        "model": "claude-test",
        "max_tokens": 128,
        "system": "Existing system contract — preserved untouched.",
        "messages": [{"role": "user", "content": "refactor auth middleware"}],
    }))

    await processor.process(ProxyContext(), req)

    # System field stays the original behavior contract.
    assert req.body["system"] == "Existing system contract — preserved untouched."
    # Plan appears as a prefilled (wrapped) assistant turn at the tail.
    messages = req.body["messages"]
    assert messages[-1] == {"role": "assistant", "content": _WRAPPED_PLAN}


async def test_planning_processor_prefills_responses_input_string() -> None:
    """Responses API with a string ``input``: wrap into list form and
    append the plan as an assistant prefill turn."""
    fake = _FakePlannerClient(_decision_with_plan_json())
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model"),
        client=fake,
    )
    req = ChatRequest.openai_responses(cast(Any, {
        "model": "gpt-test",
        "input": "refactor auth middleware",
    }))

    await processor.process(ProxyContext(), req)

    input_value = req.body["input"]
    assert isinstance(input_value, list)
    assert input_value[0] == {"role": "user", "content": "refactor auth middleware"}
    assert input_value[-1] == {"role": "assistant", "content": _WRAPPED_PLAN}


class _UsagePlannerClient:
    """Planner double that returns a token-usage block, so the producer has
    counts to record for the intake sink."""

    def __init__(self, content: str, usage: object) -> None:
        self.content = content
        self.usage = usage

    async def plan(
        self,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> PlannerCompletion:
        return PlannerCompletion(content=self.content, usage=self.usage)


async def test_process_stashes_planner_submodel_call_for_intake() -> None:
    """A successful planner call is recorded for the intake sink to emit."""
    fake = _UsagePlannerClient(
        _decision_with_plan_json(),
        SimpleNamespace(
            prompt_tokens=512,
            completion_tokens=64,
            prompt_tokens_details=SimpleNamespace(cached_tokens=8),
        ),
    )
    processor = PlanningRequestProcessor(
        PlanningConfig(model="planner-model", cadence_n=1),
        client=fake,
    )
    ctx = ProxyContext()

    await processor.process(ctx, _openai_request_first_turn())

    assert ctx.submodel_calls == [
        {
            "model": "planner-model",
            "prompt_tokens": 512,
            "completion_tokens": 64,
            "cached_tokens": 8,
            "router_type": "plan_execute",
            "routed_to": "planner",
        }
    ]
