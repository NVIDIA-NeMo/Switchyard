# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental LLM classifier request processor."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from switchyard.lib.endpoints import outcome_metrics
from switchyard.lib.processors.llm_classifier import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    ClassifierCompletion,
    LLMClassifierConfig,
    LLMClassifierError,
    LLMClassifierRequestProcessor,
    ReasonCode,
    RouteSignals,
    RouteTier,
    TaskType,
    parse_route_signals,
)
from switchyard.lib.proxy_context import CTX_SWITCHYARD_FALLBACK, ProxyContext
from switchyard.lib.session_affinity import SessionAffinity
from switchyard_rust.core import ChatRequest


class _FakeUsage:
    """Duck-typed OpenAI-shaped usage object for tests."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakePromptTokensDetails(cached_tokens)


class _FakePromptTokensDetails:
    def __init__(self, cached_tokens: int) -> None:
        self.cached_tokens = cached_tokens


class _FakeClassifierClient:
    def __init__(
        self,
        response: str | Exception,
        *,
        usage: _FakeUsage | None = None,
    ) -> None:
        self.response = response
        self.usage = usage
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
        return ClassifierCompletion(content=self.response, usage=self.usage)


@pytest.fixture(autouse=True)
def _reset_classifier_fail_open_metrics() -> None:
    outcome_metrics._reset_for_tests()
    yield
    outcome_metrics._reset_for_tests()


def _messages_with_prior_assistant_turns(n: int) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": "debug this traceback"}]
    for i in range(n):
        messages.append({"role": "assistant", "content": f"turn {i + 1}"})
    return messages


def _request(**body_overrides: Any) -> ChatRequest:
    body: dict[str, Any] = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "debug this traceback"}],
    }
    body.update(body_overrides)
    return ChatRequest.openai_chat(cast(Any, body))


def _signals_json(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "task_type": "debugging",
        "complexity": "complex",
        "reasoning_depth": "multi_step",
        "tool_planning_required": False,
        "precision_requirement": "high",
        "context_dependency": "conversation",
        "structured_output_risk": "low",
        "recommended_tier": "complex",
        "confidence": 0.86,
        "reason_code": "debugging",
        "abstain": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


async def test_request_processor_stamps_classifier_signals() -> None:
    fake = _FakeClassifierClient(_signals_json())
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )
    req = _request()
    ctx = ProxyContext()

    returned = await processor.process(ctx, req)

    assert returned is req
    signals = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
    assert isinstance(signals, RouteSignals)
    assert signals.task_type is TaskType.DEBUGGING
    assert signals.recommended_tier is RouteTier.COMPLEX
    assert signals.reason_code is ReasonCode.DEBUGGING
    assert fake.calls[0]["model"] == "router-model"
    assert '"request_type": "openai_chat"' in fake.calls[0]["request_summary"]
    assert "debug this traceback" in fake.calls[0]["request_summary"]


async def test_classifier_skips_llm_call_when_session_pinned() -> None:
    """Once a conversation is pinned, the classifier runs once per task, not per turn."""
    fake = _FakeClassifierClient(_signals_json())
    affinity = SessionAffinity(enabled=True)
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
        affinity=affinity,
    )
    req = _request()

    # First turn: not yet pinned → the classifier makes its LLM call.
    await processor.process(ProxyContext(), req)
    assert len(fake.calls) == 1

    # The tier selector would pin the conversation after the first turn.
    await affinity.pin(ProxyContext(), req, "weak")

    # Later turns of the same conversation skip the LLM call entirely.
    ctx2 = ProxyContext()
    returned = await processor.process(ctx2, req)
    assert returned is req
    assert len(fake.calls) == 1  # unchanged — classified exactly once
    assert CTX_DETERMINISTIC_ROUTE_SIGNALS not in ctx2.metadata


async def test_classifier_runs_until_warmup_pin_exists() -> None:
    """Warmup keeps the classifier live until the selector can commit a pin."""
    fake = _FakeClassifierClient(_signals_json())
    affinity = SessionAffinity(enabled=True, warmup_turns=2)
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
        affinity=affinity,
    )

    req1 = _request()
    await processor.process(ProxyContext(), req1)
    await affinity.pin(ProxyContext(), req1, "weak")

    req2 = _request(messages=_messages_with_prior_assistant_turns(1))
    await processor.process(ProxyContext(), req2)
    await affinity.pin(ProxyContext(), req2, "weak")

    req3 = _request(messages=_messages_with_prior_assistant_turns(2))
    await processor.process(ProxyContext(), req3)
    await affinity.pin(ProxyContext(), req3, "weak")

    assert len(fake.calls) == 3

    req4 = _request(messages=_messages_with_prior_assistant_turns(3))
    ctx4 = ProxyContext()
    returned = await processor.process(ctx4, req4)
    assert returned is req4
    assert len(fake.calls) == 3
    assert CTX_DETERMINISTIC_ROUTE_SIGNALS not in ctx4.metadata


async def test_classifier_runs_when_affinity_disabled() -> None:
    """A disabled affinity coordinator never gates the classifier."""
    fake = _FakeClassifierClient(_signals_json())
    affinity = SessionAffinity(enabled=False)
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
        affinity=affinity,
    )
    req = _request()
    await affinity.pin(ProxyContext(), req, "weak")  # no-op while disabled

    await processor.process(ProxyContext(), req)
    await processor.process(ProxyContext(), req)
    assert len(fake.calls) == 2  # classifier runs every turn


def test_parse_route_signals_accepts_json_fence() -> None:
    signals = parse_route_signals(f"```json\n{_signals_json()}\n```")

    assert signals.task_type is TaskType.DEBUGGING
    assert signals.confidence == 0.86


async def test_request_processor_fail_open_stamps_abstain_signals() -> None:
    fake = _FakeClassifierClient("not json")
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )
    ctx = ProxyContext()

    await processor.process(ctx, _request())

    signals = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
    assert isinstance(signals, RouteSignals)
    assert signals.abstain is True
    assert signals.confidence == 0.0
    assert signals.reason_code is ReasonCode.AMBIGUOUS
    assert ctx.metadata[CTX_SWITCHYARD_FALLBACK] == "classifier_error"

    metrics = "\n".join(outcome_metrics.render_lines())
    assert 'switchyard_classifier_fail_open_triggered_total{reason="parse_error"} 1' in metrics
    assert 'switchyard_classifier_fail_open_triggered_total{reason="timeout"} 0' in metrics


async def test_fail_open_annotates_signals_dump(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _FakeClassifierClient(RuntimeError("upstream timeout"))
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model", dump_signals_to_stderr=True),
        client=fake,
    )
    await processor.process(ProxyContext(), _request())

    line = next(
        ln for ln in capsys.readouterr().err.splitlines()
        if ln.startswith("classifier_signals=")
    )
    payload = json.loads(line.removeprefix("classifier_signals="))
    assert payload["fail_open"] is True
    assert "upstream timeout" in payload["error"]
    assert payload["abstain"] is True
    metrics = "\n".join(outcome_metrics.render_lines())
    assert 'switchyard_classifier_fail_open_triggered_total{reason="timeout"} 1' in metrics


class _StatusClassifierError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream returned {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (_StatusClassifierError(500), "upstream_5xx"),
        (_StatusClassifierError(401), "upstream_4xx"),
        (RuntimeError("SSL CERTIFICATE_VERIFY_FAILED"), "ssl"),
    ],
)
async def test_fail_open_records_bounded_reason_metric(
    error: Exception,
    reason: str,
) -> None:
    fake = _FakeClassifierClient(error)
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )

    await processor.process(ProxyContext(), _request())

    metrics = "\n".join(outcome_metrics.render_lines())
    assert (
        f'switchyard_classifier_fail_open_triggered_total{{reason="{reason}"}} 1'
        in metrics
    )


async def test_request_processor_fail_closed_raises() -> None:
    fake = _FakeClassifierClient(RuntimeError("boom"))
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model", fail_open=False),
        client=fake,
    )

    with pytest.raises(LLMClassifierError):
        await processor.process(ProxyContext(), _request())


async def test_openai_client_sends_strict_json_schema_response_format() -> None:
    from switchyard.lib.processors.llm_classifier.request_processor import (
        OpenAIChatLLMClassifierClient,
    )
    from switchyard.lib.processors.llm_classifier.signals import (
        CodingAgentRouteDecision,
    )

    captured: dict[str, Any] = {}

    class _StubLLMClient:
        async def acompletion(self, **kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Msg:
                content = json.dumps({
                    "recommended_tier": "medium",
                    "confidence": 0.7,
                    "abstain": False,
                    "turn_type": "exploration",
                    "code_modification_scope": "none",
                    "tool_call_count_estimate": 0,
                    "requires_codebase_context": False,
                })

            class _Choice:
                message = _Msg()

            class _Result:
                choices = [_Choice()]

            return _Result()

    client = OpenAIChatLLMClassifierClient(
        cast(Any, _StubLLMClient()),
        signal_schema=CodingAgentRouteDecision,
        structured_output_mode="json_schema",
    )

    await client.classify(model="m", system_prompt="sp", request_summary="rs")

    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "CodingAgentRouteDecision"
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    # Every property is required in strict mode, even ones with pydantic defaults.
    assert set(schema["required"]) == set(schema["properties"].keys())
    # Banned-by-strict-mode keywords are stripped.
    assert "minimum" not in schema["properties"]["confidence"]
    assert "default" not in schema["properties"]["abstain"]


async def test_openai_client_json_object_mode_skips_schema() -> None:
    from switchyard.lib.processors.llm_classifier.request_processor import (
        OpenAIChatLLMClassifierClient,
    )

    captured: dict[str, Any] = {}

    class _StubLLMClient:
        async def acompletion(self, **kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Msg:
                content = "{}"

            class _Choice:
                message = _Msg()

            class _Result:
                choices = [_Choice()]

            return _Result()

    client = OpenAIChatLLMClassifierClient(
        cast(Any, _StubLLMClient()),
        structured_output_mode="json_object",
    )

    await client.classify(model="m", system_prompt="sp", request_summary="rs")

    assert captured["response_format"] == {"type": "json_object"}


async def test_request_summary_is_truncated() -> None:
    fake = _FakeClassifierClient(_signals_json())
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model", max_request_chars=256),
        client=fake,
    )

    await processor.process(
        ProxyContext(),
        _request(messages=[{"role": "user", "content": "x" * 1_000}]),
    )

    summary = fake.calls[0]["request_summary"]
    assert len(summary) <= 256
    assert summary.endswith("...<truncated>")


async def test_request_summary_drops_prior_turns_and_tool_schemas() -> None:
    """Slimmed summary keeps system + last user + tool names; drops the rest.

    Why: classifier tax on multi-turn agent traffic (TB, Claude Code) was
    ~17s/req with full tool schemas + prior turns. Strip non-signal bulk so
    Nemotron processes O(100) tokens instead of O(4k).
    """
    fake = _FakeClassifierClient(_signals_json())
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )

    await processor.process(
        ProxyContext(),
        _request(
            messages=[
                {"role": "system", "content": "you are an agent"},
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "PRIOR_ASSISTANT_BULK"},
                {"role": "tool", "tool_call_id": "x", "content": "TOOL_RESULT_BULK"},
                {"role": "user", "content": "the latest ask"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "run_bash",
                        "description": "run a shell command",
                        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                    },
                },
            ],
        ),
    )

    summary = fake.calls[0]["request_summary"]
    assert "you are an agent" in summary
    assert "the latest ask" in summary
    assert "run_bash" in summary
    assert "run a shell command" in summary
    assert "PRIOR_ASSISTANT_BULK" not in summary
    assert "TOOL_RESULT_BULK" not in summary
    assert "parameters" not in summary  # tool input schema stripped


async def test_request_summary_preserves_first_user_for_terminus2_shape() -> None:
    """terminus-2 bundles task framing into the FIRST user message (not system).

    On turn 2+ the last user message is just terminal output with no task
    context. Without preserving the first user message, the classifier
    would be blind to what task is being solved — confirmed against live
    classifier-mode runs where the slim summary on turn 2+ contained only
    "New Terminal Output: ..." with no framing.
    """
    fake = _FakeClassifierClient(_signals_json())
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )

    await processor.process(
        ProxyContext(),
        _request(
            messages=[
                # No system role — framing is here as a user message
                {"role": "user", "content": "TASK_FRAMING: solve permission inheritance"},
                {"role": "assistant", "content": "PRIOR_ASSISTANT_BULK"},
                {"role": "user", "content": "New Terminal Output: mkdir done"},
            ],
        ),
    )

    summary = fake.calls[0]["request_summary"]
    # Both first and last user messages survive
    assert "TASK_FRAMING" in summary
    assert "New Terminal Output: mkdir done" in summary
    # Prior assistant turn is still dropped
    assert "PRIOR_ASSISTANT_BULK" not in summary


async def test_request_summary_dedupes_single_user_turn() -> None:
    """Single-turn case: first and last user are the same message — no dup."""
    fake = _FakeClassifierClient(_signals_json())
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )

    await processor.process(
        ProxyContext(),
        _request(messages=[{"role": "user", "content": "ONLY_TURN"}]),
    )

    summary = fake.calls[0]["request_summary"]
    # Should appear exactly once in the serialized body
    assert summary.count("ONLY_TURN") == 1


async def test_request_summary_drops_anthropic_tool_input_schema() -> None:
    fake = _FakeClassifierClient(_signals_json())
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-model"),
        client=fake,
    )

    await processor.process(
        ProxyContext(),
        _request(
            messages=[{"role": "user", "content": "anthropic-style request"}],
            tools=[
                {
                    "name": "search",
                    "description": "search the web",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            ],
        ),
    )

    summary = fake.calls[0]["request_summary"]
    assert "search the web" in summary
    assert "input_schema" not in summary


async def test_classifier_records_tokens_into_stats_when_accumulator_wired() -> None:
    """Classifier overhead lands in the dedicated classifier bucket.

    Without this wiring the classifier's LLM spend was invisible to
    ``/v1/routing/stats`` — exactly the gap that flattered the
    classifier-vs-force-strong cost comparison in TB-lite runs.
    """
    from switchyard.lib.stats_accumulator import StatsAccumulator

    accumulator = StatsAccumulator()
    fake = _FakeClassifierClient(
        _signals_json(),
        usage=_FakeUsage(prompt_tokens=420, completion_tokens=80, cached_tokens=10),
    )
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-classifier"),
        client=fake,
        stats_accumulator=accumulator,
    )

    await processor.process(ProxyContext(), _request())
    await processor.process(ProxyContext(), _request())

    snapshot = accumulator.snapshot_sync()
    # Backend bucket untouched — the classifier doesn't show up among
    # routed-traffic models even though it ran twice.
    assert snapshot["models"] == {}
    assert snapshot["total_requests"] == 0
    # Classifier block carries the per-call totals + cost.
    classifier = snapshot["classifier"]
    assert classifier["total_requests"] == 2
    assert classifier["models"]["router-classifier"]["calls"] == 2
    assert classifier["models"]["router-classifier"]["prompt_tokens"] == 840
    assert classifier["models"]["router-classifier"]["completion_tokens"] == 160
    assert classifier["models"]["router-classifier"]["cached_tokens"] == 20
    # Latency was recorded — at least one sample in the histogram.
    assert classifier["models"]["router-classifier"]["model_call_latency"]["count"] == 2


async def test_classifier_skips_stats_recording_when_no_accumulator() -> None:
    """Backwards compat: omitting the accumulator must not raise."""
    fake = _FakeClassifierClient(
        _signals_json(),
        usage=_FakeUsage(prompt_tokens=100, completion_tokens=20),
    )
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-classifier"),
        client=fake,
    )

    # No accumulator wired; processor still resolves successfully.
    await processor.process(ProxyContext(), _request())


async def test_classifier_records_error_on_failure_path() -> None:
    """Fail-open path stamps abstain signals and records the error.

    The classifier call failed, so there are no tokens to charge — but the
    failure itself must surface on ``/v1/routing/stats`` so a silent
    fail-open does not hide the failure rate.  ``total_requests`` and
    ``total_errors`` both increment; per-model ``calls`` stays at zero
    (it counts completed, token-bearing calls), and per-model ``errors``
    increments so ``errors / (calls + errors)`` reads as the true
    failure rate.
    """
    from switchyard.lib.stats_accumulator import StatsAccumulator

    accumulator = StatsAccumulator()
    fake = _FakeClassifierClient(RuntimeError("upstream 503"))
    processor = LLMClassifierRequestProcessor(
        LLMClassifierConfig(model="router-classifier"),
        client=fake,
        stats_accumulator=accumulator,
    )

    await processor.process(ProxyContext(), _request())

    snapshot = accumulator.snapshot_sync()
    classifier = snapshot["classifier"]
    assert classifier["total_requests"] == 1
    assert classifier["total_errors"] == 1
    model_stats = classifier["models"]["router-classifier"]
    assert model_stats["calls"] == 0
    assert model_stats["errors"] == 1
    assert model_stats["prompt_tokens"] == 0
    assert model_stats["completion_tokens"] == 0
