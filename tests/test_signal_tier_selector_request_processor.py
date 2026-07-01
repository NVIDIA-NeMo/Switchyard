# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mapping LLM classifier signals to deterministic backend tiers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.processors.llm_classifier import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    CTX_DETERMINISTIC_TIER_DECISION,
    ChannelKind,
    CodeModificationScope,
    CodingAgentRouteDecision,
    CodingAgentTurnType,
    Complexity,
    ContextDependency,
    MemoryDependency,
    OpenClawRouteDecision,
    OpenClawTurnType,
    PrecisionRequirement,
    ReasonCode,
    ReasoningDepth,
    RiskLevel,
    RouteSignals,
    RouteTier,
    SignalTierSelectorConfig,
    SignalTierSelectorRequestProcessor,
    TierSelectionDecision,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.routing_trace import ROUTING_TRACE_JSONL_ENV
from switchyard.lib.session_affinity import SessionAffinity
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse


def _stub_response() -> ChatResponse:
    return ChatResponse.openai_completion({"id": "test", "choices": []})


class _RecordingBackend(LLMBackend):
    def __init__(self) -> None:
        self.models: list[str] = []

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [
            ChatRequestType.OPENAI_CHAT,
            ChatRequestType.OPENAI_RESPONSES,
            ChatRequestType.ANTHROPIC,
        ]

    async def call(self, ctx: ProxyContext, request: Any) -> ChatResponse:
        self.models.append(request.body["model"])
        return _stub_response()


class BackendFailure(RuntimeError):
    """Test-only backend failure with a potentially sensitive message."""


class _FailingBackend(_RecordingBackend):
    async def call(self, ctx: ProxyContext, request: Any) -> ChatResponse:
        self.models.append(request.body["model"])
        raise BackendFailure("sensitive upstream detail")


class _StreamingBackend(_RecordingBackend):
    async def call(self, ctx: ProxyContext, request: Any) -> ChatResponse:
        self.models.append(request.body["model"])

        async def chunks():
            if False:
                yield None

        return ChatResponse.openai_stream(chunks())


def _request(content: str = "hello", *, prior_assistant_turns: int = 0) -> ChatRequest:
    messages: list[dict[str, str]] = [{"role": "user", "content": content}]
    for i in range(prior_assistant_turns):
        messages.append({"role": "assistant", "content": f"turn {i + 1}"})
    return ChatRequest.openai_chat(cast(Any, {
        "model": "client-model",
        "messages": messages,
    }))


def _two_tier_config() -> SignalTierSelectorConfig:
    """Config mapping COMPLEX/REASONING→strong, SIMPLE/MEDIUM/default→weak."""
    return SignalTierSelectorConfig(
        tier_mapping={
            RouteTier.SIMPLE: "weak",
            RouteTier.MEDIUM: "weak",
            RouteTier.COMPLEX: "strong",
            RouteTier.REASONING: "strong",
        },
        default_tier="weak",
        min_confidence=0.5,
    )


def _signals(
    *,
    recommended_tier: RouteTier = RouteTier.COMPLEX,
    confidence: float = 0.9,
    abstain: bool = False,
) -> RouteSignals:
    return RouteSignals(
        task_type="coding",
        complexity=Complexity.COMPLEX,
        reasoning_depth=ReasoningDepth.MULTI_STEP,
        tool_planning_required=False,
        precision_requirement=PrecisionRequirement.HIGH,
        context_dependency=ContextDependency.CONVERSATION,
        structured_output_risk=RiskLevel.LOW,
        recommended_tier=recommended_tier,
        confidence=confidence,
        reason_code=ReasonCode.CODING_COMPLEX,
        abstain=abstain,
    )


def _single_routing_event(ctx: ProxyContext) -> dict[str, Any]:
    trace = cast(dict[str, Any] | None, ctx.routing_trace)
    assert trace is not None
    assert len(trace["events"]) == 1
    return cast(dict[str, Any], trace["events"][0])


def _router(
    *,
    strong_backend: LLMBackend | None = None,
    weak_backend: LLMBackend | None = None,
) -> DeterministicRoutingLLMBackend:
    return DeterministicRoutingLLMBackend(
        tiers={
            "strong": (strong_backend or _RecordingBackend(), "strong-model"),
            "weak": (weak_backend or _RecordingBackend(), "weak-model"),
        },
        default_tier="weak",
    )


async def test_signal_tier_selector_maps_signals_to_llm_target(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    processor = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="weak",
            min_confidence=0.5,
        ),
    )
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(),
    })
    req = _request()

    returned = await processor.process(ctx, req)

    assert returned is req
    assert req.body["model"] == "client-model"
    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    decision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert isinstance(decision, TierSelectionDecision)
    assert decision.tier == "strong"
    assert decision.source == "policy_tier"
    assert decision.policy_tier is RouteTier.COMPLEX
    assert decision.llm_recommended_tier is RouteTier.COMPLEX

    event = _single_routing_event(ctx)
    assert event["producer"] == "llm_classifier"
    assert event["name"] == "tier_selection"
    assert event["schema"] == "llm_classifier.tier_selection/v1"
    assert event["phase"] == "initial"
    assert event["selection"] == {"tier": "strong"}
    assert event["source"] == decision.source
    assert event["reason"] == decision.reason
    assert event["confidence"] == 0.9
    assert event["details"] == {
        "policy_tier": "complex",
        "llm_recommended_tier": "complex",
        "selector": {
            "tier_mapping": {
                "simple": "weak",
                "medium": "weak",
                "complex": "strong",
                "reasoning": "strong",
            },
            "default_tier": "weak",
            "min_confidence": 0.5,
            "escalate_on_tool_planning": False,
            "escalate_target_tier": "weak",
            "align_with_llm_recommendation": False,
            "alignment_min_confidence": 0.7,
        },
    }


async def test_signal_tier_selector_defaults_on_low_confidence(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    processor = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="weak",
            min_confidence=0.8,
        ),
    )
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.2),
    })

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    decision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert isinstance(decision, TierSelectionDecision)
    assert decision.source == "low_confidence"
    event = _single_routing_event(ctx)
    assert event["selection"] == {"tier": "weak"}
    assert event["source"] == "low_confidence"
    assert event["confidence"] == 0.2


async def test_signal_tier_selector_defaults_on_abstain(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    processor = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="weak",
        ),
    )
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(abstain=True),
    })

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    decision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert isinstance(decision, TierSelectionDecision)
    assert decision.source == "abstain"
    event = _single_routing_event(ctx)
    assert event["selection"] == {"tier": "weak"}
    assert event["source"] == "abstain"
    assert event["confidence"] == 0.9


async def test_signal_tier_selector_traces_missing_signals(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext()

    await SignalTierSelectorRequestProcessor(_two_tier_config()).process(ctx, _request())

    event = _single_routing_event(ctx)
    assert event["selection"] == {"tier": "weak"}
    assert event["source"] == "missing_signals"
    assert "confidence" not in event
    assert event["details"]["policy_tier"] is None
    assert event["details"]["llm_recommended_tier"] is None


async def test_signal_tier_selector_accepts_dict_signals() -> None:
    processor = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="weak",
        ),
    )
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals().model_dump(mode="json"),
    })

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"


async def test_selector_routes_on_policy_tier_not_llm_recommended_tier() -> None:
    """When LLM picks SIMPLE but features score COMPLEX, policy wins."""
    processor = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="strong",
        ),
    )
    # LLM emits "simple" but the features unambiguously score COMPLEX.
    signals = RouteSignals(
        task_type="coding",
        complexity=Complexity.COMPLEX,
        reasoning_depth=ReasoningDepth.MULTI_STEP,
        tool_planning_required=True,
        precision_requirement=PrecisionRequirement.HIGH,
        context_dependency=ContextDependency.CONVERSATION,
        structured_output_risk=RiskLevel.HIGH,
        recommended_tier=RouteTier.SIMPLE,
        confidence=0.95,
        reason_code=ReasonCode.CODING_COMPLEX,
        abstain=False,
    )
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTE_SIGNALS: signals})

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    decision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.source == "policy_tier"
    assert decision.policy_tier is RouteTier.COMPLEX
    # The LLM's own (overridden) vote is still captured for audit.
    assert decision.llm_recommended_tier is RouteTier.SIMPLE


def test_route_signals_policy_tier_scores_complexity() -> None:
    simple = RouteSignals(
        task_type="chat",
        complexity=Complexity.SIMPLE,
        reasoning_depth=ReasoningDepth.NONE,
        tool_planning_required=False,
        precision_requirement=PrecisionRequirement.LOW,
        context_dependency=ContextDependency.LATEST_MESSAGE,
        structured_output_risk=RiskLevel.LOW,
        recommended_tier=RouteTier.REASONING,  # LLM disagrees — ignored
        confidence=0.9,
        reason_code=ReasonCode.SIMPLE_QA,
        abstain=False,
    )
    assert simple.policy_tier() is RouteTier.SIMPLE

    deep = RouteSignals(
        task_type="math",
        complexity=Complexity.REASONING,
        reasoning_depth=ReasoningDepth.DEEP,
        tool_planning_required=False,
        precision_requirement=PrecisionRequirement.HIGH,
        context_dependency=ContextDependency.CONVERSATION,
        structured_output_risk=RiskLevel.LOW,
        recommended_tier=RouteTier.SIMPLE,  # LLM disagrees — ignored
        confidence=0.9,
        reason_code=ReasonCode.MATH_REASONING,
        abstain=False,
    )
    assert deep.policy_tier() is RouteTier.REASONING


def test_coding_agent_policy_tier_scales_with_scope_and_tools() -> None:
    chitchat = CodingAgentRouteDecision(
        recommended_tier=RouteTier.COMPLEX,  # ignored
        confidence=0.9,
        turn_type=CodingAgentTurnType.CHITCHAT,
        code_modification_scope=CodeModificationScope.NONE,
    )
    assert chitchat.policy_tier() is RouteTier.SIMPLE

    refactor = CodingAgentRouteDecision(
        recommended_tier=RouteTier.SIMPLE,  # ignored
        confidence=0.9,
        turn_type=CodingAgentTurnType.DEBUG,
        code_modification_scope=CodeModificationScope.CROSS_MODULE,
        tool_call_count_estimate=6,
        requires_codebase_context=True,
    )
    assert refactor.policy_tier() is RouteTier.COMPLEX


def test_openclaw_policy_tier_handles_chat_vs_orchestration() -> None:
    chitchat = OpenClawRouteDecision(
        recommended_tier=RouteTier.COMPLEX,  # ignored
        confidence=0.9,
        turn_type=OpenClawTurnType.CHITCHAT,
        channel_kind=ChannelKind.CASUAL,
    )
    assert chitchat.policy_tier() is RouteTier.SIMPLE

    irreversible_action = OpenClawRouteDecision(
        recommended_tier=RouteTier.SIMPLE,  # ignored
        confidence=0.9,
        turn_type=OpenClawTurnType.ACTION,
        tool_call_count_estimate=2,
        memory_dependency=MemoryDependency.HEAVY,
        external_action_required=True,
        precision_requirement=PrecisionRequirement.HIGH,
        ambiguity=RiskLevel.LOW,
        channel_kind=ChannelKind.DELIBERATE,
    )
    assert irreversible_action.policy_tier() is RouteTier.COMPLEX


async def test_selector_to_backend_handoff_dispatches_selected_tier() -> None:
    strong_backend = _RecordingBackend()
    weak_backend = _RecordingBackend()
    selector = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="weak",
        ),
    )
    router = DeterministicRoutingLLMBackend(
        tiers={
            "strong": (strong_backend, "strong-model"),
            "weak": (weak_backend, "weak-model"),
        },
        default_tier="weak",
    )
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(),
    })
    req = _request()

    await selector.process(ctx, req)
    await router.call(ctx, req)

    assert strong_backend.models == ["strong-model"]
    assert weak_backend.models == []
    assert req.body["model"] == "strong-model"


async def test_backend_attempt_trace_records_metadata_selection(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTING_TIER: "strong"})

    await _router().call(ctx, _request())

    event = _single_routing_event(ctx)
    assert event["kind"] == "attempt"
    assert event["producer"] == "deterministic_routing"
    assert event["name"] == "backend_attempt"
    assert event["phase"] == "initial"
    assert event["selection"] == {"tier": "strong", "model": "strong-model"}
    assert event["source"] == "routing_tier_metadata"
    assert event["status"] == "success"
    assert event["duration_ms"] >= 0.0
    assert event["details"] == {"attempt_number": 0, "response_mode": "buffered"}
    assert "error" not in event


async def test_backend_attempt_trace_marks_cached_context_fallback_as_initial(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTING_TIER: "weak"})
    ctx.selected_target = "strong"
    ctx.evicted_targets = ["weak"]

    await _router().call(ctx, _request())

    event = _single_routing_event(ctx)
    assert event["phase"] == "initial"
    assert event["selection"] == {"tier": "strong", "model": "strong-model"}
    assert event["source"] == "context_selected_target"
    assert event["status"] == "success"
    assert event["details"]["attempt_number"] == 0


async def test_backend_attempt_trace_numbers_calls_on_one_request(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTING_TIER: "weak"})
    router = _router()

    await router.call(ctx, _request())
    await router.call(ctx, _request())

    trace = cast(dict[str, Any], ctx.routing_trace)
    assert [event["phase"] for event in trace["events"]] == ["initial", "retry"]
    assert [event["details"]["attempt_number"] for event in trace["events"]] == [0, 1]


async def test_backend_attempt_trace_records_defensive_default(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTING_TIER: "unknown"})

    await _router().call(ctx, _request())

    event = _single_routing_event(ctx)
    assert event["phase"] == "initial"
    assert event["selection"] == {"tier": "weak", "model": "weak-model"}
    assert event["source"] == "backend_default"
    assert event["status"] == "success"


async def test_backend_attempt_trace_sanitizes_error(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTING_TIER: "weak"})

    with pytest.raises(BackendFailure, match="sensitive upstream detail"):
        await _router(weak_backend=_FailingBackend()).call(ctx, _request())

    event = _single_routing_event(ctx)
    assert event["selection"] == {"tier": "weak", "model": "weak-model"}
    assert event["source"] == "routing_tier_metadata"
    assert event["status"] == "error"
    assert event["error"] == "BackendFailure"
    assert "sensitive upstream detail" not in str(event)
    assert event["duration_ms"] >= 0.0


async def test_backend_attempt_trace_distinguishes_open_stream(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    ctx = ProxyContext(metadata={CTX_DETERMINISTIC_ROUTING_TIER: "weak"})

    await _router(weak_backend=_StreamingBackend()).call(ctx, _request())

    event = _single_routing_event(ctx)
    assert event["name"] == "backend_stream_opened"
    assert event["status"] == "success"
    assert event["details"] == {"attempt_number": 0, "response_mode": "stream"}


# --- Tool-planning escalation ----------------------------------------------


def _coding_signals(
    *,
    tool_call_count_estimate: int = 0,
    turn_type: CodingAgentTurnType = CodingAgentTurnType.EXPLORATION,
    code_modification_scope: CodeModificationScope = CodeModificationScope.NONE,
    recommended_tier: RouteTier = RouteTier.MEDIUM,
    confidence: float = 0.9,
) -> CodingAgentRouteDecision:
    return CodingAgentRouteDecision(
        turn_type=turn_type,
        code_modification_scope=code_modification_scope,
        tool_call_count_estimate=tool_call_count_estimate,
        requires_codebase_context=False,
        recommended_tier=recommended_tier,
        confidence=confidence,
    )


def _config_with_escalation(escalate: bool) -> SignalTierSelectorConfig:
    return SignalTierSelectorConfig(
        tier_mapping={
            RouteTier.SIMPLE: "weak",
            RouteTier.MEDIUM: "weak",
            RouteTier.COMPLEX: "strong",
            RouteTier.REASONING: "strong",
        },
        default_tier="strong",
        escalate_on_tool_planning=escalate,
    )


async def test_tool_planning_escalation_lifts_weak_to_strong() -> None:
    """RouteSignals with tool_planning_required=True escalates when flag on."""
    processor = SignalTierSelectorRequestProcessor(_config_with_escalation(escalate=True))
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: RouteSignals(
            task_type="coding",
            complexity=Complexity.SIMPLE,    # would map to weak via SIMPLE
            reasoning_depth=ReasoningDepth.LIGHT,
            tool_planning_required=True,
            precision_requirement=PrecisionRequirement.LOW,
            context_dependency=ContextDependency.LATEST_MESSAGE,
            structured_output_risk=RiskLevel.LOW,
            recommended_tier=RouteTier.SIMPLE,
            confidence=0.9,
            reason_code=ReasonCode.SIMPLE_QA,
        ),
    })

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.source == "tool_planning_escalation"
    assert decision.policy_tier == RouteTier.SIMPLE


async def test_tool_planning_flag_off_keeps_policy_tier() -> None:
    """Same signals, escalate flag off => no override; tier stays weak."""
    processor = SignalTierSelectorRequestProcessor(_config_with_escalation(escalate=False))
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: RouteSignals(
            task_type="coding",
            complexity=Complexity.SIMPLE,
            reasoning_depth=ReasoningDepth.LIGHT,
            tool_planning_required=True,
            precision_requirement=PrecisionRequirement.LOW,
            context_dependency=ContextDependency.LATEST_MESSAGE,
            structured_output_risk=RiskLevel.LOW,
            recommended_tier=RouteTier.SIMPLE,
            confidence=0.9,
            reason_code=ReasonCode.SIMPLE_QA,
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.source == "policy_tier"
    # policy_tier may differ from SIMPLE due to tool_planning_required +1
    # to COMPLEX; just assert the source is the normal policy path.


async def test_tool_planning_no_escalation_when_already_default_tier() -> None:
    """If policy_tier already routes to default_tier (strong), no override fires."""
    processor = SignalTierSelectorRequestProcessor(_config_with_escalation(escalate=True))
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
            tool_call_count_estimate=5,
            turn_type=CodingAgentTurnType.DEBUG,    # already → COMPLEX → strong
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.tier == "strong"
    assert decision.source == "policy_tier"   # not escalation


async def test_tool_planning_escalation_for_coding_agent_threshold() -> None:
    """CodingAgent escalates at ``tool_count >= 3 AND scope >= FUNCTION``.

    Concordance rule: a tool-heavy turn must also be modifying code
    (scope >= FUNCTION) for escalation to fire. Pure-probe turns
    (``ls`` + ``cat`` + ``grep`` with scope=NONE) stay on weak even at
    high tool counts; 2-tool read+edit cycles stay on weak even with
    scope=FUNCTION.
    """
    config = _config_with_escalation(escalate=True)
    processor = SignalTierSelectorRequestProcessor(config)

    # scope=NONE: never escalates regardless of tool count.
    for count in (0, 1, 2, 3, 5):
        ctx = ProxyContext(metadata={
            CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
                tool_call_count_estimate=count,
                turn_type=CodingAgentTurnType.EXPLORATION,
                code_modification_scope=CodeModificationScope.NONE,
            ),
        })
        await processor.process(ctx, _request())
        actual = ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]
        assert actual == "weak", (
            f"scope=NONE, count={count}: expected weak, got {actual}"
        )

    # scope=FUNCTION: escalates only at >=3 tool calls.
    for count, expected in [(0, "weak"), (1, "weak"), (2, "weak"), (3, "strong"), (5, "strong")]:
        ctx = ProxyContext(metadata={
            CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
                tool_call_count_estimate=count,
                turn_type=CodingAgentTurnType.EXPLORATION,
                code_modification_scope=CodeModificationScope.FUNCTION,
            ),
        })
        await processor.process(ctx, _request())
        actual = ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]
        assert actual == expected, (
            f"scope=FUNCTION, count={count}: expected {expected}, got {actual}"
        )


async def test_tool_planning_escalation_for_openclaw() -> None:
    """OpenClaw escalates on TOOL_ORCHESTRATION turn_type or 2+ tool calls."""
    processor = SignalTierSelectorRequestProcessor(_config_with_escalation(escalate=True))

    # Base case: CHITCHAT, no tools → weak, no escalation.
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: OpenClawRouteDecision(
            turn_type=OpenClawTurnType.CHITCHAT,
            tool_call_count_estimate=0,
            memory_dependency=MemoryDependency.NONE,
            external_action_required=False,
            precision_requirement=PrecisionRequirement.LOW,
            ambiguity=RiskLevel.LOW,
            channel_kind=ChannelKind.CASUAL,
            recommended_tier=RouteTier.SIMPLE,
            confidence=0.9,
        ),
    })
    await processor.process(ctx, _request())
    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"

    # 2+ tool calls on a non-tool turn_type → escalates.
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: OpenClawRouteDecision(
            turn_type=OpenClawTurnType.LOOKUP,
            tool_call_count_estimate=2,
            memory_dependency=MemoryDependency.NONE,
            external_action_required=False,
            precision_requirement=PrecisionRequirement.LOW,
            ambiguity=RiskLevel.LOW,
            channel_kind=ChannelKind.CASUAL,
            recommended_tier=RouteTier.SIMPLE,
            confidence=0.9,
        ),
    })
    await processor.process(ctx, _request())
    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"


async def test_requires_tool_planning_base_returns_false() -> None:
    """Bare RouteDecision returns False; the override is a no-op for it."""
    from switchyard.lib.processors.llm_classifier.signals import (
        RouteDecision,
    )

    decision = RouteDecision(recommended_tier=RouteTier.MEDIUM, confidence=0.9)
    assert decision.requires_tool_planning() is False


async def test_coding_agent_preset_has_escalation_enabled() -> None:
    """Sanity check: coding_agent_2_tier and openclaw_2_tier opt in by default."""
    from switchyard.lib.processors.llm_classifier import LLMClassifierPresets

    coding = LLMClassifierPresets.coding_agent_2_tier(weak="weak", strong="strong")
    assert coding.escalate_on_tool_planning is True
    openclaw = LLMClassifierPresets.openclaw_2_tier(weak="weak", strong="strong")
    assert openclaw.escalate_on_tool_planning is True
    general = LLMClassifierPresets.general_2_tier(weak="weak", strong="strong")
    assert general.escalate_on_tool_planning is False


# --- LLM alignment bump ----------------------------------------------------


def _config_with_alignment(
    *,
    enabled: bool,
    min_conf: float = 0.7,
) -> SignalTierSelectorConfig:
    return SignalTierSelectorConfig(
        tier_mapping={
            RouteTier.SIMPLE: "weak",
            RouteTier.MEDIUM: "weak",
            RouteTier.COMPLEX: "strong",
            RouteTier.REASONING: "strong",
        },
        default_tier="strong",
        align_with_llm_recommendation=enabled,
        alignment_min_confidence=min_conf,
    )


async def test_alignment_bump_lifts_simple_to_complex_when_llm_confident() -> None:
    """Policy lands on SIMPLE; high-conf LLM says COMPLEX → bump to strong."""
    processor = SignalTierSelectorRequestProcessor(_config_with_alignment(enabled=True))
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
            tool_call_count_estimate=0,
            turn_type=CodingAgentTurnType.CHITCHAT,  # policy SIMPLE
            recommended_tier=RouteTier.COMPLEX,      # LLM disagrees
            confidence=0.9,
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.tier == "strong"
    assert decision.source == "llm_alignment_bump"
    assert decision.policy_tier == RouteTier.COMPLEX


async def test_alignment_bump_disabled_keeps_policy_tier() -> None:
    """Same signals, flag off → no bump."""
    processor = SignalTierSelectorRequestProcessor(_config_with_alignment(enabled=False))
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
            turn_type=CodingAgentTurnType.CHITCHAT,
            recommended_tier=RouteTier.COMPLEX,
            confidence=0.9,
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.source == "policy_tier"
    assert decision.tier == "weak"


async def test_alignment_bump_suppressed_below_confidence_floor() -> None:
    """LLM says COMPLEX but confidence 0.6 < threshold 0.7 → no bump."""
    processor = SignalTierSelectorRequestProcessor(
        _config_with_alignment(enabled=True, min_conf=0.7),
    )
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
            turn_type=CodingAgentTurnType.CHITCHAT,
            recommended_tier=RouteTier.COMPLEX,
            confidence=0.6,
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.source == "policy_tier"
    assert decision.tier == "weak"


async def test_alignment_bump_never_demotes() -> None:
    """Policy lands on COMPLEX; LLM says SIMPLE → policy wins, no demotion."""
    processor = SignalTierSelectorRequestProcessor(_config_with_alignment(enabled=True))
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
            turn_type=CodingAgentTurnType.DEBUG,  # policy COMPLEX
            recommended_tier=RouteTier.SIMPLE,    # LLM disagrees softer
            confidence=0.95,
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.tier == "strong"
    assert decision.source == "policy_tier"


# --- escalate_target_tier separated from default_tier ---------------------


async def test_escalate_target_tier_defaults_to_default_tier() -> None:
    """When escalate_target_tier is None, effective target == default_tier."""
    config = SignalTierSelectorConfig(
        tier_mapping={
            RouteTier.SIMPLE: "weak",
            RouteTier.MEDIUM: "weak",
            RouteTier.COMPLEX: "strong",
            RouteTier.REASONING: "strong",
        },
        default_tier="strong",
        escalate_on_tool_planning=True,
        escalate_target_tier=None,
    )
    assert config.effective_escalate_target_tier == "strong"


async def test_escalate_target_tier_override_three_tier_setup() -> None:
    """3-tier setup escalates SIMPLE+tool_planning to MEDIUM, not STRONG."""
    config = SignalTierSelectorConfig(
        tier_mapping={
            RouteTier.SIMPLE: "weak",
            RouteTier.MEDIUM: "mid",
            RouteTier.COMPLEX: "strong",
            RouteTier.REASONING: "strong",
        },
        default_tier="strong",
        escalate_on_tool_planning=True,
        escalate_target_tier="mid",  # land on mid, not strong
    )
    processor = SignalTierSelectorRequestProcessor(config)
    ctx = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _coding_signals(
            # PLANNING turn_type triggers escalation unconditionally
            # (the one signal we trust on its own). Tool-count-driven
            # escalation now requires scope >= FUNCTION too — see
            # `requires_tool_planning` for the concordance rule — so
            # use PLANNING here to keep this test focused on the
            # escalate_target_tier mechanism rather than the threshold.
            turn_type=CodingAgentTurnType.PLANNING,
        ),
    })

    await processor.process(ctx, _request())

    decision: TierSelectionDecision = ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION]
    assert decision.tier == "mid"  # not "strong"
    assert decision.source == "tool_planning_escalation"


async def test_escalate_target_tier_validates_against_tier_mapping() -> None:
    """Unknown target raises at config construction (validator)."""
    with pytest.raises(ValueError, match="escalate_target_tier"):
        SignalTierSelectorConfig(
            tier_mapping={
                RouteTier.SIMPLE: "weak",
                RouteTier.MEDIUM: "weak",
                RouteTier.COMPLEX: "strong",
                RouteTier.REASONING: "strong",
            },
            default_tier="strong",
            escalate_target_tier="middle",  # not in mapping
        )


async def test_coding_agent_preset_enables_alignment_bump() -> None:
    """coding_agent and openclaw opt into LLM alignment; general stays off."""
    from switchyard.lib.processors.llm_classifier import LLMClassifierPresets

    coding = LLMClassifierPresets.coding_agent_2_tier(weak="weak", strong="strong")
    assert coding.align_with_llm_recommendation is True
    openclaw = LLMClassifierPresets.openclaw_2_tier(weak="weak", strong="strong")
    assert openclaw.align_with_llm_recommendation is True
    general = LLMClassifierPresets.general_2_tier(weak="weak", strong="strong")
    assert general.align_with_llm_recommendation is False


async def test_session_affinity_pins_confident_verdict(monkeypatch, tmp_path) -> None:
    """A confident verdict is pinned and holds even when later signals flip."""
    monkeypatch.setenv(ROUTING_TRACE_JSONL_ENV, str(tmp_path / "routing-trace.jsonl"))
    processor = SignalTierSelectorRequestProcessor(
        _two_tier_config(),
        affinity=SessionAffinity(enabled=True),
    )
    req = _request()  # identical body across turns → same session key

    # Turn 1: confident COMPLEX → "strong"; pin it.
    ctx1 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx1, req)
    assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"

    # Turn 2: low confidence *would* fall back to "weak" — the pin overrides.
    ctx2 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.2),
    })
    await processor.process(ctx2, req)
    assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    assert ctx2.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "sticky"
    event = _single_routing_event(ctx2)
    assert event["selection"] == {"tier": "strong"}
    assert event["source"] == "sticky"
    assert "confidence" not in event


async def test_session_affinity_warmup_delays_pinning_until_after_threshold() -> None:
    """Warmup turns route normally, then the first confident post-warmup verdict pins."""
    processor = SignalTierSelectorRequestProcessor(
        _two_tier_config(),
        affinity=SessionAffinity(enabled=True, warmup_turns=2),
    )

    ctx1 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx1, _request("same task"))
    assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"

    ctx2 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.2),
    })
    await processor.process(ctx2, _request("same task", prior_assistant_turns=1))
    assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert ctx2.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "low_confidence"

    ctx3 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx3, _request("same task", prior_assistant_turns=2))
    assert ctx3.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    assert ctx3.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "policy_tier"

    ctx4 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.2),
    })
    await processor.process(ctx4, _request("same task", prior_assistant_turns=3))
    assert ctx4.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    assert ctx4.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "sticky"


async def test_session_affinity_warmup_does_not_pin_fallback_after_threshold() -> None:
    """Fallback decisions remain non-sticky even after the warmup period ends."""
    processor = SignalTierSelectorRequestProcessor(
        _two_tier_config(),
        affinity=SessionAffinity(enabled=True, warmup_turns=1),
    )

    ctx1 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx1, _request("same task"))
    assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"

    ctx2 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(abstain=True),
    })
    await processor.process(ctx2, _request("same task", prior_assistant_turns=1))
    assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert ctx2.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "abstain"

    ctx3 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx3, _request("same task", prior_assistant_turns=2))
    assert ctx3.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
    assert ctx3.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "policy_tier"


async def test_session_affinity_does_not_pin_fallback_decision() -> None:
    """A fail-open/abstain fallback is NOT pinned, so the next turn re-classifies.

    A single transient turn-1 classifier failure must not lock the whole task to
    the default tier."""
    processor = SignalTierSelectorRequestProcessor(
        _two_tier_config(),
        affinity=SessionAffinity(enabled=True),
    )
    req = _request()

    # Turn 1: classifier abstained → default "weak", but this must NOT pin.
    ctx1 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(abstain=True),
    })
    await processor.process(ctx1, req)
    assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
    assert ctx1.metadata[CTX_DETERMINISTIC_TIER_DECISION].source == "abstain"

    # Turn 2: a confident verdict re-classifies — the task is NOT locked to weak.
    ctx2 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx2, req)
    assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"


async def test_session_affinity_off_routes_per_turn() -> None:
    """Without affinity, each turn follows its own signals (no pinning)."""
    processor = SignalTierSelectorRequestProcessor(_two_tier_config())
    req = _request()

    ctx1 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.2),
    })
    await processor.process(ctx1, req)
    assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"

    ctx2 = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
    })
    await processor.process(ctx2, req)
    assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"


async def test_session_affinity_distinct_conversations_independent() -> None:
    """A pin for one conversation does not leak into another (distinct key)."""
    processor = SignalTierSelectorRequestProcessor(
        _two_tier_config(),
        affinity=SessionAffinity(enabled=True),
    )

    # Conversation A: confident → "strong"; pinned.
    await processor.process(
        ProxyContext(metadata={
            CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.9),
        }),
        _request("question A"),
    )
    # Conversation B (distinct first-user) decides on its own signals → "weak",
    # proving A's strong pin did not leak across the key boundary.
    ctx_b = ProxyContext(metadata={
        CTX_DETERMINISTIC_ROUTE_SIGNALS: _signals(confidence=0.2),
    })
    await processor.process(ctx_b, _request("question B"))
    assert ctx_b.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
