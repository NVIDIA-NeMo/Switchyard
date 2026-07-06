# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed signal vocabulary for LLM classifier routing.

Each routing profile (general chat, coding agent, OpenClaw personal agent)
defines its own pydantic schema. All schemas share a minimal contract —
:class:`RouteDecision` — so the downstream tier selector can stay
profile-agnostic.

The base contract exposes:

* :attr:`RouteDecision.recommended_tier` — the LLM's own vote, captured
  verbatim from the classifier output. Kept for audit / comparison only;
  routing does **not** read it.
* :attr:`RouteDecision.confidence` — calibration signal used by the
  tier-selector's confidence floor.
* :attr:`RouteDecision.abstain` — explicit "too ambiguous to classify"
  marker that forces the selector to its default tier.
* :meth:`RouteDecision.policy_tier` — a deterministic, code-derived
  tier computed from the profile-specific feature set. **This is what
  the tier selector routes on.** Each subclass implements it as a small
  scoring scheme over its observable features.

Splitting "extract features" (LLM) from "decide tier" (Python) makes the
policy testable, replayable from captured signals, and tunable without
re-prompting. When ``policy_tier()`` disagrees with the LLM's
``recommended_tier``, that's a signal for the operator to investigate —
both values are persisted in :class:`TierSelectionDecision`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

#: ``ProxyContext.metadata`` key for the latest semantic route decision record.
CTX_DETERMINISTIC_ROUTE_SIGNALS = "_deterministic_route_signals"

#: In-progress trace payload shared across the classifier, selector, and backend.
CTX_LLM_CLASSIFIER_TRACE_PAYLOAD = "_llm_classifier_trace_payload"


class TaskType(str, Enum):
    """Coarse task family inferred from the incoming LLM request."""

    CHAT = "chat"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    CODING = "coding"
    DEBUGGING = "debugging"
    MATH = "math"
    PLANNING = "planning"
    CREATIVE_WRITING = "creative_writing"
    AGENTIC_TASK = "agentic_task"
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    OTHER = "other"


class Complexity(str, Enum):
    """Estimated capability tier needed to answer well."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    REASONING = "reasoning"


class ReasoningDepth(str, Enum):
    """How much multi-step reasoning the request appears to require."""

    NONE = "none"
    LIGHT = "light"
    MULTI_STEP = "multi_step"
    DEEP = "deep"


class PrecisionRequirement(str, Enum):
    """How costly small correctness errors are likely to be."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContextDependency(str, Enum):
    """How much surrounding context the final model likely needs."""

    LATEST_MESSAGE = "latest_message"
    CONVERSATION = "conversation"
    EXTERNAL_CONTEXT = "external_context"


class RiskLevel(str, Enum):
    """Small ordinal risk scale for policy-relevant routing signals."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RouteTier(str, Enum):
    """Advisory abstract tier before policy maps it to a concrete model."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    REASONING = "reasoning"


_TIER_ORDER: tuple[RouteTier, ...] = (
    RouteTier.SIMPLE,
    RouteTier.MEDIUM,
    RouteTier.COMPLEX,
    RouteTier.REASONING,
)


def _argmax_tier(scores: dict[RouteTier, int]) -> RouteTier:
    """Return the highest-scoring tier; ties resolve to the *stronger* tier.

    The "ties → stronger" rule is the safer side of the cost/quality
    trade-off, matching the chain's overall fail-open posture.
    """
    if not scores:
        return RouteTier.MEDIUM
    best_score = max(scores.values())
    for tier in reversed(_TIER_ORDER):
        if scores.get(tier, 0) == best_score:
            return tier
    return RouteTier.MEDIUM


class ReasonCode(str, Enum):
    """Stable, operator-friendly reason code for the extracted signal set."""

    SIMPLE_QA = "simple_qa"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    CODING_SIMPLE = "coding_simple"
    CODING_COMPLEX = "coding_complex"
    DEBUGGING = "debugging"
    MATH_REASONING = "math_reasoning"
    TOOL_AGENTIC = "tool_agentic"
    LONG_CONTEXT = "long_context"
    STRUCTURED_OUTPUT = "structured_output"
    CREATIVE_GENERATION = "creative_generation"
    RESEARCH_SYNTHESIS = "research_synthesis"
    AMBIGUOUS = "ambiguous"
    OTHER = "other"


# ---- Coding-agent profile vocabulary ----------------------------------------


class CodingAgentTurnType(str, Enum):
    """Per-turn intent in a coding-agent harness like Claude Code or Cursor."""

    CHITCHAT = "chitchat"
    PLANNING = "planning"
    EXPLORATION = "exploration"
    EDIT = "edit"
    DEBUG = "debug"
    EXPLANATION = "explanation"
    CLARIFICATION = "clarification"
    SUMMARIZE = "summarize"


class CodeModificationScope(str, Enum):
    """Estimated blast radius of code changes the turn is likely to emit."""

    NONE = "none"
    SINGLE_LINE = "single_line"
    FUNCTION = "function"
    FILE = "file"
    MULTI_FILE = "multi_file"
    CROSS_MODULE = "cross_module"


# ---- OpenClaw profile vocabulary --------------------------------------------


class OpenClawTurnType(str, Enum):
    """Per-message intent on an OpenClaw personal-assistant channel.

    OpenClaw runs across messaging platforms (Telegram, Discord, Slack,
    Signal, iMessage, WhatsApp) with workspace-file memory (SOUL.md,
    USER.md, MEMORY.md). Turns trend short and conversational; tool calls
    are common but usually narrow.
    """

    CHITCHAT = "chitchat"
    LOOKUP = "lookup"
    MEMORY_RECALL = "memory_recall"
    PLANNING = "planning"
    TOOL_ORCHESTRATION = "tool_orchestration"
    ACTION = "action"
    EXPLANATION = "explanation"
    CLARIFICATION = "clarification"


class MemoryDependency(str, Enum):
    """How much OpenClaw memory the turn needs to read."""

    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"


class ChannelKind(str, Enum):
    """Posture of the underlying messaging channel.

    Per the OpenClaw docs: ``casual`` channels (e.g. WhatsApp group chat)
    bias toward fast/cheap models; ``deliberate`` channels (e.g. a curated
    Telegram bot) bias toward a stronger model.
    """

    CASUAL = "casual"
    DELIBERATE = "deliberate"


# ---- Decision schemas -------------------------------------------------------


class RouteDecision(BaseModel):
    """Minimal contract every profile schema must satisfy.

    The deterministic tier selector reads :meth:`policy_tier`, ``confidence``,
    and ``abstain`` — never ``recommended_tier`` directly. Profile schemas
    add their own observable feature set and override :meth:`policy_tier`
    to score against those features.
    """

    model_config = ConfigDict(frozen=True)

    recommended_tier: RouteTier
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool = False

    def policy_tier(self) -> RouteTier:
        """Deterministic tier derived from the schema's features.

        Base implementation falls back to the LLM-emitted ``recommended_tier``
        because the bare :class:`RouteDecision` has no other features to score
        against. Subclasses with feature fields override this with their own
        scoring scheme.
        """
        return self.recommended_tier

    def requires_tool_planning(self) -> bool:
        """Return True when this turn needs careful tool orchestration.

        The deterministic tier selector consults this when
        :attr:`SignalTierSelectorConfig.escalate_on_tool_planning` is
        enabled — a True return value forces escalation to the
        ``default_tier`` (typically strong) even when ``policy_tier()``
        would otherwise route to a cheaper tier. Use for profiles whose
        weak-tier picks regress on multi-step tool-driven turns.

        Base implementation returns ``False`` — schemas without a
        tool-use signal can keep the override a no-op. Subclasses with
        relevant fields override per their own semantics.
        """
        return False

    @classmethod
    def make_abstain(cls, fallback_tier: RouteTier) -> RouteDecision:
        """Construct an abstain-stamped instance for fail-open behavior.

        The base implementation works for any subclass whose extra fields have
        safe defaults. Profiles with required-without-default fields override
        this to supply their own safe defaults.
        """
        return cls(
            recommended_tier=fallback_tier,
            confidence=0.0,
            abstain=True,
        )


class RouteSignals(RouteDecision):
    """General-purpose signal record (default profile, backwards-compatible).

    Suited for mixed traffic — chat, summarization, extraction, generic coding.
    The coding-agent and OpenClaw profiles are better served by their dedicated
    profile schemas.
    """

    task_type: TaskType
    complexity: Complexity
    reasoning_depth: ReasoningDepth
    tool_planning_required: bool
    precision_requirement: PrecisionRequirement
    context_dependency: ContextDependency
    structured_output_risk: RiskLevel
    reason_code: ReasonCode

    def policy_tier(self) -> RouteTier:
        scores: dict[RouteTier, int] = dict.fromkeys(_TIER_ORDER, 0)

        if self.complexity is Complexity.SIMPLE:
            scores[RouteTier.SIMPLE] += 2
        elif self.complexity is Complexity.MEDIUM:
            scores[RouteTier.MEDIUM] += 2
        elif self.complexity is Complexity.COMPLEX:
            scores[RouteTier.COMPLEX] += 2
        elif self.complexity is Complexity.REASONING:
            scores[RouteTier.REASONING] += 2

        if self.reasoning_depth is ReasoningDepth.DEEP:
            scores[RouteTier.REASONING] += 2
        elif self.reasoning_depth is ReasoningDepth.MULTI_STEP:
            scores[RouteTier.COMPLEX] += 1

        if self.tool_planning_required:
            scores[RouteTier.COMPLEX] += 1

        if self.precision_requirement is PrecisionRequirement.HIGH:
            scores[RouteTier.COMPLEX] += 1

        if self.structured_output_risk is RiskLevel.HIGH:
            scores[RouteTier.COMPLEX] += 1

        return _argmax_tier(scores)

    def requires_tool_planning(self) -> bool:
        return self.tool_planning_required

    @classmethod
    def make_abstain(cls, fallback_tier: RouteTier) -> RouteSignals:
        return cls(
            task_type=TaskType.OTHER,
            complexity=Complexity.MEDIUM,
            reasoning_depth=ReasoningDepth.LIGHT,
            tool_planning_required=False,
            precision_requirement=PrecisionRequirement.MEDIUM,
            context_dependency=ContextDependency.CONVERSATION,
            structured_output_risk=RiskLevel.MEDIUM,
            recommended_tier=fallback_tier,
            confidence=0.0,
            reason_code=ReasonCode.AMBIGUOUS,
            abstain=True,
        )


class CodingAgentRouteDecision(RouteDecision):
    """Decision schema for coding-agent harnesses (Claude Code, Codex, Cursor).

    These harnesses interleave conversational turns, exploration, planning,
    edits, and debugging, with a human in the loop. ``turn_type`` plus
    ``code_modification_scope`` carry most of the routing-meaningful signal.
    """

    turn_type: CodingAgentTurnType = CodingAgentTurnType.EXPLORATION
    code_modification_scope: CodeModificationScope = CodeModificationScope.NONE
    tool_call_count_estimate: int = Field(default=0, ge=0)
    requires_codebase_context: bool = False

    def policy_tier(self) -> RouteTier:
        scores: dict[RouteTier, int] = dict.fromkeys(_TIER_ORDER, 0)

        if self.turn_type in (
            CodingAgentTurnType.CHITCHAT,
            CodingAgentTurnType.CLARIFICATION,
            CodingAgentTurnType.SUMMARIZE,
        ):
            scores[RouteTier.SIMPLE] += 2
        elif self.turn_type in (
            CodingAgentTurnType.EXPLORATION,
            CodingAgentTurnType.EXPLANATION,
            CodingAgentTurnType.EDIT,
        ):
            scores[RouteTier.MEDIUM] += 2
        elif self.turn_type in (
            CodingAgentTurnType.PLANNING,
            CodingAgentTurnType.DEBUG,
        ):
            scores[RouteTier.COMPLEX] += 2

        if self.code_modification_scope in (
            CodeModificationScope.NONE,
            CodeModificationScope.SINGLE_LINE,
        ):
            scores[RouteTier.SIMPLE] += 1
        elif self.code_modification_scope in (
            CodeModificationScope.FUNCTION,
            CodeModificationScope.FILE,
        ):
            scores[RouteTier.MEDIUM] += 1
        elif self.code_modification_scope in (
            CodeModificationScope.MULTI_FILE,
            CodeModificationScope.CROSS_MODULE,
        ):
            scores[RouteTier.COMPLEX] += 2

        # Tool-call scoring: a two-band scheme so policy_tier can register
        # multi-step orchestration without needing the escalation override
        # to do all the heavy lifting. Previously only the >=4 threshold
        # bumped, leaving 2-3 tool turns scored at MEDIUM with no signal.
        if self.tool_call_count_estimate >= 4:
            scores[RouteTier.COMPLEX] += 1
        elif self.tool_call_count_estimate >= 2:
            scores[RouteTier.MEDIUM] += 1

        # Codebase context is signal that the turn needs to read across
        # the repo, but it's not on its own a hard-task marker once the
        # weak tier has a long context window (V4 Pro / K2.6 both
        # 200K+; V4 Pro is 1M). Earlier sizing as +1 COMPLEX assumed a
        # short-context weak tier where codebase reach was a stress
        # signal — now it's just orientation. Keep the +1 nudge but
        # land it on MEDIUM so it doesn't single-handedly tip a turn
        # into COMPLEX.
        if self.requires_codebase_context:
            scores[RouteTier.MEDIUM] += 1

        return _argmax_tier(scores)

    def requires_tool_planning(self) -> bool:
        """Coding-agent turns escalate only on real multi-step orchestration.

        Concordance rule: two independent signals must agree that the
        turn would defeat the weak tier before we override a weak-tier
        verdict. Three configurations qualify:

        * ``turn_type == PLANNING`` — the classifier explicitly labeled
          the turn as planning; unconditional escalation (the one
          unambiguous signal).
        * ``tool_call_count_estimate >= 3`` **and** scope reaches
          FUNCTION or deeper — a tool-heavy turn that also touches
          code. The AND-coupling kills the "ls + cat + grep" pure-probe
          case where many tools fire but nothing is being changed.

        2-tool turns (the typical "read file + edit" cycle) stay on
        the weak tier — modern weak models (V4 Pro, K2.6) handle that
        loop without help. The ≥3 threshold matches the boundary where
        turn shape starts to look like genuine multi-step planning
        rather than the agent's normal read→write rhythm.
        """
        scope_is_modifying = self.code_modification_scope in (
            CodeModificationScope.FUNCTION,
            CodeModificationScope.FILE,
            CodeModificationScope.MULTI_FILE,
            CodeModificationScope.CROSS_MODULE,
        )
        return (
            self.turn_type == CodingAgentTurnType.PLANNING
            or (self.tool_call_count_estimate >= 3 and scope_is_modifying)
        )


class OpenClawRouteDecision(RouteDecision):
    """Decision schema for OpenClaw personal-assistant channels.

    OpenClaw agents run across messaging platforms (Telegram, Discord, Slack,
    Signal, iMessage, WhatsApp) and read workspace files for personality
    (``SOUL.md``), user profile (``USER.md``), and long-term memory
    (``MEMORY.md``). Turns are short, conversational, often latency-sensitive,
    and tool calls are common but usually narrow.

    The routing-meaningful signals are:

    * ``turn_type`` — chitchat vs lookup vs tool orchestration etc.
    * ``tool_call_count_estimate`` — orchestration cost.
    * ``memory_dependency`` — does the turn need to read across MEMORY.md?
    * ``external_action_required`` — is the agent about to send a message,
      post somewhere, or hit an external API? Combined with
      ``precision_requirement=high`` this is the strongest "use the strong
      tier" signal (irreversible actions are unforgiving).
    * ``ambiguity`` — chat is "yes / ok / sure"-heavy and ambiguous turns
      benefit disproportionately from a stronger model that can ask a good
      clarifying question.
    * ``channel_kind`` — per the OpenClaw docs, casual channels bias
      toward a fast/cheap model; deliberate channels toward a stronger one.
    """

    turn_type: OpenClawTurnType = OpenClawTurnType.CHITCHAT
    tool_call_count_estimate: int = Field(default=0, ge=0)
    memory_dependency: MemoryDependency = MemoryDependency.NONE
    external_action_required: bool = False
    precision_requirement: PrecisionRequirement = PrecisionRequirement.LOW
    ambiguity: RiskLevel = RiskLevel.LOW
    channel_kind: ChannelKind = ChannelKind.CASUAL

    def policy_tier(self) -> RouteTier:
        scores: dict[RouteTier, int] = dict.fromkeys(_TIER_ORDER, 0)

        if self.turn_type in (
            OpenClawTurnType.CHITCHAT,
            OpenClawTurnType.LOOKUP,
            OpenClawTurnType.MEMORY_RECALL,
            OpenClawTurnType.CLARIFICATION,
        ):
            scores[RouteTier.SIMPLE] += 2
        elif self.turn_type in (
            OpenClawTurnType.PLANNING,
            OpenClawTurnType.EXPLANATION,
        ):
            scores[RouteTier.MEDIUM] += 2
        elif self.turn_type in (
            OpenClawTurnType.TOOL_ORCHESTRATION,
            OpenClawTurnType.ACTION,
        ):
            scores[RouteTier.COMPLEX] += 2

        if self.tool_call_count_estimate >= 4:
            scores[RouteTier.COMPLEX] += 1
        elif self.tool_call_count_estimate >= 1:
            scores[RouteTier.MEDIUM] += 1

        if self.memory_dependency is MemoryDependency.HEAVY:
            scores[RouteTier.COMPLEX] += 2
        elif self.memory_dependency is MemoryDependency.LIGHT:
            scores[RouteTier.MEDIUM] += 1

        if self.external_action_required and (
            self.precision_requirement is PrecisionRequirement.HIGH
        ):
            scores[RouteTier.COMPLEX] += 2
        elif self.external_action_required:
            scores[RouteTier.MEDIUM] += 1

        if self.ambiguity is RiskLevel.HIGH:
            scores[RouteTier.COMPLEX] += 1
        elif self.ambiguity is RiskLevel.MEDIUM:
            scores[RouteTier.MEDIUM] += 1

        if self.channel_kind is ChannelKind.CASUAL:
            scores[RouteTier.SIMPLE] += 1
        else:
            scores[RouteTier.COMPLEX] += 1

        return _argmax_tier(scores)

    def requires_tool_planning(self) -> bool:
        """OpenClaw turns escalate on explicit tool orchestration or 2+ tool calls.

        ``TOOL_ORCHESTRATION`` already maps to COMPLEX via ``policy_tier``
        so this rule mostly catches multi-step ``PLANNING`` /
        ``EXPLANATION`` turns whose ``tool_call_count_estimate``
        signals real orchestration intent.
        """
        return (
            self.tool_call_count_estimate >= 2
            or self.turn_type == OpenClawTurnType.TOOL_ORCHESTRATION
        )


__all__ = [
    "CTX_DETERMINISTIC_ROUTE_SIGNALS",
    "ChannelKind",
    "CodeModificationScope",
    "CodingAgentRouteDecision",
    "CodingAgentTurnType",
    "Complexity",
    "ContextDependency",
    "MemoryDependency",
    "OpenClawRouteDecision",
    "OpenClawTurnType",
    "PrecisionRequirement",
    "ReasonCode",
    "ReasoningDepth",
    "RiskLevel",
    "RouteDecision",
    "RouteSignals",
    "RouteTier",
    "TaskType",
]
