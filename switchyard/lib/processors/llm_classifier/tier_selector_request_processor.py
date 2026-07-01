# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request processor that maps LLM classifier signals to a backend tier."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)
from switchyard.lib.processors.llm_classifier.signals import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    CTX_LLM_CLASSIFIER_TRACE_PAYLOAD,
    RouteDecision,
    RouteSignals,
    RouteTier,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_affinity import SessionAffinity
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: ``ProxyContext.metadata`` key for the deterministic tier selector decision.
CTX_DETERMINISTIC_TIER_DECISION = "_deterministic_tier_decision"

DecisionSource = Literal[
    "policy_tier",
    "low_confidence",
    "abstain",
    "missing_signals",
    "unknown_policy_tier",
    "tool_planning_escalation",
    "llm_alignment_bump",
    "sticky",
]

#: Confident classifier verdicts — the only decisions pinned for affinity.
#: An allowlist, so a newly added (possibly fallback) source is non-sticky by
#: default and can't silently lock a task to the default tier.
_STICKY_SOURCES: frozenset[str] = frozenset(
    {"policy_tier", "tool_planning_escalation", "llm_alignment_bump"}
)


class TierSelectionDecision(BaseModel):
    """Audit record for one signal-to-tier decision.

    Captures both the LLM's own vote (``llm_recommended_tier``) and the
    Python policy's computed verdict (``policy_tier``). They will frequently
    agree; when they disagree it's a signal worth investigating, so both
    are persisted on every decision.
    """

    model_config = ConfigDict(frozen=True)

    tier: str
    source: DecisionSource
    reason: str
    policy_tier: RouteTier | None = None
    llm_recommended_tier: RouteTier | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SignalTierSelectorConfig(BaseModel):
    """Policy for mapping :class:`RouteDecision` to backend tier labels.

    ``tier_mapping`` translates abstract classifier tiers such as
    ``RouteTier.SIMPLE`` into concrete backend labels configured on
    :class:`DeterministicRoutingLLMBackend`. The mapping is applied to the
    *policy* tier (``signals.policy_tier()``), not to the LLM's
    ``recommended_tier`` field.
    """

    model_config = ConfigDict(frozen=True)

    tier_mapping: Mapping[RouteTier, str] = Field(
        default_factory=lambda: {
            RouteTier.SIMPLE: "simple",
            RouteTier.MEDIUM: "medium",
            RouteTier.COMPLEX: "complex",
            RouteTier.REASONING: "reasoning",
        },
    )
    default_tier: str = Field(default="medium", min_length=1)
    """Fallback target on abstain / missing signals / low confidence /
    unknown policy_tier. Kept distinct from ``escalate_target_tier``
    so the "safer-on-uncertainty" path (typically strong) and the
    explicit escalation target can diverge once a third middle tier
    exists; in a 2-tier setup they coincide by default."""

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    escalate_on_tool_planning: bool = Field(default=False)
    """When ``True``, any policy-decided tier other than the escalation
    target is overridden when ``signals.requires_tool_planning()`` fires.
    One-way gate — never demotes. Used to bias coding-agent / OpenClaw
    traffic toward the strong tier on multi-step tool-driven turns
    where weak models tend to fumble; off by default for general
    traffic (``RouteSignals`` already factors ``tool_planning_required``
    into its ``policy_tier`` scoring)."""

    escalate_target_tier: str | None = Field(default=None)
    """Tier label the escalation override should target. ``None`` falls
    back to ``default_tier`` (the 2-tier convention). Set explicitly
    when a 3+-tier setup wants escalation to land on the *middle*
    tier (e.g. SIMPLE+tool_planning → Sonnet) rather than always
    jumping to the safe-on-uncertainty fallback (which is typically
    the *strongest* tier). Kept structurally separate from
    ``default_tier`` so the two concerns — "what fires on uncertainty"
    and "what fires on confident-but-flagged" — stay decoupled."""

    align_with_llm_recommendation: bool = Field(default=False)
    """When ``True``, treat a high-confidence LLM ``recommended_tier``
    vote of COMPLEX/REASONING as a one-way bump if the deterministic
    ``policy_tier()`` landed on SIMPLE/MEDIUM. Surfaces signal the
    feature-based scoring under-weights — when the classifier
    confidently says "this is hard" but the discrete features didn't
    quite cross a threshold, trust the LLM's gestalt read. Gated on
    ``alignment_min_confidence`` and never demotes. Off by default;
    enable in agent-shaped profiles (coding_agent, openclaw) where
    the LLM has session-level awareness the schema features lack."""

    alignment_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    """Minimum ``signals.confidence`` for ``align_with_llm_recommendation``
    to fire. Below this floor the bump is suppressed even if the LLM's
    recommended tier is harder — protects against low-confidence
    over-routing."""

    @field_validator("tier_mapping")
    @classmethod
    def _tier_mapping_values_non_empty(
        cls,
        value: Mapping[RouteTier, str],
    ) -> Mapping[RouteTier, str]:
        if not value:
            raise ValueError("tier_mapping must not be empty")
        empty = [tier.value for tier, label in value.items() if not label]
        if empty:
            raise ValueError(f"tier_mapping has empty labels for {empty}")
        return value

    @model_validator(mode="after")
    def _default_tier_known(self) -> SignalTierSelectorConfig:
        labels = set(self.tier_mapping.values())
        if self.default_tier not in labels:
            raise ValueError(
                f"default_tier {self.default_tier!r} must be one of "
                f"tier_mapping values {sorted(labels)}",
            )
        if (
            self.escalate_target_tier is not None
            and self.escalate_target_tier not in labels
        ):
            raise ValueError(
                f"escalate_target_tier {self.escalate_target_tier!r} must be "
                f"one of tier_mapping values {sorted(labels)}",
            )
        return self

    @property
    def effective_escalate_target_tier(self) -> str:
        """Concrete escalation target — explicit field if set, else default_tier."""
        return (
            self.escalate_target_tier
            if self.escalate_target_tier is not None
            else self.default_tier
        )


class SignalTierSelectorRequestProcessor:
    """Convert a stamped :class:`RouteDecision` into a deterministic backend tier.

    This processor performs no LLM calls and does not mutate the request body.
    It routes on ``signals.policy_tier()`` — a deterministic function of the
    classifier's extracted features — and only writes
    ``CTX_DETERMINISTIC_ROUTING_TIER`` for
    :class:`DeterministicRoutingLLMBackend`; the backend remains responsible
    for rewriting ``request.body["model"]`` to the selected model.

    The LLM's own ``recommended_tier`` is still captured in
    :class:`TierSelectionDecision` for audit, so disagreements between the
    LLM and the policy are observable.
    """

    def __init__(
        self,
        config: SignalTierSelectorConfig | None = None,
        *,
        affinity: SessionAffinity | None = None,
    ) -> None:
        self._config = config or SignalTierSelectorConfig()
        # Shared per-conversation tier pin. When set and enabled, the first
        # turn's tier is recorded and every later turn reuses it; the same
        # coordinator gates the classifier's LLM call (classify once per task).
        self._affinity = affinity

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        signals = _read_signals(ctx)
        decision = self._decide(ctx, request, signals)

        ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = decision.tier
        ctx.metadata[CTX_DETERMINISTIC_TIER_DECISION] = decision

        trace_payload = ctx.metadata.get(CTX_LLM_CLASSIFIER_TRACE_PAYLOAD)
        if isinstance(trace_payload, dict):
            trace_payload["decision"] = {
                "tier": decision.tier,
                "source": decision.source,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "policy_tier": decision.policy_tier.value if decision.policy_tier else None,
                "llm_recommended_tier": (
                    decision.llm_recommended_tier.value
                    if decision.llm_recommended_tier
                    else None
                ),
            }

        log.debug(
            "SignalTierSelectorRequestProcessor: tier=%s source=%s "
            "policy=%s llm=%s reason=%s",
            decision.tier,
            decision.source,
            decision.policy_tier.value if decision.policy_tier else None,
            decision.llm_recommended_tier.value if decision.llm_recommended_tier else None,
            decision.reason,
        )
        return request

    def _decide(
        self, ctx: ProxyContext, request: ChatRequest, signals: RouteDecision | None
    ) -> TierSelectionDecision:
        """Select a tier, reusing a per-conversation pin when affinity is on.

        On a reused pin the classifier was skipped upstream, so ``signals`` may
        be absent — the pin alone decides the tier."""
        pinned = self._affinity.pinned(ctx, request) if self._affinity else None
        if pinned is not None:
            return TierSelectionDecision(
                tier=pinned,
                source="sticky",
                reason="reusing pinned tier for this conversation",
            )
        decision = self._select(signals)
        # Pin only a confident verdict — pinning a fail-open fallback would lock
        # the task to the default tier and the classifier would never re-run.
        if self._affinity is not None and decision.source in _STICKY_SOURCES:
            self._affinity.pin(ctx, request, decision.tier)
        return decision

    def _select(self, signals: RouteDecision | None) -> TierSelectionDecision:
        if signals is None:
            return self._default_decision(
                source="missing_signals",
                reason="no RouteDecision found on ProxyContext metadata",
            )

        if signals.abstain:
            return self._default_decision(
                source="abstain",
                reason="classifier abstained",
                signals=signals,
            )

        if signals.confidence < self._config.min_confidence:
            return self._default_decision(
                source="low_confidence",
                reason=(
                    f"classifier confidence {signals.confidence:.3f} "
                    f"< min_confidence {self._config.min_confidence:.3f}"
                ),
                signals=signals,
            )

        policy_tier = signals.policy_tier()

        # LLM alignment bump: when the classifier confidently says
        # "harder than feature scoring suggests", trust the LLM's vote.
        # One-way (never demotes). Lets the LLM surface session-level
        # signal that the discrete-feature scoring under-weights.
        bumped_from: RouteTier | None = None
        if (
            self._config.align_with_llm_recommendation
            and signals.confidence >= self._config.alignment_min_confidence
            and policy_tier in (RouteTier.SIMPLE, RouteTier.MEDIUM)
            and signals.recommended_tier in (RouteTier.COMPLEX, RouteTier.REASONING)
        ):
            bumped_from = policy_tier
            policy_tier = signals.recommended_tier

        tier = self._config.tier_mapping.get(policy_tier)
        if tier is None:
            return self._default_decision(
                source="unknown_policy_tier",
                reason=f"no backend tier mapped for policy_tier {policy_tier.value!r}",
                signals=signals,
            )

        if bumped_from is not None:
            return TierSelectionDecision(
                tier=tier,
                source="llm_alignment_bump",
                reason=(
                    f"LLM alignment bump: policy_tier={bumped_from.value!r} "
                    f"-> {policy_tier.value!r} (LLM recommended {policy_tier.value!r} "
                    f"@ confidence={signals.confidence:.3f} >= "
                    f"{self._config.alignment_min_confidence:.3f}); mapped to {tier!r}"
                ),
                policy_tier=policy_tier,
                llm_recommended_tier=signals.recommended_tier,
                confidence=signals.confidence,
            )

        escalate_target = self._config.effective_escalate_target_tier
        if (
            self._config.escalate_on_tool_planning
            and tier != escalate_target
            and signals.requires_tool_planning()
        ):
            return TierSelectionDecision(
                tier=escalate_target,
                source="tool_planning_escalation",
                reason=(
                    f"tool_planning escalation: policy_tier={policy_tier.value!r} "
                    f"mapped to {tier!r}; escalated to {escalate_target!r}"
                ),
                policy_tier=policy_tier,
                llm_recommended_tier=signals.recommended_tier,
                confidence=signals.confidence,
            )

        return TierSelectionDecision(
            tier=tier,
            source="policy_tier",
            reason=f"policy_tier() returned {policy_tier.value!r}",
            policy_tier=policy_tier,
            llm_recommended_tier=signals.recommended_tier,
            confidence=signals.confidence,
        )

    def _default_decision(
        self,
        *,
        source: DecisionSource,
        reason: str,
        signals: RouteDecision | None = None,
    ) -> TierSelectionDecision:
        return TierSelectionDecision(
            tier=self._config.default_tier,
            source=source,
            reason=reason,
            policy_tier=signals.policy_tier() if signals else None,
            llm_recommended_tier=signals.recommended_tier if signals else None,
            confidence=signals.confidence if signals else None,
        )


def _read_signals(ctx: ProxyContext) -> RouteDecision | None:
    raw = ctx.metadata.get(CTX_DETERMINISTIC_ROUTE_SIGNALS)
    if isinstance(raw, RouteDecision):
        return raw
    if isinstance(raw, Mapping):
        return RouteSignals.model_validate(dict(raw))
    return None


__all__ = [
    "CTX_DETERMINISTIC_TIER_DECISION",
    "SignalTierSelectorConfig",
    "SignalTierSelectorRequestProcessor",
    "TierSelectionDecision",
]
