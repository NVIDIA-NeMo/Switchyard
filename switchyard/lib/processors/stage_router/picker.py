# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Python shell over the Rust stage_router picker.

The routing logic — the two-axis scorer, the escalate/de-escalate shortcuts, and
tier selection — lives in Rust (``switchyard_components::stage_router::pick_tier``,
exposed as ``stage_pick_tier``). This module only adds the Python-side pieces the
Rust core deliberately leaves to the caller: the ``no_signal`` case, the async LLM
classifier, and decision-source recording."""

import logging
from typing import TYPE_CHECKING, cast

from switchyard.lib.processors.stage_router.decision_log import (
    CONTEXT_KEY,
    DecisionSource,
    StageRouterDecisionLog,
)

if TYPE_CHECKING:
    from switchyard.lib.processors.stage_router.classifier import TierClassifier
    from switchyard.lib.proxy_context import ProxyContext

log = logging.getLogger(__name__)

EFFICIENT: int = 0
CAPABLE: int = 1

#: Rust returns the tier as a name; map it back to the picker's int constants.
_TIER: dict[str, int] = {"efficient": EFFICIENT, "capable": CAPABLE}


async def pick_capable_first(
    ctx: "ProxyContext",
    confidence_threshold: float,
    classifier: "TierClassifier | None" = None,
    decision_log: StageRouterDecisionLog | None = None,
) -> int:
    """CAPABLE default. EFFICIENT only when the scorer is confidently negative."""
    return await _pick(ctx, "capable_first", confidence_threshold, classifier, decision_log)


async def pick_efficient_first(
    ctx: "ProxyContext",
    confidence_threshold: float,
    classifier: "TierClassifier | None" = None,
    decision_log: StageRouterDecisionLog | None = None,
) -> int:
    """EFFICIENT default. CAPABLE only when the scorer is confidently positive."""
    return await _pick(ctx, "efficient_first", confidence_threshold, classifier, decision_log)


async def _pick(
    ctx: "ProxyContext",
    picker_mode: str,
    confidence_threshold: float,
    classifier: "TierClassifier | None",
    decision_log: StageRouterDecisionLog | None,
) -> int:
    from switchyard_rust.components import (  # local import: heavy native module
        get_tool_result_signal,
        stage_pick_tier,
    )

    signal = get_tool_result_signal(ctx)
    if signal is None:
        default = EFFICIENT if picker_mode == "efficient_first" else CAPABLE
        return _record(ctx, decision_log, "no_signal", default)

    # The Rust core runs the escalate/de-escalate shortcuts and the scorer.
    outcome = stage_pick_tier(signal, picker_mode, confidence_threshold)
    if outcome.resolved:
        # resolved ⇒ tier and source are set (guaranteed by the Rust core).
        source = cast("DecisionSource", outcome.source)
        return _record(ctx, decision_log, source, _TIER[cast(str, outcome.tier)])

    # Scorer wasn't confident. Consult the classifier; fall open to the default
    # tier when there is none or it can't decide.
    default = _TIER[outcome.default_tier]
    if classifier is None:
        return _record(ctx, decision_log, "fall_open", default)
    verdict = await classifier.classify(ctx, signal)
    if verdict == "capable":
        return _record(ctx, decision_log, "llm-classifier", CAPABLE)
    if verdict == "efficient":
        return _record(ctx, decision_log, "llm-classifier", EFFICIENT)
    return _record(ctx, decision_log, "fall_open", default)


def _record(
    ctx: "ProxyContext",
    decision_log: StageRouterDecisionLog | None,
    source: DecisionSource,
    tier: int,
) -> int:
    try:
        ctx.metadata[CONTEXT_KEY] = source
    except Exception:
        # ProxyContext.metadata may be a strict map; never let a stamping
        # failure block routing.
        log.debug("failed to stamp decision source", exc_info=True)
    if decision_log is not None:
        decision_log.record(source)
    return tier


__all__ = ["CAPABLE", "EFFICIENT", "pick_capable_first", "pick_efficient_first"]
