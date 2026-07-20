# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two pickers (capable-first / efficient-first) that share override + scorer
logic; differ only in their fallback tier on low-confidence turns."""

import logging
from typing import TYPE_CHECKING

from switchyard.lib.processors.stage_router.decision_log import (
    CONTEXT_KEY,
    DecisionSource,
    StageRouterDecisionLog,
)
from switchyard.lib.processors.stage_router.dimensions import from_signal
from switchyard.lib.processors.stage_router.scorer import DEFAULT_WEIGHTS, score

if TYPE_CHECKING:
    from collections.abc import Mapping

    from switchyard.lib.processors.stage_router.classifier import TierClassifier
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.components import ToolResultSignal

log = logging.getLogger(__name__)

EFFICIENT: int = 0
CAPABLE: int = 1

# Override thresholds — tunable in one place. Promote to YAML if calibration
# diverges across deployments.
#: Force CAPABLE when the latest tool result hit a CRITICAL severity pattern.
SEVERITY_CRITICAL: float = 1.0


async def pick_capable_first(
    ctx: "ProxyContext",
    confidence_threshold: float,
    classifier: "TierClassifier | None" = None,
    weights: "Mapping[str, float]" = DEFAULT_WEIGHTS,
    decision_log: StageRouterDecisionLog | None = None,
) -> int:
    """CAPABLE default. EFFICIENT only when the scorer is confidently negative."""
    return await _pick(
        ctx,
        default_tier=CAPABLE,
        confidence_threshold=confidence_threshold,
        classifier=classifier,
        weights=weights,
        decision_log=decision_log,
    )


async def pick_efficient_first(
    ctx: "ProxyContext",
    confidence_threshold: float,
    classifier: "TierClassifier | None" = None,
    weights: "Mapping[str, float]" = DEFAULT_WEIGHTS,
    decision_log: StageRouterDecisionLog | None = None,
) -> int:
    """EFFICIENT default. CAPABLE only when the scorer is confidently positive."""
    return await _pick(
        ctx,
        default_tier=EFFICIENT,
        confidence_threshold=confidence_threshold,
        classifier=classifier,
        weights=weights,
        decision_log=decision_log,
    )


async def _pick(
    ctx: "ProxyContext",
    default_tier: int,
    confidence_threshold: float,
    classifier: "TierClassifier | None",
    weights: "Mapping[str, float]",
    decision_log: StageRouterDecisionLog | None,
) -> int:
    from switchyard_rust.components import (
        get_tool_result_signal,  # local import: heavy native module
    )

    signal = get_tool_result_signal(ctx)
    if signal is None:
        return _record(ctx, decision_log, "no_signal", default_tier)

    override = _apply_overrides(signal)
    if override is not None:
        return _record(ctx, decision_log, "override", override)

    # Settled run: a recent test-pass backed by a recent code change (write or edit),
    # with no error in the window — safe on the cheap tier for both pickers. The
    # severity gate keeps a windowed HARD error from being swallowed as "settled";
    # such a turn falls through to the scorer, where the error routes it to CAPABLE.
    if (
        signal.tests_passed
        and (signal.recent_write_count + signal.recent_edit_count) >= 1
        and not signal.severity > 0.0
    ):
        return _record(ctx, decision_log, "tests_passed", EFFICIENT)

    dimensions = from_signal(signal)
    # Both pickers share the scorer: severity / spinning / exploring corroborate
    # toward CAPABLE, production / clean-streak toward EFFICIENT. The
    # confidence_threshold is the corroboration dial (see scorer docstring); the two
    # pickers differ only in the low-confidence default tier below.
    result = score(dimensions, weights=weights)
    if result.confidence >= confidence_threshold:
        tier = CAPABLE if result.score > 0 else EFFICIENT
        return _record(ctx, decision_log, "dimensions", tier)

    if classifier is None:
        return _record(ctx, decision_log, "fall_open", default_tier)
    verdict = await classifier.classify(ctx, signal)
    if verdict == "capable":
        return _record(ctx, decision_log, "llm-classifier", CAPABLE)
    if verdict == "efficient":
        return _record(ctx, decision_log, "llm-classifier", EFFICIENT)
    return _record(ctx, decision_log, "fall_open", default_tier)


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


def _apply_overrides(signal: "ToolResultSignal") -> int | None:
    """Non-negotiable, signal-derived shortcuts that bypass the scorer.

    A CRITICAL severity or a context compaction always forces CAPABLE; both outrank
    the settled-run (`tests_passed`) shortcut handled in :func:`_pick`.

    Compaction resets the router's accumulated signals, so a task that had escalated
    de-escalates straight back to weak. Once the context has been compacted we force —
    and, because the summary stays in the prefix on every subsequent turn, hold — the
    strong tier, which is where a task hard enough to overflow its context belongs.
    """
    if signal.compacted:
        return CAPABLE
    if signal.severity >= SEVERITY_CRITICAL:
        return CAPABLE
    return None


__all__ = ["CAPABLE", "EFFICIENT", "pick_capable_first", "pick_efficient_first"]
