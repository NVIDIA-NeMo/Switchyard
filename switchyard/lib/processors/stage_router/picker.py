# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two pickers (strong-default / weak-default) that share override + scorer
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

WEAK: int = 0
STRONG: int = 1

# Override thresholds — tunable in one place. Promote to YAML if calibration
# diverges across deployments.
#: Force STRONG when the latest tool result hit a CRITICAL severity pattern.
SEVERITY_CRITICAL: float = 1.0
#: Force WEAK when `tests_passed` AND the agent has been working long enough
#: (turn_depth) with few writes — interpreted as the run already settled.
CLEAN_TESTS_MIN_TURN_DEPTH: int = 10
CLEAN_TESTS_MAX_WRITES: int = 1


async def pick_strong_default(
    ctx: "ProxyContext",
    confidence_threshold: float,
    classifier: "TierClassifier | None" = None,
    weights: "Mapping[str, float]" = DEFAULT_WEIGHTS,
    decision_log: StageRouterDecisionLog | None = None,
) -> int:
    """STRONG default. WEAK only when the scorer is confidently negative."""
    return await _pick(
        ctx,
        default_tier=STRONG,
        confidence_threshold=confidence_threshold,
        classifier=classifier,
        weights=weights,
        decision_log=decision_log,
    )


async def pick_weak_default(
    ctx: "ProxyContext",
    confidence_threshold: float,
    classifier: "TierClassifier | None" = None,
    weights: "Mapping[str, float]" = DEFAULT_WEIGHTS,
    decision_log: StageRouterDecisionLog | None = None,
) -> int:
    """WEAK default. STRONG only when the scorer is confidently positive."""
    return await _pick(
        ctx,
        default_tier=WEAK,
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

    dimensions = from_signal(signal)
    result = score(dimensions, weights=weights)
    if result.confidence >= confidence_threshold:
        tier = STRONG if result.score > 0 else WEAK
        return _record(ctx, decision_log, "dimensions", tier)

    if classifier is None:
        return _record(ctx, decision_log, "fall_open", default_tier)
    verdict = await classifier.classify(ctx, signal)
    if verdict == "strong":
        return _record(ctx, decision_log, "llm-classifier", STRONG)
    if verdict == "weak":
        return _record(ctx, decision_log, "llm-classifier", WEAK)
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
    """Non-negotiable, signal-derived shortcuts that bypass the scorer."""
    if signal.severity >= SEVERITY_CRITICAL:
        return STRONG
    if (
        signal.tests_passed
        and signal.turn_depth >= CLEAN_TESTS_MIN_TURN_DEPTH
        and signal.write_count <= CLEAN_TESTS_MAX_WRITES
    ):
        return WEAK
    return None


__all__ = ["STRONG", "WEAK", "pick_strong_default", "pick_weak_default"]
