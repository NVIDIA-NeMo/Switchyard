# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StageRouter — weighted scorer + selective LLM-classifier."""

from switchyard.lib.processors.stage_router.classifier import (
    CAPABLE_TIER,
    EFFICIENT_TIER,
    TierClassifier,
)
from switchyard.lib.processors.stage_router.decision_log import (
    CONTEXT_KEY,
    DecisionSource,
    StageRouterDecisionLog,
)
from switchyard.lib.processors.stage_router.dimensions import (
    CodingAgentDimensions,
    from_signal,
)
from switchyard.lib.processors.stage_router.handoff_notes import (
    DEFAULT_ESCALATION_NOTE,
    HandoffNoteInjector,
)
from switchyard.lib.processors.stage_router.picker import (
    CAPABLE,
    EFFICIENT,
    pick_capable_first,
    pick_efficient_first,
)
from switchyard.lib.processors.stage_router.scorer import (
    DEFAULT_WEIGHTS,
    ScoreResult,
    score,
)

__all__ = [
    "CONTEXT_KEY",
    "DEFAULT_ESCALATION_NOTE",
    "DEFAULT_WEIGHTS",
    "CAPABLE",
    "CAPABLE_TIER",
    "HandoffNoteInjector",
    "StageRouterDecisionLog",
    "CodingAgentDimensions",
    "DecisionSource",
    "ScoreResult",
    "TierClassifier",
    "EFFICIENT",
    "EFFICIENT_TIER",
    "from_signal",
    "pick_capable_first",
    "pick_efficient_first",
    "score",
]
