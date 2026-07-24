# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StageRouter — Rust two-axis scorer + picker, with a selective LLM classifier.

The routing logic lives in Rust (``switchyard_components::stage_router``); this
package is the Python shell: the picker orchestration, the async classifier,
handoff notes, and decision logging."""

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

__all__ = [
    "CONTEXT_KEY",
    "DEFAULT_ESCALATION_NOTE",
    "CAPABLE",
    "CAPABLE_TIER",
    "HandoffNoteInjector",
    "StageRouterDecisionLog",
    "DecisionSource",
    "TierClassifier",
    "EFFICIENT",
    "EFFICIENT_TIER",
    "pick_capable_first",
    "pick_efficient_first",
]
