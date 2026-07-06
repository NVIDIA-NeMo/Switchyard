# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier picker component — stamps ``ctx.selected_target``/``selected_model``.
Fails open to the weak tier on picker exceptions."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from switchyard.lib.processors.stage_router import (
    CONTEXT_KEY,
    STRONG,
    WEAK,
    StageRouterDecisionLog,
    pick_strong_default,
    pick_weak_default,
)
from switchyard.lib.processors.stage_router.classifier import RECENT_MESSAGES_KEY, TierClassifier

if TYPE_CHECKING:
    from switchyard.lib.backends.llm_target import LlmTarget
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.stats_accumulator import StatsAccumulator
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: Async picker signature. The factory pre-binds knobs and the optional classifier.
TierPicker = Callable[["ProxyContext"], Awaitable[int]]

#: YAML-resolvable picker names; mirrors :class:`StageRouterPickerMode`.
BUILTIN_PICKERS: dict[str, Callable[..., Awaitable[int]]] = {
    "stage_router_strong_default": pick_strong_default,
    "stage_router_weak_default": pick_weak_default,
}


class StageRouterRequestProcessor:
    """Picks a tier and stamps it on the context. Policy lives in the picker."""

    def __init__(
        self,
        *,
        targets: Sequence[LlmTarget],
        picker: TierPicker,
        classifier: TierClassifier | None = None,
        decision_log: StageRouterDecisionLog | None = None,
    ) -> None:
        if len(targets) != 2:
            raise ValueError(f"stage_router requires exactly 2 targets, got {len(targets)}")
        self._target_ids = [t.id for t in targets]
        self._target_models = [t.model for t in targets]
        self._picker = picker
        self._classifier = classifier
        self._max_index = len(targets) - 1
        self._decision_log = decision_log if decision_log is not None else StageRouterDecisionLog()
        self._stats_accumulator: StatsAccumulator | None = None

    def attach_stats_accumulator(self, stats_accumulator: StatsAccumulator) -> None:
        """Attach serving-level stats to stage_router-only routing components."""
        self._stats_accumulator = stats_accumulator
        if self._classifier is not None:
            self._classifier.attach_stats_accumulator(stats_accumulator)

    def decision_stats(self) -> dict[str, int]:
        """Snapshot of decision-source counts since process start."""
        return self._decision_log.snapshot()

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        # Stash trailing messages for the classifier when one is configured.
        try:
            body = request.body
            if isinstance(body, dict):
                messages = body.get("messages")
                if isinstance(messages, list):
                    ctx.metadata[RECENT_MESSAGES_KEY] = messages
        except Exception:
            log.debug("failed to stash request messages on ctx", exc_info=True)
        idx = await self._resolve_index(ctx)
        ctx.selected_target = self._target_ids[idx]
        ctx.selected_model = self._target_models[idx]
        await self._record_decision_source(ctx)
        log.debug(
            "stage_router pick: idx=%d target=%s model=%s",
            idx, ctx.selected_target, ctx.selected_model,
        )
        return request

    async def _resolve_index(self, ctx: ProxyContext) -> int:
        try:
            idx = await self._picker(ctx)
        except Exception:
            log.exception("stage_router picker raised; falling back to index 0 (weak)")
            return WEAK
        return max(0, min(idx, self._max_index))

    async def _record_decision_source(self, ctx: ProxyContext) -> None:
        """Copy the picker source stamp into shared routing stats when available."""
        if self._stats_accumulator is None:
            return
        source = ctx.metadata.get(CONTEXT_KEY)
        if not isinstance(source, str) or not source:
            return
        try:
            await cast(Any, self._stats_accumulator).record_routing_decision(
                "stage_router",
                source,
            )
        except Exception:
            log.debug("failed to record stage_router decision source", exc_info=True)


__all__ = [
    "BUILTIN_PICKERS",
    "STRONG",
    "WEAK",
    "StageRouterRequestProcessor",
    "TierPicker",
    "pick_strong_default",
    "pick_weak_default",
]
