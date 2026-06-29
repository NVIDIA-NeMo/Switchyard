# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier picker component — stamps ``ctx.selected_target``/``selected_model``.
Fails open to the weak tier on picker exceptions."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from switchyard.lib import metrics, spans
from switchyard.lib.processors.cascade import (
    CONTEXT_KEY,
    STRONG,
    WEAK,
    CascadeDecisionLog,
    pick_strong_default,
    pick_weak_default,
)
from switchyard.lib.processors.cascade.classifier import RECENT_MESSAGES_KEY, TierClassifier
from switchyard.lib.proxy_context import CTX_ROUTER_NAME

if TYPE_CHECKING:
    from switchyard.lib.backends.llm_target import LlmTarget
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: Async picker signature. The factory pre-binds knobs and the optional classifier.
TierPicker = Callable[["ProxyContext"], Awaitable[int]]

#: YAML-resolvable picker names; mirrors :class:`CascadePickerMode`.
BUILTIN_PICKERS: dict[str, Callable[..., Awaitable[int]]] = {
    "cascade_strong_default": pick_strong_default,
    "cascade_weak_default": pick_weak_default,
}


class CascadeRequestProcessor:
    """Picks a tier and stamps it on the context. Policy lives in the picker."""

    def __init__(
        self,
        *,
        targets: Sequence[LlmTarget],
        picker: TierPicker,
        classifier: TierClassifier | None = None,
        decision_log: CascadeDecisionLog | None = None,
    ) -> None:
        if len(targets) != 2:
            raise ValueError(f"cascade requires exactly 2 targets, got {len(targets)}")
        self._target_ids = [t.id for t in targets]
        self._target_models = [t.model for t in targets]
        self._picker = picker
        self._classifier = classifier
        self._max_index = len(targets) - 1
        self._decision_log = decision_log if decision_log is not None else CascadeDecisionLog()

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
        tier = "strong" if idx == STRONG else "weak"
        source = ctx.metadata.get(CONTEXT_KEY)
        ctx.metadata[CTX_ROUTER_NAME] = "cascade"
        with spans.route_decision_span(
            router="cascade",
            tier=tier,
            selected_model=ctx.selected_model,
            selected_target=ctx.selected_target,
            source=source if isinstance(source, str) else None,
        ):
            pass
        metrics.record_routing_decision(
            router="cascade",
            source=source if isinstance(source, str) and source else None,
            tier=tier,
        )
        log.debug(
            "cascade pick: idx=%d target=%s model=%s",
            idx, ctx.selected_target, ctx.selected_model,
        )
        return request

    async def _resolve_index(self, ctx: ProxyContext) -> int:
        try:
            idx = await self._picker(ctx)
        except Exception:
            log.exception("cascade picker raised; falling back to index 0 (weak)")
            return WEAK
        return max(0, min(idx, self._max_index))


__all__ = [
    "BUILTIN_PICKERS",
    "STRONG",
    "WEAK",
    "CascadeRequestProcessor",
    "TierPicker",
    "pick_strong_default",
    "pick_weak_default",
]
