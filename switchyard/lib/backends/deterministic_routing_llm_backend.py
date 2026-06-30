# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic router dispatch backend.

This backend executes a pre-stamped routing decision. It does not score the
request or duplicate routing rules; upstream experimental processors choose a
tier and write it to ``ProxyContext.metadata``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from switchyard.lib import metrics, spans
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.proxy_context import CTX_ROUTER_NAME
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequestType

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest, ChatResponse

log = logging.getLogger(__name__)

#: ``ProxyContext.metadata`` key under which the upstream processor stamps the
#: chosen tier label. Read by :meth:`DeterministicRoutingLLMBackend._pick_tier`.
CTX_DETERMINISTIC_ROUTING_TIER = "_deterministic_routing_tier"


class DeterministicRoutingLLMBackend(LLMBackend):
    """Dispatch to the tier picked by the upstream router processor.

    Args:
        tiers: Mapping of tier label to ``(LLMBackend, model_name)``.
            Pass pre-built backends here when you want full control. For the
            common case, use :meth:`from_tiers` so the standard OpenAI /
            Anthropic backends get built from :class:`LlmTarget` config.
        default_tier: Tier label to dispatch to when the request processor
            did not run or stamped an unrecognised label. Must be a key of
            ``tiers``.
    """

    def __init__(
        self,
        *,
        tiers: dict[str, tuple[LLMBackend, str]],
        default_tier: str,
    ) -> None:
        if not tiers:
            raise ValueError(
                "DeterministicRoutingLLMBackend requires at least one tier",
            )
        if default_tier not in tiers:
            raise ValueError(
                f"default_tier {default_tier!r} not in tiers {sorted(tiers)}",
            )
        self._backends = {label: backend for label, (backend, _) in tiers.items()}
        self._models = {label: model for label, (_, model) in tiers.items()}
        self._default_tier = default_tier

        log.info(
            "DeterministicRoutingLLMBackend: tiers=%s default=%s",
            {label: model for label, (_, model) in tiers.items()},
            default_tier,
        )

    @classmethod
    def from_tiers(
        cls,
        *,
        tiers: Mapping[str, LlmTarget],
        default_tier: str,
    ) -> DeterministicRoutingLLMBackend:
        """Build a backend from a mapping of ``label -> LlmTarget``."""
        if not tiers:
            raise ValueError("from_tiers requires at least one tier")
        if default_tier not in tiers:
            raise ValueError(
                f"default_tier {default_tier!r} not in tiers {sorted(tiers)}",
            )

        built = {
            label: (_build_backend(target), target.model)
            for label, target in tiers.items()
        }
        return cls(tiers=built, default_tier=default_tier)

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [
            ChatRequestType.OPENAI_CHAT,
            ChatRequestType.OPENAI_RESPONSES,
            ChatRequestType.ANTHROPIC,
        ]

    async def startup(self) -> None:
        for backend in self._backends.values():
            await backend.startup()

    async def shutdown(self) -> None:
        for backend in reversed(list(self._backends.values())):
            await backend.shutdown()

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        from switchyard_rust.components import set_stats_route_label

        tier = self._pick_tier(ctx, request)
        model = self._models[tier]
        request.set_model(model)
        ctx.selected_target = tier
        ctx.selected_model = model
        # Stamp the tier label so ``selected_stats_tier`` returns
        # ``"strong"`` / ``"weak"`` for the response-side OTel ``tier``
        # attribution and the intake route label.
        set_stats_route_label(ctx, tier)
        ctx.metadata[CTX_ROUTER_NAME] = "deterministic"
        with spans.route_decision_span(
            router="deterministic",
            tier=tier,
            selected_model=model,
            selected_target=tier,
        ):
            pass
        metrics.record_routing_decision(router="deterministic", source="classifier", tier=tier)
        return await self._backends[tier].call(ctx, request)

    def _pick_tier(self, ctx: ProxyContext, request: ChatRequest) -> str:  # noqa: ARG002
        """Read the tier label stamped upstream; fall back defensively.

        ``ctx.selected_target`` is the routing runtime's source of truth — it
        gets rewritten to the configured ``fallback_target_on_evict`` after a
        context-window overflow, so honouring it (over the upstream-stamped
        metadata key) is what lets evict-and-reroute actually reach this
        backend's strong tier on the retry.
        """
        rerouted = ctx.selected_target
        if isinstance(rerouted, str) and rerouted in self._backends:
            return rerouted

        picked = ctx.metadata.get(CTX_DETERMINISTIC_ROUTING_TIER)
        if isinstance(picked, str) and picked in self._backends:
            return picked

        log.warning(
            "DeterministicRoutingLLMBackend: no valid tier on ctx "
            "(found %r); defaulting to %r.",
            picked,
            self._default_tier,
        )
        return self._default_tier


__all__ = [
    "CTX_DETERMINISTIC_ROUTING_TIER",
    "DeterministicRoutingLLMBackend",
]


def _build_backend(target: LlmTarget) -> LLMBackend:
    """Build a backend for the experimental deterministic router.

    This experimental path now uses the same Rust-owned native backend builder
    as production factories so routing behavior cannot drift.
    """
    from switchyard.lib.backends.multi_llm_backend import (
        build_native_backend,
        resolve_llm_target,
    )

    target = resolve_llm_target(target)
    return build_native_backend(target)
