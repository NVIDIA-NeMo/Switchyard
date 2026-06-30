# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python compatibility random-routing request processor."""

from __future__ import annotations

import random
from typing import Any

from switchyard.lib import metrics, spans
from switchyard.lib.proxy_context import CTX_ROUTER_NAME
from switchyard_rust.components import RandomRoutingProcessorConfig


class RandomRoutingRequestProcessor:
    """Rewrite requests to the randomly selected strong or weak target.

    This is a compatibility component for the current Python profile chain.
    New Rust serving code uses components-v2 profiles directly instead of
    exporting a Rust request-processor object.
    """

    def __init__(self, config: RandomRoutingProcessorConfig) -> None:
        """Create a processor from validated random-routing config."""
        self.config = config
        self._rng = random.Random(config.rng_seed)

    async def startup(self) -> None:
        """Start the processor; random routing has no owned resources."""

    async def shutdown(self) -> None:
        """Stop the processor; random routing has no owned resources."""

    def select(self, original_model: str | None = None) -> dict[str, Any]:
        """Select a target and return the routing decision dictionary."""
        draw = self._rng.random()
        if draw < self.config.strong_probability:
            tier = "strong"
            target = self.config.strong
        else:
            tier = "weak"
            target = self.config.weak
        return {
            "tier": tier,
            "selected_target": target.id,
            "selected_model": target.model,
            "original_model": original_model,
            "strong_probability": self.config.strong_probability,
            "draw": draw,
        }

    async def process(self, ctx: Any, request: Any) -> Any:
        """Rewrite the request model and stamp the selected target on context."""
        decision = self.select(request.model)
        request.set_model(decision["selected_model"])
        ctx.selected_target = decision["selected_target"]
        ctx.selected_model = decision["selected_model"]
        # Stamp tier + router so executor metrics and the routing-decision counter
        # can label this request's coin-flip outcome.
        ctx.metadata["_random_routing_tier"] = decision["tier"]
        ctx.metadata[CTX_ROUTER_NAME] = "random"
        with spans.route_decision_span(
            router="random",
            tier=decision["tier"],
            selected_model=decision["selected_model"],
            selected_target=decision["selected_target"],
            original_model=decision["original_model"],
            draw=decision["draw"],
            strong_probability=decision["strong_probability"],
        ):
            pass
        metrics.record_routing_decision(router="random", source="coin", tier=decision["tier"])
        return request

__all__ = ["RandomRoutingRequestProcessor"]
