# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python compatibility random-routing request processor."""

from __future__ import annotations

import random
from typing import Any

from switchyard_rust.components import RandomRoutingProcessorConfig


class RandomRoutingRequestProcessor:
    """Rewrite requests to the randomly selected strong or weak target.

    This is a compatibility component for the current Python profile chain.
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
        return request

__all__ = ["RandomRoutingRequestProcessor"]
