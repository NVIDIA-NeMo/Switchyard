# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind Random routing to the candidates supplied by LiteLLM."""

from __future__ import annotations

from collections.abc import Sequence

from litellm.types.router import RoutingContext

from switchyard.libsy import algorithms

from .lite_llm_request_rewriter import LiteLLMRequestRewriter
from .switchyard_routing_plugin import SwitchyardRoutingPlugin


class RandomRoutingPlugin(LiteLLMRequestRewriter):
    """Randomly select from the current unique LiteLLM candidates."""

    def __init__(
        self,
        *,
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._weights = tuple(weights) if weights is not None else None
        self._seed = seed
        self._plugins: dict[tuple[str, ...], SwitchyardRoutingPlugin] = {}

    async def run(self, context: RoutingContext) -> RoutingContext:
        """Reuse one Random algorithm for each live LiteLLM candidate pool."""
        candidates = tuple(dict.fromkeys(context.candidate_models))
        if not candidates:
            raise ValueError("Random routing requires at least one LiteLLM candidate")
        plugin = self._plugins.get(candidates)
        if plugin is None:
            plugin = SwitchyardRoutingPlugin(
                algorithms.random(candidates, weights=self._weights, seed=self._seed)
            )
            self._plugins[candidates] = plugin
        return await plugin.run(context)


__all__ = ["RandomRoutingPlugin"]
