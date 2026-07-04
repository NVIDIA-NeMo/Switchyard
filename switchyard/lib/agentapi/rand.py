# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted random router (ports rand.rs) built on the optimizer interfaces."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from switchyard.lib.agentapi.chat import ChatRequest, EnrichmentData
from switchyard.lib.agentapi.optimizer import (
    AgentApiOptAlgorithm,
    AgentApiOptimizer,
    Decision,
    ModelInference,
    OptimizerResponse,
    OptInput,
    RequestInput,
    ResponseInput,
    Return,
)


@dataclass
class WeightedModel:
    """A routing target and its relative (non-negative, finite) selection weight."""

    model: str
    weight: float


@dataclass
class RandomRoutingDecision:
    """Decision info attached to a random-routing ModelInference."""

    selected_model: str
    draw: float
    total_weight: float


class RandomRouter(AgentApiOptAlgorithm):
    """Factory minting a fresh weighted-random optimizer per session.

    Weights are relative; a target is chosen with probability
    weight / sum(weights). A seed makes routing reproducible.
    """

    def __init__(self, models: list[WeightedModel], rng_seed: int | None = None) -> None:
        self.models = models
        self.rng_seed = rng_seed

    def optimizer(self) -> AgentApiOptimizer:
        rng = random.Random(self.rng_seed) if self.rng_seed is not None else random.Random()
        return _RandomOptimizer(list(self.models), rng)


class _RandomOptimizer(AgentApiOptimizer):
    """Per-session random router over a weighted set of N targets."""

    def __init__(self, models: list[WeightedModel], rng: random.Random) -> None:
        self._models = models
        self._rng = rng
        self._pending_request: ChatRequest | None = None
        self._completed = False

    async def feed(self, input: OptInput, enrichment: EnrichmentData) -> None:
        if isinstance(input, RequestInput):
            self._pending_request = input.request
        elif isinstance(input, ResponseInput):
            # A fed response means the caller performed the model call; the next
            # optimize should hand control back to the agent.
            self._completed = True

    async def optimize(self) -> Decision:
        if self._completed:
            return Return()
        if self._pending_request is None:
            raise ValueError("optimize called before a request was fed")
        request = self._pending_request
        self._pending_request = None
        decision = self._select()
        request.model = decision.selected_model
        return ModelInference(
            OptimizerResponse(
                requests=[request],
                decision_reasoning=(
                    f"weighted random draw {decision.draw} of total weight "
                    f"{decision.total_weight}; selected {decision.selected_model}"
                ),
                decision_info=decision,
            )
        )

    def _select(self) -> RandomRoutingDecision:
        """Draw a weighted target; error if no target has positive finite weight."""
        selectable = [m for m in self._models if math.isfinite(m.weight) and m.weight > 0.0]
        total_weight = sum(m.weight for m in selectable)
        if total_weight <= 0.0:
            raise ValueError("random router has no target with positive weight")
        draw = self._rng.random() * total_weight
        cumulative = 0.0
        for model in selectable:
            cumulative += model.weight
            if draw < cumulative:
                return RandomRoutingDecision(model.model, draw, total_weight)
        # Floating-point rounding can leave the draw at the top of the range;
        # fall back to the last positively-weighted target.
        return RandomRoutingDecision(selectable[-1].model, draw, total_weight)
