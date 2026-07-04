# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-classifier router (ports llm_class.rs).

Unlike a local ML classifier, an LLM classifier needs its own model call to
score the request. That maps onto the optimizer's multi-round loop: the router
first asks the caller to run a classifier model, then — once the score is fed
back — asks the caller to run the routed target model, and finally returns
control to the agent.
"""

from __future__ import annotations

import enum
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

# Prepended to the user prompt when asking the classifier for a strong-win-rate
# score. Matches the resolved value of the Rust literal.
CLASSIFIER_PROMPT_PREAMBLE = (
    "Rate how strongly this request needs a frontier model. "
    "Reply with a single strong-win-rate score in [0, 1]:\n"
)


class ClassifierTier(enum.Enum):
    """The tier a classifier score selected."""

    STRONG = "strong"
    WEAK = "weak"

    def as_str(self) -> str:
        return self.value


@dataclass
class ClassifierRoutingDecision:
    """Decision info attached to the routed ModelInference decision."""

    score: float | None
    threshold: float
    tier: ClassifierTier
    selected_model: str


class _Phase(enum.Enum):
    """Where the router is in its classify -> route -> return lifecycle."""

    AWAITING_REQUEST = enum.auto()
    CLASSIFY = enum.auto()
    AWAITING_SCORE = enum.auto()
    ROUTE = enum.auto()
    AWAITING_RESPONSE = enum.auto()
    DONE = enum.auto()


class LlmClassifier(AgentApiOptAlgorithm):
    """Factory minting a fresh LLM-classifier optimizer per session."""

    def __init__(
        self,
        classifier_model: str,
        strong_model: str,
        weak_model: str,
        threshold: float,
    ) -> None:
        self.classifier_model = classifier_model
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.threshold = threshold

    def optimizer(self) -> AgentApiOptimizer:
        return _ClassifierOptimizer(
            self.classifier_model, self.strong_model, self.weak_model, self.threshold
        )


class _ClassifierOptimizer(AgentApiOptimizer):
    """Per-session LLM-classifier router."""

    def __init__(
        self, classifier_model: str, strong_model: str, weak_model: str, threshold: float
    ) -> None:
        self._classifier_model = classifier_model
        self._strong_model = strong_model
        self._weak_model = weak_model
        self._threshold = threshold
        self._phase = _Phase.AWAITING_REQUEST
        self._pending_request: ChatRequest | None = None
        self._score: float | None = None

    async def feed(self, input: OptInput, enrichment: EnrichmentData) -> None:
        if isinstance(input, RequestInput):
            self._pending_request = input.request
            self._phase = _Phase.CLASSIFY
        elif isinstance(input, ResponseInput):
            if self._phase is _Phase.AWAITING_SCORE:
                self._score = _parse_score(input.response.completion)
                self._phase = _Phase.ROUTE
            elif self._phase is _Phase.AWAITING_RESPONSE:
                self._phase = _Phase.DONE
            else:
                raise ValueError(
                    "classifier router received a response outside a pending model call"
                )

    async def optimize(self) -> Decision:
        if self._phase is _Phase.AWAITING_REQUEST:
            raise ValueError("optimize called before a request was fed")
        if self._phase is _Phase.CLASSIFY:
            assert self._pending_request is not None
            user_prompt = self._pending_request.prompt
            self._phase = _Phase.AWAITING_SCORE
            classifier_request = ChatRequest(
                prompt=f"{CLASSIFIER_PROMPT_PREAMBLE}{user_prompt}",
                model=self._classifier_model,
            )
            return ModelInference(
                OptimizerResponse(
                    requests=[classifier_request],
                    decision_reasoning=f"classifying request via {self._classifier_model}",
                )
            )
        if self._phase is _Phase.AWAITING_SCORE:
            raise ValueError("optimize called before the classifier response was fed")
        if self._phase is _Phase.ROUTE:
            assert self._pending_request is not None
            request = self._pending_request
            self._pending_request = None
            decision = self._decide()
            request.model = decision.selected_model
            self._phase = _Phase.AWAITING_RESPONSE
            return ModelInference(
                OptimizerResponse(
                    requests=[request],
                    decision_reasoning=(
                        f"classifier score {decision.score} vs threshold "
                        f"{decision.threshold}; selected {decision.selected_model} "
                        f"({decision.tier.as_str()})"
                    ),
                    decision_info=decision,
                )
            )
        if self._phase is _Phase.AWAITING_RESPONSE:
            raise ValueError("optimize called before the model response was fed")
        # _Phase.DONE
        return Return()

    def _decide(self) -> ClassifierRoutingDecision:
        """Build the routing decision, defaulting to strong on an unusable score."""
        if self._score is not None and self._score >= self._threshold:
            tier, model = ClassifierTier.STRONG, self._strong_model
        elif self._score is not None:
            tier, model = ClassifierTier.WEAK, self._weak_model
        else:
            # Defensive default: keep traffic flowing on the strong tier when the
            # classifier response was unparseable.
            tier, model = ClassifierTier.STRONG, self._strong_model
        return ClassifierRoutingDecision(self._score, self._threshold, tier, model)


def _parse_score(text: str) -> float | None:
    """Parse a strong-win-rate score, returning None when unparseable."""
    try:
        return float(text.strip())
    except ValueError:
        return None
