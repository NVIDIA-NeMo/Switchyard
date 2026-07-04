# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer role interfaces ported from libsy's lib.rs.

An AgentApiOptimizer is a per-session state machine fed inputs (requests,
responses, metadata) that yields routing/optimization Decisions. An
AgentApiOptAlgorithm is the factory that mints a fresh optimizer per session.

Deviation from the Rust source: the Response input variant here wraps a
ChatResponse (its completion text) rather than reusing ChatRequest, which is the
evident intent of the Rust `AgentApiOptInput::Response`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from switchyard.lib.agentapi.chat import ChatRequest, ChatResponse, EnrichmentData


class OptInput:
    """Base for optimizer inputs (request / response / metadata)."""


@dataclass
class RequestInput(OptInput):
    """An inbound request to route."""

    request: ChatRequest


@dataclass
class ResponseInput(OptInput):
    """A model response fed back after the caller performed a model call."""

    response: ChatResponse


@dataclass
class MetadataInput(OptInput):
    """Out-of-band metadata for routing."""

    metadata: dict[str, str]


@dataclass
class OptimizerResponse:
    """Payload of a ModelInference decision: the model calls to perform."""

    requests: list[ChatRequest]
    enrichment_data: list[EnrichmentData] = field(default_factory=list)
    decision_reasoning: str | None = None
    decision_info: object | None = None


class Decision:
    """Base for optimizer decisions (ModelInference / Return)."""


@dataclass
class ModelInference(Decision):
    """The caller should perform the listed model calls and feed responses back."""

    response: OptimizerResponse


@dataclass
class Return(Decision):
    """The optimizer is done; hand control back to the calling agent."""


class AgentApiOptimizer(abc.ABC):
    """Per-session stateful optimizer driven by feed() -> optimize()."""

    async def feed(self, input: OptInput, enrichment: EnrichmentData) -> None:  # noqa: B027
        """Feed a new input; default is a no-op (override to accumulate state)."""

    @abc.abstractmethod
    async def optimize(self) -> Decision:
        """Make the next routing/optimization decision from current state."""


class AgentApiOptAlgorithm(abc.ABC):
    """Factory that mints a fresh optimizer per session."""

    @abc.abstractmethod
    def optimizer(self) -> AgentApiOptimizer:
        """Create a new optimizer instance for a session."""
