# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python port of the libsy AgentApi optimizer interfaces and routers.

Public entry point is the `route` decorator, which drives an async model-call
function through a routing algorithm (`RandomRouter` or `LlmClassifier`).
"""

from switchyard.lib.agentapi.chat import ChatRequest, ChatResponse, EnrichmentData
from switchyard.lib.agentapi.decorator import route
from switchyard.lib.agentapi.llm_class import (
    ClassifierRoutingDecision,
    ClassifierTier,
    LlmClassifier,
)
from switchyard.lib.agentapi.optimizer import (
    AgentApiOptAlgorithm,
    AgentApiOptimizer,
    Decision,
    MetadataInput,
    ModelInference,
    OptimizerResponse,
    RequestInput,
    ResponseInput,
    Return,
)
from switchyard.lib.agentapi.rand import RandomRouter, RandomRoutingDecision, WeightedModel

__all__ = [
    "route",
    "RandomRouter",
    "RandomRoutingDecision",
    "WeightedModel",
    "LlmClassifier",
    "ClassifierRoutingDecision",
    "ClassifierTier",
    "ChatRequest",
    "ChatResponse",
    "EnrichmentData",
    "AgentApiOptimizer",
    "AgentApiOptAlgorithm",
    "Decision",
    "ModelInference",
    "Return",
    "OptimizerResponse",
    "RequestInput",
    "ResponseInput",
    "MetadataInput",
]
