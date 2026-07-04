# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agentapi subpackage exports its public API."""

from __future__ import annotations

import switchyard.lib.agentapi as agentapi


def test_public_exports_present():
    for name in [
        "route",
        "RandomRouter",
        "WeightedModel",
        "LlmClassifier",
        "ClassifierTier",
        "ChatRequest",
        "ChatResponse",
        "EnrichmentData",
        "AgentApiOptimizer",
        "AgentApiOptAlgorithm",
        "Decision",
        "ModelInference",
        "Return",
    ]:
        assert hasattr(agentapi, name), name
        assert name in agentapi.__all__
