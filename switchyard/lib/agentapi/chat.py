# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal request/response/enrichment types shared by the optimizers.

Ports libsy's ChatRequest / ChatResponse / EnrichementData. These are the
provider-neutral representations the optimizers reason about.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatRequest:
    """Provider-neutral request: a single prompt aimed at a model id."""

    prompt: str
    model: str


@dataclass
class ChatResponse:
    """Provider-neutral response carrying the completion text."""

    completion: str


@dataclass
class EnrichmentData:
    """Correlation/enrichment metadata used for routing decisions."""

    session_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    correlation_id: str | None = None
    extra_metadata: dict[str, str] | None = None
