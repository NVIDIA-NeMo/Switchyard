# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize non-standard ``reasoning_effort`` values before backend dispatch.

Claude Code's ``/effort`` picker offers an ``xhigh`` level (and the
Codex launcher's model catalog mirrors it). Both upstream paths reject
that value: NVIDIA Inference Hub's LiteLLM passthrough returns HTTP
500 with ``Invalid effort value: xhigh. Must be one of: 'high',
'medium', 'low', 'max'.`` for Azure Anthropic, and
``Unmapped reasoning effort: xhigh`` for Bedrock.

This processor runs early in the chain and normalizes the request body
so the upstream never sees an unsupported value. Unknown values are
mapped to ``high`` rather than stripped because the user's intent
(``xhigh`` means "as much reasoning as possible") is closer to ``high``
than to absent.
"""

from __future__ import annotations

import logging

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: OpenAI-compatible values the upstream accepts. ``"max"`` is non-standard
#: but supported by NVIDIA Hub's LiteLLM for reasoning-budget overrides.
_VALID_REASONING_EFFORT = frozenset({"low", "medium", "high", "max"})

#: Aliases the upstream rejects → the nearest valid value.
_REASONING_EFFORT_ALIASES = {
    "xhigh": "high",
}


class ReasoningEffortNormalizer:
    """Normalize ``request.body["reasoning_effort"]`` to an upstream-valid value.

    No-op when the field is absent or already a valid value. Maps known
    aliases (``xhigh`` → ``high``); for unrecognized values, replaces
    with ``high`` and emits a warning so operators can spot
    misconfigured client-side enums.
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:  # noqa: ARG002
        body = request.body
        if not isinstance(body, dict):
            return request
        effort = body.get("reasoning_effort")
        if not isinstance(effort, str) or effort in _VALID_REASONING_EFFORT:
            return request

        mapped = _REASONING_EFFORT_ALIASES.get(effort, "high")
        log.warning(
            "ReasoningEffortNormalizer: unsupported reasoning_effort=%r; "
            "normalizing to %r before dispatch",
            effort,
            mapped,
        )
        body["reasoning_effort"] = mapped
        request.replace_body(body)
        return request


__all__ = ["ReasoningEffortNormalizer"]
