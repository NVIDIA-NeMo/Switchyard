# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-id compatibility check for the vLLM ``enable_thinking`` hint."""

from __future__ import annotations

# Anthropic-served (Claude) ids reject the vLLM ``chat_template_kwargs``
# field, and Fireworks AI's OpenAI-compatible API validates request bodies
# strictly and 400s on it as well ("Extra inputs are not permitted");
# Fireworks ids always carry the provider namespace
# (``accounts/fireworks/models/...``). vLLM-served reasoning models that
# need the hint never carry these tokens.
_NO_REASONING_HINT_TAGS = ("anthropic", "bedrock", "claude", "fireworks")


def model_accepts_reasoning_hint(model: str) -> bool:
    """Whether ``model`` tolerates the vLLM ``enable_thinking=False`` hint.

    ``False`` for Anthropic/Bedrock/Claude and Fireworks-served ids (they
    400 on it), else ``True`` — usable directly as a classifier
    ``disable_reasoning`` default.
    """
    lowered = model.lower()
    return not any(tag in lowered for tag in _NO_REASONING_HINT_TAGS)


__all__ = ["model_accepts_reasoning_hint"]
