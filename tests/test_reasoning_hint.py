# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :func:`model_accepts_reasoning_hint`."""

from __future__ import annotations

import pytest

from switchyard.lib.processors.reasoning_hint import model_accepts_reasoning_hint


@pytest.mark.parametrize(
    "model",
    [
        "aws/anthropic/bedrock-claude-sonnet-4-6",
        "aws/anthropic/bedrock-claude-opus-4-8",
        "azure/anthropic/claude-opus-4-6",
        "anthropic/claude-3-5-sonnet",
        "Bedrock-Claude-Whatever",  # case-insensitive
        "accounts/fireworks/models/deepseek-v4-flash",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/glm-5p2",
    ],
)
def test_claude_family_rejects_hint(model: str) -> None:
    assert model_accepts_reasoning_hint(model) is False


@pytest.mark.parametrize(
    "model",
    [
        "nvidia/deepseek-ai/deepseek-v4-flash",
        "nvidia/deepseek-ai/evals-deepseek-v4-pro",
        "nvidia/nvidia/nemotron-3-super-v3",
        "openai/openai/gpt-5.2",
        "router-model",
    ],
)
def test_non_claude_models_accept_hint(model: str) -> None:
    assert model_accepts_reasoning_hint(model) is True
