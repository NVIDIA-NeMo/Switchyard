# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for stage_router profile classifier construction."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from switchyard.lib.processors.stage_router_request_processor import StageRouterRequestProcessor
from switchyard.lib.profiles.stage_router import StageRouterProfileConfig, _build_classifier
from switchyard.lib.profiles.stage_router_config import ClassifierConfig, StageRouterConfig
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest
from switchyard_rust.profiles import ProfileInput


class _ClassifierClient:
    """Fake async classifier client that returns one deterministic tier verdict."""

    def __init__(self, tier: str = "weak") -> None:
        """Store the tier returned by later classifier calls."""
        self._tier = tier
        self.calls = 0

    async def acompletion(self, **_kwargs: object) -> object:
        """Return a LiteLLM-shaped classifier response with token usage."""
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"tier": self._tier})),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
        )


def test_no_classifier_when_config_absent() -> None:
    assert _build_classifier(None) is None


def test_bedrock_claude_classifier_disables_reasoning() -> None:
    classifier = _build_classifier(
        ClassifierConfig(model="aws/anthropic/bedrock-claude-sonnet-4-6", api_key="sk"),
    )
    assert classifier is not None
    assert classifier._disable_reasoning is False


def test_deepseek_classifier_keeps_reasoning_disabled() -> None:
    classifier = _build_classifier(
        ClassifierConfig(model="nvidia/deepseek-ai/deepseek-v4-flash", api_key="sk"),
    )
    assert classifier is not None
    assert classifier._disable_reasoning is True


@pytest.mark.parametrize(
    ("picker", "expected_target"),
    [
        ("stage_router_strong_default", "weak"),
        ("stage_router_weak_default", "weak"),
    ],
)
async def test_runtime_stats_reach_profile_classifier(
    picker: str,
    expected_target: str,
) -> None:
    """Python stage_router classifier overhead is visible in shared routing stats."""
    stats = StatsAccumulator()
    config = StageRouterConfig.model_validate({
        "picker": picker,
        "confidence_threshold": 1.0,
        "fallback_target_on_evict": "strong",
        "strong": {
            "id": "strong",
            "model": "strong/model",
            "api_key": "strong-key",
            "base_url": "http://127.0.0.1:9/strong/v1",
        },
        "weak": {
            "id": "weak",
            "model": "weak/model",
            "api_key": "weak-key",
            "base_url": "http://127.0.0.1:9/weak/v1",
        },
        "classifier": {
            "model": "classifier/model",
            "api_key": "classifier-key",
            "base_url": "http://127.0.0.1:9/classifier/v1",
        },
    })
    profile = (
        StageRouterProfileConfig.from_config(config)
        .build()
        .with_runtime_components(stats_accumulator=stats, enable_stats=True)
    )
    processor = next(
        component
        for component in profile.iter_components()
        if isinstance(component, StageRouterRequestProcessor)
    )
    classifier = processor._classifier
    assert classifier is not None
    client = _ClassifierClient(tier="weak")
    classifier._client = client  # type: ignore[assignment]

    processed = await profile.process(ProfileInput(_chat_request()))

    snapshot: dict[str, Any] = await stats.snapshot()
    assert client.calls == 1
    assert processed.selected_target == expected_target
    assert classifier._stats is stats
    assert snapshot["classifier"]["total_requests"] == 1
    assert snapshot["classifier"]["total_errors"] == 0
    model_stats = snapshot["classifier"]["models"]["classifier/model"]
    assert model_stats["calls"] == 1
    assert model_stats["prompt_tokens"] == 11
    assert model_stats["completion_tokens"] == 7
    assert model_stats["cached_tokens"] == 3
    assert model_stats["model_call_latency"]["count"] == 1
    assert snapshot["routing_decisions"]["stage_router"]["llm-classifier"] == 1


def _chat_request() -> ChatRequest:
    """Build the OpenAI request shape passed through the stage_router profile."""
    return ChatRequest.openai_chat({
        "model": "stage_router-route",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2? Reply with just the number.",
            }
        ],
        "max_tokens": 8,
    })
