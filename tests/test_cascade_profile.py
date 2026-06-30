# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cascade profile classifier construction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from switchyard.lib import metrics
from switchyard.lib.processors.cascade_request_processor import CascadeRequestProcessor
from switchyard.lib.profiles.cascade import CascadeProfileConfig, _build_classifier
from switchyard.lib.profiles.cascade_config import CascadeConfig, ClassifierConfig
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
        ("cascade_strong_default", "weak"),
        ("cascade_weak_default", "weak"),
    ],
)
async def test_runtime_classifier_records_tokens_via_otel(
    picker: str,
    expected_target: str,
) -> None:
    """Python cascade classifier overhead is recorded via OTel (tier=classifier)."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.reset_for_test(provider.get_meter("switchyard"))
    config = CascadeConfig.model_validate({
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
        CascadeProfileConfig.from_config(config)
        .build()
        .with_runtime_components()
    )
    processor = next(
        component
        for component in profile.iter_components()
        if isinstance(component, CascadeRequestProcessor)
    )
    classifier = processor._classifier
    assert classifier is not None
    client = _ClassifierClient(tier="weak")
    classifier._client = client  # type: ignore[assignment]

    processed = await profile.process(ProfileInput(_chat_request()))

    assert client.calls == 1
    assert processed.selected_target == expected_target

    # Classifier token spend is tagged tier="classifier" via OTel.
    counters: dict[str, int] = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for point in metric.data.data_points:
                    if point.attributes.get("tier") == "classifier":
                        counters[metric.name] = counters.get(metric.name, 0) + point.value
    assert counters["switchyard.prompt_tokens"] == 11
    assert counters["switchyard.completion_tokens"] == 7
    assert counters["switchyard.cached_tokens"] == 3
    metrics.reset_for_test(None)


def _chat_request() -> ChatRequest:
    """Build the OpenAI request shape passed through the cascade profile."""
    return ChatRequest.openai_chat({
        "model": "cascade-route",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2? Reply with just the number.",
            }
        ],
        "max_tokens": 8,
    })
