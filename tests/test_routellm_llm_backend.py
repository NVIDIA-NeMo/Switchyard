# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RouteLLM backend dispatch now runs through Rust ``MultiLlmBackend``."""

from __future__ import annotations

import pytest

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.routellm_request_processor import (
    CTX_ROUTELLM_TIER,
    RouteLLMRequestProcessor,
)
from switchyard.lib.profiles.routellm import RouteLLMConfig, RouteLLMProfileConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.components import MultiLlmBackend, StatsLlmBackend
from switchyard_rust.core import ChatRequest


class _FakeClassifier:
    def __init__(self, score: float) -> None:
        self.score = score

    def calculate_strong_win_rate(self, prompt: str) -> float:
        return self.score


def _target(target_id: str, model: str) -> LlmTarget:
    return LlmTarget(
        id=target_id,
        model=model,
        format=BackendFormat.OPENAI,
        api_key=f"sk-{target_id}",
        base_url="https://example.invalid/v1",
    )


def _config(*, enable_stats: bool = False) -> RouteLLMConfig:
    return RouteLLMConfig(
        strong=_target("strong", "strong-model"),
        weak=_target("weak", "weak-model"),
        threshold=0.5,
        router_type="bert",
        enable_stats=enable_stats,
        fallback_target_on_evict="strong",
    )


def _request(model: str = "client-model") -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })


def test_profile_builds_multi_llm_backend_without_stats() -> None:
    profile = RouteLLMProfileConfig.from_config(_config(enable_stats=False)).build()
    backend = next(component for component in profile.iter_components() if isinstance(component, MultiLlmBackend))

    assert isinstance(backend, MultiLlmBackend)
    assert backend.target_ids() == ["strong", "weak"]
    assert [item.value for item in backend.supported_request_types] == [
        "openai_chat",
        "openai_responses",
        "anthropic",
        "gemini",
    ]


def test_profile_wraps_multi_llm_backend_with_stats_when_enabled() -> None:
    config = _config(enable_stats=True)
    profile = (
        RouteLLMProfileConfig.from_config(config)
        .build()
        .with_runtime_components(enable_stats=config.enable_stats)
    )
    backend = next(component for component in profile.iter_components() if isinstance(component, StatsLlmBackend))

    assert isinstance(backend, StatsLlmBackend)
    assert [item.value for item in backend.supported_request_types] == [
        "openai_chat",
        "openai_responses",
        "anthropic",
        "gemini",
    ]


async def test_request_processor_stamps_selected_target_for_multi_backend() -> None:
    processor = RouteLLMRequestProcessor(_config(), classifier=_FakeClassifier(0.8))
    ctx = ProxyContext()

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_ROUTELLM_TIER] == "strong"
    assert ctx.selected_target == "strong"
    assert ctx.selected_model == "strong-model"


async def test_low_score_stamps_weak_target_for_multi_backend() -> None:
    processor = RouteLLMRequestProcessor(_config(), classifier=_FakeClassifier(0.1))
    ctx = ProxyContext()

    await processor.process(ctx, _request())

    assert ctx.metadata[CTX_ROUTELLM_TIER] == "weak"
    assert ctx.selected_target == "weak"
    assert ctx.selected_model == "weak-model"


async def test_multi_backend_rejects_unknown_selected_target_before_network() -> None:
    profile = RouteLLMProfileConfig.from_config(_config(enable_stats=False)).build()
    backend = next(component for component in profile.iter_components() if isinstance(component, MultiLlmBackend))
    ctx = ProxyContext()
    ctx.selected_target = "missing"

    with pytest.raises(RuntimeError, match="selected target missing is not configured"):
        await backend.call(ctx, _request())


def test_profile_configures_strong_default_target_for_missing_route_selection() -> None:
    profile = RouteLLMProfileConfig.from_config(_config(enable_stats=False)).build()
    backend = next(component for component in profile.iter_components() if isinstance(component, MultiLlmBackend))

    assert isinstance(backend, MultiLlmBackend)
    assert backend.default_target_id() == "strong"
