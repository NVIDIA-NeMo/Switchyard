# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live evict-and-reroute coverage for every multi-target router.

Each test builds the router's recipe in-process against NVIDIA Inference
Hub, forces the weak target (a model with a known 131k context cap) to
overflow with a ~840k-token prompt, and asserts that:

* the inbound request still returns HTTP 200 (the compatibility chain's
  fallback path served the response on the strong target),
* the response model is the strong target's model id, and
* the shared :class:`StatsAccumulator` records the failing target as
  ``errors=1, calls=0`` and the fallback target as ``calls=1`` with the
  full prompt-token attribution.

This pins the four moving parts that must all line up for the feature
to work end-to-end on a live backend — provider-shape 400 detection,
the compatibility chain's retry, the Python ↔ Rust exception surface,
and per-target stats attribution.

Prerequisites:
    - ``NVIDIA_API_KEY`` env var (set; tests skip otherwise).

Run with::

    NVIDIA_API_KEY=nvapi-... uv run pytest tests/e2e/test_evict_reroute_e2e.py -v
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.server.switchyard_app import build_switchyard_app

_STRONG_MODEL = "aws/anthropic/bedrock-claude-opus-4-7"
_WEAK_MODEL = "nvidia/nvidia/nemotron-3-super-120b-long-ctx"
_BASE_URL = "https://inference-api.nvidia.com/v1"
# ~840k tokens — well beyond Nemotron-long-ctx's 131k cap, well within
# Opus 4.7's window, so the eviction is guaranteed and the fallback
# call actually succeeds.
_OVERFLOW_TEXT = "lorem ipsum dolor sit amet " * 60_000


@pytest.fixture
def nvidia_api_key() -> str:
    # Live tests against NVIDIA Inference Hub — skip in CI without a real key.
    # OPENAI_API_KEY is intentionally NOT a fallback; CI sets it to a dummy
    # value for the passthrough e2e tests and the dummy would 401 here.
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        pytest.skip("NVIDIA_API_KEY not set — required for live evict-reroute e2e")
    return key


def _strong_target(api_key: str) -> LlmTarget:
    return LlmTarget(
        id="strong",
        model=_STRONG_MODEL,
        format=BackendFormat.ANTHROPIC,
        api_key=api_key,
        base_url=_BASE_URL,
    )


def _weak_target(api_key: str) -> LlmTarget:
    return LlmTarget(
        id="weak",
        model=_WEAK_MODEL,
        format=BackendFormat.OPENAI,
        api_key=api_key,
        base_url=_BASE_URL,
    )


async def _client_for(app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=600.0,
    ) as client:
        yield client


def _assert_eviction_stats(snapshot: dict) -> None:
    """Pin the per-target attribution shape after a successful reroute."""
    weak = snapshot["models"][_WEAK_MODEL]
    strong = snapshot["models"][_STRONG_MODEL]
    assert weak["errors"] == 1
    assert weak["calls"] == 0
    assert strong["calls"] == 1
    assert strong["errors"] == 0
    assert strong["prompt_tokens"] > 100_000


@pytest.mark.timeout(120)
async def test_stage_router_evicts_weak_and_reroutes_to_strong(nvidia_api_key: str) -> None:
    """``type: stage_router`` reroutes to ``strong`` when weak overflows."""
    from switchyard.lib.profiles import (
        ProfileSwitchyard,
        StageRouterConfig,
        StageRouterProfileConfig,
    )

    stats = StatsAccumulator()
    config = StageRouterConfig(
        capable=_strong_target(nvidia_api_key),
        efficient=_weak_target(nvidia_api_key),
        picker="efficient_first",
        confidence_threshold=0.0,
        fallback_target_on_evict="strong",
    )
    switchyard = ProfileSwitchyard(
        StageRouterProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=config.enable_stats,
        )
    )
    app = build_switchyard_app(switchyard)

    async for client in _client_for(app):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stage_router-test",
                "messages": [{"role": "user", "content": _OVERFLOW_TEXT}],
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == _STRONG_MODEL
    _assert_eviction_stats(await stats.snapshot())


@pytest.mark.timeout(120)
async def test_random_routing_evicts_weak_and_reroutes_to_strong(
    nvidia_api_key: str,
) -> None:
    """``type: random_routing`` reroutes when the coin-picked weak overflows.

    ``strong_probability=0.0`` pins the first pick on weak so the test is
    deterministic; the compatibility chain then rewrites to ``strong``.
    """
    from switchyard.lib.profiles import (
        ProfileSwitchyard,
        RandomRoutingConfig,
        RandomRoutingProfileConfig,
    )

    stats = StatsAccumulator()
    config = RandomRoutingConfig(
        strong=_strong_target(nvidia_api_key),
        weak=_weak_target(nvidia_api_key),
        strong_probability=0.0,
        fallback_target_on_evict="strong",
        rng_seed=1,
    )
    switchyard = ProfileSwitchyard(
        RandomRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=config.enable_stats,
        )
    )
    app = build_switchyard_app(switchyard)

    async for client in _client_for(app):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "random-test",
                "messages": [{"role": "user", "content": _OVERFLOW_TEXT}],
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == _STRONG_MODEL
    _assert_eviction_stats(await stats.snapshot())


@pytest.mark.timeout(180)
async def test_deterministic_evicts_weak_and_reroutes_to_strong(
    nvidia_api_key: str,
) -> None:
    """``type: deterministic`` reroutes when the classifier-picked weak overflows.

    Validates the Python-LLMBackend exception surface: the
    inner native Rust backend raises ``ContextWindowExceeded``, the
    Python wrapper (``DeterministicRoutingLLMBackend``) propagates the
    typed exception, and the compatibility chain catches the preserved
    variant to trigger the retry.
    """
    from switchyard.lib.profiles import (
        DeterministicRoutingConfig,
        DeterministicRoutingProfileConfig,
        ProfileSwitchyard,
    )

    stats = StatsAccumulator()
    classifier = LlmTarget(
        id="classifier",
        model="nvidia/deepseek-ai/deepseek-v4-flash",
        format=BackendFormat.OPENAI,
        api_key=nvidia_api_key,
        base_url=_BASE_URL,
    )
    config = DeterministicRoutingConfig(
        strong=_strong_target(nvidia_api_key),
        weak=_weak_target(nvidia_api_key),
        classifier=classifier,
        profile_name="coding_agent",
        classifier_min_confidence=0.0,
        classifier_fail_open=True,
        fallback_target_on_evict="strong",
    )
    switchyard = ProfileSwitchyard(
        DeterministicRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats,
            enable_stats=config.enable_stats,
        )
    )
    app = build_switchyard_app(switchyard)

    async for client in _client_for(app):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deterministic-test",
                "messages": [{"role": "user", "content": _OVERFLOW_TEXT}],
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == _STRONG_MODEL
    _assert_eviction_stats(await stats.snapshot())
