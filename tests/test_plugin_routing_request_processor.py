# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`PluginRoutingRequestProcessor`.

The processor is unit-tested with a hand-rolled fake :class:`PluginClient`
so we can exercise the routing-policy logic — fallbacks, error handling,
metadata stamping — without spawning real subprocesses on every assertion.
``test_plugin_client.py`` covers the real-subprocess path end-to-end.
"""

from __future__ import annotations

import pytest

from switchyard.lib.plugin import (
    PluginCrashError,
    PluginRoutingError,
    PluginTimeoutError,
    RouteDecision,
    RouteError,
    RouteRequest,
    RouteResult,
)
from switchyard.lib.processors.plugin_routing_request_processor import (
    CTX_OSS_ROUTER_TIER,
    PluginRoutingRequestProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest


class _FakePluginClient:
    """Drop-in stand-in for :class:`PluginClient`.

    ``script`` is a list of canned responses (decisions or exceptions)
    consumed in order by successive :meth:`route` calls. Lets us script
    timeouts, crashes, and decisions deterministically.
    """

    def __init__(self, script: list[RouteResult | Exception]) -> None:
        self.script = list(script)
        self.calls: list[RouteRequest] = []

    async def route(self, request: RouteRequest) -> RouteResult:
        self.calls.append(request)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _processor(
    *,
    fallback_tier: str | None = None,
    tier_models: dict[str, str] | None = None,
) -> PluginRoutingRequestProcessor:
    proc = PluginRoutingRequestProcessor(
        plugin_command=["true"],
        tier_models=tier_models or {"strong": "openai/gpt-5", "weak": "openai/gpt-5-mini"},
        fallback_tier=fallback_tier,
    )
    return proc


def _request() -> ChatRequest:
    return ChatRequest.openai_chat(body={
        "model": "ignored-by-processor",
        "messages": [{"role": "user", "content": "hello"}],
    })


def _ctx() -> ProxyContext:
    return ProxyContext()


def _bind_fake(proc: PluginRoutingRequestProcessor, fake: _FakePluginClient) -> None:
    """Patch the private slot directly — startup() spawns a real subprocess."""
    proc._client = fake  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_decision_rewrites_model_and_stamps_metadata() -> None:
    proc = _processor()
    fake = _FakePluginClient([RouteDecision(tier="weak", metadata={"score": 0.4})])
    _bind_fake(proc, fake)

    ctx = _ctx()
    request = _request()
    result = await proc.process(ctx, request)

    assert result.body["model"] == "openai/gpt-5-mini"
    assert ctx.metadata[CTX_OSS_ROUTER_TIER] == "weak"
    assert ctx.metadata["_oss_router_model"] == "openai/gpt-5-mini"
    assert ctx.metadata["_oss_router_plugin_metadata"] == {"score": 0.4}
    # Verify the request summary the plugin received was sane.
    assert fake.calls[0].available_tiers == ("strong", "weak")
    assert fake.calls[0].summary.message_count == 1


# ---------------------------------------------------------------------------
# Plugin error envelope
# ---------------------------------------------------------------------------


async def test_route_error_with_fallback_dispatches_to_plugin_hint() -> None:
    proc = _processor(fallback_tier="strong")
    fake = _FakePluginClient([
        RouteError(code=-32000, message="model dead", fallback_tier="weak"),
    ])
    _bind_fake(proc, fake)

    ctx = _ctx()
    result = await proc.process(ctx, _request())

    # Plugin hinted "weak"; that's a known tier so it wins over the
    # operator's strong-fallback default.
    assert result.body["model"] == "openai/gpt-5-mini"
    assert ctx.metadata[CTX_OSS_ROUTER_TIER] == "weak"


async def test_route_error_unknown_fallback_falls_back_to_operator_config() -> None:
    proc = _processor(fallback_tier="strong")
    fake = _FakePluginClient([
        RouteError(code=-32000, message="model dead", fallback_tier="ULTRA"),
    ])
    _bind_fake(proc, fake)

    ctx = _ctx()
    result = await proc.process(ctx, _request())

    # Plugin's "ULTRA" hint isn't a registered tier, so the operator's
    # strong fallback wins.
    assert result.body["model"] == "openai/gpt-5"
    assert ctx.metadata[CTX_OSS_ROUTER_TIER] == "strong"


async def test_route_error_no_fallback_raises() -> None:
    proc = _processor(fallback_tier=None)
    fake = _FakePluginClient([RouteError(code=-32000, message="boom")])
    _bind_fake(proc, fake)

    with pytest.raises(PluginRoutingError):
        await proc.process(_ctx(), _request())


# ---------------------------------------------------------------------------
# Plugin transport failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [
    PluginTimeoutError("timed out"),
    PluginCrashError("crashed"),
    PluginRoutingError("malformed"),
])
async def test_transport_failure_falls_back_when_configured(exc: Exception) -> None:
    proc = _processor(fallback_tier="weak")
    _bind_fake(proc, _FakePluginClient([exc]))

    ctx = _ctx()
    result = await proc.process(ctx, _request())

    assert result.body["model"] == "openai/gpt-5-mini"
    assert ctx.metadata[CTX_OSS_ROUTER_TIER] == "weak"
    metadata = ctx.metadata["_oss_router_plugin_metadata"]
    assert metadata["fallback_exception"] == type(exc).__name__


@pytest.mark.parametrize("exc", [
    PluginTimeoutError("timed out"),
    PluginCrashError("crashed"),
    PluginRoutingError("malformed"),
])
async def test_transport_failure_reraises_without_fallback(exc: Exception) -> None:
    proc = _processor(fallback_tier=None)
    _bind_fake(proc, _FakePluginClient([exc]))

    with pytest.raises(type(exc)):
        await proc.process(_ctx(), _request())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


async def test_processor_requires_startup_before_first_call() -> None:
    proc = _processor()
    with pytest.raises(RuntimeError, match="startup"):
        await proc.process(_ctx(), _request())


def test_processor_rejects_empty_tier_models() -> None:
    with pytest.raises(ValueError, match="at least one tier"):
        PluginRoutingRequestProcessor(plugin_command=["true"], tier_models={})


def test_processor_rejects_unknown_fallback_tier() -> None:
    with pytest.raises(ValueError, match="not in tier_models"):
        PluginRoutingRequestProcessor(
            plugin_command=["true"],
            tier_models={"strong": "m"},
            fallback_tier="weak",
        )


# ---------------------------------------------------------------------------
# Request summary content
# ---------------------------------------------------------------------------


async def test_request_summary_extracts_tools_from_body() -> None:
    proc = _processor()
    fake = _FakePluginClient([RouteDecision(tier="strong", metadata={})])
    _bind_fake(proc, fake)

    request = ChatRequest.openai_chat(body={
        "model": "ignored",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {"name": "bash"}},
            {"type": "function", "function": {"name": "edit"}},
        ],
    })
    await proc.process(_ctx(), request)

    summary = fake.calls[0].summary
    assert summary.has_tool_use
    assert summary.tool_names == ("bash", "edit")


async def test_metadata_exposed_keys_only() -> None:
    proc = PluginRoutingRequestProcessor(
        plugin_command=["true"],
        tier_models={"strong": "m1", "weak": "m2"},
        expose_metadata_keys=("safe_key",),
    )
    fake = _FakePluginClient([RouteDecision(tier="strong", metadata={})])
    _bind_fake(proc, fake)

    ctx = _ctx()
    ctx.metadata["safe_key"] = "value-1"
    ctx.metadata["secret_key"] = "should-not-leak"
    await proc.process(ctx, _request())

    forwarded = fake.calls[0].metadata
    assert forwarded == {"safe_key": "value-1"}


# ---------------------------------------------------------------------------
# Profile wiring
# ---------------------------------------------------------------------------


def test_oss_router_recipe_builds_full_chain() -> None:
    from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
    from switchyard.lib.profiles import OSSRouterProfileConfig, ProfileSwitchyard
    from switchyard.lib.profiles.oss_router import OSSRouterConfig, OSSRouterTier
    from switchyard_rust.translation import TranslationEngine

    config = OSSRouterConfig(
        plugin_command=["python", "-u", "/dev/null"],  # never spawned by build_*
        tiers=(
            OSSRouterTier(label="strong", tier=LlmTarget(
                id="strong",
                model="openai/test-strong",
                format=BackendFormat.OPENAI,
                api_key="k",
                base_url="https://example/v1",
            )),
            OSSRouterTier(label="weak", tier=LlmTarget(
                id="weak",
                model="openai/test-weak",
                format=BackendFormat.OPENAI,
                api_key="k",
                base_url="https://example/v1",
            )),
        ),
        fallback_tier="weak",
        fallback_target_on_evict="strong",
    )
    switchyard = ProfileSwitchyard(
        OSSRouterProfileConfig.from_config(config).build()
    )

    # Runtime components are populated; the plugin hasn't actually been
    # spawned yet (startup() hasn't been awaited).
    components = switchyard.iter_components()
    assert any(isinstance(c, PluginRoutingRequestProcessor) for c in components)
    assert any(isinstance(c, TranslationEngine) for c in components)


def test_oss_router_config_rejects_duplicate_tier_labels() -> None:
    from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
    from switchyard.lib.profiles.oss_router import OSSRouterConfig, OSSRouterTier

    tier = OSSRouterTier(label="strong", tier=LlmTarget(
        id="strong", model="m", format=BackendFormat.OPENAI, api_key="k", base_url="u",
    ))
    with pytest.raises(ValueError, match="unique"):
        OSSRouterConfig(
            plugin_command=["true"],
            tiers=(tier, tier),
            fallback_target_on_evict="strong",
        )


def test_oss_router_config_rejects_unknown_fallback_tier() -> None:
    from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
    from switchyard.lib.profiles.oss_router import OSSRouterConfig, OSSRouterTier

    tier = OSSRouterTier(label="strong", tier=LlmTarget(
        id="strong", model="m", format=BackendFormat.OPENAI, api_key="k", base_url="u",
    ))
    with pytest.raises(ValueError, match="fallback"):
        OSSRouterConfig(
            plugin_command=["true"],
            tiers=(tier,),
            fallback_tier="weak",
            fallback_target_on_evict="strong",
        )
