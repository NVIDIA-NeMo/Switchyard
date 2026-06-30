# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profile-backed plan-execute construction."""

from __future__ import annotations

from typing import Any

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.plan_execute import PlanningRequestProcessor
from switchyard.lib.profiles import (
    PlanExecuteConfig,
    PlanExecutePresets,
    PlanExecuteProfileConfig,
    ProfileSwitchyard,
)
from switchyard.lib.profiles.plan_execute import _ExecutorStatsTargetProcessor
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest, ChatResponse
from switchyard_rust.translation import TranslationEngine


def _config(
    *,
    enable_stats: bool = True,
    planner_model: str = "anthropic/claude-opus-4.6",
) -> PlanExecuteConfig:
    return PlanExecuteConfig(
        planner=LlmTarget(
            id="planner",
            model=planner_model,
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        executor=LlmTarget(
            id="executor",
            model="moonshotai/kimi-k2.6",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        cadence_n=2,
        enable_stats=enable_stats,
        fallback_target_on_evict="planner",
    )


def _plan_execute_switchyard(
    config: PlanExecuteConfig,
    *,
    pre_routing_request_processors: list[Any] | None = None,
    extra_request_processors: list[Any] | None = None,
    extra_response_processors: list[Any] | None = None,
) -> ProfileSwitchyard:
    """Build the profile-backed runtime used by these tests."""
    return ProfileSwitchyard(
        PlanExecuteProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            pre_request_processors=pre_routing_request_processors or (),
            post_request_processors=extra_request_processors or (),
            response_processors=extra_response_processors or (),
        )
    )


def _planner_config(switchyard: ProfileSwitchyard) -> object:
    """Pull the built :class:`PlanningConfig` out of the wired planner."""
    planner = next(
        c for c in switchyard.iter_components()
        if isinstance(c, PlanningRequestProcessor)
    )
    return planner._config


class _NoopRequestProcessor:
    async def process(self, _ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        return request


class _NoopResponseProcessor:
    async def process(self, _ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        return response


class TestProfileStructure:
    def test_returns_profile_backed_switchyard_adapter(self) -> None:
        switchyard = _plan_execute_switchyard(_config())
        assert isinstance(switchyard, ProfileSwitchyard)

    def test_planner_processor_present(self) -> None:
        switchyard = _plan_execute_switchyard(_config())
        planners = [
            c for c in switchyard.iter_components()
            if isinstance(c, PlanningRequestProcessor)
        ]
        assert len(planners) == 1

    def test_translator_present(self) -> None:
        switchyard = _plan_execute_switchyard(_config())
        translators = [
            c for c in switchyard.iter_components()
            if isinstance(c, TranslationEngine)
        ]
        assert len(translators) == 1

    def test_executor_stats_marker_runs_before_planner(self) -> None:
        switchyard = _plan_execute_switchyard(_config())
        components = list(switchyard.iter_components())
        marker_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, _ExecutorStatsTargetProcessor)
        )
        planner_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, PlanningRequestProcessor)
        )
        assert marker_idx < planner_idx

    def test_extra_processors_are_wired(self) -> None:
        request_processor = _NoopRequestProcessor()
        response_processor = _NoopResponseProcessor()
        switchyard = _plan_execute_switchyard(
            _config(),
            extra_request_processors=[request_processor],
            extra_response_processors=[response_processor],
        )
        components = list(switchyard.iter_components())
        assert request_processor in components
        assert response_processor in components

    def test_pre_routing_runs_before_planner(self) -> None:
        pre = _NoopRequestProcessor()
        switchyard = _plan_execute_switchyard(
            _config(),
            pre_routing_request_processors=[pre],
        )
        components = list(switchyard.iter_components())
        pre_idx = components.index(pre)
        planner_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, PlanningRequestProcessor)
        )
        assert pre_idx < planner_idx

    async def test_executor_stats_marker_stamps_executor_tier(self) -> None:
        marker = _ExecutorStatsTargetProcessor("executor-target")
        ctx = ProxyContext()
        request = ChatRequest.openai_chat({
            "model": "client-route",
            "messages": [{"role": "user", "content": "hi"}],
        })

        await marker.process(ctx, request)

        # The marker stamps the executor target so downstream OTel tier
        # resolution attributes the call to the executor tier.
        assert ctx.selected_target == "executor-target"


class TestPlannerParamCompatibility:
    """Anthropic-family planners must drop ``temperature`` (Bedrock Claude
    4.7 400s on it) but *keep* ``response_format={"type": "json_object"}``:
    LiteLLM turns it into a forced tool call, and the planner reads the JSON
    back from ``tool_calls`` (see ``_completion_content``)."""

    def test_anthropic_planner_drops_temperature_keeps_response_format(self) -> None:
        switchyard = _plan_execute_switchyard(
            _config(planner_model="aws/anthropic/bedrock-claude-opus-4-7"),
        )
        config = _planner_config(switchyard)
        assert config.temperature is None
        assert config.response_format == {"type": "json_object"}

    def test_non_anthropic_planner_keeps_defaults(self) -> None:
        switchyard = _plan_execute_switchyard(
            _config(planner_model="nvidia/deepseek/v4-pro"),
        )
        config = _planner_config(switchyard)
        assert config.temperature == 0.0
        assert config.response_format == {"type": "json_object"}


class TestCodingAgentDefaultPreset:
    """Pins the validated planner+executor pairing on the shipping default."""

    def test_planner_is_strong_model(self) -> None:
        preset = PlanExecutePresets.coding_agent_default(api_key="sk-test")
        assert preset.planner.model == "anthropic/claude-opus-4.6"

    def test_executor_is_weak_model(self) -> None:
        preset = PlanExecutePresets.coding_agent_default(api_key="sk-test")
        # Weak executor — OpenRouter-available Kimi.
        assert preset.executor.model == "moonshotai/kimi-k2.6"

    def test_cadence_n_validated_default(self) -> None:
        preset = PlanExecutePresets.coding_agent_default(api_key="sk-test")
        # Validated cadence per commit c9339748: 2 beat both 1 (too noisy)
        # and 4 (too sparse) on the TB-Lite Nemotron-Nano sweep.
        assert preset.cadence_n == 2

    def test_preset_label_stamped(self) -> None:
        preset = PlanExecutePresets.coding_agent_default(api_key="sk-test")
        assert preset.preset == "coding_agent_default"

    def test_credentials_propagate_to_both_tiers(self) -> None:
        preset = PlanExecutePresets.coding_agent_default(
            api_key="sk-secret",
            base_url="https://custom.invalid/v1",
        )
        assert preset.planner.endpoint.api_key == "sk-secret"
        assert preset.executor.endpoint.api_key == "sk-secret"
        assert preset.planner.endpoint.base_url == "https://custom.invalid/v1"
        assert preset.executor.endpoint.base_url == "https://custom.invalid/v1"
