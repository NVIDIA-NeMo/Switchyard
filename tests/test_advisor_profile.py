# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wiring tests for the advisor profile, config, preset, and public exports."""

from __future__ import annotations

import pydantic
import pytest

import switchyard
from switchyard.lib.backends.advisor_loop_backend import AdvisorLoopBackend
from switchyard.lib.backends.advisor_tool_call_backend import AdvisorToolCallBackend
from switchyard.lib.backends.llm_target import BackendFormat
from switchyard.lib.processors.reasoning_effort_normalizer import (
    ReasoningEffortNormalizer,
)
from switchyard.lib.processors.stats_request_processor import StatsRequestProcessor
from switchyard.lib.processors.stats_response_processor_accumulator import (
    StatsResponseProcessor,
)
from switchyard.lib.profiles import (
    AdvisorConfig,
    AdvisorPresets,
    AdvisorProfileConfig,
    ProfileSwitchyard,
)
from switchyard.lib.profiles.table import profile_config_type
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequestType


def _config(**overrides) -> AdvisorConfig:
    # Formats pinned: an omitted format coerces to openai, which the
    # review_gate validator rejects for the executor tier.
    base: dict = {
        "executor": {"model": "exec-model", "base_url": "http://exec.test", "api_key": "k",
                     "format": "anthropic"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "anthropic"},
    }
    base.update(overrides)
    return AdvisorConfig(**base)


def _openai_config(**overrides) -> AdvisorConfig:
    base: dict = {
        "executor": {"model": "qwen/qwen3-max", "base_url": "http://exec.test",
                     "api_key": "k", "format": "openai"},
        "advisor": {"model": "deepseek/deepseek-r2", "base_url": "http://adv.test",
                    "api_key": "k", "format": "openai"},
    }
    base.update(overrides)
    return AdvisorConfig(**base)


def _advisor_switchyard(
    config: AdvisorConfig,
    *,
    stats_accumulator: StatsAccumulator | None = None,
) -> ProfileSwitchyard:
    """Build the profile-backed runtime used by these tests."""
    return ProfileSwitchyard(
        AdvisorProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats_accumulator,
            enable_stats=config.enable_stats,
        )
    )


class TestProfileStructure:
    def test_registered_profile_type_is_advisor(self) -> None:
        assert profile_config_type(AdvisorProfileConfig) == "advisor"

    def test_returns_profile_backed_switchyard_adapter(self) -> None:
        assert isinstance(_advisor_switchyard(_config()), ProfileSwitchyard)

    def test_backend_defaults_to_tool_call(self) -> None:
        cfg = _config()
        assert cfg.strategy == "tool_call"
        components = list(_advisor_switchyard(cfg).iter_components())
        assert any(isinstance(c, AdvisorToolCallBackend) for c in components)

    def test_review_gate_strategy_builds_loop_backend(self) -> None:
        components = list(
            _advisor_switchyard(_config(strategy="review_gate")).iter_components()
        )
        assert any(isinstance(c, AdvisorLoopBackend) for c in components)

    def test_reasoning_effort_normalizer_present(self) -> None:
        components = list(_advisor_switchyard(_config()).iter_components())
        assert any(isinstance(c, ReasoningEffortNormalizer) for c in components)

    def test_stats_processors_wired_when_enabled(self) -> None:
        components = list(_advisor_switchyard(_config()).iter_components())
        assert any(isinstance(c, StatsRequestProcessor) for c in components)
        assert any(isinstance(c, StatsResponseProcessor) for c in components)

    def test_stats_processors_absent_when_disabled(self) -> None:
        components = list(
            _advisor_switchyard(_config(enable_stats=False)).iter_components()
        )
        assert not any(isinstance(c, StatsRequestProcessor) for c in components)
        assert not any(isinstance(c, StatsResponseProcessor) for c in components)

    def test_runtime_attaches_accumulator_to_backend(self) -> None:
        # The advisor backends do their own stats accounting (like
        # LatencyServiceLLMBackend) and cannot be wrapped by StatsLlmBackend;
        # with_runtime_components must attach the shared accumulator through
        # the ``_stats`` compatibility hook instead.
        stats = StatsAccumulator()
        switchyard_adapter = _advisor_switchyard(_config(), stats_accumulator=stats)
        backend = next(
            c for c in switchyard_adapter.iter_components()
            if isinstance(c, AdvisorToolCallBackend)
        )
        assert backend._stats is stats

    def test_openai_tiers_build_tool_call_backend_on_openai_wire(self) -> None:
        backend = next(
            c for c in _advisor_switchyard(_openai_config()).iter_components()
            if isinstance(c, AdvisorToolCallBackend)
        )
        assert backend.supported_request_types == [ChatRequestType.OPENAI_CHAT]

    def test_anthropic_executor_advertises_anthropic_wire(self) -> None:
        backend = next(
            c for c in _advisor_switchyard(_config()).iter_components()
            if isinstance(c, AdvisorToolCallBackend)
        )
        assert backend.supported_request_types == [ChatRequestType.ANTHROPIC]


class TestAdvisorConfig:
    def test_coerces_dict_targets(self) -> None:
        cfg = _config()
        assert cfg.executor.model == "exec-model"
        assert cfg.advisor.model == "adv-model"

    def test_rejects_empty_target_model(self) -> None:
        # The Rust-backed LlmTarget rejects empty model ids during coercion,
        # before the config-level non-empty validator can fire.
        with pytest.raises(pydantic.ValidationError, match="must not be empty"):
            _config(executor={"model": "", "base_url": "http://e", "api_key": "k"})

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _config(strategy="both")

    def test_rejects_responses_format_on_either_tier(self) -> None:
        for tier in ("executor", "advisor"):
            with pytest.raises(pydantic.ValidationError, match="responses"):
                _config(**{tier: {"model": "m", "base_url": "http://t", "api_key": "k",
                                  "format": "responses"}})

    def test_review_gate_rejects_openai_executor(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="review_gate"):
            _openai_config(strategy="review_gate")

    def test_tool_call_accepts_mixed_and_openai_tiers(self) -> None:
        assert _openai_config().strategy == "tool_call"
        mixed = _config(advisor={"model": "deepseek/deepseek-r2", "base_url": "http://adv.test",
                                 "api_key": "k", "format": "openai"})
        assert mixed.advisor.format == BackendFormat.OPENAI
        assert mixed.executor.format == BackendFormat.ANTHROPIC


class TestOpusPairPreset:
    """Pins the validated executor+advisor pairing on the shipping default."""

    def test_preset_pairs_opus_47_and_48(self) -> None:
        cfg = AdvisorPresets.opus47_exec_opus48_advisor(api_key="nvapi-test")
        assert cfg.executor.model == "aws/anthropic/bedrock-claude-opus-4-7"
        assert cfg.advisor.model == "aws/anthropic/bedrock-claude-opus-4-8"
        assert cfg.preset == "opus47_exec_opus48_advisor"
        assert cfg.executor.endpoint.base_url == "https://inference-api.nvidia.com/v1"
        # Both tiers are native Anthropic-Messages (no OpenAI translation →
        # caching survives). The executor suppresses x-api-key and
        # authenticates via Bearer.
        assert cfg.executor.format == BackendFormat.ANTHROPIC
        assert cfg.advisor.format == BackendFormat.ANTHROPIC
        assert cfg.executor.endpoint.api_key == ""
        assert cfg.executor.extra_headers == {"Authorization": "Bearer nvapi-test"}

    def test_preset_defaults_to_tool_call_strategy(self) -> None:
        cfg = AdvisorPresets.opus47_exec_opus48_advisor(api_key="k")
        assert cfg.strategy == "tool_call"
        gate = AdvisorPresets.opus47_exec_opus48_advisor(
            api_key="k", strategy="review_gate",
        )
        assert gate.strategy == "review_gate"

    def test_preset_model_overrides(self) -> None:
        cfg = AdvisorPresets.opus47_exec_opus48_advisor(
            api_key="k", executor_model="custom/exec", advisor_model="custom/adv",
        )
        assert cfg.executor.model == "custom/exec"
        assert cfg.advisor.model == "custom/adv"


def test_public_exports() -> None:
    assert switchyard.AdvisorConfig is AdvisorConfig
    assert switchyard.AdvisorPresets is AdvisorPresets
    assert switchyard.AdvisorProfileConfig is AdvisorProfileConfig
    for name in ("AdvisorConfig", "AdvisorPresets", "AdvisorProfileConfig"):
        assert name in switchyard.__all__
