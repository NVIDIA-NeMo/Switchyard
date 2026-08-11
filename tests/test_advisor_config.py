# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`AdvisorConfig` validation, presets, and public exports."""

from __future__ import annotations

import pydantic
import pytest

import switchyard
from switchyard.lib.backends.advisor_config import AdvisorConfig
from switchyard.lib.backends.advisor_loop_backend import AdvisorLoopBackend
from switchyard.lib.backends.advisor_presets import AdvisorPresets
from switchyard.lib.backends.llm_target import BackendFormat


def _config(**overrides) -> AdvisorConfig:
    base: dict = {
        "executor": {"model": "exec-model", "base_url": "http://exec.test", "api_key": "k",
                     "format": "anthropic"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "anthropic"},
    }
    base.update(overrides)
    return AdvisorConfig(**base)


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

    def test_rejects_responses_format_on_either_tier(self) -> None:
        for tier in ("executor", "advisor"):
            with pytest.raises(pydantic.ValidationError, match="responses"):
                _config(**{tier: {"model": "m", "base_url": "http://t", "api_key": "k",
                                  "format": "responses"}})

    def test_redo_feedback_prefix_is_configurable(self) -> None:
        cfg = _config(redo_feedback_prefix="REVIEWER SAYS: ")
        assert cfg.redo_feedback_prefix == "REVIEWER SAYS: "

    def test_accepts_mixed_wire_tiers(self) -> None:
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

    def test_preset_model_overrides(self) -> None:
        cfg = AdvisorPresets.opus47_exec_opus48_advisor(
            api_key="k", executor_model="custom/exec", advisor_model="custom/adv",
        )
        assert cfg.executor.model == "custom/exec"
        assert cfg.advisor.model == "custom/adv"


def test_public_exports() -> None:
    assert switchyard.AdvisorConfig is AdvisorConfig
    assert switchyard.AdvisorLoopBackend is AdvisorLoopBackend
    assert switchyard.AdvisorPresets is AdvisorPresets
    for name in ("AdvisorConfig", "AdvisorLoopBackend", "AdvisorPresets"):
        assert name in switchyard.__all__
