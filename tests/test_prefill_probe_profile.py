# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile construction and validation for learned prefill-probe routing."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import switchyard
import switchyard_rust.components as rust_components
from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
    AnthropicCacheBreakpointBackend,
)
from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend
from switchyard.lib.profiles.prefill_probe_config import (
    PrefillProbeConfig,
    PrefillProbeRoutingPolicyConfig,
)
from switchyard.lib.profiles.prefill_probe_profile_config import (
    DEFAULT_PREFILL_PROBE_BASE_URL,
    PrefillProbeProfileConfig,
)
from switchyard_rust.components import BackendFormat


def test_native_processor_binding_is_exported() -> None:
    assert rust_components.PrefillProbeRequestProcessor.__name__ == (
        "PrefillProbeRequestProcessor"
    )


class _FakePrefillProbeRequestProcessor:
    instances: list[_FakePrefillProbeRequestProcessor] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


@pytest.fixture
def fake_native_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> type[_FakePrefillProbeRequestProcessor]:
    """Replace checkpoint loading while retaining exact profile arguments."""
    _FakePrefillProbeRequestProcessor.instances.clear()
    monkeypatch.setattr(
        rust_components,
        "PrefillProbeRequestProcessor",
        _FakePrefillProbeRequestProcessor,
        raising=False,
    )
    return _FakePrefillProbeRequestProcessor


def _config_data(
    *,
    strong_format: BackendFormat = BackendFormat.OPENAI,
) -> dict[str, Any]:
    return {
        "probe": {
            "id": "probe",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "base_url": "http://probe.invalid:8000/v1",
            "format": BackendFormat.OPENAI,
        },
        "strong": {
            "id": "capable",
            "model": "aws/anthropic/bedrock-claude-opus-4-7",
            "base_url": "https://completion.invalid/v1",
            "api_key": "test-key",
            "format": strong_format,
        },
        "weak": {
            "id": "efficient",
            "model": "nvidia/nemotron-3-super-120b-long-ctx",
            "base_url": "https://completion.invalid/v1",
            "api_key": "test-key",
            "format": BackendFormat.OPENAI,
        },
        "strong_checkpoint_head": "opus-4.7",
        "weak_checkpoint_head": "nemotron-3-super",
        "hidden_states_dir": "/tmp/switchyard-hidden-states",
        "checkpoint_dir": "/checkpoints/prefill-router",
        "routing_policy": {
            "type": "cost_aware",
            "lambda": 0.75,
            "weak_cost": 0.1,
            "strong_cost": 1.0,
        },
        "fallback_target_on_evict": "capable",
        "tier_timeout_s": 600.0,
        "enable_stats": True,
    }


def _built_profile(
    fake_native_processor: type[_FakePrefillProbeRequestProcessor],
    *,
    strong_format: BackendFormat = BackendFormat.OPENAI,
) -> tuple[PrefillProbeConfig, Any]:
    del fake_native_processor
    config = PrefillProbeConfig.model_validate(
        _config_data(strong_format=strong_format)
    )
    return config, PrefillProbeProfileConfig.from_config(config).build()


def _deterministic_backend(profile: Any) -> DeterministicRoutingLLMBackend:
    return next(
        component
        for component in profile.iter_components()
        if isinstance(component, DeterministicRoutingLLMBackend)
    )


def test_policy_uses_lambda_alias_with_explicit_semantics() -> None:
    policy = PrefillProbeRoutingPolicyConfig.model_validate({
        "type": "cost_aware",
        "lambda": 0.75,
        "weak_cost": 0.1,
        "strong_cost": 1.0,
    })

    assert policy.lambda_ == 0.75
    assert policy.model_dump(by_alias=True)["lambda"] == 0.75


def test_profile_passes_exact_probe_and_policy_configuration(
    fake_native_processor: type[_FakePrefillProbeRequestProcessor],
) -> None:
    config, profile = _built_profile(fake_native_processor)

    assert len(fake_native_processor.instances) == 1
    processor = fake_native_processor.instances[0]
    assert processor.kwargs == {
        "probe_base_url": "http://probe.invalid:8000/v1",
        "probe_model": "Qwen/Qwen3.6-35B-A3B",
        "hidden_states_dir": "/tmp/switchyard-hidden-states",
        "checkpoint_dir": "/checkpoints/prefill-router",
        "strong_checkpoint_head": "opus-4.7",
        "weak_checkpoint_head": "nemotron-3-super",
        "strong_target_id": "capable",
        "weak_target_id": "efficient",
        "routing_lambda": 0.75,
        "weak_cost": 0.1,
        "strong_cost": 1.0,
    }
    assert config.probe.model not in {config.strong.model, config.weak.model}
    assert profile.iter_components()[0] is processor


def test_missing_probe_base_url_uses_local_default(
    fake_native_processor: type[_FakePrefillProbeRequestProcessor],
) -> None:
    data = _config_data()
    data["probe"] = {
        "id": "probe",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "format": BackendFormat.OPENAI,
    }
    config = PrefillProbeConfig.model_validate(data)

    PrefillProbeProfileConfig.from_config(config).build()

    processor = fake_native_processor.instances[0]
    assert processor.kwargs["probe_base_url"] == DEFAULT_PREFILL_PROBE_BASE_URL


def test_profile_builds_configured_tiers_with_strong_default(
    fake_native_processor: type[_FakePrefillProbeRequestProcessor],
) -> None:
    config, profile = _built_profile(fake_native_processor)
    backend = _deterministic_backend(profile)

    assert set(backend._backends) == {"capable", "efficient"}
    assert backend._models == {
        "capable": config.strong.model,
        "efficient": config.weak.model,
    }
    assert backend._default_tier == "capable"
    assert profile._fallback_target_on_evict == "capable"


def test_anthropic_tier_keeps_cache_wrapper_outermost_with_stats(
    fake_native_processor: type[_FakePrefillProbeRequestProcessor],
) -> None:
    config, profile = _built_profile(
        fake_native_processor,
        strong_format=BackendFormat.ANTHROPIC,
    )
    profile = profile.with_runtime_components(
        enable_stats=config.enable_stats,
    )
    backend = _deterministic_backend(profile)

    strong_backend = backend._backends["capable"]
    assert isinstance(strong_backend, AnthropicCacheBreakpointBackend)
    assert isinstance(strong_backend._inner, StatsLlmBackend)
    assert isinstance(backend._backends["efficient"], StatsLlmBackend)


@pytest.mark.parametrize(
    ("policy", "error"),
    [
        ({"type": "cost_aware", "lambda": -0.1, "weak_cost": 0.0, "strong_cost": 1.0}, "lambda"),
        ({"type": "cost_aware", "lambda": 1.1, "weak_cost": 0.0, "strong_cost": 1.0}, "lambda"),
        ({"type": "cost_aware", "lambda": 0.5, "weak_cost": -1.0, "strong_cost": 1.0}, "weak_cost"),
        (
            {
                "type": "cost_aware",
                "lambda": 0.5,
                "weak_cost": 0.0,
                "strong_cost": float("inf"),
            },
            "strong_cost",
        ),
    ],
)
def test_invalid_policy_values_fail_validation(
    policy: dict[str, Any],
    error: str,
) -> None:
    data = _config_data()
    data["routing_policy"] = policy

    with pytest.raises(ValidationError, match=error):
        PrefillProbeConfig.model_validate(data)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        ("duplicate_target", "strong.id and weak.id must differ"),
        ("duplicate_head", "checkpoint_head"),
        ("invalid_fallback", "fallback_target_on_evict"),
        ("blank_checkpoint", "checkpoint_dir"),
        ("blank_hidden_states", "hidden_states_dir"),
        ("invalid_timeout", "tier_timeout_s"),
    ],
)
def test_invalid_profile_relationships_fail_validation(
    mutate: str,
    error: str,
) -> None:
    data = _config_data()
    if mutate == "duplicate_target":
        data["weak"] = {
            **data["weak"],
            "id": "capable",
        }
    elif mutate == "duplicate_head":
        data["weak_checkpoint_head"] = data["strong_checkpoint_head"]
    elif mutate == "invalid_fallback":
        data["fallback_target_on_evict"] = "probe"
    elif mutate == "blank_checkpoint":
        data["checkpoint_dir"] = " "
    elif mutate == "blank_hidden_states":
        data["hidden_states_dir"] = ""
    elif mutate == "invalid_timeout":
        data["tier_timeout_s"] = 0.0

    with pytest.raises(ValidationError, match=error):
        PrefillProbeConfig.model_validate(data)


def test_public_package_exports_profile_types() -> None:
    assert switchyard.PrefillProbeConfig is PrefillProbeConfig
    assert switchyard.PrefillProbeProfileConfig is PrefillProbeProfileConfig
    assert (
        switchyard.PrefillProbeRoutingPolicyConfig
        is PrefillProbeRoutingPolicyConfig
    )
